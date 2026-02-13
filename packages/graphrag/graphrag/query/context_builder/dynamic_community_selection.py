# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Algorithm to dynamically select relevant communities with respect to a query."""

import asyncio
import logging
from collections import Counter
from copy import deepcopy
from time import time
from typing import TYPE_CHECKING, Any

from graphrag_llm.tokenizer import Tokenizer

from graphrag.data_model.community import Community
from graphrag.data_model.community_report import CommunityReport
from graphrag.index.tracing import get_trace_context
from graphrag.query.context_builder.rate_prompt import RATE_QUERY
from graphrag.query.context_builder.rate_relevancy import rate_relevancy

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion

logger = logging.getLogger(__name__)


class DynamicCommunitySelection:
    """Dynamic community selection to select community reports that are relevant to the query.

    Any community report with a rating EQUAL or ABOVE the rating_threshold is considered relevant.
    """

    def __init__(
        self,
        community_reports: list[CommunityReport],
        communities: list[Community],
        model: "LLMCompletion",
        tokenizer: Tokenizer,
        rate_query: str = RATE_QUERY,
        use_summary: bool = False,
        threshold: int = 1,
        keep_parent: bool = False,
        num_repeats: int = 1,
        max_level: int = 2,
        concurrent_coroutines: int = 8,
        timeout_sec: int = 0,
        model_params: dict[str, Any] | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.rate_query = rate_query
        self.num_repeats = num_repeats
        self.use_summary = use_summary
        self.threshold = threshold
        self.keep_parent = keep_parent
        self.max_level = max_level
        self.timeout_sec = timeout_sec
        self.semaphore = asyncio.Semaphore(concurrent_coroutines)
        self.model_params = model_params if model_params else {}

        self.reports = {report.community_id: report for report in community_reports}
        self.communities = {community.short_id: community for community in communities}

        # mapping from level to communities
        self.levels: dict[str, list[str]] = {}

        for community in communities:
            if community.level not in self.levels:
                self.levels[community.level] = []
            if community.short_id in self.reports:
                self.levels[community.level].append(community.short_id)

        # start from root communities (level 0)
        self.starting_communities = self.levels["0"]

    async def select(self, query: str) -> tuple[list[CommunityReport], dict[str, Any]]:
        """
        Select relevant communities with respect to the query.

        Args:
            query: the query to rate against
        """
        start = time()
        queue = deepcopy(self.starting_communities)
        level = 0

        ratings = {}  # store the ratings for each community
        llm_info: dict[str, Any] = {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "output_tokens": 0,
        }
        relevant_communities = set()
        selection_timed_out = False

        # Create Langfuse span for dynamic community selection
        trace_ctx = get_trace_context()
        dcs_span = None
        if trace_ctx and trace_ctx.should_trace:
            try:
                dcs_span = trace_ctx.create_span(
                    name="dynamic_community_selection",
                    input={"query": query, "starting_communities": len(queue)},
                    metadata={"threshold": self.threshold, "max_level": self.max_level},
                )
            except Exception:
                logger.debug("Failed to create Langfuse span for dynamic community selection")

        while queue:
            # Create per-level span
            level_span = None
            if dcs_span is not None:
                try:
                    level_span = dcs_span.start_span(
                        name=f"level_{level}",
                        input={"num_communities": len(queue), "communities": queue[:20]},
                    )
                except Exception:
                    logger.debug("Failed to create Langfuse span for level %d", level)

            gather_tasks = [
                asyncio.create_task(
                    rate_relevancy(
                        query=query,
                        description=(
                            self.reports[community].summary
                            if self.use_summary
                            else self.reports[community].full_content
                        ),
                        model=self.model,
                        tokenizer=self.tokenizer,
                        rate_query=self.rate_query,
                        num_repeats=self.num_repeats,
                        semaphore=self.semaphore,
                        community_id=community,
                        parent_span=level_span,
                        **self.model_params,
                    )
                )
                for community in queue
            ]

            # Use asyncio.wait with remaining time budget so timeout works mid-level
            remaining_timeout = None
            timed_out = False
            if self.timeout_sec > 0:
                elapsed = time() - start
                remaining_timeout = max(0, self.timeout_sec - elapsed)
                if remaining_timeout <= 0:
                    # Already timed out, cancel all tasks
                    for task in gather_tasks:
                        task.cancel()
                    timed_out = True

            if not timed_out:
                done, pending = await asyncio.wait(gather_tasks, timeout=remaining_timeout)
                if pending:
                    timed_out = True
                    logger.warning(
                        "Dynamic community selection timeout (%ds) during level %d: "
                        "%d/%d communities rated, cancelling %d pending.",
                        self.timeout_sec,
                        level,
                        len(done),
                        len(gather_tasks),
                        len(pending),
                    )
                    for task in pending:
                        task.cancel()
            else:
                done = set()

            # Collect results: match completed tasks back to their communities
            completed_results = {}
            for task, community in zip(gather_tasks, queue, strict=True):
                if task in done and not task.cancelled():
                    completed_results[community] = task.result()

            communities_to_rate = []
            for community in queue:
                if community not in completed_results:
                    continue
                result = completed_results[community]
                rating = result["rating"]
                logger.debug(
                    "dynamic community selection: community %s rating %s",
                    community,
                    rating,
                )
                ratings[community] = rating
                llm_info["llm_calls"] += result["llm_calls"]
                llm_info["prompt_tokens"] += result["prompt_tokens"]
                llm_info["output_tokens"] += result["output_tokens"]
                if rating >= self.threshold:
                    relevant_communities.add(community)
                    # find children nodes of the current node and append them to the queue
                    # TODO check why some sub_communities are NOT in report_df
                    if community in self.communities:
                        for child in self.communities[community].children:
                            # Convert child to string to match self.reports key type
                            child_str = str(child)
                            if child_str in self.reports:
                                communities_to_rate.append(child_str)
                            else:
                                logger.debug(
                                    "dynamic community selection: cannot find community %s in reports",
                                    child,
                                )
                    # remove parent node if the current node is deemed relevant
                    if not self.keep_parent and community in self.communities:
                        relevant_communities.discard(self.communities[community].parent)

            # End per-level span
            if level_span is not None:
                try:
                    level_span.update(output={
                        "num_relevant": sum(1 for c in queue if ratings.get(c, 0) >= self.threshold),
                        "num_children_queued": len(communities_to_rate),
                    })
                    level_span.end()
                except Exception:
                    logger.debug("Failed to end Langfuse span for level %d", level)

            # Flush Langfuse after each level for real-time visibility
            if trace_ctx and trace_ctx.should_trace:
                trace_ctx.flush()

            # If timed out mid-level, stop traversal with partial results
            if timed_out:
                selection_timed_out = True
                queue = []
            else:
                queue = communities_to_rate
            level += 1

            if (
                (len(queue) == 0)
                and (len(relevant_communities) == 0)
                and (str(level) in self.levels)
                and (level <= self.max_level)
            ):
                logger.debug(
                    "dynamic community selection: no relevant community "
                    "reports, adding all reports at level %s to rate.",
                    level,
                )
                # append all communities at the next level to queue
                queue = self.levels[str(level)]

        community_reports = [
            self.reports[community] for community in relevant_communities
        ]
        end = time()

        # End dynamic community selection span
        if dcs_span is not None:
            try:
                dcs_span.update(output={
                    "duration_s": int(end - start),
                    "num_relevant": len(relevant_communities),
                    "total_communities": len(self.reports),
                    "timed_out": selection_timed_out,
                    "rating_distribution": dict(sorted(Counter(ratings.values()).items())),
                    "llm_calls": llm_info["llm_calls"],
                    "prompt_tokens": llm_info["prompt_tokens"],
                    "output_tokens": llm_info["output_tokens"],
                })
                dcs_span.end()
            except Exception:
                logger.debug("Failed to end Langfuse span for dynamic community selection")

        logger.debug(
            "dynamic community selection (took: %ss)\n"
            "\trating distribution %s\n"
            "\t%s out of %s community reports are relevant\n"
            "\tprompt tokens: %s, output tokens: %s",
            int(end - start),
            dict(sorted(Counter(ratings.values()).items())),
            len(relevant_communities),
            len(self.reports),
            llm_info["prompt_tokens"],
            llm_info["output_tokens"],
        )

        llm_info["ratings"] = ratings
        return community_reports, llm_info
