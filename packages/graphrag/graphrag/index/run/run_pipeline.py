# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Different methods to run the pipeline."""

import contextlib
import json
import logging
import re
import secrets
import time
from collections.abc import AsyncIterable
from dataclasses import asdict
from typing import Any

import pandas as pd
from graphrag_cache import create_cache
from graphrag_storage import Storage, create_storage

from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.run.utils import create_run_context
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.pipeline import Pipeline
from graphrag.index.typing.pipeline_run_result import PipelineRunResult
from graphrag.utils.storage import load_table_from_storage, write_table_to_storage

logger = logging.getLogger(__name__)


async def run_pipeline(
    pipeline: Pipeline,
    config: GraphRagConfig,
    callbacks: WorkflowCallbacks,
    is_update_run: bool = False,
    additional_context: dict[str, Any] | None = None,
    input_documents: pd.DataFrame | None = None,
) -> AsyncIterable[PipelineRunResult]:
    """Run all workflows using a simplified pipeline."""
    input_storage = create_storage(config.input_storage)
    output_storage = create_storage(config.output_storage)
    cache = create_cache(config.cache)

    # load existing state in case any workflows are stateful
    state_json = await output_storage.get("context.json")
    state = json.loads(state_json) if state_json else {}

    if additional_context:
        state.setdefault("additional_context", {}).update(additional_context)

    if is_update_run:
        logger.info("Running incremental indexing.")

        update_storage = create_storage(config.update_output_storage)
        # we use this to store the new subset index, and will merge its content with the previous index
        update_timestamp = time.strftime("%Y%m%d-%H%M%S")
        timestamped_storage = update_storage.child(update_timestamp)
        delta_storage = timestamped_storage.child("delta")
        # copy the previous output to a backup folder, so we can replace it with the update
        # we'll read from this later when we merge the old and new indexes
        previous_storage = timestamped_storage.child("previous")
        await _copy_previous_output(output_storage, previous_storage)

        state["update_timestamp"] = update_timestamp

        # if the user passes in a df directly, write directly to storage so we can skip finding/parsing later
        if input_documents is not None:
            await write_table_to_storage(input_documents, "documents", delta_storage)
            pipeline.remove("load_update_documents")

        context = create_run_context(
            input_storage=input_storage,
            output_storage=delta_storage,
            previous_storage=previous_storage,
            cache=cache,
            callbacks=callbacks,
            state=state,
        )

    else:
        logger.info("Running standard indexing.")

        # if the user passes in a df directly, write directly to storage so we can skip finding/parsing later
        if input_documents is not None:
            await write_table_to_storage(input_documents, "documents", output_storage)
            pipeline.remove("load_input_documents")

        context = create_run_context(
            input_storage=input_storage,
            output_storage=output_storage,
            cache=cache,
            callbacks=callbacks,
            state=state,
        )

    async for table in _run_pipeline(
        pipeline=pipeline,
        config=config,
        context=context,
    ):
        yield table


async def _run_pipeline(
    pipeline: Pipeline,
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> AsyncIterable[PipelineRunResult]:
    start_time = time.time()

    last_workflow = "<startup>"

    # Initialize Langfuse tracing if enabled
    langfuse_client = None
    root_span = None
    trace_ctx = None
    
    if config.langfuse.enabled:
        try:
            from langfuse import Langfuse

            from graphrag.index.tracing import (
                TraceContext,
                set_langfuse_config,
                set_trace_context,
            )

            # Apply pipeline-level sampling
            should_trace = secrets.SystemRandom().random() < config.langfuse.sample_rate

            if should_trace:
                # Initialize Langfuse client with fail-fast error handling
                try:
                    import os
                    import re
                    
                    # Resolve environment variables in config values
                    def resolve_env_var(value: str | None) -> str | None:
                        """Resolve ${VAR} or ${VAR:default} patterns in config values."""
                        if not value or not isinstance(value, str):
                            return value
                        
                        # Match ${VAR} or ${VAR:default}
                        pattern = r"\$\{([^:}]+)(?::([^}]*))?\}"
                        
                        def replacer(match):
                            var_name = match.group(1)
                            default_value = match.group(2) if match.group(2) is not None else ""
                            return os.getenv(var_name, default_value)
                        
                        return re.sub(pattern, replacer, value)
                    
                    langfuse_client = Langfuse(
                        public_key=resolve_env_var(config.langfuse.public_key),
                        secret_key=resolve_env_var(config.langfuse.secret_key),
                        host=resolve_env_var(config.langfuse.host),
                        flush_at=config.langfuse.flush_at,
                    )
                except Exception as e:
                    msg = f"Failed to initialize Langfuse client: {e}"
                    logger.exception(msg)
                    raise ValueError(msg) from e

                # Extract session_id and user_id from additional_context
                additional_ctx = context.state.get("additional_context", {})
                session_id = additional_ctx.get("session_id")
                user_id = additional_ctx.get("user_id")

                # Create trace (Langfuse 3.x: use start_as_current_span and enter context)
                trace_ctx_mgr = langfuse_client.start_as_current_span(
                    name="graphrag_indexing_pipeline",
                    metadata={
                        "pipeline": pipeline.names(),
                        "config_summary": {
                            "concurrent_requests": config.concurrent_requests,
                            "async_mode": config.async_mode.value if hasattr(config.async_mode, "value") else str(config.async_mode),
                        },
                    },
                )
                trace = trace_ctx_mgr.__enter__()
                
                # Update trace-level attributes
                if session_id or user_id:
                    langfuse_client.update_current_trace(
                        session_id=session_id,
                        user_id=user_id,
                    )

                # Set trace context
                trace_ctx = TraceContext(
                    trace=trace,
                    session_id=session_id,
                    user_id=user_id,
                    should_trace=True,
                    context_manager=trace_ctx_mgr,
                )
                set_trace_context(trace_ctx)
                set_langfuse_config(config.langfuse)

                # Create root span for the entire pipeline
                root_span = trace_ctx.create_span(
                    name="pipeline_execution",
                    metadata={"workflows": pipeline.names()},
                )
            else:
                logger.info("Langfuse tracing is sampled out for this pipeline run (sample_rate=%.2f)", config.langfuse.sample_rate)

        except ImportError as e:
            msg = f"Langfuse is enabled but the langfuse package is not installed: {e}"
            logger.exception(msg)
            raise ImportError(msg) from e

    try:
        await _dump_json(context)

        logger.info("Executing pipeline...")
        for name, workflow_function in pipeline.run():
            last_workflow = name
            context.callbacks.workflow_start(name, None)
            work_time = time.time()
            result = await workflow_function(config, context)
            context.callbacks.workflow_end(name, result)
            yield PipelineRunResult(
                workflow=name, result=result.result, state=context.state, error=None
            )
            context.stats.workflows[name] = {"overall": time.time() - work_time}
            if result.stop:
                logger.info("Halting pipeline at workflow request")
                break

        context.stats.total_runtime = time.time() - start_time
        logger.info("Indexing pipeline complete.")
        await _dump_json(context)

        # End root span successfully
        if root_span is not None:
            root_span.end(
                output={"status": "completed", "total_runtime": context.stats.total_runtime},
            )

    except Exception as e:
        logger.exception("error running workflow %s", last_workflow)
        
        # End root span with error
        if root_span is not None:
            root_span.end(
                output={"status": "error", "last_workflow": last_workflow},
            )
        
        yield PipelineRunResult(
            workflow=last_workflow, result=None, state=context.state, error=e
        )
    finally:
        # Exit the trace context if it was created
        if trace_ctx and trace_ctx.context_manager:
            with contextlib.suppress(Exception):
                trace_ctx.context_manager.__exit__(None, None, None)

        # Flush Langfuse events
        if langfuse_client is not None:
            try:
                langfuse_client.flush()
                logger.info("Langfuse events flushed successfully.")
            except Exception as e:  # noqa: BLE001 - intentionally catch all exceptions during cleanup
                logger.warning("Failed to flush Langfuse events: %s", e)
        
        # Clean up trace context
        if config.langfuse.enabled:
            try:
                from graphrag.index.tracing import (
                    set_langfuse_config,
                    set_trace_context,
                )
                set_trace_context(None)
                set_langfuse_config(None)
            except ImportError:
                pass


async def _dump_json(context: PipelineRunContext) -> None:
    """Dump the stats and context state to the storage."""
    await context.output_storage.set(
        "stats.json", json.dumps(asdict(context.stats), indent=4, ensure_ascii=False)
    )
    # Dump context state, excluding additional_context
    temp_context = context.state.pop(
        "additional_context", None
    )  # Remove reference only, as object size is uncertain
    try:
        state_blob = json.dumps(context.state, indent=4, ensure_ascii=False)
    finally:
        if temp_context:
            context.state["additional_context"] = temp_context

    await context.output_storage.set("context.json", state_blob)


async def _copy_previous_output(
    storage: Storage,
    copy_storage: Storage,
):
    for file in storage.find(re.compile(r"\.parquet$")):
        base_name = file.replace(".parquet", "")
        table = await load_table_from_storage(base_name, storage)
        await write_table_to_storage(table, base_name, copy_storage)
