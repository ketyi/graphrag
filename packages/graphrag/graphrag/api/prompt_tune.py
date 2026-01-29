# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
Auto Templating API.

This API provides access to the auto templating feature of graphrag, allowing external applications
to hook into graphrag and generate prompts from private data.

WARNING: This API is under development and may undergo changes in future releases.
Backwards compatibility is not guaranteed at this time.
"""

import logging
import random
import uuid

from graphrag_llm.completion import create_completion
from pydantic import PositiveInt, validate_call

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.tracing import (
    set_explicit_tracing,
    set_trace_context,
)
from graphrag.logger.standard_logging import init_loggers
from graphrag.prompt_tune.defaults import MAX_TOKEN_COUNT, PROMPT_TUNING_MODEL_ID
from graphrag.prompt_tune.generator.community_report_rating import (
    generate_community_report_rating,
)
from graphrag.prompt_tune.generator.community_report_summarization import (
    create_community_summarization_prompt,
)
from graphrag.prompt_tune.generator.community_reporter_role import (
    generate_community_reporter_role,
)
from graphrag.prompt_tune.generator.domain import generate_domain
from graphrag.prompt_tune.generator.entity_relationship import (
    generate_entity_relationship_examples,
)
from graphrag.prompt_tune.generator.entity_summarization_prompt import (
    create_entity_summarization_prompt,
)
from graphrag.prompt_tune.generator.entity_types import generate_entity_types
from graphrag.prompt_tune.generator.extract_graph_prompt import (
    create_extract_graph_prompt,
)
from graphrag.prompt_tune.generator.language import detect_language
from graphrag.prompt_tune.generator.persona import generate_persona
from graphrag.prompt_tune.loader.input import load_docs_in_chunks
from graphrag.prompt_tune.types import DocSelectionType
from graphrag.tokenizer.get_tokenizer import get_tokenizer

logger = logging.getLogger(__name__)


@validate_call(config={"arbitrary_types_allowed": True})
async def generate_indexing_prompts(
    config: GraphRagConfig,
    limit: PositiveInt = 15,
    selection_method: DocSelectionType = DocSelectionType.RANDOM,
    domain: str | None = None,
    language: str | None = None,
    max_tokens: int = MAX_TOKEN_COUNT,
    discover_entity_types: bool = True,
    min_examples_required: PositiveInt = 2,
    n_subset_max: PositiveInt = 300,
    k: PositiveInt = 15,
    verbose: bool = False,
    session_id: str | None = None,
    user_id: str | None = None,
) -> tuple[str, str, str]:
    """Generate indexing prompts.

    Parameters
    ----------
    - config: The GraphRag configuration.
    - output_path: The path to store the prompts.
    - chunk_size: The chunk token size to use for input text units.
    - limit: The limit of chunks to load.
    - selection_method: The chunk selection method.
    - domain: The domain to map the input documents to.
    - language: The language to use for the prompts.
    - max_tokens: The maximum number of tokens to use on entity extraction prompts
    - discover_entity_types: Generate entity types.
    - min_examples_required: The minimum number of examples required for entity extraction prompts.
    - n_subset_max: The number of text chunks to embed when using auto selection method.
    - k: The number of documents to select when using auto selection method.
    - session_id: Optional session ID for tracing.
    - user_id: Optional user ID for tracing.

    Returns
    -------
    tuple[str, str, str]: entity extraction prompt, entity summarization prompt, community summarization prompt
    """
    init_loggers(config=config, verbose=verbose, filename="prompt-tuning.log")

    # Initialize Langfuse tracing if enabled
    langfuse_client = None
    trace_context = None
    if config.langfuse.enabled:
        # Check sampling
        if random.random() >= config.langfuse.sample_rate:
            logger.info("Skipping Langfuse tracing due to sampling rate.")
        else:
            try:
                import langfuse  # type: ignore

                # Initialize Langfuse client
                langfuse_client = langfuse.Langfuse(
                    public_key=config.langfuse.public_key,
                    secret_key=config.langfuse.secret_key,
                    host=config.langfuse.host,
                )

                # Generate session_id if not provided
                if not session_id:
                    session_id = str(uuid.uuid4())

                # Create root trace for prompt tuning
                from graphrag.index.tracing import TraceContext

                trace = langfuse_client.trace(  # type: ignore
                    name="prompt_tuning",
                    session_id=session_id,
                    user_id=user_id,
                    metadata={
                        "limit": limit,
                        "selection_method": selection_method.value,
                        "domain": domain,
                        "language": language,
                        "max_tokens": max_tokens,
                        "discover_entity_types": discover_entity_types,
                    },
                )
                trace_context = TraceContext(
                    trace=trace,
                    session_id=session_id,
                    user_id=user_id,
                    should_trace=True,
                )
                set_trace_context(trace_context)
                logger.info("Initialized Langfuse tracing for prompt tuning.")
            except ImportError:
                logger.exception(
                    "Langfuse is enabled in config but the langfuse package is not installed. "
                    "Install it with: pip install langfuse"
                )
                msg = "Langfuse package not installed"
                raise ImportError(msg) from None
            except Exception as e:
                logger.exception("Failed to initialize Langfuse")
                msg = f"Failed to initialize Langfuse: {e!s}"
                raise RuntimeError(msg) from e

    try:
        # Set explicit tracing flag to prevent middleware duplication
        set_explicit_tracing(True)

        # Retrieve documents
        logger.info("Chunking documents...")
        span = None
        if trace_context:
            span = trace_context.create_span(name="chunk_documents")
        try:
            doc_list = await load_docs_in_chunks(
                config=config,
                limit=limit,
                select_method=selection_method,
                logger=logger,
                n_subset_max=n_subset_max,
                k=k,
            )
        finally:
            if span:
                span.end()

        # Create LLM from config
        # TODO: Expose a way to specify Prompt Tuning model ID through config
        logger.info("Retrieving language model configuration...")
        default_llm_settings = config.get_completion_model_config(PROMPT_TUNING_MODEL_ID)

        logger.info("Creating language model...")
        llm = create_completion(default_llm_settings)

        if not domain:
            logger.info("Generating domain...")
            span = None
            if trace_context:
                span = trace_context.create_generation(
                    name="generate_domain",
                    input={"docs_count": len(doc_list)},
                )
            try:
                domain = await generate_domain(llm, doc_list)
                if span:
                    span.update(output=domain)
            finally:
                if span:
                    span.end()

        if not language:
            logger.info("Detecting language...")
            span = None
            if trace_context:
                span = trace_context.create_generation(
                    name="detect_language",
                    input={"docs_count": len(doc_list)},
                )
            try:
                language = await detect_language(llm, doc_list)
                if span:
                    span.update(output=language)
            finally:
                if span:
                    span.end()

        logger.info("Generating persona...")
        span = None
        if trace_context:
            span = trace_context.create_generation(
                name="generate_persona",
                input={"domain": domain},
            )
        try:
            persona = await generate_persona(llm, domain)
            if span:
                span.update(output=persona)
        finally:
            if span:
                span.end()

        logger.info("Generating community report ranking description...")
        span = None
        if trace_context:
            span = trace_context.create_generation(
                name="generate_community_report_rating",
                input={"domain": domain, "persona": persona, "docs_count": len(doc_list)},
            )
        try:
            community_report_ranking = await generate_community_report_rating(
                llm, domain=domain, persona=persona, docs=doc_list
            )
            if span:
                span.update(output=community_report_ranking)
        finally:
            if span:
                span.end()

        entity_types = None
        extract_graph_llm_settings = config.get_completion_model_config(
            config.extract_graph.completion_model_id
        )
        if discover_entity_types:
            logger.info("Generating entity types...")
            span = None
            if trace_context:
                span = trace_context.create_generation(
                    name="generate_entity_types",
                    input={
                        "domain": domain,
                        "persona": persona,
                        "docs_count": len(doc_list),
                        "json_mode": True,
                    },
                )
            try:
                entity_types = await generate_entity_types(
                    llm,
                    domain=domain,
                    persona=persona,
                    docs=doc_list,
                    json_mode=True,
                )
                if span:
                    span.update(output=entity_types)
            finally:
                if span:
                    span.end()

        logger.info("Generating entity relationship examples...")
        span = None
        if trace_context:
            span = trace_context.create_generation(
                name="generate_entity_relationship_examples",
                input={
                    "persona": persona,
                    "entity_types": entity_types,
                    "docs_count": len(doc_list),
                    "language": language,
                    "json_mode": False,
                },
            )
        try:
            examples = await generate_entity_relationship_examples(
                llm,
                persona=persona,
                entity_types=entity_types,
                docs=doc_list,
                language=language,
                json_mode=False,  # config.llm.model_supports_json should be used, but these prompts are used in non-json mode by the index engine
            )
            if span:
                span.update(output={"examples_count": len(examples)})
        finally:
            if span:
                span.end()

        logger.info("Generating entity extraction prompt...")
        extract_graph_prompt = create_extract_graph_prompt(
            entity_types=entity_types,
            docs=doc_list,
            examples=examples,
            language=language,
            json_mode=False,  # config.llm.model_supports_json should be used, but these prompts are used in non-json mode by the index engine
            tokenizer=get_tokenizer(model_config=extract_graph_llm_settings),
            max_token_count=max_tokens,
            min_examples_required=min_examples_required,
        )

        logger.info("Generating entity summarization prompt...")
        entity_summarization_prompt = create_entity_summarization_prompt(
            persona=persona,
            language=language,
        )

        logger.info("Generating community reporter role...")
        span = None
        if trace_context:
            span = trace_context.create_generation(
                name="generate_community_reporter_role",
                input={"domain": domain, "persona": persona, "docs_count": len(doc_list)},
            )
        try:
            community_reporter_role = await generate_community_reporter_role(
                llm, domain=domain, persona=persona, docs=doc_list
            )
            if span:
                span.update(output=community_reporter_role)
        finally:
            if span:
                span.end()

        logger.info("Generating community summarization prompt...")
        community_summarization_prompt = create_community_summarization_prompt(
            persona=persona,
            role=community_reporter_role,
            report_rating_description=community_report_ranking,
            language=language,
        )

        logger.debug("Generated domain: %s", domain)
        logger.debug("Detected language: %s", language)
        logger.debug("Generated persona: %s", persona)

        return (
            extract_graph_prompt,
            entity_summarization_prompt,
            community_summarization_prompt,
        )
    finally:
        # Restore explicit tracing flag
        set_explicit_tracing(False)

        # Flush Langfuse traces
        if langfuse_client and config.langfuse.flush_at:
            logger.info("Flushing Langfuse traces...")
            langfuse_client.flush()
            logger.info("Langfuse traces flushed.")
