# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Langfuse middleware for tracing LLM calls."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graphrag_llm.types import AsyncLLMFunction, LLMFunction


def with_langfuse(
    *,
    sync_middleware: "LLMFunction",
    async_middleware: "AsyncLLMFunction",
) -> tuple["LLMFunction", "AsyncLLMFunction"]:
    """Wrap model functions with Langfuse tracing middleware.

    Parameters
    ----------
    sync_middleware : LLMFunction
        The synchronous model function to wrap.
    async_middleware : AsyncLLMFunction
        The asynchronous model function to wrap.

    Returns
    -------
    tuple[LLMFunction, AsyncLLMFunction]
        The synchronous and asynchronous model functions wrapped with Langfuse middleware.
    """

    def _langfuse_middleware(**kwargs: Any):
        # For now, just pass through - sync tracing can be added later if needed
        return sync_middleware(**kwargs)

    async def _langfuse_middleware_async(**kwargs: Any):
        """Async Langfuse middleware that traces LLM calls."""
        # Try to get trace context, but gracefully handle if not available
        trace_ctx = None
        is_explicit = False
        try:
            # Import here to avoid issues during package installation
            from graphrag.index.tracing import get_trace_context, is_explicit_tracing  # type: ignore
            trace_ctx = get_trace_context()
            is_explicit = is_explicit_tracing()
        except (ImportError, AttributeError):
            # If tracing module not available, trace_ctx remains None
            pass

        # Skip middleware tracing if:
        # 1. No trace context or tracing is disabled
        # 2. We're in an explicit tracing block (extractor is handling spans)
        if trace_ctx is None or not trace_ctx.should_trace or is_explicit:
            return await async_middleware(**kwargs)

        # Extract LLM call parameters
        messages = kwargs.get("messages", [])
        model = kwargs.get("model")
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")

        # Create a generation span for this LLM call (automatic tracing)
        generation = trace_ctx.create_generation(
            name="llm_call_auto",
            input=messages,
            model=model,
            metadata={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "source": "middleware",
            },
        )

        try:
            # Execute the LLM call
            response = await async_middleware(**kwargs)
        except Exception as e:
            # Track error in Langfuse
            if generation is not None:
                generation.end(
                    output=None,
                    metadata={"error": str(e), "error_type": type(e).__name__},
                )
            raise
        else:
            # End the generation with output and usage info
            if generation is not None:
                output = getattr(response, "content", None)
                usage = getattr(response, "usage", None)
                
                usage_dict = None
                if usage is not None:
                    usage_dict = {
                        "input": getattr(usage, "prompt_tokens", 0),
                        "output": getattr(usage, "completion_tokens", 0),
                        "total": getattr(usage, "total_tokens", 0),
                    }

                generation.end(
                    output=output,
                    usage=usage_dict,
                )

            return response

    return (_langfuse_middleware, _langfuse_middleware_async)  # type: ignore
