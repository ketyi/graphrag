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
        """Async Langfuse middleware that traces LLM calls.

        Note: Middleware tracing is disabled - all tracing is now done explicitly
        in extractors with proper span names and usage tracking. This middleware
        just passes through to the underlying function.
        """
        return await async_middleware(**kwargs)

    return (_langfuse_middleware, _langfuse_middleware_async)  # type: ignore
