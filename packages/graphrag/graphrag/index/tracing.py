# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Tracing infrastructure for GraphRAG indexing using Langfuse."""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langfuse.client import (  # type: ignore
        StatefulGenerationClient,
        StatefulSpanClient,
        StatefulTraceClient,
    )

    from graphrag.config.models.langfuse_config import LangfuseConfig

# Context variables for thread-safe trace context propagation
_trace_context: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar(
    "trace_context", default=None
)
_langfuse_config: contextvars.ContextVar[LangfuseConfig | None] = contextvars.ContextVar(
    "langfuse_config", default=None
)
# Flag to indicate when we're in an explicit tracing block (e.g., extractor with gleaning)
# When True, middleware should skip automatic span creation to avoid duplicates
_explicit_tracing: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "explicit_tracing", default=False
)


class TraceContext:
    """Context for Langfuse tracing during pipeline execution."""

    def __init__(
        self,
        trace: StatefulTraceClient,
        session_id: str | None,
        user_id: str | None,
        should_trace: bool = True,
    ):
        """Initialize trace context.

        Parameters
        ----------
        trace : StatefulTraceClient
            The Langfuse trace client.
        session_id : str | None
            The session ID for this trace.
        user_id : str | None
            The user ID for this trace.
        should_trace : bool
            Whether tracing is enabled for this execution (based on sampling).
        """
        self.trace = trace
        self.session_id = session_id
        self.user_id = user_id
        self.should_trace = should_trace

    def create_span(
        self,
        name: str,
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> StatefulSpanClient | None:
        """Create a span within the current trace.

        Parameters
        ----------
        name : str
            The name of the span.
        input : Any | None
            The input to the span.
        metadata : dict[str, Any] | None
            Additional metadata for the span.
        **kwargs : Any
            Additional arguments to pass to the span.

        Returns
        -------
        StatefulSpanClient | None
            The span client, or None if tracing is disabled.
        """
        if not self.should_trace:
            return None

        return self.trace.span(
            name=name,
            input=input,
            metadata=metadata,
            **kwargs,
        )

    def create_generation(
        self,
        name: str,
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> StatefulGenerationClient | None:
        """Create a generation within the current trace.

        Parameters
        ----------
        name : str
            The name of the generation.
        input : Any | None
            The input to the generation.
        metadata : dict[str, Any] | None
            Additional metadata for the generation.
        model : str | None
            The model used for the generation.
        **kwargs : Any
            Additional arguments to pass to the generation.

        Returns
        -------
        StatefulGenerationClient | None
            The generation client, or None if tracing is disabled.
        """
        if not self.should_trace:
            return None

        return self.trace.generation(
            name=name,
            input=input,
            metadata=metadata,
            model=model,
            **kwargs,
        )


def get_trace_context() -> TraceContext | None:
    """Get the current trace context.

    Returns
    -------
    TraceContext | None
        The current trace context, or None if not set.
    """
    return _trace_context.get()


def set_trace_context(context: TraceContext | None) -> None:
    """Set the current trace context.

    Parameters
    ----------
    context : TraceContext | None
        The trace context to set.
    """
    _trace_context.set(context)


def get_langfuse_config() -> LangfuseConfig | None:
    """Get the current Langfuse configuration.

    Returns
    -------
    LangfuseConfig | None
        The current Langfuse configuration, or None if not set.
    """
    return _langfuse_config.get()


def set_langfuse_config(config: LangfuseConfig | None) -> None:
    """Set the current Langfuse configuration.

    Parameters
    ----------
    config : LangfuseConfig | None
        The Langfuse configuration to set.
    """
    _langfuse_config.set(config)


def is_explicit_tracing() -> bool:
    """Check if we're currently in an explicit tracing block.

    Returns
    -------
    bool
        True if explicit tracing is active, False otherwise.
    """
    return _explicit_tracing.get()


def set_explicit_tracing(enabled: bool) -> None:
    """Set the explicit tracing flag.

    Parameters
    ----------
    enabled : bool
        Whether explicit tracing is enabled.
    """
    _explicit_tracing.set(enabled)
