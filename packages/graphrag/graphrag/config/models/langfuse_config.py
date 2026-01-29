# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for Langfuse tracing configuration."""

from pydantic import BaseModel, Field

from graphrag.config.defaults import graphrag_config_defaults


class LangfuseConfig(BaseModel):
    """The configuration section for Langfuse tracing."""

    enabled: bool = Field(
        description="Whether Langfuse tracing is enabled.",
        default=graphrag_config_defaults.langfuse.enabled,
    )
    public_key: str | None = Field(
        description="The Langfuse public key.",
        default=graphrag_config_defaults.langfuse.public_key,
    )
    secret_key: str | None = Field(
        description="The Langfuse secret key.",
        default=graphrag_config_defaults.langfuse.secret_key,
    )
    host: str | None = Field(
        description="The Langfuse host URL.",
        default=graphrag_config_defaults.langfuse.host,
    )
    sample_rate: float = Field(
        description="The sampling rate for tracing (0.0 to 1.0).",
        default=graphrag_config_defaults.langfuse.sample_rate,
    )
    flush_at: int = Field(
        description="The number of events to buffer before flushing.",
        default=graphrag_config_defaults.langfuse.flush_at,
    )
