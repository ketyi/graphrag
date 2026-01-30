# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Persona generating module for fine-tuning GraphRAG prompts."""

from typing import TYPE_CHECKING

from graphrag.prompt_tune.defaults import DEFAULT_TASK
from graphrag.prompt_tune.prompt.persona import GENERATE_PERSONA_PROMPT

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion
    from graphrag_llm.types import LLMCompletionResponse


async def generate_persona(
    model: "LLMCompletion", domain: str, task: str = DEFAULT_TASK
) -> str:
    """Generate an LLM persona to use for GraphRAG prompts.

    Parameters
    ----------
    - model (LLMCompletion): The LLM to use for generation
    - domain (str): The domain to generate a persona for
    - task (str): The task to generate a persona for. Default is DEFAULT_TASK
    """
    from graphrag.index.tracing import get_trace_context
    
    formatted_task = task.format(domain=domain)
    persona_prompt = GENERATE_PERSONA_PROMPT.format(sample_task=formatted_task)

    trace_context = get_trace_context()
    span = None
    if trace_context:
        span = trace_context.create_generation(
            name="generate_persona",
            input=persona_prompt,
            metadata={"domain": domain, "task": formatted_task},
        )

    response: LLMCompletionResponse = await model.completion_async(
        messages=persona_prompt
    )  # type: ignore

    if span:
        span.update(output=response.content)
        span.end()

    return response.content
