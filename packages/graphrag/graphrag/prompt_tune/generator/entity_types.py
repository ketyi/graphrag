# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Entity type generation module for fine-tuning."""

from typing import TYPE_CHECKING

from graphrag_llm.utils import (
    CompletionMessagesBuilder,
)
from pydantic import BaseModel

from graphrag.prompt_tune.defaults import DEFAULT_TASK
from graphrag.prompt_tune.prompt.entity_types import (
    ENTITY_TYPE_GENERATION_JSON_PROMPT,
    ENTITY_TYPE_GENERATION_PROMPT,
)

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion
    from graphrag_llm.types import LLMCompletionResponse


class EntityTypesResponse(BaseModel):
    """Entity types response model."""

    entity_types: list[str]


async def generate_entity_types(
    model: "LLMCompletion",
    domain: str,
    persona: str,
    docs: str | list[str],
    task: str = DEFAULT_TASK,
    json_mode: bool = False,
) -> str | list[str]:
    """
    Generate entity type categories from a given set of (small) documents.

    Example Output:
    "entity_types": ['military unit', 'organization', 'person', 'location', 'event', 'date', 'equipment']
    """
    from graphrag.index.tracing import get_trace_context
    
    formatted_task = task.format(domain=domain)

    docs_str = "\n".join(docs) if isinstance(docs, list) else docs

    entity_types_prompt = (
        ENTITY_TYPE_GENERATION_JSON_PROMPT
        if json_mode
        else ENTITY_TYPE_GENERATION_PROMPT
    ).format(task=formatted_task, input_text=docs_str)

    messages = (
        CompletionMessagesBuilder()
        .add_system_message(persona)
        .add_user_message(entity_types_prompt)
        .build()
    )

    trace_context = get_trace_context()
    span = None
    if trace_context:
        span = trace_context.create_generation(
            name="generate_entity_types",
            input={"system": persona, "user": entity_types_prompt},
            metadata={
                "domain": domain,
                "formatted_task": formatted_task,
                "json_mode": json_mode,
                "docs_count": len(docs) if isinstance(docs, list) else 1,
            },
        )

    if json_mode:
        response: LLMCompletionResponse[
            EntityTypesResponse
        ] = await model.completion_async(
            messages=messages,
            response_format=EntityTypesResponse,
        )  # type: ignore
        parsed_model = response.formatted_response
        result = parsed_model.entity_types if parsed_model else []
        
        if span:
            span.update(output=result)
            span.end()
        
        return result

    non_json_response: LLMCompletionResponse = await model.completion_async(
        messages=messages
    )  # type: ignore
    
    if span:
        span.update(output=non_json_response.content)
        span.end()
    
    return non_json_response.content
