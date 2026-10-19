from dataclasses import dataclass, field
from time import perf_counter

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from .prompts import CHAT_TEMPLATE


@dataclass
class GenerationResult:
    user_input: str
    response: str
    latency_seconds: float
    retrieved_contexts: list[str] = field(default_factory=list)


def run_query(
    user_input: str,
    llm: Runnable,
    prompt_template: str = CHAT_TEMPLATE,
) -> GenerationResult:
    """Run the same single-question generation used by chat and offline QA."""
    started = perf_counter()
    template = PromptTemplate(
        input_variables=["user_input"],
        template=prompt_template,
    )
    chain = template | llm
    response = chain.invoke({"user_input": user_input})
    return GenerationResult(
        user_input=user_input,
        response=response,
        latency_seconds=perf_counter() - started,
    )
