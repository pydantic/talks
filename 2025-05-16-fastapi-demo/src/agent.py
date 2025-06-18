import os
from pathlib import Path
from textwrap import dedent
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai.agent import Agent
from pydantic_ai.common_tools.tavily import tavily_search_tool


ROOT_DIR = Path(__file__).parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")


class BotResponse(BaseModel):
    answer: str
    reasoning: str
    reference: str | None = None
    confidence_percentage: Annotated[int, Field(ge=0, le=100)]


SYSTEM_PROMPT = dedent(
    """
    You're an all-knowing expert in the PydanticAI agent framework.
    You will receive questions from users of PydanticAI about how to use the framework effectively.

    Where necessary, use Tavily to search for PydanticAI information. The documentation can be found here: https://ai.pydantic.dev/
    The LLM txt can be found here: https://ai.pydantic.dev/llms.txt
    
    For any given answer, where possible provide references to the documentation or other relevant resources.
    Give a confidence percentage for your answer, from 0 to 100.
    """
)


def build_docs_agent() -> Agent[None, BotResponse]:
    api_key = os.getenv("TAVILY_API_KEY")
    assert api_key is not None

    return Agent(
        "openai:gpt-4.1",
        tools=[tavily_search_tool(api_key)],
        output_type=BotResponse,
        system_prompt=SYSTEM_PROMPT,
    )


def build_grammar_agent() -> Agent[None, BotResponse]:
    return Agent(
        "openai:gpt-4o-mini",
        output_type=BotResponse,
        system_prompt="Fix the punctuation in the BotResponse object fields.",
    )


async def answer_question(
    agent: Agent[None, BotResponse], question: str
) -> BotResponse:
    result = await agent.run(user_prompt=question)
    return result.output


async def improve_answer(
    agent: Agent[None, BotResponse], initial_answer: str
) -> BotResponse:
    result = await agent.run(user_prompt=initial_answer)
    return result.output
