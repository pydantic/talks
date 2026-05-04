import asyncio
from dataclasses import dataclass
from datetime import date

import logfire
from pydantic import AwareDatetime, BaseModel
from pydantic_ai import Agent
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext, LLMJudge
from pydantic_evals.online import OnlineEvalConfig, wait_for_evaluations
from pydantic_evals.online_capability import OnlineEvaluation

logfire.configure(service_name='support-api')
logfire.instrument_pydantic_ai()


class Event(BaseModel, use_attribute_docstrings=True):
    title: str
    timestamp: AwareDatetime
    guests: list[str]
    location: str | None = None


@dataclass
class SingleLLMCall(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        chat_spans = ctx.span_tree.find({'has_attributes': {'gen_ai.operation.name': 'chat'}})
        if len(chat_spans) == 1:
            return EvaluationReason(value=True)
        else:
            return EvaluationReason(value=False, reason='Multiple chat spans found, extraction failed')


event_extract_agent = Agent(
    'gateway/anthropic:claude-haiku-4-5',
    output_type=list[Event],
    instructions='Extract information about upcoming meetings',
    capabilities=[
        OnlineEvaluation(
            evaluators=[
                LLMJudge(
                    rubric='All events are extracted',
                    model='gateway/openai:gpt-5.4',
                    include_input=True,
                ),
                SingleLLMCall(),
            ],
            config=OnlineEvalConfig(default_sample_rate=1.0),
        )
    ],
)


@event_extract_agent.instructions
def add_current_date():
    return f'The current date is {date.today()}.'


async def main():
    result = await event_extract_agent.run(
        "Hi Jean, I'm in Paris this Thursday speaking at an event at L'Hôtel du Collectionneur "
        'which looks to be fairly close to your office, '
        'any chance we could meet for coffee around 11.30 on Thursday?'
    )
    print(result.output)

    await wait_for_evaluations()


if __name__ == '__main__':
    asyncio.run(main())
