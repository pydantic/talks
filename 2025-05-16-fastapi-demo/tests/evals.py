"""
Pydantic AI Evaluation for testing the PydanticAI docs agent functionality.
This eval tests how well the agent can answer questions about PydanticAI framework.
"""

import asyncio
from typing import Any

import logfire
from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, LLMJudge

from src.agent import build_agent, answer_question, BotResponse


class AgentQuery(BaseModel):
    """Input model for agent queries."""
    question: str = Field(description="Question to ask the PydanticAI docs agent")


class EvalMetadata(BaseModel):
    """Metadata for evaluation cases."""
    difficulty: str = Field(description="Difficulty level: easy, medium, hard")
    topic: str = Field(description="Topic area being tested")
    expected_keywords: list[str] = Field(description="Keywords that should appear in response")


class ConfidenceEvaluator(Evaluator[AgentQuery, BotResponse]):
    """Evaluator that checks if the agent's confidence meets minimum threshold."""
    
    def __init__(self, min_confidence: int = 70):
        self.min_confidence = min_confidence
    
    def evaluate(self, ctx: EvaluatorContext[AgentQuery, BotResponse]) -> float:
        if ctx.output.confidence_percentage >= self.min_confidence:
            return 1.0
        return ctx.output.confidence_percentage / 100.0


class KeywordPresenceEvaluator(Evaluator[AgentQuery, BotResponse]):
    """Evaluator that checks if expected keywords appear in the response."""
    
    def evaluate(self, ctx: EvaluatorContext[AgentQuery, BotResponse]) -> float:
        if not hasattr(ctx, 'metadata') or not ctx.metadata:
            return 0.0
            
        expected_keywords = ctx.metadata.get('expected_keywords', [])
        if not expected_keywords:
            return 1.0
            
        answer_lower = ctx.output.answer.lower()
        found_keywords = [
            keyword for keyword in expected_keywords 
            if keyword.lower() in answer_lower
        ]
        
        return len(found_keywords) / len(expected_keywords)


# Task function that wraps the agent
async def query_pydantic_ai_agent(query: AgentQuery) -> BotResponse:
    """Task function that queries the PydanticAI docs agent."""
    agent = build_agent()
    return await answer_question(agent, query.question)


# Create the evaluation dataset
pydantic_ai_dataset = Dataset[AgentQuery, BotResponse, dict[str, Any]](
    cases=[
        Case(
            name='basic_agent_creation',
            inputs=AgentQuery(question="How do I create a basic PydanticAI agent?"),
            metadata={
                'difficulty': 'easy',
                'topic': 'agent_creation',
                'expected_keywords': ['Agent', 'model', 'system_prompt', 'pydantic_ai']
            },
                         evaluators=(
                 ConfidenceEvaluator(min_confidence=80),
                 KeywordPresenceEvaluator(),
                 LLMJudge(
                     rubric="Response should clearly explain how to create a PydanticAI agent with code examples",
                     include_input=True
                 ),
             )
        ),
        
        Case(
            name='user_prompt_modification',
            inputs=AgentQuery(question="How do I change the user prompt in PydanticAI?"),
            metadata={
                'difficulty': 'medium',
                'topic': 'prompt_handling',
                'expected_keywords': ['run', 'run_sync', 'user_prompt', 'agent']
            },
                         evaluators=(
                 ConfidenceEvaluator(min_confidence=75),
                 KeywordPresenceEvaluator(),
                 LLMJudge(
                     rubric="Response should explain how to modify user prompts with practical examples",
                     include_input=True
                 ),
             )
        ),
        
        Case(
            name='tools_integration',
            inputs=AgentQuery(question="How do I add tools to a PydanticAI agent?"),
            metadata={
                'difficulty': 'medium',
                'topic': 'tools',
                'expected_keywords': ['tools', 'function', 'decorator', '@tool']
            },
                         evaluators=(
                 ConfidenceEvaluator(min_confidence=70),
                 KeywordPresenceEvaluator(),
                 LLMJudge(
                     rubric="Response should explain tools integration with clear examples and best practices",
                     include_input=True
                 ),
             )
        ),
    ],
    evaluators=[
        # Global evaluators that apply to all cases
        LLMJudge(
            rubric="Response should be helpful, accurate, and well-structured for PydanticAI documentation questions",
            model='openai:gpt-4o-mini'  # Use a cost-effective model for evaluation
        ),
    ]
)


async def run_evaluation(send_to_logfire: bool = True):
    """Run the PydanticAI docs agent evaluation."""
    
    if send_to_logfire:
        logfire.configure(
            send_to_logfire=True,
            service_name='pydantic-ai-docs-evals',
            environment='development'
        )
    
    print("🚀 Running PydanticAI Docs Agent Evaluation")
    print("=" * 60)
    
    # Run the evaluation
    report = await pydantic_ai_dataset.evaluate(query_pydantic_ai_agent)
    
    # Print detailed results
    report.print(
        include_input=True,
        include_output=True,
        include_durations=True,
        include_averages=True
    )
    
    # Save results to file
    pydantic_ai_dataset.to_file('pydantic_ai_evals.yaml')
    print(f"\n📁 Dataset saved to: pydantic_ai_evals.yaml")
    
    return report


if __name__ == "__main__":
    asyncio.run(run_evaluation())
