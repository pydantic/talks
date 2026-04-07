# Prompt Optimization with GEPA

This example demonstrates automated prompt optimization using GEPA (Genetic-Pareto Prompt Evolution) with pydantic-ai and pydantic-evals.

## Overview

The example shows how to:

- Define an evaluation dataset with test cases and custom evaluators
- Build a GEPA adapter that integrates pydantic-evals with GEPA
- Use `Agent.override()` to inject candidate prompts during optimization
- Run automated prompt optimization that improves based on evaluation feedback

## Running the Example

```bash
# Sync dependencies
uv sync

# Run evaluation with initial instructions
uv run -m main eval

# Compare initial vs expert instructions
uv run -m main compare

# Run optimization (this will take a while and use API credits)
uv run -m main optimize --max-calls 50
```

## Files

- `task.py` - The contact extraction task and agent definition
- `evals.py` - Evaluation dataset with test cases and custom evaluators
- `adapter.py` - GEPA adapter that bridges pydantic-evals with GEPA
- `main.py` - CLI script for running evaluation and optimization

## Related Blog Post

See the full blog post: [Automated Prompt Optimization with GEPA, Pydantic AI, and Pydantic Evals](https://pydantic.dev/articles/prompt-optimization-with-gepa)
