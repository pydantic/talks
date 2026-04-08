# Prompt Optimization with GEPA and Cached Wikipedia Pages

This example demonstrates automated prompt optimization using GEPA (Genetic-Pareto Prompt Evolution)
with pydantic-ai and pydantic-evals, using cached Wikipedia pages for all 650 UK MPs.

## Overview

The example shows how to:

- Generate a golden evaluation dataset from cached MP pages with a strong model
- Load evaluation cases from JSON instead of hand-authoring them in code
- Evaluate either all political relatives or just ancestor/parent-generation relatives
- Build a GEPA adapter that integrates pydantic-evals with GEPA
- Use `Agent.override()` to inject candidate prompts and task models during optimization
- Run automated prompt optimization that improves based on evaluation feedback

## Running the Example

```bash
# Sync dependencies
uv sync

# Generate a golden dataset for the first 100 MPs
uv run -m main generate-cases --limit 100 --model openai:gpt-5

# Evaluate ancestor-only extraction on the test split
uv run -m main eval --split test --focus ancestors --prompt-style initial

# Compare initial vs expert prompts on the same task/model
uv run -m main compare --split test --focus ancestors

# Run optimization using train/val splits from the generated file
uv run -m main optimize --train-split train --val-split val --focus ancestors --max-calls 50
```

## Files

- `task.py` - Extraction schema, page preprocessing, relation filtering, and agent definition
- `cases.py` - Golden dataset generation and JSON persistence
- `evals.py` - Dataset loading and evaluation metrics
- `adapter.py` - GEPA adapter that bridges pydantic-evals with GEPA
- `main.py` - CLI script for running evaluation and optimization

## Notes

- The MP pages are read from the local `mps/` archive, not from live Wikipedia requests.
- `generate-cases` is resumable. Re-run it with a higher `--limit`, `--all`, or a different `--output`.
- The generated file stores the full set of political relatives. Evaluation can then filter to
  `--focus ancestors` without regenerating the golden data.
