# Logfire Pycon 2024

Code snippets from:
* [this](https://slides.com/samuelcolvin-pydantic/deck) talk at Pycon US on 2024-05-16
* [this](https://slides.com/samuelcolvin-pydantic/logfire-pycon-2024) at EuroPython on 2024-07-12 (most of the talk was a live demo, so only a few slides)

The rest of the example from the [logfire-demo](https://github.com/pydantic/logfire-demo) project.

## Running the demo

1. Install uv
2. Run `uv sync` to install dependencies
3. Run `uv run python {example file name}`

## SQL Queries

The queries from the SQL slide were:

```sql
select start_timestamp, (attributes->'response_data'->'usage'->>'total_tokens')::int as usage, attributes->'response_data'->'message'->>'content' as message
from records 
where otel_scope_name='logfire.openai' and attributes->'response_data'->'message' ? 'content'
order by start_timestamp desc
```

and

```sql
select sum((attributes->'response_data'->'usage'->>'total_tokens')::int)
from records where otel_scope_name='logfire.openai'
```
