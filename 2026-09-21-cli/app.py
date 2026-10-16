"""FastAPI app wrapping the weather agent: one request, one trace.

Run with: uv run uvicorn app:app --port 8000
Then:     curl 'localhost:8000/weather?q=Paris%20and%20London'
"""

import logfire
from fastapi import FastAPI

from weather_agent import weather_agent

logfire.configure(service_name='weather-app')
logfire.instrument_pydantic_ai()
logfire.instrument_monty()

app = FastAPI()
logfire.instrument_fastapi(app)


@app.get('/weather')
async def weather(q: str = 'Paris and London') -> dict[str, str]:
    result = await weather_agent.run(f"What's the weather in {q}, in Celsius?")
    logfire.info('answered {q}', q=q, output=result.output)
    return {'answer': result.output}
