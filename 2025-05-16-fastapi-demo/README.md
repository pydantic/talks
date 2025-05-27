# FastAPI Demo with Math, Database and PydanticAI

This is a FastAPI application that demonstrates:
- Mathematical operations (division, Fibonacci)
- Database operations with SQLAlchemy
- PydanticAI agent integration with Tavily search
- Logfire observability

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Create a `.env` file in the root directory with the following environment variables:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   LOGFIRE_TOKEN=your_logfire_token_here
   DATABASE_URL=sqlite:///./test.db
   ```

3. Run the application:
   ```bash
   uv run uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
   ```

## API Endpoints

- `GET /divide/{numerator}/{denominator}` - Divide two numbers
- `GET /fibonacci/{n}` - Calculate nth Fibonacci number
- `POST /items/` - Create a new item in the database
- `GET /items/` - List all items with pagination
- `GET /items/{item_id}` - Get a specific item by ID
- `POST /agent/query` - Query the PydanticAI agent with a question

## Example Usage

Query the PydanticAI agent:
```bash
curl -X POST "http://localhost:8000/agent/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "How do I use PydanticAI tools?"}'
```
