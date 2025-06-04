# FastAPI Demo with Math, Database, PydanticAI and MCP

This is a FastAPI application that demonstrates:
- Mathematical operations (division, Fibonacci)
- Database operations with SQLAlchemy
- PydanticAI agent integration with Tavily search
- MCP (Model Context Protocol) integration with Playwright MCP server
- Logfire observability

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Ensure you have Node.js installed for the MCP filesystem server:
   ```bash
   node --version  # Should be v16 or higher
   ```

3. Create a `.env` file in the root directory with the following environment variables:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   LOGFIRE_TOKEN=your_logfire_token_here
   DATABASE_URL=sqlite:///./test.db
   ```

4. Run the application:
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
- `POST /mcp/query` - Query the MCP-enabled agent with Playwright MCP

## Example Usage

Query the PydanticAI agent:
```bash
curl -X POST "http://localhost:8000/agent/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "How do I use PydanticAI tools?"}'
```

Query the MCP agent with filesystem capabilities:
```bash
curl -X POST "http://localhost:8000/mcp/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "Create a simple Python script that prints hello world and save it to hello.py"}'
```

Run evals:
```bash
PYTHONPATH=. uv run python tests/evals.py
```
