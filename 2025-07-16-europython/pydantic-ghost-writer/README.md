# Pydantic Ghost Writer

A multi-agent MCP demo project that generates blog posts and can automatically create pull requests to the Pydantic website.

## Get started

### Install dependencies

Python 3.12 (or greater) is required to run the app.

```bash
uv sync
```

### Configure your LLM provider token

Depending on the provider used, set the appropriate environment variable.

This demo project uses Anthropic, and can be configured as such:

```bash
export ANTHROPIC_API_KEY="your_anthropic_key_here"
```

For more details, refer to the [list of available models](https://ai.pydantic.dev/models/).

### Set up Pydantic Logfire

If you want to send logs to Pydantic Logfire, make sure to set up a project:

```bash
logfire projects use
```

See [the Logfire documentation](https://logfire.pydantic.dev/docs/how-to-guides/create-write-tokens/)
for more details.

### Optional: Set GitHub token for PR creation

To enable automatic PR creation to the `pydantic.dev` repository:

```bash
export GITHUB_TOKEN="your_github_personal_access_token"
```

**GitHub Token Permissions Required:**

- `repo` (Full control of private repositories)
- `workflow` (Update GitHub Action workflows)

**How to create a GitHub Personal Access Token:**

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Select the required scopes listed above
4. Copy and export the token as shown above

> [!NOTE]
> This will ask the user for confirmation through the CLI, which will not work if exposing
> the writer agent using the MCP server.

### Run the CLI

```bash
uv run python -m ghost_writer input.json
```

Or, if you want to have interactive input from the CLI:

```bash
uv run python -m ghost_writer
```

## Features

- **AI-powered blog post generation** using Pydantic AI agents
- **Content review and iteration** with automated scoring
- **Brand guidelines integration** for authentic Pydantic voice
- **Web content extraction** from reference URLs
- **Automatic PR creation** to `pydantic.dev` repository (optional)

## Workflow

1. **Content Generation**: The writer agent creates blog posts based on your topic and requirements
2. **Quality Review**: Content is automatically reviewed and scored for quality
3. **Iteration**: Low-scoring content is automatically improved
4. **PR Creation**: Optionally create a pull request directly to the `pydantic.dev` repository
