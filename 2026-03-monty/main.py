import asyncio
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import logfire
import pydantic_core
from pydantic_ai import Agent, ModelRequest, ModelRequestNode, UserPromptPart
from pydantic_ai.models.anthropic import AnthropicModelSettings
from pydantic_graph import End
from pydantic_monty import Monty, MontyError, MontyRepl, MontyRuntimeError, run_repl_async
from rich.console import Console
from rich.markdown import Markdown

from external_functions import display_table, plot, show_plot, sql_query

logfire.configure(console=False)
logfire.instrument_pydantic_ai()

THIS_DIR = Path(__file__).parent


def _generate_stubs() -> str:
    """Generate type stubs for external_functions.py using stubgen."""
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            ['uv', 'run', 'stubgen', 'external_functions.py', '--include-docstrings', '-o', tmpdir],
            capture_output=True,
            text=True,
            cwd=THIS_DIR,
            check=True,
        )
        return (Path(tmpdir) / 'external_functions.pyi').read_text()


stubs = _generate_stubs()
instructions = f"""
You MUST return markdown with either a comment and python code to execute
in a "```python" code block, or an explanation of your process to end.

Use `sql_query` to query the data, `plot` and `show_plot` to create visualizations,
and `display_table` to show formatted tables.

The runtime uses a restricted Python subset:
- you CANNOT use the standard library except builtin functions and the following modules: `sys`, `typing`, `asyncio`, `re`, `math`
- this means the following modules are NOT available: `json`, `collections`, `json`, `datetime`, `itertools`, `functools`, etc.
- you CAN use plain dicts, lists, and builtins instead
- you CAN define and reuse functions
- you CANNOT use third party libraries
- you CANNOT define classes
- you MUST import modules you want to use e.g. `import asyncio`
- the python executor is a REPL — variables, state and functions you define persist between calls, so you can build on previous results without redefining them


The last expression evaluated is the return value.

You can use `print()` to get debug information while developing the code.

Parallelism: use `asyncio.gather` to fire multiple calls at the same time instead of awaiting each one sequentially.

You can use the following functions and types:

```python
{stubs}
```

You should start by describing the SQL schema.
"""

agent = Agent(
    'gateway/anthropic:claude-opus-4-5',
    instructions=instructions,
    model_settings=AnthropicModelSettings(anthropic_cache_messages=True),
)

prompt = 'Investigate why downloads increased recently, is this real usage or something else.'


async def main():
    print_output: list[str] = []

    def monty_print(_: object, content: str) -> None:
        print_output.append(content)

    repl = MontyRepl()

    external_functions = {
        'sql_query': sql_query,
        'plot': plot,
        'show_plot': show_plot,
        'display_table': display_table,
    }
    console = Console()
    combined: list[str] = []

    async with agent.iter(prompt) as agent_run:
        node = agent_run.next_node
        while True:
            while not isinstance(node, End):
                node = await agent_run.next(node)

            extracted = ExtractCode.extract(node.data.output)
            logfire.info(f'{extracted}')
            if extracted.comment:
                print()
                console.print(Markdown(f'---\n{node.data.output.strip()}\n'))

            if not extracted.code:
                print('model stopped')
                break

            try:
                with logfire.span('running monty repl', code=extracted.code):
                    Monty(
                        '\n'.join(combined + [extracted.code]),
                        type_check=True,
                        type_check_stubs=stubs,
                    )
                    output = await run_repl_async(
                        repl,
                        extracted.code,
                        external_functions=external_functions,
                        print_callback=monty_print,
                    )
            except (MontyError, MontyRuntimeError) as e:
                msg = f'Error running code: {e.display() if isinstance(e, MontyRuntimeError) else str(e)}'
            else:
                combined.append(extracted.code)
                msg = pydantic_core.to_json(output).decode()

            if print_output:
                msg += f'\n\nPrint Output:\n---\n{"".join(print_output)}\n---'
                print_output.clear()
            node = await agent_run.next(new_node(msg))


def new_node(msg: str) -> ModelRequestNode[None, str]:
    return ModelRequestNode(request=ModelRequest(instructions=instructions, parts=[UserPromptPart(content=msg)]))


@dataclass
class ExtractCode:
    """Extract Python code from an LLM response.

    Priority:
    1. First ```python code fence
    2. First code fence of any language
    3. Entire response as-is
    """

    code: str | None
    """Code extract from response.

    First ```(python|py) code fence, or first unexplained

    """
    comment: str | None

    @classmethod
    def extract(cls, response: str) -> ExtractCode:
        # Find all ```python/py blocks first, fall back to any code fence
        blocks = re.findall(r'```(?:python|py)\s*\n(.*?)```', response, re.DOTALL)
        if not blocks:
            blocks = re.findall(r'```\w*\s*\n(.*?)```', response, re.DOTALL)

        if blocks:
            code = '\n'.join(block.strip() for block in blocks)
            # Extract comment as text before the first code fence
            first = re.search(r'```', response)
            comment = response[: first.start()].strip() if first else None
            return cls(code=code, comment=comment or None)

        return cls(code=None, comment=response.strip())


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('exiting...')
