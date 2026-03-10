import asyncio
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import logfire
import pydantic_core
from pydantic_ai import Agent, ModelRequest, ModelRequestNode, UserPromptPart
from pydantic_graph import End
from pydantic_monty import MontyError, MontyRepl, MontyRuntimeError, run_repl_async

from external_functions import display_table, plot, show_plot, sql_query

logfire.configure()
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

You MUST return only one code block to execute. DO NOT return multiple code blocks.

Use `sql_query` to query the data, `plot` and `show_plot` to create visualizations,
and `display_table` to show formatted tables.

The runtime uses a restricted Python subset:
- you CANNOT use the standard library except builtin functions and the following modules: `sys`, `typing`, `asyncio`
- this means `json`, `collections`, `json`, `re`, `math`, `datetime`, `itertools`, `functools`, etc. are NOT available — use plain dicts, lists, and builtins instead
- you CANNOT use third party libraries
- you CANNOT define classes
- the python executor is a REPL — variables and state persist between code blocks, so you can build on previous results without redefining them
- you CAN define and reuse functions

The last expression evaluated is the return value.

You can use `print()` to get debug information while developing the code.

Parallelism: use `asyncio.gather` to fire multiple calls at the same time instead of awaiting each one sequentially.

You can use the following functions and types:

```python
{stubs}
```
"""

agent = Agent('gateway/anthropic:claude-sonnet-4-5', instructions=instructions)

prompt = 'Investigate why downloads increased recently.'


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

    async with agent.iter(prompt) as agent_run:
        node = agent_run.next_node
        while True:
            while not isinstance(node, End):
                node = await agent_run.next(node)

            extracted = ExtractCode.extract(node.data.output)
            logfire.info(f'{extracted}')
            if extracted.comment:
                print(f'LLM: {extracted.comment}')

            if not extracted.code:
                print('model stopped')
                response = input('> ')
                if response.lower() in {f'exit', ''}:
                    break
                else:
                    node = await agent_run.next(new_node(response))
                    continue

            try:
                with logfire.span('running monty repl', code=extracted.code):
                    output = await run_repl_async(
                        repl,
                        extracted.code,
                        external_functions=external_functions,
                        print_callback=monty_print,
                    )
            except (MontyError, MontyRuntimeError) as e:
                msg = f'Error running code: {e.display() if isinstance(e, MontyRuntimeError) else str(e)}'
            else:
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
        # Try ```python or ```py fences first
        m = re.search(r'```(?:python|py)\s*\n(.*?)```', response, re.DOTALL)
        if not m:
            # Try any code fence
            m = re.search(r'```\w*\s*\n(.*?)```', response, re.DOTALL)

        if m:
            code = m.group(1).strip()
            # Extract comment as text before the code fence
            comment = response[: m.start()].strip() or None
            return cls(code=code, comment=comment)

        return cls(code=None, comment=response.strip())


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('exiting...')
