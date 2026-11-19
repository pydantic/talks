"""A minimal terminal chat loop for the agent.

pydantic-ai's built-in `Agent.to_cli()` prints a `Called tool` line for every tool return, which with
Codemode means a wall of `run_code` lines per answer. This loop streams the model's text the same way but
collapses tool activity into one spinner line that disappears once the answer arrives.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from pydantic_ai import Agent
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent, ModelMessage
from rich import get_console
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.status import Status

HISTORY_PATH = Path.home() / '.pydantic-ai' / 'ai-trade-history.txt'
EXIT_COMMANDS = {'/exit', '/quit', 'exit', 'quit'}


async def ask(
    agent: Agent[None, str],
    console: Console,
    prompt: str,
    messages: Sequence[ModelMessage],
) -> list[ModelMessage]:
    """Run one turn, streaming markdown output to `console`, and return the full message history."""
    status = Status('[dim]Working on it…[/dim]', console=console)
    status.start()
    live = Live('', refresh_per_second=15, console=console, vertical_overflow='visible')
    live_started = False

    pieces: list[str] = []  # completed chunks of model text, in order
    current = ''  # text streaming from the current model request
    tool_calls = 0
    pending: set[str] = set()

    def render() -> Markdown:
        parts = [*pieces, current] if current else list(pieces)
        if pending:
            noun = 'call' if tool_calls == 1 else 'calls'
            parts.append(f'> _running code… ({tool_calls} {noun})_')
        return Markdown('\n\n'.join(parts), code_theme='monokai')

    with status, ExitStack() as stack:
        async with agent.iter(prompt, message_history=messages) as run:
            async for node in run:
                if Agent.is_model_request_node(node):
                    async with node.stream(run.ctx) as stream:
                        if not live_started:
                            status.stop()
                            stack.enter_context(live)
                            live_started = True
                        async for content in stream.stream_output(debounce_by=None):
                            current = str(content)
                            live.update(render())
                elif Agent.is_call_tools_node(node):
                    if current:
                        pieces.append(current)
                        current = ''
                    async with node.stream(run.ctx) as stream:
                        async for event in stream:
                            if isinstance(event, FunctionToolCallEvent):
                                pending.add(event.tool_call_id)
                                tool_calls += 1
                            elif isinstance(event, FunctionToolResultEvent):
                                pending.discard(event.tool_call_id)
                            live.update(render())
            live.update(render())
            assert run.result is not None
            return run.result.all_messages()


async def run_repl(agent: Agent[None, str], prog_name: str) -> None:
    console = get_console()  # shared with charts.py so its prints sit above the live display
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    session: PromptSession[str] = PromptSession(history=FileHistory(str(HISTORY_PATH)))
    messages: list[ModelMessage] = []
    console.print(f'[dim]{prog_name} — ask a question, Ctrl-D or /exit to quit[/dim]')
    while True:
        try:
            text = await session.prompt_async(f'{prog_name} ➤ ')
        except EOFError, KeyboardInterrupt:
            console.print('[dim]Exiting…[/dim]')
            return
        text = text.strip()
        if not text:
            continue
        if text.lower() in EXIT_COMMANDS:
            console.print('[dim]Exiting…[/dim]')
            return
        try:
            messages = await ask(agent, console, text, messages)
        except KeyboardInterrupt:
            console.print('[dim]Interrupted[/dim]')
        except Exception as e:  # keep the loop alive on model/tool errors
            console.print(f'[red]{type(e).__name__}:[/red] {e}')
