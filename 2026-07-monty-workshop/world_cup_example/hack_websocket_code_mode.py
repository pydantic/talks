"""A `CodeMode` variant that runs sandbox code on remote Monty workers over a WebSocket.

The stock `CodeMode` capability executes `run_code` snippets on a pool of local
`monty` subprocesses (`pydantic_monty.Monty` + the sync snapshot API). This module
swaps that execution backend for `pydantic_monty.AsyncMontyWebsocket`: each REPL
session dials a `ws://`/`wss://` URL (a relay, or any server that bridges the
connection to a worker) and drives the remote worker through the async snapshot
API. Everything else -- the `run_code` tool schema and description, the sandboxed
function catalog, tool dispatch through the agent's `ToolManager`, retry/error
mapping -- is inherited from the harness implementation.

Usage (drop-in replacement for `CodeMode` in `agent.py`):

    from .websocket_code_mode import WebsocketCodeMode

    agent = Agent(
        ...,
        capabilities=[
            WebsocketCodeMode(
                url='ws://127.0.0.1:8799',
                mount=MountDir(host_path=..., virtual_path='/output', mode='read-write'),
            )
        ],
    )

Note: the dialed URL must already contain any session/rendezvous routing the
relay needs (e.g. a `/<uuid>/parent` path).
"""

from __future__ import annotations

import asyncio
from collections.abc import Container, Coroutine
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import AbstractToolset, RunContext
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UserError
from pydantic_ai.messages import ToolCallPart, ToolReturn, ToolReturnPart
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import AgentDepsT, ToolDenied
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai_harness import CodeMode
from pydantic_ai_harness._monty_exec import PrintCapture, is_sandbox_panic
from pydantic_ai_harness.code_mode._toolset import (
    _TOOL_RETURN_CONTENT_TA,  # pyright: ignore[reportPrivateUsage]
    CodeModeToolset,
    _contains_multimodal,  # pyright: ignore[reportPrivateUsage]
    _global_mode_is_sequential,  # pyright: ignore[reportPrivateUsage]
    _RunCodeTool,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_monty import (
    AsyncFunctionSnapshot,
    AsyncMontySession,
    AsyncMontyWebsocket,
    AsyncNameLookupSnapshot,
    AsyncSnapshot,
    ExternalException,
    ExternalReturnValue,
    ExternalSettledResult,
    MontyComplete,
    MontyCrashedError,
    MontyRuntimeError,
    MontySyntaxError,
    MontyTypingError,
)

# Dispatch callback: given the sandbox function name and keyword arguments,
# perform the host-side tool call and return the (serialized) result.
DispatchFn = Coroutine[Any, Any, Any]
# A coroutine not yet scheduled on the event loop, or its running Task.
PendingCall = asyncio.Task[Any] | Coroutine[Any, Any, Any]


@dataclass
class _WebsocketRunState:
    """Remote Monty resources shared by every toolset view created during one agent run.

    Async sibling of the harness's `_MontyRunState`: the pool dials a WebSocket URL
    instead of spawning local subprocesses, so setup/teardown are awaitable.
    """

    url: str
    max_processes: int | None = None
    checkout_timeout: float | None = None
    request_timeout: float | None = None
    pool: AsyncMontyWebsocket | None = None
    session: AsyncMontySession | None = None
    has_executed_feed: bool = False
    _pool_stack: AsyncExitStack = field(default_factory=AsyncExitStack, repr=False)
    _session_stack: AsyncExitStack = field(default_factory=AsyncExitStack, repr=False)

    async def get_session(self, *, type_check: bool, type_check_stubs: str | None) -> AsyncMontySession:
        """Return the run's live remote REPL session, dialing the pool on first use."""
        if self.pool is None:
            self.pool = await self._pool_stack.enter_async_context(
                AsyncMontyWebsocket(
                    self.url,
                    max_processes=self.max_processes,
                    checkout_timeout=self.checkout_timeout,
                    request_timeout=self.request_timeout,
                )
            )
        if self.session is None:
            self.session = await self._session_stack.enter_async_context(
                self.pool.checkout(type_check=type_check, type_check_stubs=type_check_stubs)
            )
        return self.session

    async def reset(self) -> None:
        """Release the current connection and make the next call start a fresh REPL."""
        await self._session_stack.aclose()
        self._session_stack = AsyncExitStack()
        self.session = None
        self.has_executed_feed = False

    async def close(self) -> None:
        """Release the session and close the owning pool."""
        await self.reset()
        await self._pool_stack.aclose()
        self._pool_stack = AsyncExitStack()
        self.pool = None


@dataclass
class _AsyncMontyExecutor:
    """Drives a remote Monty REPL to completion, dispatching external calls to a host callback.

    Async sibling of the harness's `MontyExecutor`: identical control flow, but every
    `resume` on the websocket-backed snapshots is awaitable. Single-use -- construct a
    fresh executor for each run.
    """

    dispatch: Any  # Callable[[str, dict[str, Any]], Coroutine[Any, Any, Any]]
    valid_names: Container[str]
    sequential_names: set[str] = field(default_factory=set)
    global_sequential: bool = False

    # Parallel calls deferred but not yet resolved, keyed by Monty call id.
    _pending: dict[int, PendingCall] = field(default_factory=dict, init=False)
    # Parallel results awaited early at a sequential barrier, before their FutureSnapshot is reached.
    _pre_resolved: dict[int, ExternalSettledResult] = field(default_factory=dict, init=False)

    async def run(self, state: AsyncSnapshot) -> MontyComplete:
        """Drive the REPL from `state` until it completes."""
        try:
            while not isinstance(state, MontyComplete):
                if isinstance(state, AsyncNameLookupSnapshot):
                    # Leave the name undefined so the sandbox raises `NameError`.
                    state = await state.resume()
                elif isinstance(state, AsyncFunctionSnapshot):
                    state = await self._handle_function(state)
                else:
                    state = await self._resolve_futures(state)
        finally:
            cancelled: list[asyncio.Task[Any]] = []
            for call in self._pending.values():
                if isinstance(call, asyncio.Task):
                    call.cancel()
                    cancelled.append(call)
                else:
                    call.close()
            if cancelled:
                await asyncio.gather(*cancelled, return_exceptions=True)
        return state

    async def _handle_function(self, snapshot: AsyncFunctionSnapshot) -> AsyncSnapshot:
        """Dispatch (or defer) a single external function call."""
        if snapshot.is_os_function:
            # OS calls (env, clock, filesystem) are answered from the feed's mounts and the
            # `os=` handler captured at `feed_start`, falling back to monty's unhandled default.
            return await snapshot.resume_auto()

        name = snapshot.function_name
        if name not in self.valid_names:
            return await snapshot.resume({'exception': NameError(f'Unknown function: {name}')})

        if snapshot.args:
            return await snapshot.resume(
                {'exception': TypeError(f'{name}() does not accept positional arguments; use keyword arguments')}
            )

        if name in self.sequential_names:
            # Rendered as `def` (sync), so the sandbox code doesn't `await` the result --
            # resolve inline. Await pending parallel tasks first (barrier) for ordering.
            for cid in list(self._pending):
                self._pre_resolved[cid] = await _await_external(self._pending.pop(cid))
            return await snapshot.resume(await _await_external(self.dispatch(name, snapshot.kwargs)))

        # Deferred execution -- resolved later at AsyncFutureSnapshot.
        call = self.dispatch(name, snapshot.kwargs)
        if self.global_sequential:
            # Keep the bare coroutine unscheduled; it's awaited one-at-a-time to avoid interleaving.
            self._pending[snapshot.call_id] = call
        else:
            # Schedule now as a Task so concurrently-deferred calls actually run in parallel.
            self._pending[snapshot.call_id] = asyncio.ensure_future(call)
        return await snapshot.resume({'future': ...})

    async def _resolve_futures(self, snapshot: Any) -> AsyncSnapshot:
        """Resolve the deferred calls an `AsyncFutureSnapshot` is waiting on."""
        pending_ids = snapshot.pending_call_ids
        results: dict[int, ExternalSettledResult] = {}
        for cid in pending_ids:
            if cid in self._pre_resolved:
                results[cid] = self._pre_resolved.pop(cid)
            elif self.global_sequential:
                results[cid] = await _await_external(self._pending.pop(cid))

        gather_ids = [cid for cid in pending_ids if cid not in results]
        if gather_ids:
            settled = await asyncio.gather(*(self._pending[cid] for cid in gather_ids), return_exceptions=True)
            for cid, outcome in zip(gather_ids, settled):
                del self._pending[cid]
                results[cid] = _wrap_gathered(outcome)

        return await snapshot.resume(results)


async def _await_external(call: PendingCall) -> ExternalReturnValue | ExternalException:
    """Await a single deferred call and wrap its outcome for Monty."""
    try:
        result = await call
    except Exception as exc:  # noqa: BLE001 -- any tool failure must be surfaced inside the sandbox
        return ExternalException(exception=exc)
    return ExternalReturnValue(return_value=result)


def _wrap_gathered(outcome: Any) -> ExternalReturnValue | ExternalException:
    """Wrap an `asyncio.gather(return_exceptions=True)` outcome for Monty."""
    if isinstance(outcome, Exception):
        return ExternalException(exception=outcome)
    if isinstance(outcome, BaseException):  # pragma: no cover
        raise outcome
    return ExternalReturnValue(return_value=outcome)


@dataclass
class WebsocketCodeModeToolset(CodeModeToolset[AgentDepsT]):
    """`CodeModeToolset` that executes `run_code` on remote workers over a WebSocket.

    Inherits tool splitting, catalog rendering, and the `run_code` tool definition
    from `CodeModeToolset`; only the execution backend (`__aenter__`/`__aexit__`
    lifecycle and the `run_code` branch of `call_tool`) is replaced.
    """

    url: str = 'ws://127.0.0.1:8799'
    """`ws://`/`wss://` URL to dial -- a relay, or any server that bridges to a worker."""

    max_processes: int | None = None
    """Cap on concurrent connections (defaults to the CPU count)."""

    checkout_timeout: float | None = None
    """Seconds `checkout()` waits for capacity before raising `TimeoutError`; `None` waits forever."""

    request_timeout: float | None = 60.0
    """Hard per-call deadline in seconds; a worker exceeding it is killed and the call retries."""

    _ws_state: _WebsocketRunState | None = field(default=None, init=False, repr=False, compare=False)

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Update the wrapped toolset for this step while preserving remote REPL state."""
        new_self = await super().for_run_step(ctx)
        if new_self is not self:
            assert isinstance(new_self, WebsocketCodeModeToolset)
            new_self._ws_state = self._ws_state
        return new_self

    async def __aenter__(self) -> WebsocketCodeModeToolset[AgentDepsT]:
        """Enter the wrapped toolset and prepare lazy remote-pool resources for this run."""
        await self.wrapped.__aenter__()
        self._ws_state = _WebsocketRunState(
            url=self.url,
            max_processes=self.max_processes,
            checkout_timeout=self.checkout_timeout,
            request_timeout=self.request_timeout,
        )
        return self

    async def __aexit__(self, *args: object) -> bool | None:
        """Exit the wrapped toolset, then close the remote pool."""
        ws_state = self._ws_state
        assert ws_state is not None
        self._ws_state = None
        try:
            return await self.wrapped.__aexit__(*args)
        finally:
            await ws_state.close()

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        """Execute Python code on the remote worker, or pass through to a native tool."""
        if not isinstance(tool, _RunCodeTool):
            # Native (non-sandboxed) tool -- pass through to the wrapped toolset.
            return await self.wrapped.call_tool(name, tool_args, ctx, tool)

        code = tool_args['code']
        restart = tool_args.get('restart', False)

        ws_state = self._ws_state
        assert ws_state is not None, '`WebsocketCodeModeToolset` must be entered before calling `run_code`'

        if restart:
            await ws_state.reset()

        fresh_repl = not ws_state.has_executed_feed

        callable_defs = tool.callable_defs
        sanitized_to_original = tool.sanitized_to_original

        # Build a ToolManager for the sandbox's inner tools so that sandboxed
        # tool calls go through the standard validation/execution path.
        parent_tm = ctx.tool_manager
        assert parent_tm is not None, 'WebsocketCodeModeToolset requires ctx.tool_manager to be set'
        tool_manager = ToolManager(
            toolset=self.wrapped,
            root_capability=parent_tm.root_capability,
            ctx=ctx,
            tools=tool.wrapped_tools,
        )

        global_sequential = _global_mode_is_sequential(tool_manager.get_parallel_execution_mode)
        sequential_tools = {name for name, td in callable_defs.items() if td.sequential}

        # Collect nested tool calls and returns keyed by tool_call_id so they
        # can be attached as metadata on the run_code ToolReturnPart.
        nested_calls: dict[str, ToolCallPart] = {}
        nested_returns: dict[str, ToolReturnPart] = {}
        call_counter = 0

        async def dispatch_tool_call(sandbox_name: str, kwargs: dict[str, Any]) -> Any:
            """Dispatch a single tool call from inside the sandbox."""
            nonlocal call_counter
            original_name = sanitized_to_original.get(sandbox_name, sandbox_name)
            call_counter += 1
            parent_id = ctx.tool_call_id or 'pyd_ai_code_mode'
            tool_call_id = f'{parent_id}__{call_counter}'
            call_part = ToolCallPart(tool_name=original_name, args=kwargs, tool_call_id=tool_call_id)
            nested_calls[tool_call_id] = call_part

            try:
                result = await tool_manager.handle_call(call_part, wrap_validation_errors=False)
            except (CallDeferred, ApprovalRequired) as e:
                raise UserError(
                    f'Tool {original_name!r} raised {type(e).__name__} inside code mode, '
                    'but no `HandleDeferredToolCalls` capability resolved it. Add a handler '
                    'capability on the agent so deferred and approval-required calls can '
                    'be resolved inline.'
                ) from e

            if isinstance(result, ToolDenied):
                nested_returns[tool_call_id] = ToolReturnPart(
                    tool_name=original_name,
                    content=result.message,
                    tool_call_id=tool_call_id,
                    outcome='denied',
                )
                raise RuntimeError(f'Tool {original_name!r} call denied: {result.message}')  # noqa: TRY004

            return_metadata: Any = None
            if isinstance(result, ToolReturn):
                return_metadata = result.metadata
                result = result.return_value

            nested_returns[tool_call_id] = ToolReturnPart(
                tool_name=original_name,
                content=result,
                tool_call_id=tool_call_id,
                metadata=return_metadata,
            )

            # Serialize to JSON-compatible form so Monty receives only plain data.
            return _TOOL_RETURN_CONTENT_TA.dump_python(result)

        # Type-check only the first executed snippet (same policy as the local backend).
        type_check = fresh_repl and bool(callable_defs)
        type_check_stubs = self._build_type_check_stubs(callable_defs) if type_check else None

        capture = PrintCapture()

        try:
            session = await ws_state.get_session(type_check=type_check, type_check_stubs=type_check_stubs)
            try:
                monty_state = await session.feed_start(
                    code,
                    print_callback=capture.callback,
                    os=self.os_access,
                    mount=self.mount,
                    skip_type_check=not type_check,
                )
                completed = await _AsyncMontyExecutor(
                    dispatch=dispatch_tool_call,
                    valid_names=callable_defs,
                    sequential_names=sequential_tools,
                    global_sequential=global_sequential,
                ).run(monty_state)
            except MontyRuntimeError:
                # The session is idle again and keeps assignments made before the failing line.
                ws_state.has_executed_feed = True
                raise
            ws_state.has_executed_feed = True
        except MontySyntaxError as e:
            if fresh_repl:
                # No code ran, so discard the checkout-time type stubs.
                await ws_state.reset()
            raise ModelRetry(f'Syntax error in code:\n{capture.prepend_to(e.display())}') from e
        except MontyTypingError as e:
            await ws_state.reset()
            raise ModelRetry(f'Type error in code:\n{capture.prepend_to(e.display())}') from e
        except MontyRuntimeError as e:
            raise ModelRetry(f'Runtime error:\n{capture.prepend_to(e.display())}') from e
        except MontyCrashedError as e:
            # The remote worker died mid-feed (crash, request timeout, or dropped
            # connection); the REPL state died with it. Reset so the retry dials fresh.
            await ws_state.reset()
            raise ModelRetry(
                'The code crashed the sandbox worker and the session was reset. Revise the code and try again.'
            ) from e
        except BaseException as e:
            if not is_sandbox_panic(e):
                await ws_state.reset()
                raise
            await ws_state.reset()
            raise ModelRetry(
                'The code aborted inside the sandbox and the session was reset. Revise the code and try again.'
            ) from e

        result = completed.output
        printed = capture.joined

        # Validate result to reconstruct multimodal types (e.g. BinaryContent from
        # serialized dicts) so they flow through to the model natively.
        if result is not None:
            result = _TOOL_RETURN_CONTENT_TA.validate_python(result)

        if not printed:
            return_value: Any = result if result is not None else {}
        elif result is None:
            return_value = {'output': printed}
        elif _contains_multimodal(result):
            return_value = [printed, *result] if isinstance(result, list) else [printed, result]
        else:
            return_value = {'output': printed, 'result': result}

        return ToolReturn(
            return_value=return_value,
            metadata={'code_mode': True, 'tool_calls': nested_calls, 'tool_returns': nested_returns},
        )


@dataclass
class WebsocketCodeMode(CodeMode[AgentDepsT]):
    """`CodeMode` that executes `run_code` snippets on remote Monty workers over a WebSocket.

    Same model-facing behaviour as `CodeMode` (tool catalog, retries, mounts, OS access);
    only the execution backend differs: instead of spawning local `monty` subprocesses,
    each REPL session dials `url` via `pydantic_monty.AsyncMontyWebsocket`.
    """

    url: str = 'ws://127.0.0.1:8799'
    """`ws://`/`wss://` URL to dial, including any routing path the relay needs."""

    max_processes: int | None = None
    """Cap on concurrent connections (defaults to the CPU count)."""

    checkout_timeout: float | None = None
    """Seconds a checkout waits for capacity before raising `TimeoutError`; `None` waits forever."""

    request_timeout: float | None = 60.0
    """Hard per-call deadline in seconds for each remote turn; `None` waits indefinitely."""

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Wrap the agent's assembled toolset with the websocket-backed code-mode toolset."""
        return WebsocketCodeModeToolset(
            wrapped=toolset,
            tool_selector=self.tools,
            max_retries=self.max_retries,
            dynamic_catalog=self.dynamic_catalog,
            os_access=self.os_access,
            mount=self.mount,
            url=self.url,
            max_processes=self.max_processes,
            checkout_timeout=self.checkout_timeout,
            request_timeout=self.request_timeout,
        )
