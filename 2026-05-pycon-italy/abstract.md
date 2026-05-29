# Durable Agents: long running AI workflows in a flakey world

AI agents have evolved beyond simple question-&-answer interactions. Today’s systems orchestrate multi-step research, coordinate specialised sub-agents, and synthesise information across dozens of sources, tasks that can take minutes or hours rather than seconds. This shift toward more ambitious, longer-running workflows represents genuine progress in what AI can accomplish.

But there’s a catch: most AI frameworks are still trapped in what we call the “chat paradigm.” They’re synchronous, short-lived, and assume every interaction completes quickly. When a research agent has been working for fifteen minutes—querying data sources, processing documents, coordinating sub-agents—and then the API times out or the server restarts, all that progress is lost. The user has to start over from scratch.

This fragility creates real costs. You’re wasting compute on redundant work. You’re frustrating users who expected results. You’re undermining the economic case for AI automation. And as agents take on more valuable, longer-running tasks, these failure costs compound. The infrastructure that worked fine for chat doesn’t scale to production-grade workflows.

Durable execution solves this by automatically checkpointing workflow state as your agent runs. If anything fails—API timeout, rate limit, server crash, deployment—the system replays completed steps using cached results and continues from the exact point of failure. No redundant API calls, no lost progress. Your fifteen-minute research task recovers in seconds, not minutes.

This isn’t just about saving compute. Durability unlocks new architectural possibilities: agents that work asynchronously across hours or days, human-in-the-loop workflows where agents pause for approval, and multi-agent systems that handle partial failures gracefully. When recovery is cheap, you can design more ambitious systems.

In this talk, we’ll explore how Pydantic AI’s native Temporal integration makes durable execution practical. Temporal is a battle-tested platform used by Netflix, Uber, and Stripe for critical backend systems. Combined with Pydantic AI’s type-safe agent framework, you get agents that survive failures and maintain state across restarts.

We’ll cover concrete patterns including:

Wrapping agents for durability: Using TemporalAgent to automatically offload model requests and tool calls to durable activities, with minimal code changes to existing agents
State management: Keeping workflows responsive while preserving progress across failures
Streaming results: Delivering incremental outputs to users even as longer operations continue
Failure recovery: Configuring retry policies, timeouts, and graceful degradation
I’ll demonstrate these patterns with a multi-agent research system that coordinates parallel information gathering and synthesing structured results—then simulate failures to show exactly how recovery looks like.

This talk assumes familiarity with Python async/await patterns and basic experience building LLM applications. No prior Temporal knowledge is required—we’ll introduce the core concepts as we go. The patterns apply whether you’re using Pydantic AI specifically or thinking about durability for agent systems more broadly.

You’ll leave understanding when durable execution matters, how to implement it, and what production-ready AI architecture looks like. The gap between AI demos and AI products isn’t just about model capability—it’s about infrastructure that handles production realities.
