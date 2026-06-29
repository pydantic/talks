# Your agent needs a sandbox, not a desert

Everyone agrees agents need code execution.

That agreement lasts right up until you ask how to do it.

The default answer is usually something like "My agent needs a full Linux VM to succeed". That's a very convenient answer for sandbox providers, but I think it's often incorrect.

In many real-world agent workflows, the model does not need a whole computer. It does not need arbitrary packages, shell access, CPython, node, let alone `awk` `sed` and `gcc`. It needs a small amount of safe, expressive compute: enough to write code, call tools, and keep intermediate state out of the context window.

That is the idea behind Monty: a minimal Python interpreter, written in Rust, designed specifically for running code written by agents.

In this talk, I'll argue that for a surprisingly large class of agent systems, a curated set of tools in a custom runtime is better than a full sandbox. Not because full sandboxes are bad, but because they solve a much larger problem than most embedded agents actually have. And you pay for that mismatch in complexity, cost, operational pain, and 100,000X higher latency.

Sandboxes are great, but there's such a thing as too much sand - in many scenarios the constraints and limitations of a custom built, minimal sandbox are a feature, not a bug.
