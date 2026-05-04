Create a new deck with deckx in @deck/deck.mdx for the talk described in @abstract.md.

Look at previous talks I've given about Monty like @../2026-05-codemode/ and @../2026-03-monty/ for inspiration.

The deckx deck should use tabs.

The deck should have the following structure:

* cover slide
* whoami - samuel colvin, creator of Pyantic
* what is "codemode" - a few bullets, then we need to create a simple react component diagram to demonstrate how we add two tools (`run_code` and `find_tool`s) "in front" of the standard tools the agent has access to
* Sandbox, not a Desert - a few bullet points - "a curated list beats "the kitchen sink"", the point is that most agents (we're not talking about coding agents here) are required to perform a fairly specific task, even if that tasks seems pretty broad "Analyse BI data to find efficencies" the actual range of tools they require is quite small. Add a component showing two boxes: one has 3 tools (`sql_query`, `read_file`, `write_file`, `load_skill`) the other has hundres of tools including as many linux utilities as you can think of `awk`, `sed`, `gcc`, `less`, `more` etc all in a massive confusing word cloud.
* Introducing Monty - a few bullets, including some downsides - read https://github.com/pydantic/monty. Finish the slide with a big `uv add pydantic-monty / npm i @pydantic/monty`
* The advantages - a chart showing latency 1us vs 1s, snapshot size 1kb vs 1gb, price $0 vs $?
* "Demo" statement slide
* Thank you
