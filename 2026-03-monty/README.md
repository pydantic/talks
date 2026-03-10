



# Pydantic Monty

## Py AI - March 2026

Code execution is a great idea ... until you need to use it.














## Inspiration


* **Codemode** from Cloudflare
* **Programmatic Tool Calling** from Anthropic
* **Code Execution with MCP** from Anthropic
* **SmolAgents** from huggingface
* and many more...



















## Monty

A minimal, secure Python interpreter written in Rust, for running code written by AI agents.













## Hello World

```bash
pip install pydantic-monty
```

```python title="hello-world.py"
import pydantic_monty

m = pydantic_monty.Monty("print(f'Hello {who}')", inputs=['who'])

m.run(inputs={'who': 'World'})
```












## Why

Making agents executing code trivial and reliable.

* **Block:** Completely block access to the host environment
* **Allow:** Call functions on the host - only functions you give it access to
* **Typecheck:** Run typechecking - with ty
* **Allow:** mounted filesystem access - complete control over which files are accessible
* **Durable Execution:** Be snapshotted to bytes at external function calls
* **Latency:** Startup is <1us
* **Portability:** Be called from Rust, Python, or Javascript
* **Simplicity:** no (extra) sandbox required












**Tradeoff:** This is NOT to run your existing Python application, that will probably never be possible.


**That's fine with me**
















## The impotence vs. chaos Continuum

When you harness an LLM, you're making a trade-off between the control you retain and the capability you grant.

At one extreme, the LLM picks a function name and fills in some JSON.

At the other, you've handed a neural network your mouse and keyboard.

And there are other trade-offs...











## Start left, move right

It's like whitelisting - instead of blocking things you don't want (and hoping you've covered everything), you start with nothing, add some functionality ... and wait for users to demand more.










## Demo...

😱













## Learn more

* github.com/pydantic/monty
* x.com/pydantic
