



# Pydantic Monty

Code execution is great ... until you need to run that code in a production setting.



















## Monty

A minimal, secure Python interpreter written in Rust for use by AI.














## Inspiration


* **Codemode** from Cloudflare
* **Programmatic Tool Calling** from Anthropic
* **Code Execution with MCP** from Anthropic
* and many more...











## Why

Making agents executing code trivial and reliable.

* Completely block access to the host environment
* Call functions on the host - only functions you give it access to
* Run typechecking - with ty
* mounted filesystem access - complete control over which files are accessible
* Be snapshotted to bytes at external function calls
* Startup is <1us
* Be called from Rust, Python, or Javascript
* no (extra) sandbox required

This is NOT to run your existing Python application, that will probably never be possible.


**That's fine with me**








## Hello World

```bash
pip install pydantic-monty
```

```python title="hello-world.py"
import pydantic_monty

m = pydantic_monty.Monty("print(f'Hello {who}')", inputs=['who'])

m.run(inputs={'who': 'World'})
```









## Demo...

😱













## Learn more

* github.com/pydantic/monty
* x.com/pydantic
