from pydantic_monty import Monty

with Monty() as monty:
    with monty.checkout() as session:
        session.feed_run("print(f'Hello {who}')", inputs={'who': 'World'})
