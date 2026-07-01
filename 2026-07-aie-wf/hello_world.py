import pydantic_monty

m = pydantic_monty.Monty("print(f'Hello {who}')", inputs=['who'])

m.run(inputs={'who': 'World'})
