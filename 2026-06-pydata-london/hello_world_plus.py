import pydantic_monty


def get_who(greating: str) -> str:
    if greating == 'Hello':
        return 'World'
    else:
        return 'Pycon'


m = pydantic_monty.Monty(
    """
greating = 'Hello'
who = get_who(greating)
f'{greating} {who}'
""",
)

output = m.run(external_functions={'get_who': get_who})
print('output:', output)
