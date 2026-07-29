from pydantic_monty import Monty


def get_who(greating: str) -> str:
    if greating == 'Hello':
        return 'World'
    else:
        return 'Pydantic Workshop'


code = """
greating = 'Hi'
who = get_who(greating)
f'{greating} {who}'
"""

with Monty() as monty:
    with monty.checkout() as session:
        output = session.feed_run(code, external_lookup={'get_who': get_who})
        print('output:', output)
