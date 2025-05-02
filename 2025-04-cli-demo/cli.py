import argparse
import random
from time import sleep
from urllib.parse import quote_plus

import logfire
from opentelemetry.trace import format_trace_id

logfire.configure(
    # Maybe token info here?
    # If it's an internal service you might as well hardcode the token in the code you distribute to your team
    console=False,
    service_name='cli',
    service_version='0.1.0',
)


def fibonacci(n: int) -> int:
    """Compute the Fibonacci number recursively."""
    with logfire.span('fibonacci({n})', n=n):
        if random.uniform(0, 1) < 0.25:
            sleep(1)
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        else:
            return fibonacci(n - 1) + fibonacci(n - 2)


@logfire.instrument
def divide(numerator: int, denominator: int) -> float:
    """Divide two numbers."""
    return numerator / denominator


def main():
    # Create the top-level parser
    parser = argparse.ArgumentParser(description='Demo CLI with multiple commands')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    subparsers.required = True

    # Create parser for the "fib" command
    fib_parser = subparsers.add_parser('fib', help='Compute Fibonacci number')
    fib_parser.add_argument('n', type=int, help='Which Fibonacci number to compute')

    # Create parser for the "divide" command
    divide_parser = subparsers.add_parser('divide', help='Divide two numbers')
    divide_parser.add_argument('numerator', type=int, help='Numerator')
    divide_parser.add_argument('denominator', type=int, help='Denominator')

    # Parse arguments
    args = parser.parse_args()

    # Process commands
    with logfire.span('main', command=args.command) as span:
        try:
            if args.command == 'fib':
                result = fibonacci(args.n)
                print(f'The {args.n}th Fibonacci number is: {result}')
            elif args.command == 'divide':
                result = divide(args.numerator, args.denominator)
                print(f'The result of dividing {args.numerator} by {args.denominator} is: {result}')
        except Exception as e:
            if span.context is not None:
                trace_id = format_trace_id(span.context.trace_id)
                query = f"trace_id='{trace_id}'"
                url = f'https://logfire-eu.pydantic.info/adriangb/starter-project?q={quote_plus(query)}'
                print(f'Error occurred during command "{args.command}".\n\n{e}\n\nTrace: {url}')


if __name__ == '__main__':
    main()
