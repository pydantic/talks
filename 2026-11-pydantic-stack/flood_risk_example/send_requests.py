"""Send a few questions to the running web app, so there are recent traces and online evals to look at.

Run with `make requests` while `make app-online-evals` is running. Each question starts a new chat,
goes through the same `/api/chat` endpoint as the page, and is evaluated by the app if online evals are on.
`make requests N=3` sends only the first three, and `APP_URL` points it at an app on another port.
"""

import json
import os
import sys
import time

import httpx2

APP_URL = os.getenv('APP_URL', 'http://localhost:8000')

QUESTIONS = [
    'Should we insure 34 State St?',
    'Should we insure 25 School St? Take the aerial image into account.',
    'How far is 2 Mechanic St from water, and how high above it?',
    'Which properties on Langdon St would we decline?',
    'Should we insure 12 Pearl St?',
    'Which five properties are closest to water?',
    'Compare 197 State St and 160 Main St: which is the better risk?',
    'Triage the properties on Elm St and show me the referrals.',
]


def ask(client: httpx2.Client, question: str) -> None:
    """Ask one question in a fresh chat and print a one-line summary of what came back."""
    client.post('/api/reset').raise_for_status()
    start = time.perf_counter()
    text: dict[int, str] = {}
    maps: list[str] = []
    requests = 0
    error = None
    with client.stream('POST', '/api/chat', json={'message': question}) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith('data: '):
                continue
            event = json.loads(line.removeprefix('data: '))
            if event['type'] == 'text':
                text[event['segment']] = event['text']
            elif event['type'] == 'map':
                maps.append(event['title'])
            elif event['type'] == 'requests':
                requests = event['count']
            elif event['type'] == 'error':
                error = event['message']

    answer = ' '.join(' '.join(text.values()).split())
    summary = f'ERROR: {error}' if error else answer[:110] + ('…' if len(answer) > 110 else '')
    print(f'{time.perf_counter() - start:5.1f}s  {requests:4d} API requests  {question}\n        {summary}')


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(QUESTIONS)
    with httpx2.Client(base_url=APP_URL, timeout=300) as client:
        try:
            config = client.get('/api/config').raise_for_status().json()
        except httpx2.ConnectError:
            sys.exit(f'The web app is not running at {APP_URL}. Start it with `make app-online-evals`.')
        if not config['online_evals']:
            print('Online evals are OFF in the running app: these requests will be traced but not evaluated.')
            print('Restart it with `make app-online-evals` to get evaluator results.\n')
        for question in QUESTIONS[:limit]:
            ask(client, question)


if __name__ == '__main__':
    main()
