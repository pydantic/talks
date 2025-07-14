import argparse
import asyncio
import json
import sys
from pathlib import Path

from rich import print
from rich.markdown import Markdown
from rich.prompt import Prompt

from ghost_writer.main import generate_blog_post


def _path(path_str: str) -> Path:
    if not (path := Path(path_str)).is_file():
        raise argparse.ArgumentTypeError(f'{path_str!r} must be an existing file')

    return path


async def main(info_file: Path | None) -> None:
    print('Pydantic Ghost Writer - Blog Post Generator')
    print('=' * 50)

    if info_file is not None:
        data = json.loads(info_file.read_bytes())
    else:
        data = {}
        data['topic'] = Prompt.ask('Blog post topic/headline')
        data['author'] = Prompt.ask('Author name', default='Pydantic Team').strip()
        data['author_role'] = Prompt.ask("Author role (e.g., 'Founder', 'Core Developer')").strip()

        print('\nContent guidance:')
        data['user_requirements'] = Prompt.ask('  Additional requirements/direction').strip()
        data['opinions'] = Prompt.ask('  Specific opinions or takes to include').strip()
        data['examples'] = Prompt.ask('  Specific examples or case studies to mention').strip()

        reference_links: list[str] = []
        while True:
            link = Prompt.ask('  Enter a reference link (or press Enter to finish)').strip()
            if not link:
                break
            reference_links.append(link)
        data['reference_links'] = reference_links

    print('\nGenerating blog post...')

    response = await generate_blog_post(
        topic=data['topic'],
        author=data.get('author', 'Pydantic Team'),
        author_role=data.get('author_role', ''),
        user_requirements=data.get('user_requirements', ''),
        opinions=data.get('opinions', ''),
        examples=data.get('examples', ''),
        reference_links=data.get('reference_links', []),
    )

    print('\n' + '=' * 50)
    print('GENERATED BLOG POST:')
    print('=' * 50)
    print(Markdown(response))


if __name__ == '__main__':
    color = {'color': True} if sys.version_info >= (3, 14) else {}
    parser = argparse.ArgumentParser(
        'ghost_writer',
        description='A Pydantic AI agent to write content',
        **color,
    )
    parser.add_argument(
        'file',
        help='JSON file containing blog post info',
        nargs='?',
        type=_path,
        default=None,
    )

    args = parser.parse_args()
    asyncio.run(main(args.file))
