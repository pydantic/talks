import asyncio

from rich import print
from rich.prompt import Prompt

from ghost_writer.main import generate_blog_post


async def main():
    print('Pydantic Ghost Writer - Blog Post Generator')
    print('=' * 50)

    # Enhanced blog-specific inputs
    topic = Prompt.ask('Blog post topic/headline')
    author = Prompt.ask("Author name (or press Enter for 'Pydantic Team')").strip() or 'Pydantic Team'
    author_role = Prompt.ask("Author role (e.g., 'Founder', 'Core Developer')").strip()

    print('\nContent guidance:')
    user_requirements = Prompt.ask('  Additional requirements/direction').strip()
    opinions = Prompt.ask('  Specific opinions or takes to include').strip()
    examples = Prompt.ask('  Specific examples or case studies to mention').strip()

    # Get reference links
    reference_links: list[str] = []
    while True:
        link = Prompt.ask('  Enter a reference link (or press Enter to finish)').strip()
        if not link:
            break
        reference_links.append(link)

    print('\nGenerating blog post...')
    response = await generate_blog_post(
        topic=topic,
        author=author,
        author_role=author_role,
        user_requirements=user_requirements,
        opinions=opinions,
        examples=examples,
        reference_links=reference_links,
    )

    print('\n' + '=' * 50)
    print('GENERATED BLOG POST:')
    print('=' * 50)
    print(response)


if __name__ == '__main__':
    asyncio.run(main())
