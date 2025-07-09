import asyncio
from pathlib import Path

import logfire
from pydantic_ai import Agent
from typing_extensions import TypedDict

# Configure Logfire for observability
logfire.configure(scrubbing=False)
logfire.instrument_mcp()
logfire.instrument_pydantic_ai()

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(
    role: str,
    content_type: str,
) -> str:
    """Load and combine prompts for specific content type and role."""
    prompt_parts = []

    # Load shared/global rules
    try:
        with open("prompts/shared/global_styleguide.txt", "r") as f:
            prompt_parts.append(f.read())
    except FileNotFoundError:
        pass

    # Load brand voice (if writer)
    if role == "writer":
        try:
            with open("prompts/shared/brand_guidelines.txt", "r") as f:
                prompt_parts.append(f.read())
        except FileNotFoundError:
            pass

    # Load content-specific prompt
    try:
        with open(f"prompts/{role}/{content_type}.txt", "r") as f:
            prompt_parts.append(f.read())
    except FileNotFoundError:
        prompt_parts.append(f"No specific prompt found for {role}/{content_type}")

    return "\n\n".join(prompt_parts)


# Writer agent for generating blog content
writer_agent = Agent(
    "anthropic:claude-3-5-sonnet-latest",
    instructions=load_prompt(role="writer", content_type="blog_post"),
)


# Reviewer agent for quality control
class Score(TypedDict):
    score: int
    reason: str


reviewer_agent = Agent(
    "anthropic:claude-3-5-sonnet-latest",
    output_type=Score,
    instructions=load_prompt(role="reviewer", content_type="blog_post"),
)


# Tools for reviewer agent to access guidelines
@reviewer_agent.tool_plain
async def get_shared_guidelines() -> str:
    """Get all shared guidelines from the prompts/shared directory."""

    guidelines: list[str] = []

    shared_dir = PROMPTS_DIR / "shared"
    for text_file in shared_dir.glob("*.txt"):
        guidelines.append(f"=== {text_file.name} ===\n{text_file.read_text()}")

    return "\n\n".join(guidelines)


@reviewer_agent.tool_plain
async def get_writer_instructions() -> str:
    """Get the blog post writing instructions that the writer should follow."""
    return (PROMPTS_DIR / "writer" / "blog_post.txt").read_text()


# Tool to allow writer to get feedback from reviewer
@writer_agent.tool_plain
async def review_page_content(content: str) -> Score:
    """Review the content and return a score with feedback."""
    result = await reviewer_agent.run(f"Review this content:\n\n{content}")
    return result.output


# Tool to get brand guidelines
@writer_agent.tool_plain
async def get_brand_guidelines(query: str = "") -> str:
    """
    Get Pydantic brand guidelines and voice examples.

    Args:
        query: Optional - specific aspect you need guidance on
    """
    guidelines = (PROMPTS_DIR / "shared" / "brand_guidelines.txt").read_text()

    return f"Brand guidelines for {query!r}:\n\n{guidelines}"


# Tool to fetch web content
@writer_agent.tool_plain
async def fetch_web_content(url: str) -> str:
    """
    Fetch and return the content from a web URL.

    Args:
        url: The URL to fetch content from
    """
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            # Limit content length to avoid overwhelming the context
            content = response.text[:8000]
            return f"Content from {url}:\n\n{content}"
    except Exception as e:
        return f"Could not fetch {url}: {e}"


@writer_agent.tool_plain
async def debug_web_content(url: str) -> str:
    """Debug: Show what content we're actually getting from a URL."""
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            content = response.text[:2000]  # Show first 2000 chars
            return f"Raw content from {url}:\n\n{content}"
    except Exception as e:
        return f"Error: {e}"


async def extract_technical_content(url: str) -> str:
    """Extract technical content optimized for code snippets and documentation."""
    try:
        import trafilatura
        from markdownify import markdownify as md

        # Fetch and extract with technical optimizations
        downloaded = trafilatura.fetch_url(url)

        # Try HTML output first (better for code)
        html_content = trafilatura.extract(
            downloaded,
            output_format="html",
            favor_precision=True,
            include_formatting=True,
            include_tables=True,
        )

        if html_content:
            # Convert to Markdown for better code preservation
            markdown_content = md(html_content, heading_style="ATX")
            return f"Technical content from {url}:\n\n{markdown_content[:8000]}"
        else:
            # Fallback to plain text
            text_content = trafilatura.extract(downloaded)
            return f"Content from {url}:\n\n{text_content[:8000]}"

    except Exception as e:
        return f"Could not extract technical content from {url}: {e}"


# Main function to generate blog content
async def generate_blog_post(
    topic: str,
    author: str = "",
    author_role: str = "",
    user_prompt: str = "",
    opinions: str = "",
    examples: str = "",
    reference_links: list[str] = [],
) -> str:
    """Generate a blog post about the given topic."""

    # Build the prompt with user input and references
    prompt_parts = [f"Write a blog post about {topic}."]
    if author:
        prompt_parts.append(f"\nAuthor: {author}")
    if author_role:
        prompt_parts.append(f"\nAuthor Role: {author_role}")
    if opinions:
        prompt_parts.append(f"\nOpinions: {opinions}")
    if examples:
        prompt_parts.append(f"\nExamples: {examples}")
    if user_prompt:
        prompt_parts.append(f"\nUser requirements: {user_prompt}")

    if reference_links:
        prompt_parts.append("\nReference content from the provided links:")
        for link in reference_links:
            # Use the new technical content extraction
            try:
                content = await extract_technical_content(link)

                # DEBUG: Print first 500 chars to see improvement
                print(f"\nDEBUG - First 500 chars from {link} (cleaned):")
                print(content[:500])
                print("=" * 50)

                prompt_parts.append(f"\n{content}")
            except Exception as e:
                print(f"Could not fetch {link}: {e}")
                prompt_parts.append(f"\nCould not fetch {link}: {e}")

    prompt_parts.append(
        "Use the review_page_content tool to check your work and iterate if needed."
    )

    full_prompt = "\n".join(prompt_parts)
    response = await writer_agent.run(full_prompt)
    return response.output


# Test function for direct execution
async def main():
    print("Pydantic Ghost Writer - Blog Post Generator")
    print("=" * 50)

    # Enhanced blog-specific inputs
    topic = input("Blog post topic/headline: ")
    author = (
        input("Author name (or press Enter for 'Pydantic Team'): ").strip()
        or "Pydantic Team"
    )
    author_role = input("Author role (e.g., 'Founder', 'Core Developer'): ").strip()

    print("\nContent guidance:")
    user_prompt = input("Additional requirements/direction: ").strip()
    opinions = input("Specific opinions or takes to include: ").strip()
    examples = input("Specific examples or case studies to mention: ").strip()

    # Get reference links
    reference_links: list[str] = []
    while True:
        link = input("Enter a reference link (or press Enter to finish): ").strip()
        if not link:
            break
        reference_links.append(link)

    print("\nGenerating blog post...")
    response = await generate_blog_post(
        topic=topic,
        author=author,
        author_role=author_role,
        user_prompt=user_prompt,
        opinions=opinions,
        examples=examples,
        reference_links=reference_links,
    )

    print("\n" + "=" * 50)
    print("GENERATED BLOG POST:")
    print("=" * 50)
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
