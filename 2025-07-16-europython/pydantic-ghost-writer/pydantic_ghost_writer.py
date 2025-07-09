import asyncio
import logfire
from typing_extensions import TypedDict

from pydantic_ai import Agent

# Configure Logfire for observability
logfire.configure(scrubbing=False)
logfire.instrument_mcp() 
logfire.instrument_pydantic_ai()

# Writer agent for generating blog content
writer_agent = Agent(
    'anthropic:claude-3-5-sonnet-latest',
    instructions="""
You are a documentation writer for Pydantic.

Before writing, use the get_brand_guidelines tool to understand Pydantic's authentic voice and positioning.
Key principles to follow:
- Write by developers, for developers
- Focus on developer experience (DX)
- Be authentic, not corporate
- Technical but accessible
- Practical value and real-world usage

You need to write content that would score 8+ on quality.
Use the review_page_content tool to check your work and iterate if needed.
""",
)

# Reviewer agent for quality control
class Score(TypedDict):
    score: int
    reason: str

reviewer_agent = Agent(
    'anthropic:claude-3-5-sonnet-latest',
    output_type=Score,
    instructions="""
You are a technical content reviewer for Pydantic.

Evaluate content on:
- Technical accuracy
- Clarity for developers
- Authentic Pydantic voice (not corporate, genuine)
- Practical value
- Code examples quality (if present)

Score out of 10 and provide specific feedback.
""",
)

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
    try:
        with open('brand_guidelines.txt', 'r') as f:
            guidelines = f.read()
        
        if query:
            return f"Brand guidelines for '{query}':\n\n{guidelines}"
        else:
            return guidelines
    except FileNotFoundError:
        return "Brand guidelines file not found. Please ensure brand_guidelines.txt is in the project directory."

# Main function to generate blog content
async def generate_blog_post(topic: str, user_prompt: str = None, reference_links: list[str] = None) -> str:
    """Generate a blog post about the given topic."""
    
    # Build the prompt with user input and references
    prompt_parts = [f"Write a blog post about {topic}."]
    
    if user_prompt:
        prompt_parts.append(f"\nUser requirements: {user_prompt}")
    
    if reference_links:
        prompt_parts.append("\nReference links to consider:")
        for link in reference_links:
            prompt_parts.append(f"- {link}")
    
    prompt_parts.append("""
The post should:
- Be engaging and informative for Python developers
- Include practical examples where relevant
- Reflect Pydantic's authentic voice
- Be well-structured with clear sections
- Aim for 500-800 words

Use the review_page_content tool to check your work and iterate if the score is below 8.
""")
    
    full_prompt = "\n".join(prompt_parts)
    response = await writer_agent.run(full_prompt)
    return response.output

# Test function for direct execution
async def main():
    print("Pydantic Ghost Writer - Blog Post Generator")
    print("=" * 50)
    
    # Get user input
    topic = input("Enter the blog post topic: ")
    user_prompt = input("Enter additional requirements (optional): ").strip()
    
    # Get reference links
    reference_links = []
    while True:
        link = input("Enter a reference link (or press Enter to finish): ").strip()
        if not link:
            break
        reference_links.append(link)
    
    print("\nGenerating blog post...")
    response = await generate_blog_post(
        topic=topic,
        user_prompt=user_prompt if user_prompt else None,
        reference_links=reference_links if reference_links else None
    )
    
    print("\n" + "=" * 50)
    print("GENERATED BLOG POST:")
    print("=" * 50)
    print(response)

if __name__ == '__main__':
    asyncio.run(main())