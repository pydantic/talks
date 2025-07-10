"""
Pydantic Ghost Writer MCP Server

A MCP server that provides content generation tools
powered by Pydantic AI agents.
"""

from mcp.server.fastmcp import FastMCP

from pydantic_ghost_writer import generate_blog_post

server = FastMCP("Pydantic Ghost Writer")


@server.tool()
async def generate_blog_post(
    topic: str, user_instructions: str = "", reference_links: list[str] = []
) -> str:
    """
    Generate a blog post about the given topic using Pydantic's voice and style.

    Args:
        topic: The topic to write about
        user_instructions: Additional requirements or context from the user
        reference_links: Optional list of reference URLs to consider

    Returns:
        A well-structured blog post optimized for developer audience
    """
    return await generate_blog_post(
        topic=topic,
        user_instructions=user_instructions if user_instructions else None,
        reference_links=reference_links,
    )


def main():
    """Run the MCP server."""
    server.run()


if __name__ == "__main__":
    main()
