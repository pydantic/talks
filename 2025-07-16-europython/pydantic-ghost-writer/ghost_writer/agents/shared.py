from pathlib import Path
from typing import Literal, TypeAlias

PROMPTS_DIR = Path(__file__).parents[2] / 'prompts'


def load_prompt(role: str, content_type: str) -> str:
    return (PROMPTS_DIR / role / f'{content_type}.txt').read_text()


## Shared tools:

Guidelines: TypeAlias = Literal['brand_guidelines', 'global_styleguide', 'vocabulary']  # noqa: UP040


async def get_guidelines(guideline: Guidelines) -> str:
    """Get Pydantic guidelines and voice examples."""
    guidelines = (PROMPTS_DIR / 'shared' / f'{guideline}.txt').read_text()

    return f'Guidelines:\n\n{guidelines}'
