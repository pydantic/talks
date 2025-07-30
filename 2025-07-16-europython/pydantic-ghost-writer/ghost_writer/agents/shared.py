from pathlib import Path
from typing import Literal, TypeAlias

PROMPTS_DIR = Path(__file__).parents[2] / 'prompts'


def load_prompt(role: str, content_type: str, *, add_guidelines: bool = False) -> str:
    prompt = (PROMPTS_DIR / role / f'{content_type}.txt').read_text()

    if not add_guidelines:
        return prompt
    else:
        brand_guidelines = (PROMPTS_DIR / 'shared' / 'brand_guidelines.txt').read_text()
        global_styleguide = (PROMPTS_DIR / 'shared' / 'global_styleguide.txt').read_text()
        vocabulary = (PROMPTS_DIR / 'shared' / 'vocabulary.txt').read_text()

        return f"{prompt}\n\n## Guidelines\n\n{brand_guidelines}\n\n{global_styleguide}\n\n{vocabulary}"



## Shared tools:

Guidelines: TypeAlias = Literal['brand_guidelines', 'global_styleguide', 'vocabulary']  # noqa: UP040


async def get_guidelines(guideline: Guidelines) -> str:
    """Get Pydantic guidelines and voice examples."""
    guidelines = (PROMPTS_DIR / 'shared' / f'{guideline}.txt').read_text()

    return f'Guidelines:\n\n{guidelines}'
