"""Reads an aerial image of a property with a multimodal model and returns structured observations."""

from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, BinaryContent


class AerialObservations(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)

    water_visible: bool
    """Is a river, brook or other open water visible in the image?"""

    on_the_bank: bool
    """Does the building at the centre stand directly on the bank, with nothing between it and the water?"""

    ground_cover: Literal['mostly paved', 'mixed', 'mostly green']
    """What surrounds the building: car parks and roads, or gardens and trees?"""

    notes: str
    """One sentence on anything else relevant to flood risk, such as a basement entrance or a retaining wall."""

    def summary(self) -> str:
        water = 'on the river bank' if self.on_the_bank else 'water visible' if self.water_visible else 'no water'
        return f'{water}; surroundings {self.ground_cover}. {self.notes}'


vision_agent = Agent(
    'gateway/anthropic:claude-sonnet-5-5',
    name='vision_agent',
    output_type=AerialObservations,
    instructions=(
        'You are given an aerial photograph about 110 m across, with north at the top. '
        'The property being assessed is the building at the centre of the image. '
        'Describe only what you can see.'
    ),
)


async def inspect_image(image: bytes) -> AerialObservations:
    result = await vision_agent.run([BinaryContent(image, media_type='image/jpeg')])
    return result.output
