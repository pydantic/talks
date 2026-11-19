"""The facts a decision is made from, and the underwriting rule used to label the eval dataset."""

from typing import Literal

from pydantic import BaseModel

Zone = Literal['floodway', '1_percent', '0.2_percent', 'minimal']
OutcomeName = Literal['accept', 'refer', 'decline']

ZONE_TEXT: dict[Zone, str] = {
    'floodway': 'in the regulatory floodway, the channel that carries flood water',
    '1_percent': 'in the 1% annual chance flood zone',
    '0.2_percent': 'in the 0.2% annual chance flood zone',
    'minimal': 'outside every mapped flood zone',
}


class Facts(BaseModel):
    """What the geo risk API knows about a property."""

    property_id: str
    address: str
    kind: str
    zone: Zone
    distance_to_1_percent_zone_m: float
    height_above_water_m: float
    distance_to_water_m: float
    water_name: str | None
    aerial_notes: str | None = None

    def fact_sheet(self) -> str:
        """The facts as prose, which is what a decision model judges."""
        lines = [
            f'{self.address} ({self.kind}).',
            f'It is {ZONE_TEXT[self.zone]}.',
            (
                f'The ground there is {self.height_above_water_m:.1f} m above the nearest water, '
                f'{self.water_name or "an unnamed stream"}, which is {self.distance_to_water_m:.0f} m away.'
            ),
        ]
        if self.zone in ('0.2_percent', 'minimal'):
            lines.append(f'The 1% annual chance flood zone starts {self.distance_to_1_percent_zone_m:.0f} m away.')
        if self.aerial_notes:
            lines.append(f'From the aerial image: {self.aerial_notes}')
        return '\n'.join(lines)


def expected_outcome(facts: Facts) -> OutcomeName:
    """The outcome the underwriting guide gives for these facts.

    This is the labelling function for the offline evals. The decision model is given the same guide as
    prose, in the docstrings of `Outcome` in decision.py, and has to reach the same answer from the fact sheet.
    """
    zone, height = facts.zone, facts.height_above_water_m
    if zone == 'floodway' or (zone == '1_percent' and height < 2):
        return 'decline'
    if zone == '1_percent' or (zone == '0.2_percent' and height < 3) or (zone == 'minimal' and height < 2):
        return 'refer'
    return 'accept'


def expected_flood_excess(facts: Facts) -> bool:
    return facts.zone != 'minimal'
