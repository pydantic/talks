"""The facts a decision is made from, and the underwriting guide as code, used to label the eval dataset."""

from typing import Literal

from pydantic import BaseModel

Zone = Literal['floodway', '1_percent', '0.2_percent', 'minimal']
OutcomeName = Literal['accept', 'refer', 'decline']

# How FEMA labels each zone on a flood map, which is how the fact sheet reports it.
FEMA_ZONE: dict[Zone, str] = {
    'floodway': 'AE, regulatory floodway',
    '1_percent': 'AE',
    '0.2_percent': 'X (shaded)',
    'minimal': 'X (unshaded)',
}

# Uses with stock or fittings at ground level, which a shallow flood ruins.
GROUND_FLOOR_TRADE = {'restaurant', 'retail', 'commercial', 'mixed use'}

GUIDE = """\
FEMA zones: "AE, regulatory floodway" is the channel that carries flood water. "AE" is the 1% annual
chance zone. "X (shaded)" is the 0.2% annual chance zone. "X (unshaded)" is outside every mapped zone.
Height above water is the property's ground elevation minus the elevation of the nearest water.

Decline if any of these is true:
- The property is in the regulatory floodway.
- It is in the 1% zone and less than 2 m above the water.
- It is a restaurant, retail, commercial or mixed use property in the 1% zone and less than 3 m above the water.
- It is within 10 m of the water and less than 1.5 m above it, whatever the zone.

Otherwise refer to a senior underwriter if any of these is true:
- The property is in the 1% zone.
- It is in the 0.2% zone and less than 3 m above the water.
- It is outside every mapped zone, but within 25 m of the edge of the 1% zone and less than 3 m above the water.
- It has 4 or more units and is in the 1% or 0.2% zone.

Otherwise accept.

A higher flood excess applies to any property in the floodway, the 1% zone or the 0.2% zone, and to any
property outside the mapped zones that is within 25 m of the edge of the 1% zone and less than 3 m above the water.
"""


class Facts(BaseModel):
    """What the geo risk API knows about a property."""

    property_id: str
    address: str
    kind: str
    units: int
    zone: Zone
    distance_to_1_percent_zone_m: float
    elevation_m: float
    water_elevation_m: float
    distance_to_water_m: float
    water_name: str | None
    aerial_notes: str | None = None

    @property
    def height_above_water_m(self) -> float:
        # Elevations come to the centimetre and the fact sheet shows them in full, so this is exactly
        # the figure a reader of the sheet works out.
        return round(self.elevation_m - self.water_elevation_m, 2)

    @property
    def near_zone_edge(self) -> bool:
        """Outside the mapped zones, but close enough to the 1% zone and low enough to be at risk."""
        return self.zone == 'minimal' and self.distance_to_1_percent_zone_m <= 25 and self.height_above_water_m < 3

    def fact_sheet(self) -> str:
        """The facts as the geo risk API reports them, which is what the decision model judges.

        Nothing is worked out in advance: the zone is FEMA's label and the two elevations are given
        separately, so the guide's "height above water" has to be derived.
        """
        units = '1 unit' if self.units == 1 else f'{self.units} units'
        lines = [
            f'{self.address} ({self.kind}, {units}).',
            f'FEMA flood zone: {FEMA_ZONE[self.zone]}.',
            f'Ground elevation: {self.elevation_m:.2f} m.',
            (
                f'Nearest water: {self.water_name or "an unnamed stream"}, {self.distance_to_water_m:.0f} m away, '
                f'at an elevation of {self.water_elevation_m:.2f} m.'
            ),
        ]
        if self.zone in ('0.2_percent', 'minimal'):
            lines.append(f'Edge of the 1% annual chance zone: {self.distance_to_1_percent_zone_m:.0f} m away.')
        if self.aerial_notes:
            lines.append(f'From the aerial image: {self.aerial_notes}')
        return '\n'.join(lines)


def expected_outcome(facts: Facts) -> OutcomeName:
    """The outcome `GUIDE` gives for these facts. This labels the offline eval cases."""
    zone, height = facts.zone, facts.height_above_water_m
    if (
        zone == 'floodway'
        or (zone == '1_percent' and height < 2)
        or (zone == '1_percent' and height < 3 and facts.kind in GROUND_FLOOR_TRADE)
        or (facts.distance_to_water_m <= 10 and height < 1.5)
    ):
        return 'decline'
    if (
        zone == '1_percent'
        or (zone == '0.2_percent' and height < 3)
        or facts.near_zone_edge
        or (facts.units >= 4 and zone in ('1_percent', '0.2_percent'))
    ):
        return 'refer'
    return 'accept'


def expected_flood_excess(facts: Facts) -> bool:
    return facts.zone != 'minimal' or facts.near_zone_edge


def margin(facts: Facts) -> float:
    """How close the facts sit to one of the guide's thresholds; small means a borderline case."""
    height = facts.height_above_water_m
    margins = [abs(height - threshold) for threshold in (1.5, 2, 3)]
    margins.append(abs(facts.distance_to_water_m - 10) / 10)
    if facts.zone == 'minimal':
        margins.append(abs(facts.distance_to_1_percent_zone_m - 25) / 25)
    return min(margins)
