"""Golden dataset generation and persistence for MP political relations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator

from task import MP, PoliticalRelation

SplitName = Literal['train', 'val', 'test']
SplitFilter = Literal['all', 'train', 'val', 'test']

DEFAULT_CASES_PATH = Path('cases/golden_relations.json')


class GoldenCaseRecord(BaseModel):
    """A persisted golden test case for one MP."""

    name: str
    mp: MP
    split: SplitName
    ordinal: int
    expected_output: list[PoliticalRelation]
    generated_at: datetime

    @field_validator('expected_output')
    @classmethod
    def only_ancestors(cls, v: list[PoliticalRelation]) -> list[PoliticalRelation]:
        return [rel for rel in v if is_ancestor(rel.relation)]


class GoldenDatasetFile(BaseModel):
    """Persisted golden dataset metadata and cases."""

    version: Literal[1] = 1
    focus: Literal['all'] = 'all'
    prompt_style: Literal['expert'] = 'expert'
    source_model: str
    source_instructions: str
    updated_at: datetime
    cases: list[GoldenCaseRecord]


def load_case_records(
    path: Path = DEFAULT_CASES_PATH,
    *,
    split: SplitFilter = 'all',
    max_cases: int | None = None,
) -> list[GoldenCaseRecord]:
    """Load persisted case records with optional split and size filters."""
    if not path.exists():
        raise FileNotFoundError(f'Golden cases file not found at {path}.')
    dataset = GoldenDatasetFile.model_validate_json(path.read_bytes())

    records = sorted(dataset.cases, key=lambda case: case.ordinal)
    if split != 'all':
        records = [record for record in records if record.split == split]
    if max_cases is not None:
        records = records[:max_cases]
    return records


RELATION_IS_ANCESTOR: dict[str, bool] = {
    'ancestor': True,
    'aunt': True,
    'brother': False,
    'brother-in-law': False,
    'brother-in-law (brother of current wife Amy Richards)': False,
    'cousin': False,
    'daughter': False,
    'distant cousin': False,
    'distant relative': False,
    'domestic partner': False,
    'father': True,
    'father-in-law': True,
    'first cousin once removed': True,
    'first wife': False,
    'former employer (parliamentary assistant role)': False,
    'former wife': False,
    'grandfather': True,
    'great uncle': True,
    'great-aunt': True,
    'great-grandfather': True,
    "great-grandfather (wife's paternal great-grandfather)": True,
    'great-great-grandfather': True,
    'great-great-great-grandfather': True,
    'great-great-great-uncle': True,
    "great-great-uncle (maternal grandfather's uncle)": True,
    'great-uncle': True,
    "great-uncle's son-in-law": True,
    'half-brother': False,
    'husband': False,
    'husband/spouse': False,
    'maternal aunt': True,
    'maternal grandfather': True,
    'maternal grandmother': True,
    'mother': True,
    'niece': False,
    'partner': False,
    'paternal grandfather': True,
    "paternal grandmother's family member": True,
    'paternal great-grandfather': True,
    'self': False,
    'sister': False,
    'son': False,
    'spouse': False,
    'stepfather': True,
    'twin sister': False,
    'uncle': True,
    'wife': False,
}


def is_ancestor(relation: str) -> bool:
    """Return whether a relation should count as an ancestor (parent generation or older)."""
    return RELATION_IS_ANCESTOR[relation]
