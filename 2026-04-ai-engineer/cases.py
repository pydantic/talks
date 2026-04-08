"""Golden dataset generation and persistence for MP political relations."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from task import (
    MP,
    PoliticalRelation,
    TaskInput,
    extract_relations,
    get_instructions,
    get_mps,
    relations_agent,
)

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


class GoldenDatasetFile(BaseModel):
    """Persisted golden dataset metadata and cases."""

    version: Literal[1] = 1
    focus: Literal['all'] = 'all'
    prompt_style: Literal['expert'] = 'expert'
    source_model: str
    source_instructions: str
    updated_at: datetime
    cases: list[GoldenCaseRecord]


async def generate_golden_dataset(
    *,
    output_path: Path = DEFAULT_CASES_PATH,
    limit: int | None = 100,
    offset: int = 0,
    model: str,
    max_concurrency: int = 5,
    overwrite: bool = False,
) -> GoldenDatasetFile:
    """Generate or resume a persisted golden dataset from cached MP pages."""
    all_mps = get_mps()
    instructions = get_instructions(style='expert', focus='all')
    selected_mps = all_mps[offset : None if limit is None else offset + limit]
    ordinal_by_id = {mp.id: ordinal for ordinal, mp in enumerate(all_mps, start=1)}

    existing = None if overwrite else load_golden_dataset(output_path)
    if existing is not None:
        if existing.focus != 'all' or existing.prompt_style != 'expert':
            raise ValueError(f'Existing dataset at {output_path} has incompatible metadata. Use --overwrite.')
        if existing.source_model != model or existing.source_instructions != instructions:
            raise ValueError(
                f'Existing dataset at {output_path} was generated with a different model or instructions. '
                'Use --overwrite or a different --output path.'
            )

    existing_records = [] if existing is None else list(existing.cases)
    existing_ids = {record.mp.id for record in existing_records}
    pending = [mp for mp in selected_mps if mp.id not in existing_ids]

    if not selected_mps:
        raise ValueError('No MPs selected. Check --offset and --limit.')

    print(
        f'Preparing golden dataset from cached MP pages: '
        f'{len(selected_mps)} selected, {len(existing_records)} already present, {len(pending)} pending.'
    )

    current_records = list(existing_records)
    if pending:
        semaphore = asyncio.Semaphore(max_concurrency)
        started_at = datetime.now(timezone.utc)

        async def run_one(mp: MP) -> GoldenCaseRecord:
            async with semaphore:
                relations = await extract_relations(TaskInput(mp=mp))
                return GoldenCaseRecord(
                    name=slugify_name(mp.name),
                    mp=mp,
                    split=split_for_mp_id(mp.id),
                    ordinal=ordinal_by_id[mp.id],
                    expected_output=relations,
                    generated_at=datetime.now(timezone.utc),
                )

        with relations_agent.override(model=model, instructions=instructions):
            tasks = [asyncio.create_task(run_one(mp)) for mp in pending]
            for completed_count, task in enumerate(asyncio.as_completed(tasks), start=1):
                record = await task
                current_records.append(record)
                current_records.sort(key=lambda item: item.ordinal)
                dataset = GoldenDatasetFile(
                    source_model=model,
                    source_instructions=instructions,
                    updated_at=datetime.now(timezone.utc),
                    cases=current_records,
                )
                save_golden_dataset(dataset, output_path)
                print(f'  [{completed_count}/{len(tasks)}] {record.mp.name}: {len(record.expected_output)} relations')

        finished_at = datetime.now(timezone.utc)
        print(f'Generation window: {started_at.isoformat()} to {finished_at.isoformat()}')

    dataset = GoldenDatasetFile(
        source_model=model,
        source_instructions=instructions,
        updated_at=datetime.now(timezone.utc),
        cases=sorted(current_records, key=lambda item: item.ordinal),
    )
    save_golden_dataset(dataset, output_path)
    return dataset


def load_case_records(
    path: Path = DEFAULT_CASES_PATH,
    *,
    split: SplitFilter = 'all',
    max_cases: int | None = None,
) -> list[GoldenCaseRecord]:
    """Load persisted case records with optional split and size filters."""
    dataset = load_golden_dataset(path)
    if dataset is None:
        raise FileNotFoundError(f'Golden cases file not found at {path}. Run `uv run -m main generate-cases` first.')

    records = sorted(dataset.cases, key=lambda case: case.ordinal)
    if split != 'all':
        records = [record for record in records if record.split == split]
    if max_cases is not None:
        records = records[:max_cases]
    return records


def load_golden_dataset(path: Path = DEFAULT_CASES_PATH) -> GoldenDatasetFile | None:
    """Load the persisted golden dataset if it exists."""
    if not path.exists():
        return None
    return GoldenDatasetFile.model_validate_json(path.read_text())


def save_golden_dataset(dataset: GoldenDatasetFile, path: Path = DEFAULT_CASES_PATH) -> None:
    """Persist the golden dataset as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dataset.model_dump_json(indent=2) + '\n')


# --- Utilities ---


def slugify_name(name: str) -> str:
    """Convert a display name to a stable case name."""
    return re.sub(r'[^a-z0-9]+', '_', name.casefold()).strip('_')


def split_for_mp_id(mp_id: int) -> SplitName:
    """Assign a deterministic split so datasets can be extended incrementally."""
    remainder = mp_id % 10
    if remainder == 0:
        return 'test'
    if remainder == 1:
        return 'val'
    return 'train'


def summarize_splits(records: list[GoldenCaseRecord]) -> dict[SplitName, int]:
    """Count cases per split."""
    counts: dict[SplitName, int] = {'train': 0, 'val': 0, 'test': 0}
    for record in records:
        counts[record.split] += 1
    return counts
