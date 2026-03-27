from __future__ import annotations

from pathlib import Path

import pytest

from biofake.io import read_jsonl


@pytest.fixture()
def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture()
def mini_human_records(fixture_dir: Path) -> list[dict]:
    return read_jsonl(fixture_dir / "mini_authentic.jsonl")


@pytest.fixture()
def mini_generated_records(fixture_dir: Path) -> list[dict]:
    return read_jsonl(fixture_dir / "mini_generated.jsonl")


@pytest.fixture()
def mini_rewritten_records(fixture_dir: Path) -> list[dict]:
    return read_jsonl(fixture_dir / "mini_rewritten.jsonl")

