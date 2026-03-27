from __future__ import annotations

import json
import shlex
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import tomllib

import yaml

from biofake.schemas import ExperimentConfig, ProcessedRow, RunMetadata


def project_root() -> Path:
    return Path.cwd()


def read_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".json":
        data = json.loads(source.read_text(encoding="utf-8"))
    elif suffix == ".toml":
        data = tomllib.loads(source.read_text(encoding="utf-8"))
    else:
        with source.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source = Path(path)
    if not source.exists():
        return records
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def merge_dicts(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _coerce_override(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    if value.startswith("[") or value.startswith("{"):
        return json.loads(value)
    return value


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    updated = dict(config)
    for raw in overrides:
        key, _, value = raw.partition("=")
        if not key or not _:
            raise ValueError(f"Invalid override: {raw}")
        cursor: dict[str, Any] = updated
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = _coerce_override(value)
    return updated


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> tuple[ExperimentConfig, dict[str, Any]]:
    config_path = Path(config_path)
    base_path = config_path.parent.parent / "base.yaml"
    raw = read_yaml(base_path)

    experiment_raw = read_yaml(config_path)
    for include in experiment_raw.get("includes", []):
        include_path = (config_path.parent / include).resolve()
        raw = merge_dicts(raw, read_yaml(include_path))

    raw = merge_dicts(raw, {k: v for k, v in experiment_raw.items() if k != "includes"})
    if overrides:
        raw = apply_overrides(raw, overrides)

    return ExperimentConfig.model_validate(raw), raw


def resolved_path(path: str | Path, base: str | Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    anchor = Path(base) if base is not None else project_root()
    return (anchor / candidate).resolve()


def init_run(
    config: ExperimentConfig,
    config_path: str | Path,
    command: list[str],
    run_id: str | None = None,
) -> RunMetadata:
    resolved_runs_dir = resolved_path(config.paths.runs_dir)
    resolved_runs_dir.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = resolved_runs_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = output_dir / "resolved_config.yaml"
    command_path = output_dir / "command.txt"
    write_yaml(
        resolved_config_path,
        json.loads(config.model_dump_json()),
    )
    command_path.write_text(shlex.join(command), encoding="utf-8")
    return RunMetadata(
        run_id=run_id,
        config_path=str(config_path),
        resolved_config_path=str(resolved_config_path),
        command=" ".join(command),
        output_dir=str(output_dir),
    )


def validate_rows(records: list[dict[str, Any]]) -> list[ProcessedRow]:
    return [ProcessedRow.model_validate(record) for record in records]
