from __future__ import annotations

from typing import Any

from biofake.adversary.attacks import instantiate_attacks
from biofake.adversary.constraints import passes_basic_constraints
from biofake.io import write_jsonl
from biofake.schemas import ExperimentConfig


def rewrite_synthetic_rows(
    rows: list[dict[str, Any]],
    config: ExperimentConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    attacks = instantiate_attacks(config.adversary.attacks)
    attacked_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["split"] not in config.adversary.enabled_splits:
            continue
        for attack in attacks[: config.adversary.max_variants_per_row or len(attacks)]:
            attacked = attack.attack_row(row)
            if not passes_basic_constraints(row, attacked):
                continue
            attacked_rows.append(
                {
                    "id": f"{row['id']}_{attacked.get('attack_name', 'attack')}",
                    "split": row["split"],
                    "label": "synthetic",
                    "source": row.get("source", "pubmed_rct"),
                    "generator": row.get("generator"),
                    "attack": attacked.get("attack_family") or row.get("attack"),
                    "parent_id": row["id"],
                    "text": attacked.get("adversarial_text") or attacked.get("text"),
                    "meta": {
                        **dict(row.get("meta", {})),
                        "attack_metadata": attacked.get("attack_metadata", {}),
                        "attack_name": attacked.get("attack_name"),
                        "attack_family": attacked.get("attack_family"),
                    },
                }
            )
    write_jsonl(config.adversary.output_path, attacked_rows)
    return attacked_rows, {"attack_count": len(attacked_rows), "families": config.adversary.attacks}

