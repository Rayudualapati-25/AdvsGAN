from __future__ import annotations

from pathlib import Path
from typing import Any

from biofake.adversary.registry import build_attack
from biofake.io import read_yaml


ATTACK_CONFIG_FILES = {
    "paraphrase": "configs/adversary/paraphrase.json",
    "compress_expand": "configs/adversary/compression.json",
    "style_transfer": "configs/adversary/style_transfer.json",
}


def load_attack_configs(attack_names: list[str]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for attack_name in attack_names:
        path = ATTACK_CONFIG_FILES.get(attack_name, ATTACK_CONFIG_FILES.get(attack_name.replace("-", "_")))
        if path and Path(path).exists():
            configs.append(read_yaml(path))
        elif attack_name == "compress_expand":
            configs.append({"attack": "compression_expansion", "mode": "compress", "max_sentences": 2})
        else:
            configs.append({"attack": attack_name})
    return configs


def instantiate_attacks(attack_names: list[str]) -> list[Any]:
    return [build_attack(config) for config in load_attack_configs(attack_names)]

