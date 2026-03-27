"""Attack registry and builders."""

from __future__ import annotations

from typing import Any, Mapping

from .compression import CompressionExpansionAttack
from .paraphrase import ParaphraseAttack
from .style_transfer import StyleTransferAttack

ATTACK_REGISTRY = {
    "paraphrase": ParaphraseAttack,
    "compression_expansion": CompressionExpansionAttack,
    "compression": CompressionExpansionAttack,
    "style_transfer": StyleTransferAttack,
}


def build_attack(config: str | Mapping[str, Any]) -> Any:
    if isinstance(config, str):
        name = config
        params: dict[str, Any] = {}
    else:
        params = dict(config)
        name = str(
            params.pop("attack", params.pop("name", params.pop("family", "")))
        ).strip()
    attack_cls = ATTACK_REGISTRY.get(name)
    if attack_cls is None:
        raise KeyError(f"Unknown attack type: {name!r}")
    return attack_cls(**params)


def build_attacks(configs: list[str | Mapping[str, Any]]) -> list[Any]:
    return [build_attack(config) for config in configs]

