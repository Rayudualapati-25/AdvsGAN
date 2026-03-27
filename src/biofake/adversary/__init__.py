"""Adversarial attack primitives for BioFake."""

from .base import AdversaryAttack, AttackOutcome
from .compression import CompressionExpansionAttack
from .paraphrase import ParaphraseAttack
from .registry import ATTACK_REGISTRY, build_attack, build_attacks
from .style_transfer import StyleTransferAttack

__all__ = [
    "AdversaryAttack",
    "AttackOutcome",
    "CompressionExpansionAttack",
    "ParaphraseAttack",
    "StyleTransferAttack",
    "ATTACK_REGISTRY",
    "build_attack",
    "build_attacks",
]

