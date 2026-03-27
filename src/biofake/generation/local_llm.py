from __future__ import annotations

from typing import Any, Mapping

from biofake.generation.backends import build_generation_backend
from biofake.generation.synthetic import SyntheticGenerator


def build_local_llm(config: Mapping[str, Any]) -> SyntheticGenerator:
    backend = build_generation_backend(config)
    return SyntheticGenerator(backend=backend, config=config)

