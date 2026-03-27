from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from .backends import build_generation_backend
from .schema import GenerationRequest, GenerationResult


class SyntheticGenerator:
    """High-level generation interface with deterministic provenance."""

    def __init__(self, backend: Any | None = None, config: Mapping[str, Any] | None = None) -> None:
        self.backend = backend or build_generation_backend(config or {})
        self.config = dict(config or {})

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
        stop: Iterable[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> GenerationResult:
        request = GenerationRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            stop=tuple(stop),
            metadata=dict(metadata or {}),
        )
        return self.backend.generate(request)

    def generate_from_file(self, path: str | Path, **kwargs: Any) -> GenerationResult:
        prompt = Path(path).read_text(encoding="utf-8")
        return self.generate(prompt, **kwargs)

    def generate_many(self, prompts: Iterable[str], **kwargs: Any) -> list[GenerationResult]:
        return [self.generate(prompt, **kwargs) for prompt in prompts]
