from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    stop: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def normalized(self) -> "GenerationRequest":
        return GenerationRequest(
            prompt=str(self.prompt),
            max_tokens=max(1, int(self.max_tokens)),
            temperature=float(self.temperature),
            top_p=float(self.top_p),
            seed=self.seed,
            stop=tuple(str(item) for item in self.stop),
            metadata=dict(self.metadata),
        )

    def fingerprint(self) -> str:
        payload = {
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "stop": list(self.stop),
            "metadata": dict(self.metadata),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class GenerationProvenance:
    backend: str
    model_name: str | None = None
    model_path: str | None = None
    prompt_hash: str | None = None
    request_fingerprint: str | None = None
    seed: int | None = None
    fallback_reason: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class GenerationResult:
    text: str
    provenance: GenerationProvenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "provenance": {
                "backend": self.provenance.backend,
                "model_name": self.provenance.model_name,
                "model_path": self.provenance.model_path,
                "prompt_hash": self.provenance.prompt_hash,
                "request_fingerprint": self.provenance.request_fingerprint,
                "seed": self.provenance.seed,
                "fallback_reason": self.provenance.fallback_reason,
                "details": dict(self.provenance.details),
            },
        }
