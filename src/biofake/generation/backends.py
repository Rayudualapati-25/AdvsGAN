from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .schema import GenerationProvenance, GenerationRequest, GenerationResult


class TextGenerationBackend(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult:
        ...


@dataclass
class DeterministicFallbackBackend:
    """Template-based local backend used when no model is available."""

    model_name: str = "deterministic-fallback"
    model_path: str | None = None
    seed: int | None = None
    fallback_reason: str | None = None

    def generate(self, request: GenerationRequest) -> GenerationResult:
        request = request.normalized()
        base_seed = _stable_seed(request.prompt, request.seed if request.seed is not None else self.seed)
        rng = random.Random(base_seed)
        keywords = _extract_keywords(request.prompt)
        if not keywords:
            keywords = ["synthetic", "biomedical", "robustness"]

        background = (
            f"This synthetic abstract is grounded in the prompt terms {', '.join(keywords[:3])} "
            f"and is generated deterministically for evaluation."
        )
        methods = (
            f"We used a template-driven fallback backend with seed {base_seed} to preserve reproducibility "
            f"across CPU-only runs."
        )
        result_terms = ", ".join(keywords[: min(5, len(keywords))])
        results = (
            f"The generated text retains the topical anchors {result_terms} while remaining stable across repeated runs."
        )
        conclusion_templates = [
            "The output is suitable for provenance-aware robustness checks.",
            "The sample is intended for local testing when no llama.cpp model is available.",
            "The deterministic backend provides a repeatable control condition for deepfake experiments.",
        ]
        conclusion = rng.choice(conclusion_templates)

        text = "\n".join(
            [
                f"BACKGROUND: {background}",
                f"METHODS: {methods}",
                f"RESULTS: {results}",
                f"CONCLUSION: {conclusion}",
            ]
        )
        text = _truncate_to_token_budget(text, request.max_tokens)
        provenance = GenerationProvenance(
            backend=self.model_name,
            model_name=self.model_name,
            model_path=self.model_path,
            prompt_hash=_hash_text(request.prompt),
            request_fingerprint=request.fingerprint(),
            seed=request.seed if request.seed is not None else self.seed,
            fallback_reason=self.fallback_reason or "no local llama.cpp model configured",
            details={
                "keyword_count": len(keywords),
                "token_budget": request.max_tokens,
                "backend_seed": base_seed,
            },
        )
        return GenerationResult(text=text, provenance=provenance)


class LlamaCppBackend:
    """Adapter around the optional llama.cpp Python bindings."""

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        *,
        model_name: str | None = None,
        seed: int | None = None,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        self.model_path = Path(model_path)
        self.model_name = model_name or self.model_path.stem
        self.seed = seed
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self.kwargs = dict(kwargs)
        self._llama = self._load_model()

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("llama_cpp") is not None

    def _load_model(self) -> Any:
        if not self.model_path.exists():
            raise FileNotFoundError(f"llama.cpp model not found: {self.model_path}")
        if not self.available():
            raise ImportError("llama_cpp is not installed")
        from llama_cpp import Llama  # type: ignore[import-not-found]

        init_kwargs = {
            "model_path": str(self.model_path),
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "verbose": self.verbose,
        }
        if self.n_threads is not None:
            init_kwargs["n_threads"] = self.n_threads
        init_kwargs.update(self.kwargs)
        return Llama(**init_kwargs)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        request = request.normalized()
        seed = request.seed if request.seed is not None else self.seed
        params = {
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if seed is not None:
            params["seed"] = seed
        if request.stop:
            params["stop"] = list(request.stop)
        response = self._llama(**params)
        text = _extract_llama_text(response)
        provenance = GenerationProvenance(
            backend="llama_cpp",
            model_name=self.model_name,
            model_path=str(self.model_path),
            prompt_hash=_hash_text(request.prompt),
            request_fingerprint=request.fingerprint(),
            seed=seed,
            details={
                "n_ctx": self.n_ctx,
                "n_threads": self.n_threads,
                "n_gpu_layers": self.n_gpu_layers,
            },
        )
        return GenerationResult(text=text, provenance=provenance)


def build_generation_backend(config: Mapping[str, Any] | None = None) -> TextGenerationBackend:
    """Build a backend from a simple mapping or TOML-derived config."""

    config = _flatten_generation_config(dict(config or {}))
    backend_name = str(config.get("backend") or config.get("type") or "auto").lower()
    model_path = config.get("model_path") or config.get("path") or ""
    seed = config.get("seed")
    fallback_reason = None

    if backend_name in {"auto", "llama_cpp", "llama-cpp"} and model_path:
        try:
            return LlamaCppBackend(
                model_path=model_path,
                model_name=config.get("model_name") or config.get("name"),
                seed=seed,
                n_ctx=int(config.get("n_ctx") or 2048),
                n_threads=_maybe_int(config.get("n_threads")),
                n_gpu_layers=int(config.get("n_gpu_layers") or 0),
                verbose=bool(config.get("verbose", False)),
            )
        except Exception as exc:  # pragma: no cover - exercised when llama.cpp is installed or misconfigured
            fallback_reason = f"llama.cpp unavailable: {exc}"

    if backend_name in {"llama_cpp", "llama-cpp"} and not model_path:
        fallback_reason = "llama.cpp backend requested without a model path"

    return DeterministicFallbackBackend(
        model_name=str(config.get("fallback_model_name") or "deterministic-fallback"),
        model_path=str(model_path) if model_path else None,
        seed=_maybe_int(seed),
        fallback_reason=fallback_reason or "deterministic fallback selected",
    )


def _flatten_generation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Accept either a flat mapping or a TOML-style nested config."""

    if "generation" not in config or not isinstance(config.get("generation"), Mapping):
        return dict(config)

    merged: dict[str, Any] = dict(config["generation"])
    llama_cfg = config.get("llama_cpp")
    if isinstance(llama_cfg, Mapping):
        for key in ("n_ctx", "n_threads", "n_gpu_layers", "verbose"):
            if key not in merged and key in llama_cfg:
                merged[key] = llama_cfg[key]
    fallback_cfg = config.get("fallback")
    if isinstance(fallback_cfg, Mapping):
        if "fallback_model_name" not in merged and "model_name" in fallback_cfg:
            merged["fallback_model_name"] = fallback_cfg["model_name"]
        if "model_name" not in merged and "model_name" in fallback_cfg:
            merged["model_name"] = fallback_cfg["model_name"]
    return merged


def _extract_keywords(prompt: str, limit: int = 6) -> list[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "be",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "with",
        "using",
        "used",
        "via",
        "this",
        "these",
        "those",
        "we",
        "was",
        "were",
    }
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", prompt.lower())
    seen: dict[str, int] = {}
    for idx, token in enumerate(tokens):
        if token in stopwords:
            continue
        seen.setdefault(token, idx)
    ordered = sorted(seen.items(), key=lambda item: (item[1], item[0]))
    return [token for token, _ in ordered[:limit]]


def _stable_seed(prompt: str, seed: int | None) -> int:
    payload = f"{seed!r}:{prompt}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _truncate_to_token_budget(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])


def _maybe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _extract_llama_text(response: Any) -> str:
    if isinstance(response, Mapping):
        choices = response.get("choices")
        if isinstance(choices, Sequence) and choices:
            first = choices[0]
            if isinstance(first, Mapping):
                if "text" in first:
                    return str(first["text"]).strip()
                message = first.get("message")
                if isinstance(message, Mapping) and "content" in message:
                    return str(message["content"]).strip()
    return str(response).strip()
