from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from biofake.features.lexical import extract_texts


def _hash_token(token: str, dim: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % dim


def hashed_embeddings(texts: Iterable[str], dim: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    for text in texts:
        vector = np.zeros(dim, dtype=float)
        for token in text.lower().split():
            vector[_hash_token(token, dim)] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        rows.append(vector)
    return np.vstack(rows) if rows else np.zeros((0, dim), dtype=float)


@dataclass
class FrozenTransformerEmbeddings:
    model_name: str = "allenai/scibert_scivocab_uncased"
    batch_size: int = 8
    local_files_only: bool = True
    fallback_dim: int = 64

    def __post_init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self.backend = "hash"

    def _maybe_load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        if os.environ.get("BIOFAKE_ENABLE_TORCH", "0") != "1":
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            self._model = AutoModel.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            self._model.eval()
            self.backend = self.model_name
        except Exception:
            self._tokenizer = None
            self._model = None
            self.backend = "hash"

    def fit(self, records: Sequence[dict] | pd.DataFrame | Iterable[str]) -> "FrozenTransformerEmbeddings":
        self._maybe_load()
        return self

    def transform(self, records: Sequence[dict] | pd.DataFrame | Iterable[str]) -> np.ndarray:
        texts = extract_texts(records)
        self._maybe_load()
        if self._model is None or self._tokenizer is None or os.environ.get("BIOFAKE_ENABLE_TORCH", "0") != "1":
            return hashed_embeddings(texts, self.fallback_dim)

        import torch

        encoded_batches = []
        for index in range(0, len(texts), self.batch_size):
            batch = texts[index : index + self.batch_size]
            tokens = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self._model(**tokens)
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                encoded_batches.append(pooled.cpu().numpy())
        return np.vstack(encoded_batches) if encoded_batches else np.zeros((0, self.fallback_dim), dtype=float)
