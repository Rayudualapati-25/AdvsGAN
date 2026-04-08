from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from biofake.features.lexical import extract_texts


TOKEN_RE = re.compile(r"\b\w+\b")
SENTENCE_RE = re.compile(r"[.!?]+")
FUNCTION_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "to",
    "for", "with", "on", "at", "from", "by", "as", "into", "through",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "that", "which", "who", "whom", "this", "these", "those", "it",
})


def stylometric_array(records: Sequence[dict] | pd.DataFrame) -> np.ndarray:
    rows = []
    for text in extract_texts(records):
        tokens = TOKEN_RE.findall(text)
        token_count = max(len(tokens), 1)
        chars = len(text)
        lowered_tokens = [t.lower() for t in tokens]
        unique_ratio = len(set(lowered_tokens)) / token_count
        avg_token_length = sum(len(t) for t in tokens) / token_count
        uppercase_ratio = sum(c.isupper() for c in text) / max(chars, 1)
        punctuation_ratio = sum(c in ",.;:()[]%-/" for c in text) / max(chars, 1)
        digit_ratio = sum(c.isdigit() for c in text) / max(chars, 1)

        # Sentence-level features
        sentences = [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]
        sentence_count = max(len(sentences), 1)
        sentence_lengths = [len(TOKEN_RE.findall(s)) for s in sentences]
        avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0.0
        sentence_length_std = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0.0

        # Hapax legomena ratio (words appearing exactly once)
        word_counts = Counter(lowered_tokens)
        hapax_ratio = sum(1 for c in word_counts.values() if c == 1) / token_count

        # Function word ratio
        function_word_ratio = sum(1 for t in lowered_tokens if t in FUNCTION_WORDS) / token_count

        # Token length variance
        token_lengths = [len(t) for t in tokens]
        token_length_std = np.std(token_lengths) if len(token_lengths) > 1 else 0.0

        rows.append(
            [
                token_count,
                chars,
                unique_ratio,
                avg_token_length,
                uppercase_ratio,
                punctuation_ratio,
                digit_ratio,
                avg_sentence_length,
                sentence_length_std,
                hapax_ratio,
                function_word_ratio,
                token_length_std,
            ]
        )
    return np.asarray(rows, dtype=float)


def stylometric_matrix(records: Sequence[dict] | pd.DataFrame) -> sparse.csr_matrix:
    return sparse.csr_matrix(stylometric_array(records))

