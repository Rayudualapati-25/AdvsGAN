from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from biofake.features.lexical import extract_texts


TOKEN_RE = re.compile(r"\b\w+\b")


def stylometric_array(records: Sequence[dict] | pd.DataFrame) -> np.ndarray:
    rows = []
    for text in extract_texts(records):
        tokens = TOKEN_RE.findall(text)
        token_count = max(len(tokens), 1)
        chars = len(text)
        unique_ratio = len(set(token.lower() for token in tokens)) / token_count
        avg_token_length = sum(len(token) for token in tokens) / token_count
        uppercase_ratio = sum(char.isupper() for char in text) / max(chars, 1)
        punctuation_ratio = sum(char in ",.;:()[]%-/" for char in text) / max(chars, 1)
        digit_ratio = sum(char.isdigit() for char in text) / max(chars, 1)
        rows.append(
            [
                token_count,
                chars,
                unique_ratio,
                avg_token_length,
                uppercase_ratio,
                punctuation_ratio,
                digit_ratio,
            ]
        )
    return np.asarray(rows, dtype=float)


def stylometric_matrix(records: Sequence[dict] | pd.DataFrame) -> sparse.csr_matrix:
    return sparse.csr_matrix(stylometric_array(records))

