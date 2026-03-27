from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from biofake.features.lexical import extract_texts


def citation_array(records: Sequence[dict] | pd.DataFrame) -> np.ndarray:
    rows: list[list[float]] = []
    for text in extract_texts(records):
        rows.append(
            [
                len(re.findall(r"\[\d+\]", text)),
                len(re.findall(r"\(\d{4}\)", text)),
                len(re.findall(r"\b(?:background|methods|results|conclusion)\b", text.lower())),
                len(re.findall(r"%", text)),
                len(re.findall(r"\b(?:trial|randomized|cohort|study)\b", text.lower())),
            ]
        )
    return np.asarray(rows, dtype=float)


def citation_matrix(records: Sequence[dict] | pd.DataFrame) -> sparse.csr_matrix:
    return sparse.csr_matrix(citation_array(records))

