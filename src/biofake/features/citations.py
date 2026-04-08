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
        lowered = text.lower()
        rows.append(
            [
                len(re.findall(r"\[\d+\]", text)),
                len(re.findall(r"\(\d{4}\)", text)),
                len(re.findall(r"\b(?:background|methods|results|conclusion)\b", lowered)),
                len(re.findall(r"%", text)),
                len(re.findall(r"\b(?:trial|randomized|cohort|study)\b", lowered)),
                len(re.findall(r"\bp\s*[<>=]\s*[\d.]+\b", lowered)),
                len(re.findall(r"\b(?:ci|confidence interval)\b", lowered)),
                len(re.findall(r"\b(?:mg|ml|kg|mmol|µg)\b", lowered)),
                len(re.findall(r"@", text)),
                len(re.findall(r"\b(?:figure|table|fig)\b", lowered)),
                len(re.findall(r"\b\d+\.?\d*\b", text)),
            ]
        )
    return np.asarray(rows, dtype=float)


def citation_matrix(records: Sequence[dict] | pd.DataFrame) -> sparse.csr_matrix:
    return sparse.csr_matrix(citation_array(records))

