from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from biofake.features.lexical import extract_texts


SENTENCE_RE = re.compile(r"[.!?]+")
WORD_RE = re.compile(r"\b[a-zA-Z]+\b")


def _syllables(word: str) -> int:
    lowered = word.lower()
    groups = re.findall(r"[aeiouy]+", lowered)
    return max(len(groups), 1)


def readability_array(records: Sequence[dict] | pd.DataFrame) -> np.ndarray:
    features: list[list[float]] = []
    for text in extract_texts(records):
        sentences = [part for part in SENTENCE_RE.split(text) if part.strip()]
        words = WORD_RE.findall(text)
        sentence_count = max(len(sentences), 1)
        word_count = max(len(words), 1)
        syllable_count = sum(_syllables(word) for word in words)
        words_per_sentence = word_count / sentence_count
        syllables_per_word = syllable_count / word_count
        flesch = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
        features.append([sentence_count, word_count, words_per_sentence, syllables_per_word, flesch])
    return np.asarray(features, dtype=float)


def readability_matrix(records: Sequence[dict] | pd.DataFrame) -> sparse.csr_matrix:
    return sparse.csr_matrix(readability_array(records))

