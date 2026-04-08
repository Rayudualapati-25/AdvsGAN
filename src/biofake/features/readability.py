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

        # Gunning Fog Index
        complex_words = sum(1 for w in words if _syllables(w) >= 3)
        complex_word_ratio = complex_words / word_count
        gunning_fog = 0.4 * (words_per_sentence + 100.0 * complex_word_ratio)

        # Coleman-Liau Index
        char_count = sum(len(w) for w in words)
        letters_per_100 = (char_count / word_count) * 100
        sentences_per_100 = (sentence_count / word_count) * 100
        coleman_liau = 0.0588 * letters_per_100 - 0.296 * sentences_per_100 - 15.8

        # Long word ratio (6+ chars)
        long_word_ratio = sum(1 for w in words if len(w) >= 6) / word_count

        features.append([
            sentence_count, word_count, words_per_sentence, syllables_per_word,
            flesch, gunning_fog, coleman_liau, complex_word_ratio, long_word_ratio,
        ])
    return np.asarray(features, dtype=float)


def readability_matrix(records: Sequence[dict] | pd.DataFrame) -> sparse.csr_matrix:
    return sparse.csr_matrix(readability_array(records))

