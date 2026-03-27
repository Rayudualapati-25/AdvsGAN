from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def to_frame(records: Sequence[dict] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records.copy()
    return pd.DataFrame(records)


def extract_texts(records: Sequence[dict] | pd.DataFrame | Iterable[str]) -> list[str]:
    if isinstance(records, pd.DataFrame):
        return records["text"].fillna("").astype(str).tolist()
    records_list = list(records)  # type: ignore[arg-type]
    if records_list and isinstance(records_list[0], str):
        return [str(item) for item in records_list]
    return to_frame(records_list)["text"].fillna("").astype(str).tolist()


@dataclass
class LexicalFeatureBuilder:
    max_word_features: int = 20000
    max_char_features: int = 12000
    word_ngram_max: int = 2
    char_ngram_max: int = 5

    def __post_init__(self) -> None:
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, self.word_ngram_max),
            max_features=self.max_word_features,
            lowercase=True,
            sublinear_tf=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, self.char_ngram_max),
            max_features=self.max_char_features,
            lowercase=True,
            sublinear_tf=True,
        )

    def fit(self, records: Sequence[dict] | pd.DataFrame | Iterable[str]) -> "LexicalFeatureBuilder":
        texts = extract_texts(records)
        self.word_vectorizer.fit(texts)
        self.char_vectorizer.fit(texts)
        return self

    def transform(self, records: Sequence[dict] | pd.DataFrame | Iterable[str]) -> sparse.csr_matrix:
        texts = extract_texts(records)
        word = self.word_vectorizer.transform(texts)
        char = self.char_vectorizer.transform(texts)
        return sparse.hstack([word, char]).tocsr()

    def fit_transform(self, records: Sequence[dict] | pd.DataFrame | Iterable[str]) -> sparse.csr_matrix:
        texts = extract_texts(records)
        word = self.word_vectorizer.fit_transform(texts)
        char = self.char_vectorizer.fit_transform(texts)
        return sparse.hstack([word, char]).tocsr()
