from __future__ import annotations

from biofake.data.pubmed_rct import clean_pubmed_rct_text


def normalize_text(text: str) -> str:
    return clean_pubmed_rct_text(text)

