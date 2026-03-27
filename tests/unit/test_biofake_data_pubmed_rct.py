from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biofake.data import (
    ProcessedRow,
    deterministic_split_for_key,
    process_pubmed_rct_record,
    split_processed_rows,
)


class PubMedRCTDataTests(unittest.TestCase):
    def test_process_pubmed_rct_record_reconstructs_and_round_trips(self) -> None:
        record = {
            "pmid": "12345",
            "title": "Clinical response in asthma",
            "abstract": "BACKGROUND:   first line.\nMETHODS: second line.\nRESULTS: third line.\nCONCLUSIONS: fourth line.",
            "label": "positive",
        }

        row = process_pubmed_rct_record(record)
        self.assertIsInstance(row, ProcessedRow)
        self.assertEqual(row.row_id, "12345")
        self.assertEqual(row.source_id, "12345")
        self.assertEqual(row.split, deterministic_split_for_key("12345"))
        self.assertTrue(row.text.startswith("Clinical response in asthma"))
        self.assertIn("BACKGROUND:", row.abstract)
        self.assertIn("METHODS:", row.abstract)

        round_tripped = ProcessedRow.from_dict(row.to_dict())
        self.assertEqual(round_tripped.row_id, row.row_id)
        self.assertEqual(round_tripped.abstract, row.abstract)
        self.assertEqual(round_tripped.sections, row.sections)

    def test_split_processed_rows_is_stable(self) -> None:
        rows = [
            {"pmid": "1", "title": "a", "abstract": "methods: x"},
            {"pmid": "2", "title": "b", "abstract": "methods: y"},
            {"pmid": "3", "title": "c", "abstract": "methods: z"},
        ]
        first = split_processed_rows(rows)
        second = split_processed_rows(rows)
        self.assertEqual(
            [row.row_id for row in first["train"]],
            [row.row_id for row in second["train"]],
        )
        self.assertEqual(
            sorted(row.row_id for row in first["train"] + first["validation"] + first["test"]),
            ["1", "2", "3"],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
