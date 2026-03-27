from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biofake.generation import SyntheticGenerator, build_generation_backend


class GenerationBackendTests(unittest.TestCase):
    def test_auto_backend_falls_back_deterministically_without_model(self) -> None:
        backend = build_generation_backend({"backend": "auto", "model_path": ""})
        result_one = backend.generate(
            __import__("biofake.generation.schema", fromlist=["GenerationRequest"]).GenerationRequest(
                prompt="Generate a biomedical abstract about aspirin and stroke.",
                seed=7,
                max_tokens=80,
            )
        )
        result_two = backend.generate(
            __import__("biofake.generation.schema", fromlist=["GenerationRequest"]).GenerationRequest(
                prompt="Generate a biomedical abstract about aspirin and stroke.",
                seed=7,
                max_tokens=80,
            )
        )
        self.assertEqual(result_one.text, result_two.text)
        self.assertIn("BACKGROUND:", result_one.text)
        self.assertEqual(result_one.provenance.backend, "deterministic-fallback")
        self.assertIsNotNone(result_one.provenance.prompt_hash)

    def test_synthetic_generator_uses_backend_and_provenance(self) -> None:
        generator = SyntheticGenerator(config={"backend": "deterministic"})
        result = generator.generate("Create a synthetic trial abstract for oncology.", seed=11)
        self.assertIn("CONCLUSION:", result.text)
        self.assertEqual(result.provenance.backend, "deterministic-fallback")
        self.assertEqual(result.provenance.seed, 11)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
