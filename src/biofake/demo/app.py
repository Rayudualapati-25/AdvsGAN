from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st

from biofake.demo.example_texts import EXAMPLE_TEXTS
from biofake.demo.explain import explain_text
from biofake.io import load_config, read_jsonl
from biofake.models.baseline import BaselineDetector
from biofake.models.hybrid import HybridDetector


def _config_path_from_argv() -> str:
    if "--config" in sys.argv:
        index = sys.argv.index("--config")
        if index + 1 < len(sys.argv):
            return sys.argv[index + 1]
    return "configs/experiments/robust_hybrid.yaml"


@st.cache_resource
def _load_runtime() -> tuple[Any, Any | None, float | None]:
    config_path = _config_path_from_argv()
    config, _ = load_config(config_path)
    detector = None
    threshold = None
    model_path = Path(config.detector.model_artifact)
    threshold_path = Path(config.detector.threshold_artifact)
    if model_path.exists():
        detector = (
            BaselineDetector.load(str(model_path))
            if config.detector.kind == "baseline_tfidf_lr"
            else HybridDetector.load(str(model_path))
        )
    if threshold_path.exists():
        threshold = float(json.loads(threshold_path.read_text(encoding="utf-8"))["threshold"])
    return config, detector, threshold


def _predict_text(detector: Any, text: str, threshold: float | None) -> dict[str, Any]:
    if detector is None or threshold is None:
        return {"prediction": "unavailable", "probability_synthetic": 0.5}
    row = {
        "id": "demo_input",
        "split": "demo",
        "label": "human",
        "source": "demo",
        "generator": None,
        "attack": None,
        "parent_id": None,
        "text": text,
        "meta": {},
    }
    probability = float(detector.predict_proba([row])[0])
    prediction = "synthetic" if probability >= threshold else "human"
    return {"prediction": prediction, "probability_synthetic": probability}


def main() -> None:
    st.set_page_config(page_title="BioFake Demo", layout="wide")
    config, detector, threshold = _load_runtime()
    predictions = read_jsonl(config.eval.prediction_output)

    st.title("BioFake: Biomedical Text Deepfake Robustness")
    st.caption("CPU-first detector demo with optional local model artifacts.")

    col1, col2 = st.columns([2, 1])
    with col1:
        selected = st.selectbox("Load example", [example["title"] for example in EXAMPLE_TEXTS], index=0)
        default_text = next(example["text"] for example in EXAMPLE_TEXTS if example["title"] == selected)
        text = st.text_area("Input text", value=default_text, height=220)
        result = _predict_text(detector, text, threshold)
        st.metric("Predicted label", result["prediction"])
        st.metric("P(synthetic)", f"{result['probability_synthetic']:.4f}")
        st.subheader("Heuristic explanation")
        for reason in explain_text(text):
            st.write(f"- {reason}")

    with col2:
        st.subheader("Run Status")
        st.write(f"Config: `{_config_path_from_argv()}`")
        st.write(f"Model artifact present: `{detector is not None}`")
        st.write(f"Threshold present: `{threshold is not None}`")
        st.write(f"Saved predictions: `{len(predictions)}`")
        if predictions:
            st.subheader("Recent predictions")
            for row in predictions[:5]:
                st.write(
                    f"- `{row.get('id')}` label={row.get('label')} pred={row.get('prediction')} p={row.get('probability_synthetic', 0.0):.3f}"
                )


if __name__ == "__main__":
    main()
