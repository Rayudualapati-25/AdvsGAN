from biofake.evaluation import (
    accuracy_score,
    attack_success_rate,
    build_robustness_report,
    render_json_report,
    render_markdown_report,
    robustness_gap,
    robustness_ratio,
)


def test_basic_robustness_metrics():
    rows = [
        {
            "label": "yes",
            "baseline_prediction": "yes",
            "adversarial_prediction": "no",
            "attack_family": "paraphrase",
            "attack_name": "rule_based_paraphrase",
        },
        {
            "label": "no",
            "baseline_prediction": "no",
            "adversarial_prediction": "no",
            "attack_family": "style_transfer",
            "attack_name": "rule_based_style_transfer",
        },
    ]

    baseline_acc = accuracy_score(rows, prediction_field="baseline_prediction")
    attacked_acc = accuracy_score(rows, prediction_field="adversarial_prediction")
    report = build_robustness_report(rows)

    assert baseline_acc == 1.0
    assert attacked_acc == 0.5
    assert attack_success_rate(rows) == 0.5
    assert robustness_gap(baseline_acc, attacked_acc) == 0.5
    assert robustness_ratio(baseline_acc, attacked_acc) == 0.5
    assert report["sample_count"] == 2
    assert report["attack_success_rate"] == 0.5
    assert "paraphrase" in report["attack_families"]
    assert "style_transfer" in report["attack_families"]


def test_report_rendering_produces_markdown_and_json():
    rows = [
        {
            "label": "yes",
            "baseline_prediction": "yes",
            "adversarial_prediction": "no",
            "attack_family": "paraphrase",
        }
    ]
    report = build_robustness_report(rows)
    markdown = render_markdown_report(report)
    json_report = render_json_report(report)

    assert "# BioFake Robustness Report" in markdown
    assert "Attack Families" in markdown
    assert '"sample_count": 1' in json_report

