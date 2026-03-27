from biofake.adversary import build_attack
from biofake.evaluation import filter_rows_by_attack_metadata, leave_one_family_out


def test_filter_rows_by_attack_metadata_uses_families_and_fallback_flags():
    rows = [
        {"attack_family": "paraphrase", "attack_name": "a", "fallback_used": False},
        {"attack_family": "style_transfer", "attack_name": "b", "fallback_used": True},
        {"attack_family": "compression_expansion", "attack_name": "c", "fallback_used": False},
    ]

    filtered = filter_rows_by_attack_metadata(rows, families=["paraphrase", "style_transfer"], fallback_used=False)

    assert len(filtered) == 1
    assert filtered[0]["attack_family"] == "paraphrase"


def test_leave_one_family_out_builds_expected_scenarios():
    rows = [
        {"attack_family": "paraphrase"},
        {"attack_family": "style_transfer"},
        {"attack_family": "compression_expansion"},
    ]

    scenarios = leave_one_family_out(rows)

    assert "keep_all" in scenarios
    assert "drop_paraphrase" in scenarios
    assert len(scenarios["drop_paraphrase"]) == 2
    assert len(scenarios["drop_style_transfer"]) == 2
    assert len(scenarios["drop_compression_expansion"]) == 2


def test_build_attack_supports_string_name():
    attack = build_attack("style_transfer")
    assert attack.family == "style_transfer"

