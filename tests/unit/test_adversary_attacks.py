from biofake.adversary import (
    CompressionExpansionAttack,
    ParaphraseAttack,
    StyleTransferAttack,
    build_attack,
)


def test_paraphrase_attack_updates_text_and_metadata():
    attack = ParaphraseAttack()
    row = {
        "sample_id": "row-1",
        "content": "Patients show an important increase in risk in order to improve outcomes.",
    }

    attacked = attack.attack_row(row)

    assert attacked["content"] != row["content"]
    assert attacked["adversarial_text"] == attacked["content"]
    assert attacked["original_text"] == row["content"]
    assert attacked["attack_family"] == "paraphrase"
    assert attacked["attack_name"] == "rule_based_paraphrase"
    assert attacked["attack_metadata"]["fallback_used"] is False
    assert attacked["attack_metadata"]["transformations"]


def test_compression_attack_can_compress_and_expand_deterministically():
    compress = CompressionExpansionAttack(mode="compress", max_sentences=1)
    expand = CompressionExpansionAttack(mode="expand", expansion_sentence="This adds context.")
    row = {"text": "It is worth noting that the result was strong (and consistent). Another sentence follows."}

    compressed = compress.attack_row(row)
    expanded = expand.attack_row(row)

    assert compressed["text"] != row["text"]
    assert compressed["attack_family"] == "compression_expansion"
    assert compressed["attack_metadata"]["mode"] == "compress"
    assert expanded["text"].endswith("This adds context.")
    assert expanded["attack_metadata"]["mode"] == "expand"


def test_style_transfer_attack_respects_style_choice():
    attack = StyleTransferAttack(style="clinical")
    row = {"text": "We found the treatment shows a very good response."}

    attacked = attack.attack_row(row)

    assert attacked["text"] != row["text"]
    assert "clinically" in attacked["text"].lower() or "analysis indicates" in attacked["text"].lower()
    assert attacked["attack_family"] == "style_transfer"
    assert attacked["attack_metadata"]["style"] == "clinical"


def test_deterministic_fallback_for_unsupported_style_and_noop_paraphrase():
    unsupported = StyleTransferAttack(style="does-not-exist")
    noop = ParaphraseAttack(replace_phrases=False, reorder_sentences=False, synonym_map={})

    unsupported_row = unsupported.attack_row({"text": "A simple sentence."})
    noop_row = noop.attack_row({"text": "A simple sentence."})

    assert unsupported_row["text"] == "A simple sentence."
    assert unsupported_row["attack_metadata"]["fallback_used"] is True
    assert noop_row["text"] == "A simple sentence."
    assert noop_row["attack_metadata"]["fallback_used"] is True


def test_builder_accepts_config_dict():
    attack = build_attack({"attack": "paraphrase", "swap_two_sentences": True})
    assert isinstance(attack, ParaphraseAttack)

