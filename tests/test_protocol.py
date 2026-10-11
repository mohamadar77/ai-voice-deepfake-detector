from pathlib import Path

import pytest

from voice_detector.protocol import (
    ProtocolEntry,
    parse_protocol_file,
    parse_protocol_line,
    summarize_protocol,
)

# A tiny stand-in for an official CM protocol file. The blank line is
# intentional: real protocol files tolerate trailing blank lines.
_TRAIN_FIXTURE = """\
LA_0079 LA_T_1138215 - - bonafide
LA_0079 LA_T_1271820 - A01 spoof
LA_0080 LA_T_4984763 - A02 spoof

LA_0081 LA_T_8842348 - - bonafide
"""


def test_parse_line_reads_a_bona_fide_clip() -> None:
    entry = parse_protocol_line("LA_0079 LA_T_1138215 - - bonafide", "train")
    assert entry == ProtocolEntry(
        speaker_id="LA_0079",
        clip_id="LA_T_1138215",
        attack_id=None,
        label="bonafide",
        official_split="train",
    )


def test_parse_line_reads_a_spoof_clip_with_attack_id() -> None:
    entry = parse_protocol_line("LA_0080 LA_E_4984763 - A17 spoof", "evaluation")
    assert entry is not None
    assert entry.label == "spoof"
    assert entry.attack_id == "A17"
    assert entry.official_split == "evaluation"


def test_blank_line_is_skipped() -> None:
    assert parse_protocol_line("   ", "train") is None


def test_parse_file_preserves_every_clip(tmp_path: Path) -> None:
    protocol = tmp_path / "ASVspoof2019.LA.cm.train.trn.txt"
    protocol.write_text(_TRAIN_FIXTURE, encoding="utf-8")

    entries = parse_protocol_file(protocol, "train")

    assert [entry.clip_id for entry in entries] == [
        "LA_T_1138215",
        "LA_T_1271820",
        "LA_T_4984763",
        "LA_T_8842348",
    ]
    assert all(entry.official_split == "train" for entry in entries)


def test_summarize_reports_class_and_attack_distribution(tmp_path: Path) -> None:
    protocol = tmp_path / "train.txt"
    protocol.write_text(_TRAIN_FIXTURE, encoding="utf-8")

    summary = summarize_protocol(parse_protocol_file(protocol, "train"))

    assert summary["total_clips"] == 4
    assert summary["official_splits"] == ["train"]
    assert summary["label_counts"] == {"bonafide": 2, "spoof": 2}
    assert summary["unique_speakers"] == 3
    assert summary["unique_attacks"] == 2
    assert summary["attack_counts"] == {"A01": 1, "A02": 1}


def test_rejects_wrong_column_count() -> None:
    with pytest.raises(ValueError, match="columns"):
        parse_protocol_line("LA_0079 LA_T_1138215 bonafide", "train", line_number=7)


def test_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="label must be one of"):
        parse_protocol_line("LA_0079 LA_T_1138215 - - maybe", "train")


def test_rejects_bona_fide_clip_that_declares_an_attack() -> None:
    with pytest.raises(ValueError, match="must not declare an attack id"):
        parse_protocol_line("LA_0079 LA_T_1138215 - A01 bonafide", "train")


def test_rejects_spoof_clip_without_an_attack() -> None:
    with pytest.raises(ValueError, match="must declare an attack id"):
        parse_protocol_line("LA_0079 LA_T_1271820 - - spoof", "train")


def test_rejects_unknown_split() -> None:
    with pytest.raises(ValueError, match="official_split must be one of"):
        parse_protocol_line("LA_0079 LA_T_1138215 - - bonafide", "test")


def test_rejects_duplicate_clip_id(tmp_path: Path) -> None:
    protocol = tmp_path / "dup.txt"
    protocol.write_text(
        "LA_0079 LA_T_0001 - - bonafide\nLA_0080 LA_T_0001 - A01 spoof\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate clip_id"):
        parse_protocol_file(protocol, "train")
