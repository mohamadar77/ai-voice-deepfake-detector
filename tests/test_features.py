"""Fast, dataset-free tests for feature extraction and the feature cache (#7).

All audio is synthesised to ``tmp_path``; none of it touches the ASVspoof
dataset. These tests import ``librosa`` (via the feature path), so they run in
the with-deps environment (CI and the host), like ``test_preprocessing``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from voice_detector import features as F
from voice_detector.cli import main
from voice_detector.manifest import compute_manifest_hash, write_manifest_sqlite
from voice_detector.preprocessing import (
    AudioConfig,
    MfccConfig,
    PreprocessingConfig,
    clip_features_from_source,
)

# A short window so a ~1 s clip yields several windows without slow long audio.
_SMALL = PreprocessingConfig(
    audio=AudioConfig(sample_rate_hz=16000, window_seconds=0.25),
    mfcc=MfccConfig(n_mfcc=20, include_delta=True),
)


def _checksum(clip_id: str) -> str:
    return hashlib.sha256(clip_id.encode()).hexdigest()


def _sine(seconds: float, sample_rate: int = 16000, freq: float = 220.0) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(seconds * sample_rate), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _row(
    clip_id: str,
    *,
    official_split: str,
    label: str = "bonafide",
    development_role: str | None = None,
    local_path: str | None = None,
    checksum: str | None = None,
) -> dict[str, Any]:
    return {
        "clip_id": clip_id,
        "speaker_id": "LA_0001",
        "source_id": None,
        "attack_id": None if label == "bonafide" else "A01",
        "label": label,
        "official_split": official_split,
        "development_role": development_role,
        "duration_seconds": 1.0,
        "sample_rate_hz": 16000,
        "channels": 1,
        "checksum_sha256": checksum or _checksum(clip_id),
        "provenance": "synthetic test row",
        "local_path": local_path,
    }


def _clip_rows(tmp_path: Path, split: str, n: int, *, seconds: float = 0.6) -> list[dict[str, Any]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index in range(n):
        clip_id = f"{split}_{index}"
        path = tmp_path / f"{clip_id}.flac"
        sf.write(path, _sine(seconds, freq=200.0 + 20.0 * index), 16000)
        label = "bonafide" if index % 2 == 0 else "spoof"
        rows.append(_row(clip_id, official_split=split, label=label, local_path=str(path)))
    return rows


# --- cache key: role-independent, content-sensitive --------------------------


def test_cache_key_ignores_development_role() -> None:
    base = [_row("c1", official_split="development"), _row("c2", official_split="development")]
    rolled = [
        _row("c1", official_split="development", development_role="select"),
        _row("c2", official_split="development", development_role="threshold"),
    ]
    # Roles change the manifest hash but must NOT change the feature cache key.
    assert compute_manifest_hash(base) != compute_manifest_hash(rolled)
    assert F.feature_cache_key(base, _SMALL) == F.feature_cache_key(rolled, _SMALL)


def test_cache_key_changes_with_content_and_config() -> None:
    rows = [_row("c1", official_split="train")]
    key = F.feature_cache_key(rows, _SMALL)
    # A different audio checksum -> different key.
    assert key != F.feature_cache_key(
        [_row("c1", official_split="train", checksum=_checksum("other"))], _SMALL
    )
    # A different split assignment -> different key.
    assert key != F.feature_cache_key([_row("c1", official_split="development")], _SMALL)
    # A different preprocessing parameter -> different key.
    other_config = PreprocessingConfig(audio=_SMALL.audio, mfcc=MfccConfig(n_mfcc=13))
    assert key != F.feature_cache_key(rows, other_config)


def test_cache_key_is_order_independent() -> None:
    a = [_row("a", official_split="train"), _row("b", official_split="train")]
    assert F.feature_cache_key(a, _SMALL) == F.feature_cache_key(list(reversed(a)), _SMALL)


# --- extraction --------------------------------------------------------------


def test_extract_shapes_alignment_and_finite(tmp_path: Path) -> None:
    rows = _clip_rows(tmp_path, "train", 3)
    result = F.extract_split_features(rows, _SMALL, manifest_hash="mh")

    assert result.split == "train"
    assert result.n_clips == 3
    assert result.feature_dim == _SMALL.mfcc.feature_dim
    assert result.features.shape == (result.n_windows, _SMALL.mfcc.feature_dim)
    assert result.features.dtype == np.float32
    assert int(result.window_counts.sum()) == result.n_windows
    assert np.all(np.isfinite(result.features))
    # Window-to-clip map covers every row exactly once, in clip order.
    index = result.window_clip_index()
    assert index.shape == (result.n_windows,)
    assert index.min() == 0 and index.max() == result.n_clips - 1
    assert list(result.clip_ids) == sorted(result.clip_ids)  # deterministic order


def test_extract_matches_serving_path(tmp_path: Path) -> None:
    # The cached features for a clip must equal what the serving scorer computes
    # from the same source through the shared path -- no divergence.
    rows = _clip_rows(tmp_path, "development", 1)
    result = F.extract_split_features(rows, _SMALL)
    served = clip_features_from_source(rows[0]["local_path"], _SMALL)
    assert np.array_equal(result.features, served)


def test_extract_is_deterministic(tmp_path: Path) -> None:
    rows = _clip_rows(tmp_path, "train", 2)
    first = F.extract_split_features(rows, _SMALL)
    second = F.extract_split_features(rows, _SMALL)
    assert np.array_equal(first.features, second.features)
    assert first.cache_key == second.cache_key


def test_extract_refuses_missing_local_path(tmp_path: Path) -> None:
    rows = [_row("c1", official_split="train", local_path=None)]
    with pytest.raises(ValueError, match="no local_path"):
        F.extract_split_features(rows, _SMALL)


def test_extract_refuses_non_finite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _clip_rows(tmp_path, "train", 1)
    bad = np.full((2, _SMALL.mfcc.feature_dim), np.nan, dtype=np.float32)
    monkeypatch.setattr(F, "clip_features_from_source", lambda *a, **k: bad)
    with pytest.raises(ValueError, match="non-finite"):
        F.extract_split_features(rows, _SMALL)


def test_extract_rejects_mixed_splits(tmp_path: Path) -> None:
    rows = _clip_rows(tmp_path, "train", 1) + _clip_rows(tmp_path, "development", 1)
    with pytest.raises(ValueError, match="one split"):
        F.extract_split_features(rows, _SMALL)


def test_extract_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no rows"):
        F.extract_split_features([], _SMALL)


# --- cache round-trip and freshness -----------------------------------------


def test_write_read_roundtrip_without_pickle(tmp_path: Path) -> None:
    rows = _clip_rows(tmp_path, "train", 3)
    original = F.extract_split_features(rows, _SMALL, manifest_hash="mh-123")
    path = F.write_split_features(original, F.cache_path(tmp_path / "cache", "train"))
    assert path.is_file()

    loaded = F.read_split_features(path)
    assert np.array_equal(loaded.features, original.features)
    assert list(loaded.clip_ids) == list(original.clip_ids)
    assert list(loaded.labels) == list(original.labels)
    assert np.array_equal(loaded.window_counts, original.window_counts)
    assert loaded.cache_key == original.cache_key
    assert loaded.manifest_hash == "mh-123"
    assert loaded.preprocessing_version == _SMALL.version


def test_write_is_atomic_leaves_no_temp(tmp_path: Path) -> None:
    rows = _clip_rows(tmp_path, "train", 1)
    result = F.extract_split_features(rows, _SMALL)
    path = F.cache_path(tmp_path / "cache", "train")
    F.write_split_features(result, path)
    assert not path.with_name(f"{path.name}.tmp").exists()


def test_load_cached_if_fresh(tmp_path: Path) -> None:
    rows = _clip_rows(tmp_path, "train", 2)
    path = F.cache_path(tmp_path / "cache", "train")
    assert F.load_cached_if_fresh(rows, _SMALL, path) is None  # no file yet

    F.write_split_features(F.extract_split_features(rows, _SMALL), path)
    assert F.load_cached_if_fresh(rows, _SMALL, path) is not None  # matches

    # A content change (new checksum) makes the existing cache stale.
    stale_rows = [dict(rows[0], checksum_sha256=_checksum("moved")), rows[1]]
    assert F.load_cached_if_fresh(stale_rows, _SMALL, path) is None


def test_read_rejects_inconsistent_cache(tmp_path: Path) -> None:
    # A hand-corrupted npz (window_counts that do not sum to the feature rows)
    # must be rejected on read rather than silently mis-aligned downstream.
    path = tmp_path / "bad.npz"
    meta = {
        "feature_schema_version": F.FEATURE_SCHEMA_VERSION,
        "split": "train",
        "cache_key": "k",
        "preprocessing_version": _SMALL.version,
        "feature_dim": _SMALL.mfcc.feature_dim,
        "manifest_hash": None,
        "n_clips": 1,
        "n_windows": 1,
        "created_utc": "now",
    }
    np.savez(
        path,
        features=np.zeros((3, _SMALL.mfcc.feature_dim), dtype=np.float32),
        clip_ids=np.array(["c1"]),
        labels=np.array(["bonafide"]),
        window_counts=np.array([1], dtype=np.int64),
        meta=np.array(json.dumps(meta)),
    )
    with pytest.raises(ValueError, match="window_counts must sum"):
        F.read_split_features(path)


# --- extract-features CLI ----------------------------------------------------


def _build_manifest(tmp_path: Path) -> Path:
    manifest_dir = tmp_path / "manifest"
    rows = (
        _clip_rows(tmp_path / "audio_train", "train", 2)
        + _clip_rows(tmp_path / "audio_dev", "development", 2)
    )
    meta = {"manifest_hash": compute_manifest_hash(rows)}
    write_manifest_sqlite(
        rows, manifest_dir / "manifest.sqlite", meta=meta, include_local_path=True
    )
    return manifest_dir


def _run(args: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    code = main(args)
    out = capsys.readouterr().out
    return code, json.loads(out)


def test_cli_extracts_and_caches(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_dir = _build_manifest(tmp_path)
    out_dir = tmp_path / "features"
    code, payload = _run(
        [
            "extract-features",
            "--manifest-dir", str(manifest_dir),
            "--out-dir", str(out_dir),
            "--config", "configs/default.yaml",
            "--splits", "train", "development",
        ],
        capsys,
    )
    assert code == 0
    assert payload["status"] == "extracted"
    assert {s["split"] for s in payload["splits"]} == {"train", "development"}
    for split in payload["splits"]:
        assert split["reused_cache"] is False
        assert split["n_clips"] == 2
        assert split["cache_bytes"] > 0
        assert Path(split["output"]).is_file()
    assert (out_dir / "features_summary.json").is_file()


def test_cli_reuses_fresh_cache(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_dir = _build_manifest(tmp_path)
    out_dir = tmp_path / "features"
    base = [
        "extract-features",
        "--manifest-dir", str(manifest_dir),
        "--out-dir", str(out_dir),
        "--config", "configs/default.yaml",
        "--splits", "train",
    ]
    _run(base, capsys)
    code, payload = _run(base, capsys)
    assert code == 0
    assert payload["splits"][0]["reused_cache"] is True


def test_cli_force_reextracts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_dir = _build_manifest(tmp_path)
    out_dir = tmp_path / "features"
    base = [
        "extract-features",
        "--manifest-dir", str(manifest_dir),
        "--out-dir", str(out_dir),
        "--config", "configs/default.yaml",
        "--splits", "train",
    ]
    _run(base, capsys)
    code, payload = _run([*base, "--force"], capsys)
    assert code == 0
    assert payload["splits"][0]["reused_cache"] is False


def test_cli_dry_run_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_dir = _build_manifest(tmp_path)
    out_dir = tmp_path / "features"
    code, payload = _run(
        [
            "extract-features",
            "--manifest-dir", str(manifest_dir),
            "--out-dir", str(out_dir),
            "--config", "configs/default.yaml",
            "--splits", "train",
            "--dry-run",
        ],
        capsys,
    )
    assert code == 0
    assert payload["status"] == "planned"
    assert payload["splits"][0]["cache_fresh"] is False
    assert not out_dir.exists()


def test_cli_refuses_evaluation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_dir = _build_manifest(tmp_path)
    code, payload = _run(
        [
            "extract-features",
            "--manifest-dir", str(manifest_dir),
            "--config", "configs/default.yaml",
            "--splits", "evaluation",
        ],
        capsys,
    )
    assert code == 1
    assert payload["status"] == "refused"
    assert payload["reason"] == "evaluation_is_locked"


def test_cli_refuses_unknown_split(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_dir = _build_manifest(tmp_path)
    code, payload = _run(
        [
            "extract-features",
            "--manifest-dir", str(manifest_dir),
            "--config", "configs/default.yaml",
            "--splits", "nonsense",
        ],
        capsys,
    )
    assert code == 1
    assert payload["reason"] == "unknown_split"
