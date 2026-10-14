"""Offline MFCC feature extraction and a content-addressed feature cache.

This module wires the built manifest to the single shared preprocessing path
(:func:`voice_detector.preprocessing.clip_features_from_source`) and caches the
resulting per-window pooled MFCC features, one ``.npz`` per official split, in an
ignored artifacts directory. It computes **no** scaler and fits nothing: a clip
becomes a ``(n_windows, feature_dim)`` matrix of pooled statistics, exactly the
representation the serving scorer sees, so offline features and served features
come from one code path and cannot diverge.

Cache key (role-independent). The cache is keyed by a content digest over the
feature-determining fields only -- ``(clip_id, label, official_split, checksum)``
-- stacked with a fingerprint of every preprocessing parameter and the feature
schema version. It deliberately **excludes** ``development_role``: the #31
reservation freeze flips that column (and therefore the full manifest hash) but
does not change a single feature value, so keying on roles would needlessly
invalidate the whole train/development cache. The full manifest hash (roles
included) is still recorded in each cache's metadata for provenance, so a run
record can cite the exact manifest state the features were extracted from.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from voice_detector.preprocessing import PreprocessingConfig, clip_features_from_source

# Feature cache epoch. Bump when the *cache layout or key formula* changes in a
# way that must invalidate every existing cache even if the preprocessing
# version did not move. The preprocessing version tracks feature *values*; this
# tracks how they are keyed and stored.
FEATURE_SCHEMA_VERSION = "features-v1"

# Fields that determine a clip's features. ``development_role`` is intentionally
# absent (see the module docstring); duration/rate/channels are byte-determined
# by the checksum and so are redundant here.
_CONTENT_FIELDS = ("clip_id", "label", "official_split", "checksum_sha256")


@dataclass(frozen=True)
class SplitFeatures:
    """Cached features for one official split, aligned clip-by-clip.

    ``features`` holds every window's pooled vector stacked in clip order; the
    windows for clip ``i`` are the ``window_counts[i]`` consecutive rows that
    follow its predecessors. Window scores are aggregated back to one score per
    clip downstream -- the windows are never independent clips -- and
    :meth:`window_clip_index` gives the window-to-clip map for that.
    """

    split: str
    clip_ids: np.ndarray  # (n_clips,) unicode
    labels: np.ndarray  # (n_clips,) unicode
    window_counts: np.ndarray  # (n_clips,) int64
    features: np.ndarray  # (n_windows, feature_dim) float32
    cache_key: str
    preprocessing_version: str
    feature_dim: int
    manifest_hash: str | None = None

    def __post_init__(self) -> None:
        n_clips = int(self.clip_ids.shape[0])
        if not (self.labels.shape[0] == self.window_counts.shape[0] == n_clips):
            raise ValueError("clip_ids, labels and window_counts must be the same length")
        if self.features.ndim != 2:
            raise ValueError("features must be a 2-D (n_windows, feature_dim) array")
        if int(self.window_counts.sum()) != int(self.features.shape[0]):
            raise ValueError(
                "window_counts must sum to the number of feature rows "
                f"({int(self.window_counts.sum())} != {int(self.features.shape[0])})"
            )
        if n_clips and int(self.features.shape[1]) != int(self.feature_dim):
            raise ValueError(
                f"feature width {int(self.features.shape[1])} != declared feature_dim "
                f"{int(self.feature_dim)}"
            )
        if np.any(self.window_counts < 1):
            raise ValueError("every clip must contribute at least one window")

    @property
    def n_clips(self) -> int:
        return int(self.clip_ids.shape[0])

    @property
    def n_windows(self) -> int:
        return int(self.features.shape[0])

    def window_clip_index(self) -> np.ndarray:
        """Return the clip index (into ``clip_ids``) of every feature row."""

        return np.repeat(np.arange(self.n_clips, dtype=np.int64), self.window_counts)


def preprocessing_fingerprint(config: PreprocessingConfig) -> str:
    """A stable, human-readable string of every value that affects features.

    Included in the cache key so that changing any preprocessing parameter --
    even without bumping ``preprocessing.version`` -- invalidates the cache
    rather than silently reusing mismatched vectors.
    """

    audio, mfcc = config.audio, config.mfcc
    return (
        f"{config.version};sr={audio.sample_rate_hz};mono={int(audio.mono)};"
        f"win={audio.window_seconds};n_mfcc={mfcc.n_mfcc};n_fft={mfcc.n_fft};"
        f"hop={mfcc.hop_length};fmin={mfcc.fmin};fmax={mfcc.fmax};"
        f"delta={int(mfcc.include_delta)}"
    )


def _content_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda row: row["clip_id"]):
        line = "|".join(str(row[field]) for field in _CONTENT_FIELDS) + "\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def feature_cache_key(rows: Iterable[Mapping[str, Any]], config: PreprocessingConfig) -> str:
    """Role-independent cache key over split content + preprocessing fingerprint.

    Pass the rows of a single split; the key then identifies exactly that
    split's cached features and is unaffected by the #31 reservation roles.
    """

    digest = hashlib.sha256()
    digest.update(FEATURE_SCHEMA_VERSION.encode("utf-8"))
    digest.update(b"\n")
    digest.update(preprocessing_fingerprint(config).encode("utf-8"))
    digest.update(b"\n")
    digest.update(_content_digest(rows).encode("utf-8"))
    return digest.hexdigest()


def cache_path(cache_dir: Path | str, split: str) -> Path:
    """Path of the ``.npz`` cache for ``split`` under ``cache_dir``."""

    return Path(cache_dir) / f"features_{split}.npz"


def extract_split_features(
    rows: Iterable[Mapping[str, Any]],
    config: PreprocessingConfig,
    *,
    manifest_hash: str | None = None,
) -> SplitFeatures:
    """Extract pooled MFCC features for one split's clips via the manifest.

    ``rows`` are the catalog rows of a single official split; each must carry a
    ``local_path`` to its audio. Clips are processed in ``clip_id`` order for
    determinism. Every clip's feature matrix is checked to be finite; a
    non-finite or empty result refuses with the offending ``clip_id`` rather
    than caching corrupt features. No scaler is fit and nothing is normalised.
    """

    ordered = sorted(rows, key=lambda row: row["clip_id"])
    if not ordered:
        raise ValueError("no rows to extract; nothing to cache for this split")

    splits = {str(row["official_split"]) for row in ordered}
    if len(splits) != 1:
        raise ValueError(f"extract_split_features expects one split, got {sorted(splits)}")
    split = splits.pop()
    feature_dim = config.mfcc.feature_dim

    clip_ids: list[str] = []
    labels: list[str] = []
    window_counts: list[int] = []
    matrices: list[np.ndarray] = []
    for row in ordered:
        local_path = row.get("local_path")
        if not local_path:
            raise ValueError(
                f"clip {row['clip_id']!r} has no local_path; rebuild the manifest where the "
                "audio lives before extracting features"
            )
        matrix = clip_features_from_source(local_path, config)
        if matrix.ndim != 2 or matrix.shape[1] != feature_dim:
            raise ValueError(
                f"clip {row['clip_id']!r} produced features of shape {matrix.shape}, "
                f"expected (n_windows, {feature_dim})"
            )
        if matrix.shape[0] < 1:
            raise ValueError(f"clip {row['clip_id']!r} produced no windows")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"clip {row['clip_id']!r} produced non-finite features")
        clip_ids.append(str(row["clip_id"]))
        labels.append(str(row["label"]))
        window_counts.append(int(matrix.shape[0]))
        matrices.append(matrix.astype(np.float32, copy=False))

    features = np.concatenate(matrices, axis=0).astype(np.float32, copy=False)
    return SplitFeatures(
        split=split,
        clip_ids=np.array(clip_ids),
        labels=np.array(labels),
        window_counts=np.array(window_counts, dtype=np.int64),
        features=features,
        cache_key=feature_cache_key(ordered, config),
        preprocessing_version=config.version,
        feature_dim=feature_dim,
        manifest_hash=manifest_hash,
    )


def _meta(split_features: SplitFeatures) -> dict[str, Any]:
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "split": split_features.split,
        "cache_key": split_features.cache_key,
        "preprocessing_version": split_features.preprocessing_version,
        "feature_dim": split_features.feature_dim,
        "manifest_hash": split_features.manifest_hash,
        "n_clips": split_features.n_clips,
        "n_windows": split_features.n_windows,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def write_split_features(split_features: SplitFeatures, path: Path | str) -> Path:
    """Write features to a ``.npz`` atomically (temp file, then one rename).

    Atomic so a dropped SSH session mid-write never leaves a truncated cache the
    next run would read as valid (see docs/environment.md, "Long jobs").
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp")
    with temp.open("wb") as handle:
        np.savez(
            handle,
            features=split_features.features,
            clip_ids=split_features.clip_ids,
            labels=split_features.labels,
            window_counts=split_features.window_counts,
            meta=np.array(json.dumps(_meta(split_features), sort_keys=True)),
        )
    os.replace(temp, path)
    return path


def read_split_features(path: Path | str) -> SplitFeatures:
    """Read a ``.npz`` feature cache back into a :class:`SplitFeatures`.

    ``allow_pickle`` stays off: the cache holds only numeric and unicode arrays
    plus a JSON metadata string, so an untrusted or corrupt file cannot execute
    code on load. :class:`SplitFeatures` validation then rejects an inconsistent
    cache (mismatched lengths, wrong width, zero-window clips).
    """

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"].item()))
        return SplitFeatures(
            split=str(meta["split"]),
            clip_ids=data["clip_ids"],
            labels=data["labels"],
            window_counts=data["window_counts"].astype(np.int64, copy=False),
            features=data["features"].astype(np.float32, copy=False),
            cache_key=str(meta["cache_key"]),
            preprocessing_version=str(meta["preprocessing_version"]),
            feature_dim=int(meta["feature_dim"]),
            manifest_hash=meta.get("manifest_hash"),
        )


def load_cached_if_fresh(
    rows: Iterable[Mapping[str, Any]],
    config: PreprocessingConfig,
    path: Path | str,
) -> SplitFeatures | None:
    """Return the cache at ``path`` iff it matches the expected key, else ``None``.

    "Fresh" means the cached ``cache_key`` equals the key recomputed from the
    current split rows and preprocessing config. A missing or unreadable cache,
    or any key mismatch, returns ``None`` so the caller re-extracts.
    """

    path = Path(path)
    if not path.is_file():
        return None
    expected = feature_cache_key(rows, config)
    try:
        cached = read_split_features(path)
    except (OSError, ValueError, KeyError):
        return None
    return cached if cached.cache_key == expected else None
