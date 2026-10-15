"""Robustness transformations and training augmentation (#34).

Pure functions that take a waveform and return a transformed waveform with
structured metadata. Transforms are deterministic given the per-clip seed.

Evaluation conditions (codec, noise) are applied identically to both classes
and the same underlying clips with frozen parameters. Silence-trim is a
diagnostic-only transform, never used during training.

Training augmentation is applied BEFORE random crop so that codec has full
waveform context and noise SNR uses full-signal RMS.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

AUGMENTATION_VERSION = "robustness-v1"


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------

def derive_clip_seed(clip_id: str, global_seed: int) -> int:
    """SHA-256 deterministic per-clip seed.

    Matches the reservation.py pattern: ``sha256(f"{seed}|{id}")``.
    """
    digest = hashlib.sha256(f"{global_seed}|{clip_id}".encode()).hexdigest()
    return int(digest[:16], 16)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CodecResult:
    """Result of codec compression transform."""

    waveform: np.ndarray
    original_length: int
    output_length: int
    samples_delta: int
    codec: str
    bitrate_kbps: int
    container: str
    ffmpeg_version: str


@dataclass(frozen=True)
class NoiseResult:
    """Result of additive noise transform."""

    waveform: np.ndarray
    snr_db: float
    reference_rms: float
    noise_rms: float
    n_clipped: int
    max_abs_before_clip: float


@dataclass(frozen=True)
class TrimResult:
    """Result of silence-trim diagnostic."""

    waveform: np.ndarray
    original_length: int
    trimmed_length: int
    samples_removed: int
    kept_original: bool
    top_db: float
    min_duration_seconds: float


# ---------------------------------------------------------------------------
# ffmpeg utilities
# ---------------------------------------------------------------------------

class TransformationError(Exception):
    """Raised when a transform subprocess fails."""


def check_ffmpeg() -> str:
    """Check ffmpeg availability and return its version string.

    Raises ``TransformationError`` if ffmpeg is not found.
    """
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise TransformationError(
            "ffmpeg not found on PATH; install it for codec transforms"
        )
    result = subprocess.run(
        [exe, "-version"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    first_line = result.stdout.split("\n", 1)[0] if result.stdout else "unknown"
    return first_line


def _ffmpeg_version_cached() -> str:
    if not hasattr(_ffmpeg_version_cached, "_value"):
        _ffmpeg_version_cached._value = check_ffmpeg()  # type: ignore[attr-defined]
    return _ffmpeg_version_cached._value  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Codec transform
# ---------------------------------------------------------------------------

def codec_transform(
    waveform: np.ndarray,
    *,
    sample_rate: int = 16000,
    codec: str = "libopus",
    container: str = "ogg",
    bitrate_kbps: int = 24,
) -> CodecResult:
    """Apply codec compression via ffmpeg and return the round-tripped waveform.

    The output is truncated to the original length to remove codec padding.
    """
    import soundfile as sf

    ffmpeg_ver = _ffmpeg_version_cached()
    original_length = waveform.size

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        in_path = tmp / "input.wav"
        out_path = tmp / f"output.{container}"
        decoded_path = tmp / "decoded.wav"

        sf.write(str(in_path), waveform, sample_rate, subtype="PCM_16")

        _run_ffmpeg([
            "ffmpeg", "-y", "-i", str(in_path),
            "-c:a", codec,
            "-b:a", f"{bitrate_kbps}k",
            "-ar", str(sample_rate),
            "-ac", "1",
            str(out_path),
        ])

        _run_ffmpeg([
            "ffmpeg", "-y", "-i", str(out_path),
            "-c:a", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", "1",
            str(decoded_path),
        ])

        decoded, _ = sf.read(str(decoded_path), dtype="float32")

    decoded = decoded[:original_length]
    output_length = decoded.size

    return CodecResult(
        waveform=decoded,
        original_length=original_length,
        output_length=output_length,
        samples_delta=output_length - original_length,
        codec=codec,
        bitrate_kbps=bitrate_kbps,
        container=container,
        ffmpeg_version=ffmpeg_ver,
    )


def _run_ffmpeg(cmd: list[str]) -> None:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise TransformationError(
            f"ffmpeg failed (code {result.returncode}): {result.stderr[:500]}"
        )


# ---------------------------------------------------------------------------
# Noise transform
# ---------------------------------------------------------------------------

def noise_transform(
    waveform: np.ndarray,
    *,
    snr_db: float,
    clip_id: str,
    global_seed: int,
    epoch: int = 0,
) -> NoiseResult:
    """Add white Gaussian noise at the specified SNR.

    The per-clip seed is derived deterministically via SHA-256. When
    ``epoch > 0``, the seed is XORed with the epoch number so each
    training epoch sees different noise.
    """
    seed = derive_clip_seed(clip_id, global_seed) ^ epoch
    rng = np.random.default_rng(seed)

    reference_rms = float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2)))

    if reference_rms < 1e-10:
        return NoiseResult(
            waveform=waveform.copy(),
            snr_db=snr_db,
            reference_rms=reference_rms,
            noise_rms=0.0,
            n_clipped=0,
            max_abs_before_clip=float(np.max(np.abs(waveform))) if waveform.size > 0 else 0.0,
        )

    noise_rms = reference_rms / (10 ** (snr_db / 20))
    noise = rng.normal(0, noise_rms, size=waveform.shape).astype(waveform.dtype)
    noisy = waveform + noise

    max_abs_before_clip = float(np.max(np.abs(noisy)))
    n_clipped = int(np.sum(np.abs(noisy) > 1.0))
    noisy = np.clip(noisy, -1.0, 1.0)

    return NoiseResult(
        waveform=noisy,
        snr_db=snr_db,
        reference_rms=reference_rms,
        noise_rms=float(noise_rms),
        n_clipped=n_clipped,
        max_abs_before_clip=max_abs_before_clip,
    )


# ---------------------------------------------------------------------------
# Silence-trim diagnostic
# ---------------------------------------------------------------------------

def silence_trim(
    waveform: np.ndarray,
    *,
    sample_rate: int = 16000,
    top_db: float = 30.0,
    min_duration_seconds: float = 0.3,
) -> TrimResult:
    """Trim leading/trailing silence. Evaluation-only diagnostic.

    If trimming would produce a signal shorter than ``min_duration_seconds``,
    the original waveform is returned with ``kept_original=True``.
    """
    import librosa

    original_length = waveform.size
    min_samples = int(round(sample_rate * min_duration_seconds))

    trimmed, _ = librosa.effects.trim(waveform, top_db=top_db)
    trimmed_length = trimmed.size

    if trimmed_length < min_samples:
        return TrimResult(
            waveform=waveform.copy(),
            original_length=original_length,
            trimmed_length=original_length,
            samples_removed=0,
            kept_original=True,
            top_db=top_db,
            min_duration_seconds=min_duration_seconds,
        )

    return TrimResult(
        waveform=trimmed,
        original_length=original_length,
        trimmed_length=trimmed_length,
        samples_removed=original_length - trimmed_length,
        kept_original=False,
        top_db=top_db,
        min_duration_seconds=min_duration_seconds,
    )


# ---------------------------------------------------------------------------
# Training augmentation callable
# ---------------------------------------------------------------------------

@dataclass
class TrainingAugmentor:
    """Callable for on-the-fly training augmentation.

    Pass an instance to ``MelDataset`` as the ``augment_fn`` parameter (to be
    added in #33 augmented runs). Call ``set_epoch()`` before each epoch so
    the per-clip seed incorporates the epoch number.

    Augmentation is applied BEFORE random crop.
    """

    global_seed: int = 20261004
    probability: float = 0.5
    codec_probability: float = 0.5
    noise_probability: float = 0.5
    noise_snr_range: tuple[float, float] = (10.0, 30.0)
    codec_bitrate_kbps: int = 24
    sample_rate: int = 16000
    epoch: int = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __call__(
        self,
        waveform: np.ndarray,
        clip_id: str,
    ) -> np.ndarray:
        """Apply random augmentation to the waveform.

        Returns the (possibly transformed) waveform. Metadata about which
        transforms were applied is not returned here — the caller can
        reconstruct it from the deterministic seed.
        """
        seed = derive_clip_seed(f"aug|{clip_id}", self.global_seed) ^ self.epoch
        rng = np.random.default_rng(seed)

        if rng.random() >= self.probability:
            return waveform.copy()

        result = waveform
        if rng.random() < self.codec_probability:
            cr = codec_transform(
                result,
                sample_rate=self.sample_rate,
                bitrate_kbps=self.codec_bitrate_kbps,
            )
            result = cr.waveform

        if rng.random() < self.noise_probability:
            snr = rng.uniform(self.noise_snr_range[0], self.noise_snr_range[1])
            nr = noise_transform(
                result,
                snr_db=snr,
                clip_id=clip_id,
                global_seed=self.global_seed,
                epoch=self.epoch,
            )
            result = nr.waveform

        return result
