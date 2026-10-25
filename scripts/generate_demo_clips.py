"""Generate demo clips for the live presentation (slide 10).

Produces WAV files in an output directory:

  1. synthetic_tts.wav   — a 4-second TTS clip via pyttsx3/espeak (if available)
                           or a synthesised chirp+formant signal as fallback
  2. too_short.wav       — a 0.5-second clip (should be rejected: min 1 s)
  3. invalid.mp3         — an MP3 file (should be rejected: only WAV/FLAC accepted)
  4. digital_silence.wav — all zeros (should be rejected: silence check)

The human speech clip must be recorded by the presenter. Suggested
method (on the host with a USB microphone):

    arecord -f cd -d 5 -t wav demo_clips/human_speech.wav

Or record on a phone and transfer as WAV (16-bit PCM, 16+ kHz).

Usage:
    uv run python scripts/generate_demo_clips.py --output-dir demo_clips

The output directory is git-ignored (*.wav, *.mp3).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

SEED = 20261025
SR = 16_000


def _synthesize_speech_like(seconds: float, seed: int) -> np.ndarray:
    """Synthesize a speech-like signal with formant structure.

    Not real speech — a swept formant signal with amplitude modulation
    that looks and sounds vaguely speech-like on a spectrogram. Good
    enough to test the pipeline; the model will likely flag it.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n, dtype=np.float32) / SR

    # Fundamental frequency with slight vibrato
    f0 = 120 + 10 * np.sin(2 * np.pi * 5 * t)
    phase = np.cumsum(2 * np.pi * f0 / SR)
    glottal = 0.5 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.12 * np.sin(3 * phase)

    # Three formants (vowel-like, swept)
    f1 = 500 + 200 * np.sin(2 * np.pi * 0.8 * t)
    f2 = 1500 + 400 * np.sin(2 * np.pi * 0.5 * t + 1.0)
    f3 = 2500 + 300 * np.sin(2 * np.pi * 0.3 * t + 2.0)

    formants = np.zeros(n, dtype=np.float32)
    for f in [f1, f2, f3]:
        formants += np.sin(np.cumsum(2 * np.pi * f / SR)).astype(np.float32)

    # Amplitude envelope: speech-like bursts with pauses
    envelope = np.ones(n, dtype=np.float32)
    segment_len = SR // 4  # 250 ms segments
    for i in range(0, n, segment_len):
        if rng.random() < 0.25:  # 25% chance of pause
            end = min(i + segment_len, n)
            envelope[i:end] = 0.02

    # Smooth the envelope
    from scipy.signal import convolve

    kernel = np.ones(800, dtype=np.float32) / 800
    envelope = convolve(envelope, kernel, mode="same")

    signal = (glottal * 0.6 + formants * 0.15) * envelope
    signal += 0.01 * rng.standard_normal(n).astype(np.float32)

    # Normalize to -6 dBFS
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal * (0.5 / peak)

    return signal.astype(np.float32)


def _try_tts(text: str, seconds: float) -> np.ndarray | None:
    """Try to generate speech via pyttsx3/espeak. Returns None on failure."""
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.save_to_file(text, "/tmp/_demo_tts_tmp.wav")
        engine.runAndWait()
        audio, sr = sf.read("/tmp/_demo_tts_tmp.wav", dtype="float32")
        Path("/tmp/_demo_tts_tmp.wav").unlink(missing_ok=True)
        if sr != SR:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        # Trim or pad to desired length
        target_n = int(seconds * SR)
        if len(audio) > target_n:
            audio = audio[:target_n]
        elif len(audio) < target_n:
            audio = np.pad(audio, (0, target_n - len(audio)))
        return audio.astype(np.float32)
    except Exception:
        return None


def _write_wav(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    sf.write(str(path), audio, sr, format="WAV", subtype="PCM_16")


def _write_minimal_mp3(path: Path) -> None:
    """Write a minimal valid-ish MP3 file (MPEG1 Layer3 frame header + silence).

    Not a real MP3 encoder — just enough bytes that libsndfile rejects it
    as non-WAV/non-FLAC.
    """
    # MPEG1, Layer 3, 128kbps, 44100Hz, stereo
    frame_header = b"\xff\xfb\x90\x00"
    # Pad with zeros for one frame (~417 bytes for 128kbps @ 44100)
    frame_data = b"\x00" * 413
    path.write_bytes(frame_header + frame_data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    # 1. Synthetic speech-like clip (4 s)
    tts_text = "This is a test of the voice deepfake detection system."
    tts_audio = _try_tts(tts_text, 4.0)
    if tts_audio is not None:
        _write_wav(out / "synthetic_tts.wav", tts_audio)
        print(f"synthetic_tts.wav: TTS-generated, {len(tts_audio)/SR:.1f}s")
    else:
        synth = _synthesize_speech_like(4.0, SEED)
        _write_wav(out / "synthetic_tts.wav", synth)
        print(
            f"synthetic_tts.wav: formant-synthesized fallback (no TTS engine), "
            f"{len(synth)/SR:.1f}s"
        )

    # 2. Too-short clip (0.5 s) — should trigger duration rejection
    short = 0.3 * np.sin(
        2 * np.pi * 440 * np.arange(int(0.5 * SR), dtype=np.float32) / SR
    )
    short += 0.01 * rng.standard_normal(len(short)).astype(np.float32)
    _write_wav(out / "too_short.wav", short.astype(np.float32))
    print(f"too_short.wav: {len(short)/SR:.1f}s (should be rejected: <1s)")

    # 3. Invalid format (MP3-like) — should trigger format rejection
    _write_minimal_mp3(out / "invalid.mp3")
    print("invalid.mp3: fake MP3 header (should be rejected: wrong format)")

    # 4. Digital silence — should trigger silence rejection
    silence = np.zeros(int(3.0 * SR), dtype=np.float32)
    _write_wav(out / "digital_silence.wav", silence)
    print(f"digital_silence.wav: {len(silence)/SR:.1f}s all zeros (should be rejected)")

    print(f"\nAll clips written to {out}/")
    print(
        "\nTo complete the demo set, record your own voice:\n"
        "  arecord -f cd -d 5 -t wav demo_clips/human_speech.wav\n"
        "  OR record on your phone and convert:\n"
        "  ffmpeg -i recording.m4a -ar 16000 -ac 1 demo_clips/human_speech.wav"
    )


if __name__ == "__main__":
    main()
