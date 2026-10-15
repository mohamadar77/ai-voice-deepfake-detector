"""CNN training loop, checkpoint selection, and run record (#33).

Trains the CompactMelCNN on log-Mel spectrograms, selects the best epoch
checkpoint on dev_select by clip-level EER, picks the operating threshold on
dev_threshold at target FPR 0.05, and produces a versioned run record.

Nothing here touches the evaluation partition.
"""

from __future__ import annotations

import gc
import json
import os
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from voice_detector.cnn_dataset import (
    LABEL_MAP,
    EvalWindowDataset,
    MelDataset,
    MelSpec,
    compute_norm_stats,
)
from voice_detector.cnn_pilot import CompactMelCNN
from voice_detector.manifest import git_commit, package_version
from voice_detector.metrics import (
    METRICS_VERSION,
    evaluate_clip_scores,
    threshold_at_target_fpr,
)
from voice_detector.reservation import (
    ROLE_SELECT,
    ROLE_THRESHOLD,
    validate_frozen_record,
)
from voice_detector.scoring import SCORE_DIRECTION

CNN_STUDY_VERSION = "cnn-study-v1"


def _class_weights(rows: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    """Compute inverse-frequency class weights for balanced training."""

    counts = np.zeros(2, dtype=np.int64)
    for row in rows:
        counts[LABEL_MAP[str(row["label"])]] += 1
    total = counts.sum()
    weights = total / (2.0 * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32)


def _aggregate_window_scores(
    clip_ids: list[str],
    scores: list[float],
) -> tuple[list[str], list[float]]:
    """Mean of per-window scores, one score per clip."""

    from collections import defaultdict
    clip_scores: dict[str, list[float]] = defaultdict(list)
    for cid, s in zip(clip_ids, scores, strict=True):
        clip_scores[cid].append(s)
    unique_ids = list(dict.fromkeys(clip_ids))
    return unique_ids, [float(np.mean(clip_scores[cid])) for cid in unique_ids]


@torch.no_grad()
def _score_eval_dataset(
    model: nn.Module,
    dataset: EvalWindowDataset,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[list[str], list[str], np.ndarray]:
    """Score an eval dataset, aggregate windows to clips.

    Returns (clip_ids, labels, clip_scores). Scores follow SCORE_DIRECTION:
    the spoof logit (index 1) minus the bonafide logit (index 0), so higher
    means more synthetic.
    """

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_clip_ids: list[str] = []
    all_labels: list[int] = []
    all_scores: list[float] = []

    for batch in loader:
        logits = model(batch["spectrogram"])
        scores = (logits[:, 1] - logits[:, 0]).cpu().numpy()
        all_clip_ids.extend(batch["clip_id"])
        all_labels.extend(batch["label"].numpy().tolist())
        all_scores.extend(scores.tolist())

    agg_ids, agg_scores = _aggregate_window_scores(all_clip_ids, all_scores)

    label_map_inv = {v: k for k, v in LABEL_MAP.items()}
    id_to_label: dict[str, str] = {}
    for cid, lbl in zip(all_clip_ids, all_labels, strict=True):
        id_to_label[cid] = label_map_inv[lbl]
    agg_labels = [id_to_label[cid] for cid in agg_ids]

    return agg_ids, agg_labels, np.array(agg_scores, dtype=np.float64)


def train_cnn(
    train_rows: Sequence[Mapping[str, Any]],
    dev_rows: Sequence[Mapping[str, Any]],
    frozen_record: Mapping[str, Any],
    study_config: Mapping[str, Any],
    *,
    seed: int,
    run_label: str = "clean",
    robustness_config: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Train the CNN, select best checkpoint, threshold, produce run record.

    When ``robustness_config`` is provided, training augmentation from the
    frozen robustness spec is applied before random crop each epoch.

    Returns (run_record, best_state_dict, (mean, std)).
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    mel_spec = MelSpec.from_mapping(study_config)
    training_cfg = study_config.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 32))
    max_epochs = int(training_cfg.get("max_epochs", 20))
    patience = int(training_cfg.get("patience", 5))
    lr = float(training_cfg.get("learning_rate", 0.001))
    weight_decay = float(training_cfg.get("weight_decay", 0.0001))
    num_workers = int(training_cfg.get("num_workers", 0))
    use_class_weight = str(training_cfg.get("class_weight", "balanced")) == "balanced"

    assignment = validate_frozen_record(frozen_record)
    select_speakers = {s for s, r in assignment.items() if r == ROLE_SELECT}
    threshold_speakers = {s for s, r in assignment.items() if r == ROLE_THRESHOLD}

    dev_speaker_map = {
        str(row["clip_id"]): str(row["speaker_id"])
        for row in dev_rows
    }
    select_rows = [
        r for r in dev_rows
        if dev_speaker_map.get(str(r["clip_id"]), "") in select_speakers
    ]
    threshold_rows = [
        r for r in dev_rows
        if dev_speaker_map.get(str(r["clip_id"]), "") in threshold_speakers
    ]

    augment_fn = None
    if robustness_config is not None:
        from voice_detector.augmentation import TrainingAugmentor

        aug_cfg = robustness_config.get("training_augmentation", {})
        transforms = aug_cfg.get("transforms", {})
        noise_cfg = transforms.get("noise", {})
        snr_range = noise_cfg.get("snr_db_range", [10, 30])
        codec_cfg = transforms.get("codec", {})

        augment_fn = TrainingAugmentor(
            global_seed=int(robustness_config.get("global_seed", seed)),
            probability=float(aug_cfg.get("probability", 0.5)),
            codec_probability=float(codec_cfg.get("probability", 0.5)),
            noise_probability=float(noise_cfg.get("probability", 0.5)),
            noise_snr_range=(float(snr_range[0]), float(snr_range[1])),
            codec_bitrate_kbps=int(codec_cfg.get("bitrate_kbps", 24)),
            sample_rate=mel_spec.sample_rate_hz,
        )
        print(
            f"Augmentation enabled: p={augment_fn.probability}, "
            f"codec_p={augment_fn.codec_probability}, "
            f"noise_p={augment_fn.noise_probability}, "
            f"snr_range={augment_fn.noise_snr_range}"
        )

    print(f"Computing normalization statistics on {min(2000, len(train_rows))} training clips...")
    mean, std = compute_norm_stats(train_rows, mel_spec, seed=seed)

    train_dataset = MelDataset(
        train_rows, mel_spec, training=True, seed=seed, mean=mean, std=std,
        augment_fn=augment_fn,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )

    select_eval = EvalWindowDataset(select_rows, mel_spec, mean=mean, std=std)
    threshold_eval = EvalWindowDataset(threshold_rows, mel_spec, mean=mean, std=std)

    model = CompactMelCNN(n_mels=mel_spec.n_mels, n_frames=mel_spec.n_frames)
    n_params = sum(p.numel() for p in model.parameters())

    weight_tensor = _class_weights(train_rows) if use_class_weight else None
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_eer = float("inf")
    best_epoch = -1
    best_state: dict[str, Any] = {}
    epochs_without_improvement = 0
    epoch_records: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(max_epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        epoch_start = time.perf_counter()

        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch["spectrogram"])
            labels = batch["label"]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        select_ids, select_labels, select_scores = _score_eval_dataset(
            model, select_eval, batch_size=batch_size * 2, num_workers=0,
        )
        select_metrics = evaluate_clip_scores(
            select_ids, select_labels, select_scores,
            threshold=None, score_direction=SCORE_DIRECTION,
        )
        eer = select_metrics.eer if select_metrics.eer is not None else float("inf")
        epoch_time = time.perf_counter() - epoch_start

        epoch_rec = {
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 6),
            "dev_select_eer": round(eer, 6) if eer != float("inf") else None,
            "dev_select_roc_auc": select_metrics.roc_auc,
            "epoch_seconds": round(epoch_time, 2),
        }
        epoch_records.append(epoch_rec)
        print(
            f"Epoch {epoch + 1}/{max_epochs} | "
            f"loss={avg_loss:.4f} | "
            f"dev_select EER={eer:.4f} | "
            f"{epoch_time:.1f}s"
        )

        if eer < best_eer:
            best_eer = eer
            best_epoch = epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1} (patience={patience})")
            break

    total_time = time.perf_counter() - started

    model.load_state_dict(best_state)

    select_ids, select_labels, select_scores = _score_eval_dataset(
        model, select_eval, batch_size=batch_size * 2,
    )
    threshold_ids, threshold_labels, threshold_scores = _score_eval_dataset(
        model, threshold_eval, batch_size=batch_size * 2,
    )

    threshold_result = threshold_at_target_fpr(
        threshold_ids, threshold_labels, threshold_scores,
        target_fpr=0.05, score_direction=SCORE_DIRECTION,
    )

    dev_all_rows = list(select_rows) + list(threshold_rows)
    dev_all_eval = EvalWindowDataset(dev_all_rows, mel_spec, mean=mean, std=std)
    dev_all_ids, dev_all_labels, dev_all_scores = _score_eval_dataset(
        model, dev_all_eval, batch_size=batch_size * 2,
    )

    dev_full_metrics = evaluate_clip_scores(
        dev_all_ids, dev_all_labels, dev_all_scores,
        threshold=threshold_result.threshold, score_direction=SCORE_DIRECTION,
    )
    select_at_threshold = evaluate_clip_scores(
        select_ids, select_labels, select_scores,
        threshold=threshold_result.threshold, score_direction=SCORE_DIRECTION,
    )
    threshold_at_thresh = evaluate_clip_scores(
        threshold_ids, threshold_labels, threshold_scores,
        threshold=threshold_result.threshold, score_direction=SCORE_DIRECTION,
    )

    del select_eval, threshold_eval, dev_all_eval, train_dataset, train_loader
    gc.collect()

    run_record = {
        "schema_version": CNN_STUDY_VERSION,
        "run_label": run_label,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "package_version": package_version(),
        "seed": seed,
        "mel_config": {
            "mel_version": study_config.get("mel", {}).get("mel_version", "mel-v1"),
            "sample_rate_hz": mel_spec.sample_rate_hz,
            "n_fft": mel_spec.n_fft,
            "hop_length": mel_spec.hop_length,
            "n_mels": mel_spec.n_mels,
            "fmin": mel_spec.fmin,
            "fmax": mel_spec.fmax,
            "window_seconds": mel_spec.window_seconds,
        },
        "model": {
            "class": "CompactMelCNN",
            "n_params": n_params,
        },
        "training_config": {
            "loss": "cross_entropy",
            "class_weight": "balanced" if use_class_weight else "none",
            "optimizer": "adam",
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "patience": patience,
        },
        "augmentation": {
            "enabled": augment_fn is not None,
            "probability": augment_fn.probability if augment_fn else None,
            "codec_probability": augment_fn.codec_probability if augment_fn else None,
            "noise_probability": augment_fn.noise_probability if augment_fn else None,
            "noise_snr_range": list(augment_fn.noise_snr_range) if augment_fn else None,
            "codec_bitrate_kbps": augment_fn.codec_bitrate_kbps if augment_fn else None,
            "epoch_aware_seed": True if augment_fn else None,
            "robustness_version": robustness_config.get("version") if robustness_config else None,
        },
        "input_policy": study_config.get("input_policy", {}),
        "normalization": {
            "method": "per_channel_standardize",
            "fit_on": "train",
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "train_n_clips": len(train_rows),
        "dev_n_clips": len(dev_all_rows),
        "best_epoch": best_epoch,
        "total_epochs": len(epoch_records),
        "epochs": epoch_records,
        "metrics_version": METRICS_VERSION,
        "score_direction": SCORE_DIRECTION,
        "selected_checkpoint": f"epoch_{best_epoch}",
        "threshold": {
            "value": threshold_result.threshold,
            "boundary_score": threshold_result.boundary_score,
            "decision_rule": threshold_result.decision_rule,
            "target_fpr": threshold_result.target_fpr,
            "achieved_fpr": threshold_result.achieved_fpr,
            "n_bonafide": threshold_result.n_bonafide,
            "n_false_alarms": threshold_result.n_false_alarms,
        },
        "dev_full_metrics": dev_full_metrics.as_dict(),
        "dev_select_metrics": select_at_threshold.as_dict(),
        "dev_threshold_metrics": threshold_at_thresh.as_dict(),
        "total_seconds": round(total_time, 3),
    }

    return run_record, best_state, (mean, std)


def train_and_save_cnn(
    train_rows: Sequence[Mapping[str, Any]],
    dev_rows: Sequence[Mapping[str, Any]],
    frozen_record: Mapping[str, Any],
    study_config: Mapping[str, Any],
    *,
    seed: int,
    run_label: str = "clean",
    out_dir: Path | str,
    robustness_config: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Train, select, threshold, save — returns (run_record, output_paths)."""

    run_record, best_state, norm_stats = train_cnn(
        train_rows, dev_rows, frozen_record, study_config,
        seed=seed, run_label=run_label,
        robustness_config=robustness_config,
    )

    mel_spec = MelSpec.from_mapping(study_config)
    model = CompactMelCNN(n_mels=mel_spec.n_mels, n_frames=mel_spec.n_frames)
    model.load_state_dict(best_state)

    outputs = save_cnn_bundle(model, run_record, norm_stats, out_dir)
    return run_record, outputs


def save_cnn_bundle(
    model: nn.Module,
    run_record: Mapping[str, Any],
    norm_stats: tuple[torch.Tensor, torch.Tensor],
    out_dir: Path | str,
) -> dict[str, Path]:
    """Save the CNN checkpoint, run record, and normalization stats."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "cnn_checkpoint.pt"
    record_path = out_dir / "cnn_run_record.json"
    norm_path = out_dir / "cnn_norm_stats.pt"

    temp = model_path.with_name(f"{model_path.name}.tmp")
    torch.save(model.state_dict(), temp)
    os.replace(temp, model_path)

    temp = record_path.with_name(f"{record_path.name}.tmp")
    temp.write_text(json.dumps(dict(run_record), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, record_path)

    mean, std = norm_stats
    temp = norm_path.with_name(f"{norm_path.name}.tmp")
    torch.save({"mean": mean, "std": std}, temp)
    os.replace(temp, norm_path)

    return {"model": model_path, "record": record_path, "norm": norm_path}
