# DL120 execution tracker: CNN eval-score (PR 76) and the 5-candidate freeze

Working document. Not part of the evaluation record. Last updated: 2026-10-17.

Status legend: ⬜ todo · 🔄 running · ✅ passed · ❌ failed · ⛔ blocked · ➖ not applicable

Hosts: **Mac** = this checkout. **DL120** = `~/ai-voice-deepfake-detector-feat-cnn-eval-scoring` (branch worktree; its
`./artifacts/` holds the manifest, bundles and old stores, so `ART=$PWD/artifacts`) and `/root/ai-voice-deepfake-detector`
(main checkout; check whether its `artifacts/` is a separate mirror before the real run).

Rules for every step: development partition only until step R6. Rehearsal outputs live under `$R`. Stop at the first ❌
and send the output; do not work around a failure. The Command column assumes the setup blocks below were run in the
same shell.

## Setup blocks (run once per shell on DL120)

**Section B (pre-merge rehearsal, branch worktree)**
```bash
cd /root/ai-voice-deepfake-detector   # git checkout on branch feat/cnn-eval-scoring (real git_commit)
export ART=$PWD/artifacts VOICE_DETECTOR_ARTIFACTS_DIR=$PWD/artifacts
export VOICE_DETECTOR_LA_ROOT=<path to LA root>
export R=$ART/evaluation/rehearsal
B="--bundle $ART/lr_baseline --bundle $ART/cnn_clean_seed1 --bundle $ART/cnn_clean_seed2 --bundle $ART/cnn_augmented_seed1 --bundle $ART/cnn_augmented_seed2"
```

**Section D (real run, main checkout, after the merge)**
```bash
cd /root/ai-voice-deepfake-detector
export ART=$PWD/artifacts VOICE_DETECTOR_LA_ROOT=<path to LA root>
B="--bundle $ART/lr_baseline --bundle $ART/cnn_clean_seed1 --bundle $ART/cnn_clean_seed2 --bundle $ART/cnn_augmented_seed1 --bundle $ART/cnn_augmented_seed2"
F=reports/evaluation_freeze_g2.json
```

**Helper scripts for V6 and V7** (created by step P3; contents below)

## Tracker

### A. Already done

| ID | Step | Pass criteria | Status | Evidence |
| --- | --- | --- | --- | --- |
| T1 | ruff + full pytest on DL120 (branch) | all pass, `test_cnn_scoring.py` included | ✅ | 467 passed, `~/tmp/pytest.log` (17 Oct) |
| T2 | CI on PR 76 | all checks green | ✅ | 2/2 passing, mergeable |
| T3 | Code review of the diff | findings fixed or filed | ✅ | commits 7c2b5be, 6213759; issues #68–#75 |

### B. Pre-merge verification (rehearsal, branch worktree)

| ID | Step | Command | Pass criteria | Est. time | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | Work from the git checkout on the PR branch, create `$R` | `git -C /root/ai-voice-deepfake-detector log --oneline -1 && mkdir -p $R` | HEAD is 6213759 or later; `git status` clean | 1 min | ✅ | 17 Oct: `/root/ai-voice-deepfake-detector` on `feat/cnn-eval-scoring`, HEAD 6213759 == origin. |
| P1 | Five bundles present at `$ART/<id>` | `ls $ART/{lr_baseline,cnn_clean_seed1,cnn_clean_seed2,cnn_augmented_seed1,cnn_augmented_seed2}` | all five listed with model files, no stray files (bundle hash covers every file) | 2 min | ✅ | 17 Oct: five bundles at `$ART/<id>` in the main checkout; 3 files per CNN bundle, 2 for LR. |
| P1b | Copy the four CNN bundles to `$ART/<id>` (after P1a confirms seeds) | see "P1b commands" below | each copy `diff -r` clean; exactly 3 files per CNN bundle | 5 min | ✅ | 17 Oct: originals match `/root/artifacts`; four `cp -a` copies re-made in main checkout, all identical to their sources. |
| P2 | CNN run records declare the aggregation rule | `for d in $ART/cnn_*; do jq -r '.input_policy.eval_aggregation' $d/cnn_run_record.json; done` | four lines read `mean_window_logit` | 1 min | ✅ | 17 Oct: all four `mean_window_logit` (seen in P1a output). |
| P3 | Create helper scripts | `cat > $R/check_cnn.py` and `cat > $R/check_lr.py` (heredocs below) | files exist | 1 min | ✅ | scripts were lost with the deleted folder; recreate in the new `$R` before V6 |
| V1 | Freeze dry run | `uv run voice-detector eval-freeze $B --out $R/freeze.json --dry-run` | no refusal; 5 candidates; families and rules right; 5 conditions ending `silence_trim` | 1 min | ✅ | 17 Oct: declarations correct (LR thr 1.297; CNN thr 0.650, 2.638, -1.252, -1.198; seeds right; 5 conditions); `git_commit` 62137598bf93b4f2796999e537fa5ca67cbfa22f. Note: `uv run` created a fresh `.venv` in the main checkout; run `uv sync --frozen --all-groups` before V3 (torch). |
| V2 | Rehearsal freeze | `uv run voice-detector eval-freeze $B --out $R/freeze.json` | state FROZEN | 1 min | ✅ | 17 Oct: FROZEN, 5 candidates, 5 conditions, 27 slices, manifest_hash b932b859… (the role-bearing hash; b172e50c… is the role-free one by design, docs/data.md). Env check: torch 2.14.1+cu130, torchaudio 2.11.0, librosa 0.11.0, ffmpeg 8.1.3 with libopus. |
| V3 | Smoke on 500 dev clips | `uv run voice-detector eval-score --freeze-record $R/freeze.json --store $R/smoke.db --partition development --max-clips 500 --torch-threads 8` | exit 0; `complete: true`; `rows_written` 12500; sane `condition_seconds`; sidecar written | 10–20 min | ✅ | 17 Oct: complete, 12500 rows, predictions_hash 248e839c…; seconds per 500 clips x 5 candidates: clean 90.8, codec 383.3, noise_20 68.5, noise_10 68.3, silence_trim 63.8 (total 675 s, 1.35 s/clip). Noise clipping 543 / 1263 samples; codec shorter 0; trim kept_original 0. |
| V4 | Negative: rerun without `--resume` | `uv run voice-detector eval-score --freeze-record $R/freeze.json --store $R/smoke.db --partition development --max-clips 500` | refused, store non-empty | 1 min | ✅ | 17 Oct: refused with StoreStateError ("already holds 25 scored pairs; use a fresh --store or pass --resume"). |
| V5 | Negative: evaluation without dev store | `uv run voice-detector eval-score --freeze-record $R/freeze.json --store $R/never_created.db --partition evaluation` | `dev_store_required`; no store created; no evaluation clip scored | 1 min | ✅ | 17 Oct: `dev_store_required`; `never_created.db` does not exist. |
| V6 | CNN parity vs training scorer (4 bundles) | `for b in cnn_clean_seed1 cnn_clean_seed2 cnn_augmented_seed1 cnn_augmented_seed2; do uv run python $R/check_cnn.py $R/smoke.db $ART/$b; done` | `max_abs_diff` < 1e-4 each (expect ~1e-6) | 10 min | ✅ | 17 Oct: 500 clips each, max_abs_diff 3.8e-6 (clean s1), 7.6e-6 (clean s2), 1.9e-6 (aug s1), 1.9e-6 (aug s2). Real weights reproduce the training scorer to float32 noise. |
| V7 | LR regression vs old rehearsal store | `uv run python $R/check_lr.py $R/smoke.db $ART/evaluation/predictions.db lr_baseline` | max diff ≈ 0 | 2 min | ✅ | 18 Oct (host 2): 500 common clips with the PR #65 rehearsal store, max_abs_diff 9.5e-7. LR path unchanged. |
| V8 | Full dev run with parity gate (detached) | `nohup uv run voice-detector eval-score --freeze-record $R/freeze.json --store $R/dev.db --partition development --torch-threads 8 > $R/dev_score.log 2>&1 &` | exit 0; `dev_parity` all within tolerance | hours | ⬜ | after an interruption rerun the same command plus `--resume`; follow with `tail -f $R/dev_score.log` |
| V9 | Dev rehearsal report | `uv run voice-detector eval-report --freeze-record $R/freeze.json --store $R/dev.db --out-dir $R/report --bootstrap && grep '"state"' $R/freeze.json` | `rehearsal_report`; record still FROZEN; 25 results | 10 min | ⬜ | |

**Merge bar:** minimum = P0–V7 ✅. Full confidence = also V8–V9 ✅.

### C. Merge decision

| ID | Step | Command | Status | Notes |
| --- | --- | --- | --- | --- |
| M1 | Merge PR 76 into main | `gh pr merge 76 --squash` | ✅ | 18 Oct: squash-merged as c4445f5. |

### D. Real run (after M1; fresh paths, nothing from section B or the overnight run is reused)

Set once per shell: `cd /root/ai-voice-deepfake-detector; export ART=$PWD/artifacts VOICE_DETECTOR_ARTIFACTS_DIR=$PWD/artifacts VOICE_DETECTOR_LA_ROOT=/root/data/raw/LA; F=reports/evaluation_freeze_g2.json; D=$ART/evaluation/predictions_dev.db; E=$ART/evaluation/predictions_eval.db` and `B="--bundle $ART/lr_baseline --bundle $ART/cnn_clean_seed1 --bundle $ART/cnn_clean_seed2 --bundle $ART/cnn_augmented_seed1 --bundle $ART/cnn_augmented_seed2"`.

| ID | Step | Command | Pass criteria | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- |
| R1 | Pull main, sync | `git fetch && git checkout main && git pull --ff-only && uv sync --frozen --all-groups && git log --oneline -1` | HEAD c4445f5 | ✅ | 18 Oct: fast-forward 6c10342..c4445f5 on main, 110 packages checked. |
| R2 | Real freeze dry run | `uv run voice-detector eval-freeze $B --out $F --dry-run \| jq -c '.git_commit, (.candidates[] \| {candidate_id, threshold})'` | git_commit c4445f5…; thresholds as before | ✅ | 18 Oct: git_commit c4445f5083d8…; thresholds LR 1.2973, CNN 0.6496 / 2.6376 / -1.2520 / -1.1977 (unchanged). |
| R3 | **Real freeze (one-way)** | `uv run voice-detector eval-freeze $B --out $F` | FROZEN | ✅ | 18 Oct 08:59Z: FROZEN at `reports/evaluation_freeze_g2.json` on c4445f5; 5 candidates, 5 conditions, 27 slices, manifest b932b859…. Not yet committed to git. |
| R4 | Dev scoring, 12 workers, parity gate (detached) | `nohup uv run voice-detector eval-score --freeze-record $F --store $D --partition development --workers 12 > $ART/evaluation/dev_score.log 2>&1 &` | complete; parity passed; about 40 min | ✅ | 18 Oct: 24,844 clips, 621,100 rows, complete, 37.9 min (clean 192.6, codec 1534.7, noise_20 186.1, noise_10 186.5, trim 152.4); parity passed for all five (n_clips 24844; roc_auc delta <= 1.8e-8; eer delta 0; confusion delta 0 or one boundary clip). predictions_hash 8845093e… equals the overnight rehearsal run (different commit and freeze record). |
| R5 | Dev rehearsal report, 25 workers | `uv run voice-detector eval-report --freeze-record $F --store $D --out-dir $ART/evaluation/report_dev --bootstrap --workers 25` | `rehearsal_report`; record still FROZEN | ✅ | 18 Oct: 25/25 pairs in 24 min with 25 workers; `rehearsal_report`, record still FROZEN; report_hash da8f0fd0…. |
| R6 | **Locked evaluation scoring (owner go)** | `nohup uv run voice-detector eval-score --freeze-record $F --store $E --partition evaluation --dev-store $D --workers 12 > $ART/evaluation/eval_score.log 2>&1 &` | complete; no metrics printed; about 2 h | ✅ | 18 Oct: finished 14:48:17Z on c4445f5, `--workers 30` (planned 12), torch_threads 1. Sidecar: complete true, error null, 4,926 s (clean 630, codec 2,567, noise_10 596, noise_20 609, trim 525); diagnostics: 0 codec outputs shorter, 0 trim fallbacks, noise clipped 147,520 (10 dB) / 61,698 (20 dB); predictions_hash 1ef12a29…. Deviation: `eval_score.log` ends at clean 50,000/71,237, with no final JSON (cause unknown); completeness rests on the sidecar (complete = no missing pairs) and R7 re-checks it. |
| R7 | **Locked report (owner go)** | `uv run voice-detector eval-report --freeze-record $F --store $E --dev-store $D --out-dir $ART/evaluation/report_locked --bootstrap --workers 25` | REPORTED | ✅ | 18 Oct 18:21:41Z-20:37:11Z (2 h 16 min), `--workers 30`: rc 0, 25/25 pairs, record REPORTED (reported_utc 2026-10-18T20:37:09Z), report_hash 2bc69920…, predictions_hash 1ef12a29… unchanged, stores byte-identical to the pre-R7 checksums. No numbers read yet. |

`/root/real_dev.sh` runs R2-R5 unattended (stops at the first failure); R6 and R7 stay manual.

Note: R6 was actually started with `--workers 20` (the table shows the planned 12). Scores do not depend on the worker count; the effective torch/BLAS pins are the same. A `--resume` may use either count.

### R7 runbook (written before any locked result is seen)

Superseded by `docs/evaluation-protocol.md` (primary endpoint, duration option A, interval coverage, failure table); that document is the pre-registration and must be committed before R7. This section is the short checklist.

Preconditions, all on DL120 on c4445f5 (do not pull): R6 log ends with `"status": "scored"`, `"complete": true`, `"n_missing": 0`, `"rows_written": 1780925`, and shows no metrics; the record is SCORED or still FROZEN as the CLI reports; `$D` metadata still has `dev_parity=passed`. R7 needs a separate explicit owner go.

Run R7 once. If it fails before writing the report, send the output and do not re-run with changed arguments. The report must not be regenerated with different settings after the numbers are seen.

Reading order for the locked report (fixed in advance):
1. Integrity: report status `reported`, record REPORTED, freeze identity, store hashes, per-pair n_clips (71,237 for clean), class counts.
2. Clean condition, per candidate, at the frozen thresholds: EER, ROC AUC, macro F1, balanced accuracy, human false-positive rate, synthetic detection rate, confusion matrix, with speaker-grouped bootstrap intervals.
3. Dev versus evaluation gap per candidate, reported as is. A large gap is a finding, not a defect to fix.
4. Seen versus unfamiliar attacks, reported separately (AGENTS.md); sample counts for every slice.
5. Robustness conditions (codec, noise_20db, noise_10db, silence_trim) against clean, same clips, both classes.
6. Clean versus augmented CNN comparison, then the CNN versus LR comparison. Interpret with the intervals, not the point estimates.

Rules after the report: no retuning, no threshold change, no checkpoint swap, no re-run to get different numbers. Any deviation or negative result is recorded in the Log below. Describe results as a project evaluation on ASVspoof 2019 LA, never as an official leaderboard result, and as scores, not calibrated probabilities (see #79 for threshold brittleness).

After R7, commit only small summaries and the REPORTED freeze record (as separate snapshots, with a sha256 copy-and-compare as for the FROZEN record). Never commit the databases, predictions, or audio.

## P1a / P1b commands (bundles are not yet at `$ART/<candidate_id>`)

P1a: confirm which directory is which (seed, label, augmentation) before copying:
```bash
for d in $ART/manifest/cnn_study/clean $ART/manifest/cnn_study/clean-seed2 $ART/manifest/train_cnn/augmented_seed1 $ART/manifest/train_cnn/augmented_seed2; do echo $d; jq -c '{run_label, seed, aug: .augmentation.enabled, eval_agg: .input_policy.eval_aggregation}' $d/cnn_run_record.json; done
find $ART /root/artifacts -name lr_run_record.json
```
Expected: seed 20261004 for `clean` and `augmented_seed1`, 20261005 for `clean-seed2` and `augmented_seed2`; `aug` true only for the augmented two; `eval_agg` is `mean_window_logit` everywhere.

P1b: copy (the originals stay untouched), then verify:
```bash
cp -a $ART/manifest/cnn_study/clean $ART/cnn_clean_seed1
cp -a $ART/manifest/cnn_study/clean-seed2 $ART/cnn_clean_seed2
cp -a $ART/manifest/train_cnn/augmented_seed1 $ART/cnn_augmented_seed1
cp -a $ART/manifest/train_cnn/augmented_seed2 $ART/cnn_augmented_seed2
diff -r $ART/manifest/cnn_study/clean $ART/cnn_clean_seed1 && diff -r $ART/manifest/cnn_study/clean-seed2 $ART/cnn_clean_seed2 && diff -r $ART/manifest/train_cnn/augmented_seed1 $ART/cnn_augmented_seed1 && diff -r $ART/manifest/train_cnn/augmented_seed2 $ART/cnn_augmented_seed2 && echo all-identical
ls -A $ART/lr_baseline $ART/cnn_clean_seed1 $ART/cnn_clean_seed2 $ART/cnn_augmented_seed1 $ART/cnn_augmented_seed2
```
If the LR bundle is not at `$ART/lr_baseline`, copy it there the same way (it must hold `lr_run_record.json` and `lr_pipeline.joblib`).

## Host 2 repeat (new, stronger runtime)

Section B was passed on DL120 (host 1) through V6; V7 and the condition sanity check were still open. The same steps are now
repeated on a new host. Host 1 results stay above as evidence. Commands for P1-V7 are the same as in section B; the setup
block applies after N1.

| ID | Step | Command | Pass criteria | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- |
| N0 | Record host facts | `hostname; nproc; free -g \| head -2; lscpu \| grep 'Model name'; nvidia-smi -L; df -h /root \| tail -1; uv --version` | cores, RAM, disk recorded | ✅ | 18 Oct: hp-xl420gen9-01, 48 cores (Xeon E5-2650L v3 @ 1.8 GHz), 31 GB RAM, no GPU, 541 GB free. Similar class to host 1, not much faster. |
| N1 | Clone PR branch, sync env | `git clone <URL> /root/ai-voice-deepfake-detector && cd /root/ai-voice-deepfake-detector && git checkout feat/cnn-eval-scoring && uv sync --frozen --all-groups` | HEAD 6213759; torch/torchaudio/librosa import | ✅ | 18 Oct: HEAD 6213759; 110 packages checked; torch 2.14.1+cu130, torchaudio 2.11.0, librosa 0.11.0 (locked versions). |
| N2 | ffmpeg with libopus | `ffmpeg -version \| head -1 && ffmpeg -hide_banner -encoders \| grep -c libopus` | version line; count >= 1 | ✅ | 18 Oct: ffmpeg 8.1.3 (same as host 1), libopus count 1. |
| N3 | LA data present | `find $VOICE_DETECTOR_LA_ROOT -name '*.flac' \| wc -l` | 121461 | ✅ | 18 Oct: all 121,461 manifest files exist and match their manifest sha256 (parallel full pass, 0 bad). The data dir holds 122,299 `*.flac`, 838 more than the manifest; the extras are not read by scoring and are not yet identified (optional check). |
| N4 | Copy manifest + 5 bundles (+ old LR store for V7) | `rsync -a <SRC>/artifacts/{manifest,lr_baseline,cnn_clean_seed1,cnn_clean_seed2,cnn_augmented_seed1,cnn_augmented_seed2} $PWD/artifacts/` | files present | ✅ | 18 Oct: manifest, features, lr_baseline, four CNN bundles (3 files each) present, also host-1 rehearsal outputs under `evaluation/rehearsal/` (freeze.json, smoke.db). Use a fresh `R=$ART/evaluation/rehearsal_h2` so they are not overwritten; keep them as host-1 evidence for N5. |
| N5 | Identity vs host 1 | compare `manifest_hash` and each `bundle_hash` of the new dry run with host 1's `freeze.json` | all equal | ✅ | 18 Oct: `identical` (manifest hash and all five bundle hashes equal host 1); thresholds equal; git_commit 62137598…. |
| N6 | Repeat P1, V1, V2, V3, V4, V5, V6, V7, sanity check | as section B | as section B | ⬜ | |

## Overnight run (host 2, code 4166e1b with `--workers`)

Script `/root/overnight.sh`, started with `nohup bash /root/overnight.sh > /root/overnight.log 2>&1 &`. Steps, stopping at the first failure:
fresh rehearsal freeze, 500-clip smoke with 12 workers, comparison with the serial smoke store (max diff < 1e-4, 0 decision flips),
full dev run with 12 workers and the parity gate, dev rehearsal report without and with bootstrap. Expected: dev scoring about 1-1.5 h
(estimate, not measured), bootstrap time unknown. Outputs under `artifacts/evaluation/overnight/`. Resume after an interruption: the
same eval-score command plus `--resume`.

| ID | Step | Status | Result / evidence |
| --- | --- | --- | --- |
| O1 | Freeze, smoke with workers, comparison | ✅ | 18 Oct: smoke with 12 workers 500 clips; max abs diff vs serial 7.6e-6, 0 decision flips; seconds clean 13.6 / codec 43.0 / noise 4.5 / 4.5 / trim 3.8 |
| O2 | Full dev scoring with parity gate | ✅ | 18 Oct: 24,844 clips, 621,100 rows, complete, 12 workers, 2,396 s (clean 190.8, codec 1633.9, noise_20 231.8, noise_10 186.9, trim 152.1); ~130 clips/s; parity passed for all 5 candidates: n_clips 24844/24844, roc_auc delta <= 1.8e-8, eer delta 0, confusion delta 0 or one boundary clip (fp +1 / tn -1) for four of five |
| O3 | Dev rehearsal report (plain, bootstrap) | ✅ | 18 Oct: plain report 24 s (25 results, record stayed FROZEN); bootstrap report took about 5.5 h (00:54Z to 06:27Z). Locked evaluation bootstrap would be longer: needs a decision |

### Overnight follow-up: parallel bootstrap check (commit 8702a11)

| ID | Step | Command | Pass criteria | Status | Result / evidence |
| --- | --- | --- | --- | --- | --- |
| B1 | Pull new commit | `cd /root/ai-voice-deepfake-detector && git pull --ff-only && git log --oneline -1` | HEAD 8702a11 | ✅ | 18 Oct: fast-forward 4166e1b..8702a11 |
| B2 | Rerun the dev bootstrap report with workers on the overnight store | `uv run voice-detector eval-report --freeze-record $R/freeze.json --store $R/dev.db --out-dir $R/report_dev_boot_w --bootstrap --workers 25` | `rehearsal_report`, 25 results, record still FROZEN; expected 15-30 min vs 5.5 h | ✅ | 18 Oct: 25/25 pairs, `rehearsal_report`, record still FROZEN. Elapsed not logged; owner estimates under 20 min (serial: 5.5 h). |
| B3 | Compare with the serial bootstrap report | `diff <(jq -S .results $R/report_dev_boot/evaluation_report.json) <(jq -S .results $R/report_dev_boot_w/evaluation_report.json) && echo identical-results` | `identical-results` | ✅ | 18 Oct: `identical-results` on real dev data (25 pairs x 27 slices with bootstrap). |

## Helper scripts (step P3)

```bash
cat > $R/check_cnn.py <<'EOF'
import os, sys
from pathlib import Path
import numpy as np, torch
from voice_detector.cnn_dataset import EvalWindowDataset
from voice_detector.cnn_pilot import CompactMelCNN
from voice_detector.cnn_scoring import load_cnn_scorer
from voice_detector.cnn_train import _score_eval_dataset
from voice_detector.manifest import read_manifest_sqlite
from voice_detector.prediction_store import read_predictions

store, bundle = Path(sys.argv[1]), Path(sys.argv[2])
preds = {p.clip_id: p.clip_score for p in
         read_predictions(store, candidate_id=bundle.name, condition="clean")}
catalog = Path(os.environ["VOICE_DETECTOR_ARTIFACTS_DIR"]) / "manifest" / "manifest.sqlite"
rows = [r for r in read_manifest_sqlite(catalog)[0] if r["clip_id"] in preds]
scorer = load_cnn_scorer(bundle)
mel = scorer._mel_spec
norm = torch.load(bundle / "cnn_norm_stats.pt", map_location="cpu", weights_only=True)
model = CompactMelCNN(n_mels=mel.n_mels, n_frames=mel.n_frames)
model.load_state_dict(torch.load(bundle / "cnn_checkpoint.pt", map_location="cpu", weights_only=True))
ids, _, ref = _score_eval_dataset(model, EvalWindowDataset(rows, mel, mean=norm["mean"], std=norm["std"]), batch_size=64)
diff = np.abs(np.array([preds[i] for i in ids]) - ref)
print(bundle.name, "clips", len(ids), "max_abs_diff", float(diff.max()))
EOF
cat > $R/check_lr.py <<'EOF'
import sys
from voice_detector.prediction_store import read_predictions as rp

new_store, old_store, candidate = sys.argv[1:4]
new = {p.clip_id: p.clip_score for p in rp(new_store, candidate_id=candidate, condition="clean")}
old = {p.clip_id: p.clip_score for p in rp(old_store, candidate_id=candidate, condition="clean")}
common = new.keys() & old.keys()
print(candidate, "common clips", len(common), "max_abs_diff", max(abs(new[k] - old[k]) for k in common))
EOF
```

## Failure handling

| Symptom | Meaning | Action |
| --- | --- | --- |
| `bundle_outside_artifacts_root` | bundle not at `$ART/<candidate_id>` | copy it there; re-run freeze |
| `bundle_invalid` | run record or model file problem | fix the bundle, never edit a run record |
| `parity_failed` | scores disagree with the recorded dev metrics | stop; send output |
| `store_without_provenance` / `store_mismatch` | wrong or legacy store | use a fresh `--store` |
| `parity_not_passed` / `dev_store_required` | evaluation run lacks a passed dev store | finish R4 first |
| run interrupted | whole conditions are committed | same command plus `--resume` |
| `git_commit` is "unknown" in a freeze draft | `git rev-parse HEAD` fails in the process cwd (not a repo, git missing, or safe.directory/ownership) | fix git in the worktree before freezing; see the V1 note |

## Log

| Date | ID | Note |
| --- | --- | --- |
| 2026-10-17 | T1–T3 | tests, CI and review done |
| 2026-10-17 | V2 | Observation: `lr_run_record.train_manifest_hash` is the role-free `b172e50c…`; the freeze cites the role-bearing `b932b859…`. So the LR features were extracted from the manifest before the dev reservation roles were added, which is the ordering `docs/data.md` says to avoid. Scoring is unaffected (audio is decoded fresh, thresholds are fixed), and this is the gap tracked by #52/#53. The dev parity gate (V8) re-checks the dev metrics, including `n_clips`. |
| 2026-10-17 | V3 | Timing extrapolation (serial, 8 torch threads): dev 24,844 clips about 9.3 h (codec about 5.3 h of it); evaluation 71,237 clips about 26.7 h. Exceeds the 2-3 h / 8 h thresholds set in issue #70 for adding `--workers`. Decision pending. |
| 2026-10-18 | host 2 | V1, V2 repeated on hp-xl420gen9-01: dry run matches host 1; rehearsal freeze `rehearsal_h2/freeze.json` FROZEN (manifest b932b859…). Pending: LA file count + full checksum pass, V3 smoke with cross-host score comparison. |
| 2026-10-18 | host 2 V3 | Smoke 500 clips done with `--torch-threads 20`: complete, 12500 rows, predictions_hash e444d8bc… (host 1: 248e839c…, differs; cross-host score comparison pending). Seconds: clean 90.8, codec 348.8, noise_20 60.9, noise_10 57.8, trim 33.6 (total 592 s vs 675 s on host 1). Diagnostics identical to host 1 (noise clipped 543 / 1263). Extrapolation: dev about 8.2 h, evaluation about 23.4 h serial. |
| 2026-10-18 | host 2 vs host 1 | Cross-host comparison of the 500-clip smoke stores: 12500 rows each, same keys, max abs score difference 7.6e-6 (cnn_clean_seed2, codec), LR <= 9.5e-7, decision flips 0. Hashes differ (e444d8bc… vs 248e839c…) only by last-bit float differences. Run each real store on one host, do not mix hosts for one store. |
| 2026-10-18 | parity pre-check | Dev roles in the manifest: select 19,938 + threshold 4,906 = 24,844, none unassigned. All five run records report dev n_clips 24,844, so the dev parity gate compares like with like. |
| 2026-10-18 | sanity | Mean abs score shift vs clean on 500 dev clips. LR: codec 1.30, noise_20 4.08, noise_10 5.37, trim 1.83. CNN clean s1: 7.16, 14.05, 14.37, 3.64. CNN augmented s1: 1.64, 5.89, 6.43, 2.84. Every condition changes the scores; noise_10 > noise_20 throughout; the augmented CNN shifts less than the clean CNN under every condition. Shifts are score-scale dependent, so do not compare across model families. |
| 2026-10-18 | merge bar | P0-V7 passed (host 1 through V6, host 2 for V1-V3, cross-host, V7, sanity). V8-V9 (full dev run) not run. |
| 2026-10-18 | workers | `--workers N` added to PR 76 (commit 4166e1b, no review gate by owner choice): spawned worker processes, one torch thread each, BLAS pools pinned to one thread, single SQLite writer, per-condition atomic commit. Store provenance pins effective thread settings, not the worker count. |
| 2026-10-18 | overnight | Everything passed. Dev scoring with 12 workers took 40 min instead of the ~8 h serial estimate (about 12x). Parity gate passed on real data for all five candidates. Bootstrap report is the new bottleneck (about 5.5 h on dev). |
| 2026-10-18 | bootstrap | `eval-report --workers N` added to PR 76 (commit 8702a11, issue #78): one worker task per candidate x condition pair, results consumed in order, report byte-identical to serial (unit-tested). Real-data check pending (B1-B3). |
| 2026-10-18 | bootstrap check | Parallel bootstrap report on the overnight dev store matches the serial report exactly (`.results` identical), in under 20 min instead of 5.5 h. PR 76 has no open verification items. |
| 2026-10-18 | M1 | PR 76 squash-merged into main as c4445f5. Real run (section D) unblocked up to R5; R6 and R7 wait for the owner. |
| 2026-10-18 | R3-R5 | Real freeze and real dev run done on c4445f5. Dev predictions_hash 8845093e… is identical to the overnight rehearsal run: scores reproduce exactly across runs, commits and freeze records. Next: owner go for R6; recommended before it: commit the FROZEN record as evidence. |
| 2026-10-18 | R6 go | Owner go for the locked evaluation scoring. Frozen record committed as evidence in PR 80 (sha256 cfd1406c…). |
| 2026-10-18 | R6 running | Started on hp-xl420gen9-01 (the R-series host; host 2) with `--workers 20` as planned; the sidecar later showed 30. A full pytest run (481 passed in 73.7 s, torch tests included) was also run on the same host while R6 was scoring (mistaken for a separate machine). Scores are unaffected (same commit and lockfile, per-clip scores independent of load); only timing may have been affected. R7 runbook and reading order added above. |
| 2026-10-18 | R6 done | Locked evaluation scoring complete (sidecar `complete: true`, 82 min, 30 workers). The log file lacks the final JSON (status, rows_written, n_missing); recorded as a logging deviation. No evaluation predictions were opened. R7 waits for the owner go. |
| 2026-10-18 | consult R7 | Architect review: GO-AFTER-X. Written `docs/evaluation-protocol.md` (to be committed before R7). Metadata-only pre-flight on DL120 pending; optional synthetic dress rehearsal pending; R7 still waits for the owner go. |
| 2026-10-18 | R7 pre-flight | Metadata-only pre-flight passed on hp-xl420gen9-01: HEAD c4445f50, clean tree, no stray processes, 29 GB available, ffmpeg 8.1.3 with libopus; record FROZEN (sha256 cfd1406c…); one sidecar (resume false, max_clips null, error null, workers 30, torch_threads 1); evaluation store 25 pairs x 71,237 = 1,780,925 rows, quick_check ok; hashes E 1ef12a29… (matches sidecar) and D 8845093e… (matches R4); E and D freeze_identity and provenance identical (9f3ee9c2…); check_store_for_report passed for both. Pre-R7 checksums: E dacc4693…, D 8e501127…, sidecar 1f5737ce…; backed up to /root/backup_prer7 and verified. |
| 2026-10-18 | R6 log gap | `eval_score.log` was last written 09:33:25 (clean at 50,000/71,237) and created 09:33:51 (local, -0400), while the store was created 09:25:44 and last written 10:47:39. A file created after its last write means it was replaced mid-run; the scorer kept writing to the unlinked original, so later progress and the final JSON were lost. Inference, not proof; the store, counts and hashes are unaffected. |
| 2026-10-18 | R7 go and launch | Owner go; protocol doc committed and pushed first (docs/evaluation-protocol, 26586d8). Launched detached on hp-xl420gen9-01 with `--workers 30` (planned 25; report is worker-count independent). Record flipped to SCORED at 2026-10-18T18:22:01Z with predictions_hash 1ef12a29… (equals the R6 sidecar); SCORED snapshot `freeze_g2_SCORED.json` sha256 1995eaad…. R7 running; no report numbers seen. |
| 2026-10-18 | R7 done | Locked report generated and the freeze record is REPORTED (report_hash 2bc69920…, 25 results, rc 0, 2 h 16 min). Stores unchanged (sha256 match). Next: integrity checks from the report (protocol section 2), back up the REPORTED record and report, then read results in the pre-registered order. |
| 2026-10-19 | R7 integrity | Post-R7 integrity check passed on the report: git_commit c4445f50, 25 results/25 pairs, overall 71,237 (7,355 bona fide / 63,882 spoof) in all pairs, thresholds equal the record, 13 attack slices 4,914 spoof each, duration bins sum 96,081 per pair, 0 of 175 dev slices differ from the R5 rehearsal. Backups: REPORTED record sha256 88f27bda…, report c0116ce2… (/root/backup_post_r7). |
| 2026-10-19 | R7 results read | Read in the pre-registered order (descriptive only, no winner declared). Primary endpoint, clean EER on unseen attacks: LR 19.93, CNN clean s1/s2 8.30/6.68, CNN augmented s1/s2 11.77/7.35 (%). Known-algorithm A16/A19: CNN 0.67-5.12, LR 28.5. Seed spread is large (augmented 11.77 vs 7.35, intervals do not overlap). Frozen clean-dev thresholds do not transfer under codec/noise: bona fide flagged as spoof up to 92% (codec, clean s2) and 86% (noise_10db, augmented s2) while AUC stays 85-97%; clean s1 AUC is 34% at noise_10db (below chance, EER 61.4). silence_trim raises CNN EER from 7-12 to 20-23. Negative results and the threshold finding to be reported as they are. |
| 2026-10-19 | PR 82 host test | On the XL420 in a separate worktree (/root/pr82, 481738d): ruff clean; tests/test_eval_supplementary.py 24 passed (15 s); full suite 504 passed, 1 failed (`test_config.py::test_default_config_uses_repository_local_paths`, caused by the exported `VOICE_DETECTOR_ARTIFACTS_DIR` of the host shell; confirmed: with VOICE_DETECTOR_* unset, tests/test_config.py gives 2 passed); dev rehearsal of `eval-supplement` 2 min 30 s with 5 workers, report_hash e78ed516…, serial run identical. Evaluation store not touched. |
| 2026-10-20 | deployed model | Dev-only rule (accepted 2026-10-19) applied mechanically to the R5 development report: cnn_augmented_seed1 wins (mean dev EER over clean/codec/noise_20/noise_10 = 13.44; runner-up cnn_augmented_seed2 15.25). Recorded with caveats (single-seed pick, 7.21% pooled HFPR and highest CNN clean EER on locked unseen attacks, threshold non-transfer) in docs/deployed-model-decision.md; issue #86. |
