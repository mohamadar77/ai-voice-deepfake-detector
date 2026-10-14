# Host runbook: freeze the development reservation (#31)

**Status (2026-10-14):** code merged into PR
[#47](https://github.com/mohamadar77/ai-voice-deepfake-detector/pull/47) at commit
`966b000` (branch `feat/dev-reservation-freeze`). **The host run below has not yet
been executed** — the owner will run it later. Until it runs,
`reports/dev_reservation_frozen.json` does not exist and the development
reservation is not frozen.

## What this run does and why it must happen on the host

`voice-detector reserve-development` derives a deterministic, seeded,
speaker-grouped split of the ASVspoof 2019 LA **development** partition into
`dev_select` and `dev_threshold` roles, then:

- writes `reports/dev_reservation_frozen.json` — the **authoritative, committed**
  membership (speaker IDs only). This record **outranks** the git-ignored
  manifest; downstream code trusts it over the manifest's `development_role`
  column.
- stamps the `development_role` column into the manifest SQLite (host-only,
  git-ignored under `artifacts/**`).

It must run on the DL120 host (`hpe-dl120gen10-01`) because that is where the real
LA manifest lives. The membership record it commits is the deliverable; the
manifest mutation stays on the host.

**Ordering hazard:** this freeze changes the manifest hash, so it must run
**before** #7's host feature extraction. Do not extract features against a
manifest whose roles are not yet frozen.

## Guard behaviour (expected, not errors to debug around)

The reservation refuses by default — with a JSON refusal + a remediation hint,
never a traceback — when:

- the manifest lacks a `catalog_meta` table (e.g. it was built before this slice);
- the catalog hash has drifted from what the committed record expects;
- a committed `reports/dev_reservation_frozen.json` already disagrees with a
  freshly derived reservation (use `--force` only to deliberately replace it).

Because the manifest builder in this slice is what first writes `catalog_meta`,
an older host manifest **must be rebuilt** (step 3) or step 4 will cleanly refuse.

## Steps (run top to bottom on the host)

Host repo: `~/ai-voice-deepfake-detector`. LA data: `data/raw/LA`. Manifest:
`artifacts/manifest/` (git-ignored). Long jobs run detached under `tmux` with a
file log; detach with `Ctrl-b d`, reattach with `tmux attach -t <name>`.

### 1. Sync the branch + deps (quick)

```bash
cd ~/ai-voice-deepfake-detector && git fetch origin && git checkout feat/dev-reservation-freeze && git pull --ff-only && uv sync --frozen --all-groups
```

### 2. Lint + full suite on this exact commit — the "host green on the same commit" gate

```bash
tmux new -s vd-test 'cd ~/ai-voice-deepfake-detector && uvx ruff@0.16.10 check . 2>&1 | tee ~/vd-test.log && uv run pytest -q 2>&1 | tee -a ~/vd-test.log; echo "EXIT=$?" | tee -a ~/vd-test.log'
```

### 3. Rebuild the manifest with the new code (long; regenerates `catalog_meta`)

```bash
tmux new -s vd-manifest 'cd ~/ai-voice-deepfake-detector && uv run voice-detector build-manifest --la-root data/raw/LA 2>&1 | tee ~/vd-manifest.log; echo "EXIT=$?" | tee -a ~/vd-manifest.log'
```

### 4. Dry-run the reservation (previews the record and runs every guard; writes nothing)

```bash
cd ~/ai-voice-deepfake-detector && uv run voice-detector reserve-development --dry-run
```

Sanity-check the printed role sizes and per-role class/attack coverage against
#31's spec before step 5.

### 5. Real run (writes the record + stamps `development_role`)

```bash
cd ~/ai-voice-deepfake-detector && uv run voice-detector reserve-development
```

### 6. Commit the frozen record onto the PR branch + push

The sign-off trailer is written literally (not `git commit -s`), above
`Co-Authored-By`.

```bash
cd ~/ai-voice-deepfake-detector && git add reports/dev_reservation_frozen.json && git commit -m "chore(reservation): freeze authoritative dev reservation record (#31)" -m "Host run of voice-detector reserve-development against the real LA manifest; writes the committed speaker-grouped membership that outranks the git-ignored manifest." -m "Signed-off-by: Mohamad Abo-Ras <mohamad.abo.ras@gmail.com>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" && git push
```

## After the run

- The record commit lands in PR #47, so CI re-runs on the new HEAD (adding one
  JSON file does not change test outcomes).
- Record the run per the reproducibility rules: config, seed, dependency
  versions, the commit (`966b000` + the record commit), manifest hash before and
  after, runtime, and the derived role counts.
- Alternative: if the record should stay out of PR #47, run steps 1–5 but commit
  the record on `main` after #47 merges instead of step 6.
