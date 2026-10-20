#!/usr/bin/env bash
# Archive the frozen evaluation artifacts with SHA-256 checksums, outside Git (TODO-Plan §10).
#
#   scripts/archive_artifacts.sh --artifacts DIR --reported-record FILE --frozen-record FILE \
#       --dest DIR [--dry-run] [--check-published-hashes]
#
# Sources are only ever read. DEST must not exist (or be empty): nothing is overwritten.
# The archive holds the manifest catalog, the five model bundles, both freeze-record copies,
# the evaluation directory (prediction stores, reports, scoring logs, sidecars) and every
# run record, plus SHA256SUMS and ARCHIVE_INFO.txt. Feature caches are not archived: they are
# regenerable. Audio is never copied. Never put DEST inside the repository.
set -euo pipefail

ART="" REPORTED="" FROZEN="" DEST="" DRY=0 PUBLISHED=0
while [ $# -gt 0 ]; do
  case "$1" in
    --artifacts) ART="$2"; shift 2 ;;
    --reported-record) REPORTED="$2"; shift 2 ;;
    --frozen-record) FROZEN="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --check-published-hashes) PUBLISHED=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
for v in ART REPORTED FROZEN DEST; do
  [ -n "${!v}" ] || { echo "missing --$(echo "$v" | tr 'A-Z' 'a-z')" >&2; exit 2; }
done

BUNDLES=(lr_baseline cnn_clean_seed1 cnn_clean_seed2 cnn_augmented_seed1 cnn_augmented_seed2)
fail() { echo "REFUSED: $*" >&2; exit 1; }

[ -d "$ART/manifest" ] || fail "no $ART/manifest"
[ -d "$ART/evaluation" ] || fail "no $ART/evaluation"
for b in "${BUNDLES[@]}"; do [ -d "$ART/$b" ] || fail "missing bundle $ART/$b"; done
[ -f "$REPORTED" ] || fail "no reported record $REPORTED"
[ -f "$FROZEN" ] || fail "no frozen record $FROZEN"
if [ -d "$DEST" ] && [ -n "$(ls -A "$DEST")" ]; then fail "$DEST exists and is not empty"; fi
# A live SQLite database (journal or WAL present) would copy inconsistently.
if find "$ART/evaluation" "$ART/manifest" \( -name '*-wal' -o -name '*-journal' \) | grep -q .; then
  fail "a database has a -wal/-journal file: stop the process using it first"
fi
grep -q '"state": *"REPORTED"' "$REPORTED" || fail "$REPORTED is not in state REPORTED"
grep -q '"state": *"FROZEN"' "$FROZEN" || fail "$FROZEN is not in state FROZEN"

if [ "$DRY" = 1 ]; then
  echo "DRY RUN: nothing is written. Would archive into $DEST:"
  du -sh "$ART/manifest" "$ART/evaluation" "${BUNDLES[@]/#/$ART/}" "$REPORTED" "$FROZEN"
  echo "run records:"; find "$ART" -name '*run_record*.json' | sort
  df -h "$(dirname "$DEST")" | tail -1
  exit 0
fi

mkdir -p "$DEST"/{manifest,bundles,freeze,evaluation,run_records}
cp -a "$ART/manifest/." "$DEST/manifest/"
for b in "${BUNDLES[@]}"; do cp -a "$ART/$b" "$DEST/bundles/$b"; done
cp -a "$REPORTED" "$DEST/freeze/evaluation_freeze_g2.REPORTED.json"
cp -a "$FROZEN" "$DEST/freeze/evaluation_freeze_g2.FROZEN.json"
cp -a "$ART/evaluation/." "$DEST/evaluation/"
(cd "$ART" && find . -name '*run_record*.json' -print0 | while IFS= read -r -d '' f; do
  mkdir -p "$DEST/run_records/$(dirname "$f")"; cp -a "$f" "$DEST/run_records/$f"
done)

{
  echo "archived_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "source_artifacts=$ART"
  echo "bundles=${BUNDLES[*]}"
} > "$DEST/ARCHIVE_INFO.txt"

# Checksums are computed on the copy, then the copy is compared with the source bytes.
(cd "$DEST" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
(cd "$DEST" && sha256sum -c --quiet SHA256SUMS) || fail "checksum verification failed"
for b in "${BUNDLES[@]}"; do diff -r "$ART/$b" "$DEST/bundles/$b" >/dev/null || fail "bundle $b differs"; done
diff -r "$ART/evaluation" "$DEST/evaluation" >/dev/null || fail "evaluation directory differs"
cmp "$REPORTED" "$DEST/freeze/evaluation_freeze_g2.REPORTED.json"
cmp "$FROZEN" "$DEST/freeze/evaluation_freeze_g2.FROZEN.json"

if [ "$PUBLISHED" = 1 ]; then
  # Values recorded in docs/evaluation-results-g2.md (Provenance table).
  check() {
    local got; got="$(sha256sum "$DEST/$1" | cut -d' ' -f1)"
    [ "$got" = "$2" ] || fail "$1 sha256 $got != published $2"
    echo "published hash ok: $1"
  }
  check freeze/evaluation_freeze_g2.FROZEN.json cfd1406ca977362207228aa2af8a494c45c0f3cbe1216ef9509aa3ea1d8cb4d2
  check freeze/evaluation_freeze_g2.REPORTED.json 88f27bda2c317136826a5337b13cc4ac3224e948c865efa7fb120665d321fb18
  check evaluation/report_locked/evaluation_report.json c0116ce295b3c7f50a667bff79b77e3c476494b9e4cd895f36aae4421427ed2b
fi

chmod -R a-w "$DEST"
echo "ARCHIVE OK: $(wc -l < "$DEST/SHA256SUMS") files, $(du -sh "$DEST" | cut -f1) in $DEST"
