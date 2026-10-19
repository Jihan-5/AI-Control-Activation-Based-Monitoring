#!/usr/bin/env bash
# Copy the small phase-1 artifacts (no activations, no weights) into results/phase1/
# so they can be committed and pushed for review. Typically < 10 MB.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/phase1/$(date +%Y%m%d-%H%M)
mkdir -p "$OUT"
for RUN in pilot_qwen1p5b_c0 pilot_qwen1p5b_c2cue pilot_qwen1p5b_c2; do
  SRC=runs/$RUN; [ -d "$SRC" ] || continue
  DST=$OUT/$RUN; mkdir -p "$DST"
  for f in config.resolved.yaml decision_log.jsonl env/validity.json labels/pool_summary.json manifest/profile.json manifest/run_manifest.json; do
    [ -f "$SRC/$f" ] && mkdir -p "$DST/$(dirname "$f")" && cp "$SRC/$f" "$DST/$f"
  done
  [ -d "$SRC/gates" ] && cp -r "$SRC/gates" "$DST/"
  [ -f "$SRC/labels/pool.csv" ] && gzip -c "$SRC/labels/pool.csv" > "$DST/pool.csv.gz"
done
for A in runs/adapters/*/; do
  [ -d "$A" ] || continue
  DST=$OUT/adapters/$(basename "$A"); mkdir -p "$DST"
  for f in summary.json lineage.json train_log.json masking_printout.txt; do [ -f "$A/$f" ] && cp "$A/$f" "$DST/"; done
done
cp -r runs/phase1_logs "$OUT/logs" 2>/dev/null || true
du -sh "$OUT"
echo "Collected into $OUT. To send for review:"
echo "  git add $OUT && git commit -m 'Phase 1 results' && git push"
