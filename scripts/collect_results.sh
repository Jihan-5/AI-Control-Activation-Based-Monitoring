#!/usr/bin/env bash
# Copy every small artifact (no activations, no weights, no pickles) into
# results/<timestamp>/ (or $RESULTS_DIR) for committing. Typically 5-50 MB.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
OUT=${RESULTS_DIR:-results/$(date +%Y%m%d-%H%M)}
mkdir -p "$OUT"
for RUN in runs/*/; do
  NAME=$(basename "$RUN")
  case "$NAME" in adapters|*_logs) continue;; esac
  DST=$OUT/$NAME; mkdir -p "$DST"
  for f in config.resolved.yaml decision_log.jsonl timings.json env/validity.json labels/pool_summary.json \
           manifest/profile.json manifest/run_manifest.json splits/manifest.json selection/selected.json \
           calibration/thresholds.json attacks/report.json; do
    [ -f "$RUN/$f" ] && mkdir -p "$DST/$(dirname "$f")" && cp "$RUN/$f" "$DST/$f"
  done
  for d in gates results selection; do [ -d "$RUN/$d" ] && cp -r "$RUN/$d" "$DST/"; done
  [ -f "$RUN/labels/pool.csv" ] && gzip -nc "$RUN/labels/pool.csv" > "$DST/pool.csv.gz"
  [ -f "$RUN/splits/labels.csv" ] && gzip -nc "$RUN/splits/labels.csv" > "$DST/split_labels.csv.gz"
  [ -f "$DST/results/root_predictions.csv" ] && gzip -nf "$DST/results/root_predictions.csv"
done
for A in runs/adapters/*/; do
  [ -d "$A" ] || continue
  DST=$OUT/adapters/$(basename "$A"); mkdir -p "$DST"
  for f in summary.json lineage.json train_log.json masking_printout.txt; do [ -f "$A/$f" ] && cp "$A/$f" "$DST/"; done
done
for L in runs/*_logs; do [ -d "$L" ] && cp -r "$L" "$OUT/"; done
du -sh "$OUT"
echo "Collected into $OUT. To share:  git add $OUT && git commit -m 'Results $(basename "$OUT")' && git push"
