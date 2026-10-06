#!/bin/bash
# Submit a single audit/train/replay allocation; no full experimental suite.
set -euo pipefail
adni_source="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
adni_work="$(dirname -- "$adni_source")"
export ADNI_DATA_ROOT="${ADNI_DATA_ROOT:-/home/groups/comp3710/ADNI}"
export ADNI_SPLITS_DIR="${ADNI_SPLITS_DIR:-$adni_work/adni_splits_v1}"
export ADNI_RUNS_ROOT="${ADNI_RUNS_ROOT:-$adni_work/runs}"
for adni_path in "$ADNI_DATA_ROOT" "$ADNI_SPLITS_DIR" "$ADNI_RUNS_ROOT"; do
    [[ "$adni_path" == /* ]] || { printf 'Use absolute ADNI paths.\n' >&2; exit 2; }
done
adni_logs="$ADNI_RUNS_ROOT/slurm_logs"
adni_command=(sbatch --parsable --export=ALL --time="${ADNI_PREPROCESSING_TIME_LIMIT:-01:00:00}"
    --output="$adni_logs/preprocessing_check_%j.out" --error="$adni_logs/preprocessing_check_%j.err"
    "$adni_source/slurm/preprocessing_check.sbatch" "$adni_source")
if [[ "${1:-}" == --dry-run && "$#" -eq 1 ]]; then
    printf '%q ' "${adni_command[@]}"; printf '\n'; exit 0
fi
[[ "$#" -eq 0 ]] || { printf 'Usage: bash slurm/submit_preprocessing_check.sh [--dry-run]\n' >&2; exit 2; }
[[ -d "$ADNI_DATA_ROOT" && -f "$ADNI_SPLITS_DIR/COMPLETED.json" ]] || { printf 'Data or frozen manifests missing.\n' >&2; exit 2; }
command -v sbatch >/dev/null || { printf 'Submit from the cluster login node.\n' >&2; exit 2; }
mkdir -p -- "$adni_logs"
"${adni_command[@]}"
