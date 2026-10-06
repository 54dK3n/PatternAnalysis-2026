#!/bin/bash
# Submit one scratch-trained inner-only pilot from the login node.
set -euo pipefail

adni_model=convnext_lite
adni_dry_run=false
for adni_arg in "$@"; do
    case "$adni_arg" in
        cnn|convnext_lite) adni_model="$adni_arg" ;;
        --dry-run) adni_dry_run=true ;;
        *) printf 'Usage: bash slurm/submit_pilot.sh [cnn|convnext_lite] [--dry-run]\n' >&2; exit 2 ;;
    esac
done
adni_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
adni_project_root="$(dirname -- "$adni_script_dir")"
adni_work_root="$(dirname -- "$adni_project_root")"
export ADNI_DATA_ROOT="${ADNI_DATA_ROOT:-/home/groups/comp3710/ADNI}"
export ADNI_SPLITS_DIR="${ADNI_SPLITS_DIR:-$adni_work_root/adni_splits_v1}"
export ADNI_RUNS_ROOT="${ADNI_RUNS_ROOT:-$adni_work_root/runs}"
# Log paths passed to sbatch must be absolute; #SBATCH does not expand $HOME.
for adni_path in "$ADNI_DATA_ROOT" "$ADNI_SPLITS_DIR" "$ADNI_RUNS_ROOT"; do
    [[ "$adni_path" == /* ]] || { printf 'Use absolute ADNI paths: %s\n' "$adni_path" >&2; exit 2; }
done
[[ -d "$ADNI_DATA_ROOT" && -d "$ADNI_SPLITS_DIR" ]] || {
    printf 'Data or frozen split directory is missing. Check ADNI_DATA_ROOT and ADNI_SPLITS_DIR.\n' >&2; exit 2;
}
# Fail before queue submission when only part of the update was installed.
[[ -f "$adni_project_root/evaluation/reporting.py" ]] &&
    grep -q 'convnext_lite' "$adni_project_root/models/registry.py" &&
    grep -q -- '--inner-only' "$adni_project_root/engine/training.py" || {
        printf 'Project update is incomplete: install the complete source bundle first.\n' >&2; exit 2;
    }
adni_log_dir="$ADNI_RUNS_ROOT/slurm_logs"
adni_command=(sbatch --parsable --export=ALL
    --time="${ADNI_TIME_LIMIT:-02:00:00}"
    --output="$adni_log_dir/${adni_model}_%j.out"
    --error="$adni_log_dir/${adni_model}_%j.err"
    "$adni_script_dir/${adni_model}.sbatch" "$adni_project_root" "$ADNI_RUNS_ROOT")
if "$adni_dry_run"; then
    printf 'Dry run; no job submitted.\n'
    printf '%q ' "${adni_command[@]}"; printf '\n'
    exit 0
fi
command -v sbatch >/dev/null || { printf 'sbatch is unavailable; submit from the cluster login node.\n' >&2; exit 2; }
mkdir -p -- "$adni_log_dir"
adni_job="$("${adni_command[@]}")"
printf 'Submitted job: %s\nLogs: %s/%s_%s.{out,err}\n' "$adni_job" "$adni_log_dir" "$adni_model" "${adni_job%%;*}"
