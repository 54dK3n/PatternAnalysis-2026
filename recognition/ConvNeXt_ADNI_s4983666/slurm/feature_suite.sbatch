#!/bin/bash
#SBATCH --job-name=adni-feature-v2
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using slurm/submit_feature_suite.sh.}"
[[ "$#" -eq 1 ]] || { printf 'Expected source root.\n' >&2; exit 2; }
adni_source="$1"
adni_conda="${ADNI_CONDA_ROOT:-$HOME/miniconda3}"
set +u
source "$adni_conda/etc/profile.d/conda.sh"
conda activate "${ADNI_CONDA_ENV:-torch}"
set -u
cd -- "$adni_source"
python -u - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable; feature suite was not started.')
print('GPU:', torch.cuda.get_device_name(0), flush=True)
PY
adni_output="${ADNI_FEATURE_OUTPUT:-${ADNI_RUNS_ROOT:-$(dirname -- "$adni_source")/runs}/feature_suite_v2_job${SLURM_JOB_ID}}"
adni_command=(python -u run_feature_experiments.py --device cuda --workers 2
    --data-root "${ADNI_DATA_ROOT:-/home/groups/comp3710/ADNI}"
    --splits-dir "${ADNI_SPLITS_DIR:-$(dirname -- "$adni_source")/adni_splits_v1}"
    --reference-suite "${ADNI_FEATURE_REFERENCE_SUITE:-$(dirname -- "$adni_source")/runs/convnext_suite_job633000}"
    --output "$adni_output" --phase "${ADNI_FEATURE_PHASE:-all}")
if [[ "${ADNI_FEATURE_NO_REPEATS:-0}" == 1 ]]; then adni_command+=(--no-repeats); fi
printf 'Feature suite output: %s\n' "$adni_output"
exec "${adni_command[@]}"
