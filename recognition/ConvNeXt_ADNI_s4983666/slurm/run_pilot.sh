#!/bin/bash
# Run inside a Slurm GPU allocation; never train directly on the login node.
set -euo pipefail
: "${SLURM_JOB_ID:?Use submit_pilot.sh to obtain a Slurm GPU allocation.}"
[[ "$#" -eq 3 ]] || { printf 'Expected model, project root, and run root.\n' >&2; exit 2; }
adni_model="$1"
adni_project_root="$2"
adni_runs_root="$3"
case "$adni_model" in
    cnn) adni_default_lr=0.001; adni_default_decay=0.0001 ;;
    convnext_lite) adni_default_lr=0.0001; adni_default_decay=0.05 ;;
    *) printf 'Unsupported pilot model: %s\n' "$adni_model" >&2; exit 2 ;;
esac
adni_conda_root="${ADNI_CONDA_ROOT:-$HOME/miniconda3}"
[[ -f "$adni_conda_root/etc/profile.d/conda.sh" ]] || {
    printf 'Conda is missing at %s; set ADNI_CONDA_ROOT before submission.\n' "$adni_conda_root" >&2; exit 2;
}
# Some Conda activation scripts reference unset variables.
set +u
source "$adni_conda_root/etc/profile.d/conda.sh"
conda activate "${ADNI_CONDA_ENV:-torch}"
set -u
cd -- "$adni_project_root"
python -u - <<'PY'
import sys
import torch
import PIL
import matplotlib
print('Python:', sys.version, flush=True)
print('Packages:', torch.__version__, PIL.__version__, matplotlib.__version__, flush=True)
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable in this allocation; no training was started.')
print('GPU:', torch.cuda.get_device_name(0), flush=True)
PY
adni_fold="${ADNI_FOLD:-1}"
[[ "$adni_fold" =~ ^[1-5]$ ]] || { printf 'ADNI_FOLD must be 1-5.\n' >&2; exit 2; }
adni_output="$adni_runs_root/${adni_model}_inner_fold0${adni_fold}_job${SLURM_JOB_ID}"
[[ ! -e "$adni_output" ]] || { printf 'Refusing existing output: %s\n' "$adni_output" >&2; exit 2; }
mkdir -p -- "$adni_runs_root"
printf 'Exploratory inner-only run: %s\n' "$adni_output"
printf 'Outer validation, calibration and final-test images will not be scored.\n'
adni_command=(python -u train.py
    --data-root "${ADNI_DATA_ROOT:-/home/groups/comp3710/ADNI}"
    --splits-dir "${ADNI_SPLITS_DIR:-$(dirname -- "$adni_project_root")/adni_splits_v1}"
    --output "$adni_output" --model "$adni_model" --inner-only
    --fold "$adni_fold" --epochs "${ADNI_EPOCHS:-30}" --patience "${ADNI_PATIENCE:-5}"
    --batch-size "${ADNI_BATCH_SIZE:-32}" --workers "${ADNI_WORKERS:-2}"
    --threads "${ADNI_THREADS:-2}" --seed "${ADNI_SEED:-3710}"
    --lr "${ADNI_LR:-$adni_default_lr}" --weight-decay "${ADNI_WEIGHT_DECAY:-$adni_default_decay}"
    --augmentation "${ADNI_AUGMENTATION:-none}" --image-height 240 --image-width 256
    --calibration-bins 15 --reject-threshold 0.8
    --profile-warmup 10 --profile-repeats 100 --device cuda)
printf '%q ' "${adni_command[@]}"; printf '\n'
exec "${adni_command[@]}"
