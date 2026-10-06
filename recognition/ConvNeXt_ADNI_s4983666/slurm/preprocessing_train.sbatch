#!/bin/bash
#SBATCH --job-name=adni-preproc-train
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
# Parameters are supplied by submit_preprocessing_train.sh and recorded in launch.json.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using slurm/submit_preprocessing_train.sh.}"
[[ "$#" -ge 1 ]] || { printf 'Expected source root and training options.\n' >&2; exit 2; }
adni_source="$1"
shift
adni_conda="${ADNI_CONDA_ROOT:-$HOME/miniconda3}"
set +u
source "$adni_conda/etc/profile.d/conda.sh"
conda activate "${ADNI_CONDA_ENV:-torch}"
set -u
cd -- "$adni_source"
adni_python="${ADNI_PYTHON:-python}"
exec "$adni_python" -u slurm/preprocessing_train.py --run "$@"
