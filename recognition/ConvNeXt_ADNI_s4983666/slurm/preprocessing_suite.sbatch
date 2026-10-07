#!/bin/bash
#SBATCH --job-name=adni-overnight
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using slurm/submit_preprocessing_suite.sh.}"
[[ "$#" -ge 1 ]] || { printf 'Expected source root and suite options.\n' >&2; exit 2; }
adni_source="$1"
shift
adni_conda="${ADNI_CONDA_ROOT:-$HOME/miniconda3}"
set +u
source "$adni_conda/etc/profile.d/conda.sh"
conda activate "${ADNI_CONDA_ENV:-torch}"
set -u
cd -- "$adni_source"
exec "${ADNI_PYTHON:-python}" -u slurm/preprocessing_suite.py --run "$@"
