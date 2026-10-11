#!/bin/bash
#SBATCH --job-name=adni-final-eval
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
# Final evaluation of ONE frozen checkpoint (run once, after all model choices are made):
#   sbatch slurm/final_eval.sbatch <run directory containing final.pt>
# 1) fit Platt scaling + referral threshold on calibration patients;
# 2) apply both unchanged to the test patients.
set -euo pipefail
run_dir="$(cd -- "$1" && pwd -P)"
splits="${ADNI_SPLITS:-$HOME/comp3710/adni_splits_v1}"
source "${ADNI_CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${ADNI_CONDA_ENV:-torch}"
python -u predict.py --checkpoint "$run_dir/final.pt" --splits-dir "$splits" \
    --role calibration --output "$run_dir/final_calibration"
python -u predict.py --checkpoint "$run_dir/final.pt" --splits-dir "$splits" \
    --role test --calibration-file "$run_dir/final_calibration/calibration.json" \
    --output "$run_dir/final_test"
