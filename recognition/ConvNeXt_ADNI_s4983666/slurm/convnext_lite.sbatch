#!/bin/bash
#SBATCH --job-name=adni-lite-inner
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
# submit_pilot.sh passes absolute output/error paths and source/run roots.
set -euo pipefail
[[ "$#" -eq 2 ]] || { printf 'Use bash slurm/submit_pilot.sh convnext_lite\n' >&2; exit 2; }
exec bash "$1/slurm/run_pilot.sh" convnext_lite "$1" "$2"
