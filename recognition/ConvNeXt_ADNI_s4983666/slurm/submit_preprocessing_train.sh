#!/bin/bash
# Inspect with --dry-run; otherwise submit one configurable inner-development run.
set -euo pipefail
adni_source="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
adni_python="${ADNI_PYTHON:-${ADNI_CONDA_ROOT:-$HOME/miniconda3}/envs/${ADNI_CONDA_ENV:-torch}/bin/python}"
[[ -x "$adni_python" ]] || { printf 'Python environment missing; set ADNI_PYTHON to its absolute executable.\n' >&2; exit 2; }
exec "$adni_python" -u "$adni_source/slurm/preprocessing_train.py" "$@"
