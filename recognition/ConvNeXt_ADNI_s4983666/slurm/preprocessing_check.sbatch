#!/bin/bash
#SBATCH --job-name=adni-preproc-check
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
# Audit real inner inputs, train one scratch epoch, and replay its checkpoint.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using slurm/submit_preprocessing_check.sh.}"
[[ "$#" -eq 1 ]] || { printf 'Expected source root.\n' >&2; exit 2; }
adni_source="$1"
adni_conda="${ADNI_CONDA_ROOT:-$HOME/miniconda3}"
set +u
source "$adni_conda/etc/profile.d/conda.sh"
conda activate "${ADNI_CONDA_ENV:-torch}"
set -u
cd -- "$adni_source"
adni_data="${ADNI_DATA_ROOT:-/home/groups/comp3710/ADNI}"
adni_splits="${ADNI_SPLITS_DIR:-$(dirname -- "$adni_source")/adni_splits_v1}"
adni_output="${ADNI_PREPROCESSING_OUTPUT:-${ADNI_RUNS_ROOT:-$(dirname -- "$adni_source")/runs}/preprocessing_check_job${SLURM_JOB_ID}}"
python -u - "$adni_output" "$adni_data" "$adni_splits" <<'PY'
import sys
import torch
from pathlib import Path
from utils.artifacts import validate_output
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable; preprocessing check was not started.')
output = validate_output(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
output.mkdir(parents=True, exist_ok=False)
print('GPU:', torch.cuda.get_device_name(0), flush=True)
print('Preprocessing check output:', output, flush=True)
PY
printf 'Stage 1/3: real training/early-stop input audit\n'
python -u audit_preprocessing.py --data-root "$adni_data" --splits-dir "$adni_splits" \
    --output "$adni_output/input_review" --fold 1 --preprocessing scan_intensity_crop --max-images 6
printf 'Stage 2/3: one-epoch scratch ConvNeXt Lite training\n'
python -u train.py --data-root "$adni_data" --splits-dir "$adni_splits" \
    --output "$adni_output/pilot" --model convnext_lite --fold 1 --inner-only \
    --preprocessing scan_intensity_crop --augmentation none --epochs 1 --patience 5 \
    --batch-size 32 --lr 0.0001 --weight-decay 0.05 --workers 2 --threads 2 --device cuda
printf 'Stage 3/3: checkpoint prediction replay\n'
python -u predict.py --checkpoint "$adni_output/pilot/best.pt" \
    --data-root "$adni_data" --splits-dir "$adni_splits" --output "$adni_output/replay" \
    --batch-size 32 --workers 2 --threads 2 --device cuda --skip-inference-profile
python -u - "$adni_output" "$adni_splits" <<'PY'
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
import torch
from slurm.deploy_feature_source import frozen_seals
from utils.artifacts import write_json
root = Path(sys.argv[1])
audit = json.loads((root / 'input_review/summary.json').read_text())
config = json.loads((root / 'pilot/config.json').read_text())
trained = json.loads((root / 'pilot/metrics.json').read_text())
replayed = json.loads((root / 'replay/metrics.json').read_text())
checkpoint = torch.load(root / 'pilot/best.pt', map_location='cpu', weights_only=True)
assert audit['status'] == trained['status'] == replayed['status'] == 'complete'
assert config['checkpoint_format_version'] == 3 and checkpoint['config'] == config
assert config['model_name'] == 'convnext_lite_v1' and config['initialization'] == 'random'
assert config['pretrained_weights'] is None and config['augmentation'] == 'none'
assert config['evaluation_mode'] == 'inner_only' and config['epochs_limit'] == 1
assert trained['epochs_completed'] == trained['best_epoch'] == replayed['checkpoint_epoch'] == 1
assert trained['evaluation_role'] == replayed['evaluation_role'] == 'development_inner_early_stop'
assert audit['preprocessing_config'] == config['preprocessing_config'] == replayed['preprocessing_config']
assert config['preprocessing_config']['name'] == 'scan_intensity_crop'
assert audit['image_size'] == config['image_size']
assert audit['manifest_sha256'] == config['manifest_sha256'] == replayed['manifest_sha256']
assert audit['code_sha256'] == config['code_sha256']
comparison = {}
for unit in ('slice', 'scan', 'patient'):
    left = root / 'pilot' / f'early_stop_{unit}_predictions.csv'
    right_name = 'prediction_patient_predictions.csv' if unit == 'patient' else f'{unit}_predictions.csv'
    right = root / 'replay' / right_name
    # Slice/scan records include wall-clock inference timing: compare prediction
    # identity and probabilities separately from measured runtime fields.
    with left.open(newline='') as stream:
        left_rows = list(csv.DictReader(stream))
    with right.open(newline='') as stream:
        right_rows = list(csv.DictReader(stream))
    for rows in (left_rows, right_rows):
        for row in rows:
            for key in list(row):
                if 'seconds' in key or 'latency' in key:
                    row.pop(key)
    assert left_rows == right_rows, f'{unit} checkpoint prediction replay differs'
    comparison[unit] = {'identical_predictions': True, 'rows': len(left_rows)}
seals = frozen_seals(Path(sys.argv[2]))
result = {'status': 'complete', 'kind': 'one_epoch_gpu_runtime_check_not_final_evaluation',
          'job_id': os.environ['SLURM_JOB_ID'], 'node': os.environ.get('SLURMD_NODENAME'),
          'gpu': torch.cuda.get_device_name(0), 'output': str(root),
          'preprocessing_config': config['preprocessing_config'], 'image_size': config['image_size'],
          'epochs_completed': trained['epochs_completed'], 'checkpoint_format_version': 3,
          'checkpoint_sha256': hashlib.sha256((root / 'pilot/best.pt').read_bytes()).hexdigest(),
          'frozen_files_verified': len(seals), 'prediction_replay': comparison,
          'inner_early_stop_metrics': trained['metrics'], 'resources': trained['resources'],
          'limitations': ['One epoch verifies execution; it does not establish an accuracy improvement.',
                          'Early-stop patients select and evaluate this pilot; no independent final-test result.',
                          'Foreground thresholding is a head proxy; real previews require visual review.']}
write_json(root / 'summary.json', result)
print(json.dumps(result, indent=2), flush=True)
PY
