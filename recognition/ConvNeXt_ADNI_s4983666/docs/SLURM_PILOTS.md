# Rangpur inner-only GPU pilots

The observed server source root is `$HOME/comp3710/comp3710-adni`.
Frozen manifests are in `$HOME/comp3710/adni_splits_v1`; existing experiments
are in `$HOME/comp3710/runs`. Do not regenerate manifests or reuse old output
folders. Install the complete source bundle, including its new evaluation and
utility modules, rather than copying only `train.py`.

## Source update

Upload the generated archive and its checksum from your Mac to
`s4983666@rangpur.compute.eait.uq.edu.au:~/comp3710/`. On the cluster:

```bash
cd "$HOME/comp3710"
sha256sum -c coursework_update_20261005.tar.gz.sha256
cd comp3710-adni
tar --exclude=__pycache__ -czf "../code_before_update_$(date -u +%Y%m%dT%H%M%SZ).tar.gz" \
  ./*.py README.md config dataset engine evaluation models tests utils docs
tar -xzf ../coursework_update_20261005.tar.gz
sha256sum -c SERVER_UPDATE_SHA256.txt
grep -nE 'convnext_lite|inner-only' models/registry.py engine/training.py
```

Stop if a checksum or backup command fails. The archive contains source,
source documentation, configuration and tests. It does not contain ADNI images,
frozen manifests, checkpoints, environments, local AI logs or experiment outputs.
The source backup is local to the server and is not a Git commit.
Extraction updates matching source filenames. Other server files are retained;
existing experiments and manifests are outside the extraction directory.

## Submission

Run the wrapper from the login node, even if your current working directory is
`runs/`. The wrapper derives the source root from its own path:

```bash
bash "$HOME/comp3710/comp3710-adni/slurm/submit_pilot.sh" convnext_lite --dry-run
bash "$HOME/comp3710/comp3710-adni/slurm/submit_pilot.sh" convnext_lite
squeue --me
```

For a newly logged CNN control, submit `cnn` instead. Each invocation submits
exactly one job. Neither pilot is a final comparison or final-test evaluation.
The CNN control has the same patients, resolution, seed base, maximum epochs,
checkpoint selection, no augmentation and evaluation protocol as the Lite pilot;
its optimizer learning rate/decay retain the existing CNN defaults.

| Setting | Default |
|---|---|
| Account / partition | `comp3710` / `comp3710` |
| Allocation | One GPU and four CPU cores |
| Wall-time request | Two hours; runtime has not been measured for this pilot |
| Environment | `$HOME/miniconda3`, environment `torch` |
| Data | `/home/groups/comp3710/ADNI` |
| Pilot | Fold 1, scratch initialization, inner-only |
| Budget | Maximum 30 epochs, patience 5, batch 32 |
| Seed | Base 3710; training records base plus fold |
| Resolution / augmentation | Native 240 x 256 grayscale / none |
| CNN optimizer | AdamW, LR 0.001, decay 0.0001 |
| Lite optimizer | AdamW, LR 0.0001, decay 0.05 |
| Loading / CPU threads | Two workers / two PyTorch threads |
| Profiling | CUDA, batches 1 and 64, warmup 10, 100 repeats |

The 30-epoch cap is a declared experiment budget, not a coursework rule.
Early stopping can finish sooner. The predeclared raw-confidence rejection
threshold is 0.8; it is separate from the course accuracy target and is not a
fitted calibration rule. Early-stop scores reuse checkpoint-selection patients
and must be labelled exploratory. Source integrity checks still verify protected
sources, but calibration/test/outer-validation images are not model-scored.

Override settings before submission, for example:

```bash
ADNI_EPOCHS=1 bash slurm/submit_pilot.sh convnext_lite
ADNI_TIME_LIMIT=04:00:00 bash slurm/submit_pilot.sh convnext_lite
```

Supported environment overrides: `ADNI_DATA_ROOT`, `ADNI_SPLITS_DIR`,
`ADNI_RUNS_ROOT` (absolute paths), `ADNI_CONDA_ROOT`, `ADNI_CONDA_ENV`,
`ADNI_TIME_LIMIT`, `ADNI_FOLD`, `ADNI_EPOCHS`, `ADNI_PATIENCE`,
`ADNI_BATCH_SIZE`, `ADNI_WORKERS`, `ADNI_THREADS`, `ADNI_SEED`, `ADNI_LR`,
`ADNI_WEIGHT_DECAY`, and `ADNI_AUGMENTATION` (`none` or `light`). Changing a
setting defines a new experiment. Keep settings fixed for declared comparisons.
No package installation or pretrained download is performed. The job prints the
actual Python/package versions and GPU before training. Missing CUDA fails
explicitly. `run_pilot.sh` refuses execution without a Slurm job allocation.

## Logs and completion

Standard logs: `$HOME/comp3710/runs/slurm_logs/<model>_<jobid>.out` and `.err`.
Output: `$HOME/comp3710/runs/<model>_inner_fold0<fold>_job<jobid>/`.
The training output includes the configuration, epoch history, learning curves,
selected checkpoint, slice/scan/patient predictions, classification/confidence/
rejection records, failure examples and measured resource profiles. See
[the metric protocol](COURSEWORK_LOGGING.md) for definitions and limitations.

Inspect both Slurm status and error logs. A completed `metrics.json` marks the
end of this pipeline. Timeout, source-audit, package or CUDA failures may leave
partial artifacts; those do not establish a completed experiment. Retry by
submitting a new job, retaining the incomplete run for diagnosis. Do not tune
using outer validation or final test. Formal refit/calibration/testing is still
separate future work.
