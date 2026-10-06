# Scan-consistent input standardization

This opt-in implementation addresses input brightness and head position before
changing the backbone. It is an experiment, not an established accuracy gain.
All source code belongs to `recognition/ConvNeXt_ADNI_s4983666/` in the
`PatternAnalysis-2026` checkout. The older `comp3710project` checkout is not updated.

## Exact processing contract

The default `--preprocessing none` preserves historical grayscale conversion,
conditional bilinear resize and `(uint8 / 255 - 0.5) / 0.5`. Existing model names,
weights, augmentation algorithms, sampling, optimization, checkpoint selection,
frozen manifests and completed experiments retain their original meanings.

| Profile | Brightness | Geometry |
|---|---|---|
| `none` | Original fixed pixel mapping | Historical resize if needed |
| `scan_intensity` | Foreground percentile mapping shared by scan | Historical resize if needed |
| `scan_crop` | Original fixed pixel mapping | Scan-consistent integer crop / black padding |
| `scan_intensity_crop` | Foreground percentile mapping shared by scan | Scan-consistent integer crop / black padding |

1. Decode verified source bytes, apply EXIF orientation, and convert to grayscale.
2. Group the supplied complete scan slices by scan ID, with consistent owner and
   native image size. A patient with multiple scans gets separate scan statistics.
3. Foreground is the **head proxy** `uint8 > 16` by default, excluding dark background.
   No pretrained segmentation or anatomical brain extraction is performed. Its
   union bounding box conservatively includes all above-threshold pixels in the
   scan; JPEG artifacts/nonbrain content can enlarge it.
4. For intensity profiles, pool native foreground pixels across that scan's
   supplied slices. Use count-weighted nearest-rank percentiles 1 and 99 by default.
   Map `[low, high]` linearly to `[0,255]`, clip outliers and round to uint8 once.
   Pixels at/below the foreground threshold become zero. Empty/constant
   foreground scans use the recorded identity-foreground behavior, avoiding
   division by zero. This can alter relevant contrast; visual QA is required.
5. For crop profiles, use the scan-union bbox center and one fixed native-pixel
   window for every slice. There is no rotation, anatomical rescaling,
   interpolation or independently recentered slice. Out-of-frame areas are black
   padding. Every detected foreground pixel plus the requested margin must fit;
   otherwise the loader refuses the scan, rather than clipping or excluding it.
6. Apply optional training-only augmentation **after** deterministic preprocessing,
   then the fixed uint8-to-[-1,1] tensor mapping. Evaluation never augments inputs.
   Optional augmentation can still clip boundaries; use `--augmentation none`
   for this initial preprocessing comparison.

These percentiles are fixed **per-input scan** operations, not estimated
population mean/std or a fitted histogram reference. They use no diagnoses and
never pool patients, scans or roles. At inference a complete scan is required;
changing its supplied slice set changes the statistics. Automatic crop dimensions
are the only settings fitted across cases: training-scan maximum union extents
plus twice the margin, rounded up to multiples of 32 (and the model's minimum).
Held-out cases cannot refit or enlarge that window. Explicit dimensions may be
chosen prospectively and are checked against training foreground/margins.

Thresholds, percentiles and margins are proposed configuration constants, not
parameters justified by real-data results. Tune only within development training
and inner selection. Keep calibration/final-test images out of preprocessing
selection and previews. Source-integrity auditing continues to verify all original
sources; verification is not model scoring or preprocessing fitting.

## Review real inputs before training

Run in the Rangpur project environment; use a fresh output directory:

```bash
cd "$HOME/comp3710/comp3710-adni"
python audit_preprocessing.py \
  --data-root /home/groups/comp3710/ADNI \
  --splits-dir "$HOME/comp3710/adni_splits_v1" \
  --output "$HOME/comp3710/runs/preprocessing_review_fold01" \
  --fold 1 --preprocessing scan_intensity_crop --max-images 6
```

This trains no model. It exports only training/early-stop inputs, with per-scan
JSON/CSV, original + foreground/crop overlays, processed inputs and whole-scan
foreground histograms. `summary.json` records the resolved input dimensions,
training-only crop fit and source fingerprints. Inspect empty/constant foreground
counts, missed dark foreground, enlarged boxes from artifacts, altered contrast,
padding and genuine anatomy before freezing settings. No outer-validation,
calibration or final-test previews are generated.

For crop profiles, `--crop-height 0 --crop-width 0` (default) resolves dimensions
from training scans. Explicit `--crop-height H --crop-width W` must specify both.
Defaults: `--foreground-threshold 16 --intensity-lower-percentile 1
--intensity-upper-percentile 99 --crop-margin 8`. The legacy `--image-height` and
`--image-width` govern noncrop profiles; crop profiles use their resolved crop
shape and log it explicitly. A crop failure requires a reviewed, prospectively
larger window and a fresh run; no silent fallback or held-out refit is implemented.

## Train and reproduce

After the input review, an opt-in inner development run is:

```bash
python train.py \
  --data-root /home/groups/comp3710/ADNI \
  --splits-dir "$HOME/comp3710/adni_splits_v1" \
  --output "$HOME/comp3710/runs/convnext_lite_preprocessed_inner_fold01" \
  --model convnext_lite --fold 1 --inner-only \
  --preprocessing scan_intensity_crop --augmentation none \
  --epochs 30 --patience 5 --batch-size 32 --lr 0.0001 --weight-decay 0.05 \
  --workers 2 --threads 2 --device cuda
```

Thirty is a proposed cap, not a course requirement. This entry point retains its
existing minimum-early-stop-**scan**-log-loss selector and patience policy, unlike
the separate v2 feature suite's slice-loss primary selector and fixed 30 epochs.
Do not attribute cross-suite differences solely to preprocessing. For an isolated
four-profile comparison, train all four afresh with identical optimizer, seeds,
sampling, augmentation and selection policy, and record actual input dimensions.
The crop and noncrop branches intentionally have different geometric operations;
if canvas dimensions differ, report that factor and resource scope explicitly.
Keep comparable input canvases prospectively where real foreground permits it.

```bash
python predict.py \
  --checkpoint "$HOME/comp3710/runs/convnext_lite_preprocessed_inner_fold01/best.pt" \
  --data-root /home/groups/comp3710/ADNI \
  --splits-dir "$HOME/comp3710/adni_splits_v1" \
  --output "$HOME/comp3710/runs/convnext_lite_preprocessed_inner_fold01_replay"
```

Preprocessed runs use checkpoint format 3, storing the exact algorithm and
resolved crop dimensions. Prediction does not offer a preprocessing override;
it reconstructs scan statistics from the checkpoint's verified manifest role.
Historical formats 1/2 retain their original preprocessing. Unknown/incomplete
new configurations fail before source loading. The feature diagnostic and exact
shift follow-up entry points also decode format 3; their shifts perturb the
already standardized tensor without recomputing normalization. The historical
light-transform audit is skipped explicitly for these new inputs and replaced
by scan preprocessing QA, so it cannot misrepresent the model's actual input.

## Logs and boundaries

`config.json`/`metrics.json` record preprocessing settings, training-only crop fit,
per-scan QA summaries and preparation time. The preparation timing covers scan
statistics and loader setup separately from model training, excluding QA plotting.
Every new run exports `train_preprocessing_scans.json/.csv`, matching early-stop
files and private preview PNGs. Optional outer evaluation exports its own role
parameters after checkpoint selection. Failure grids reuse the evaluated complete
scan parameters rather than recomputing percentiles from a few selected mistakes.

GPU inference benchmarks still measure device-resident forward passes only;
they exclude source decoding, scan-statistic preparation, preprocessing and host
transfer. They are not end-to-end clinical latency. Initial statistics require
reading all supplied scan slices once; the normal loader rechecks each source
slice digest when it is used. Original JPEGs and frozen manifests are never edited.
Keep real logs/previews on Rangpur and archive to the authorized private Drive;
do not retain experiment evidence locally or commit it. No final-test scoring,
calibration fitting or accuracy improvement is claimed by this implementation.

Validation uses synthetic sources only: brightness/translation equivalence,
within-scan position preservation, no interpolation in crop mode, foreground
clipping refusal, training-only fitting, empty/constant input handling, source
binding, strict checkpoint decoding and actual CPU training/prediction replay.

```bash
PYTHONPATH=tests python -m unittest test_preprocessing -v
```

Pillow primitives are documented at
https://pillow.readthedocs.io/en/stable/reference/Image.html .

## Single-allocation GPU execution check

From the Rangpur login node, submit the new code with:

```bash
cd "$HOME/comp3710/comp3710-adni"
bash slurm/submit_preprocessing_check.sh --dry-run
bash slurm/submit_preprocessing_check.sh
```

One A100 allocation requests four CPUs and one hour (override with
`ADNI_PREPROCESSING_TIME_LIMIT`). It audits real fold-1 train/early-stop scans,
trains scratch `convnext_lite` for exactly one epoch with `scan_intensity_crop`
and no augmentation, and reloads the resulting format-3 checkpoint. It verifies
identical slice/scan/patient predictions, matching pipeline/source identities,
training-only crop dimensions, and all 22 frozen file seals. Training performs the
existing inference resource profile; replay skips the duplicate profile.

Outputs remain on Rangpur in
`$HOME/comp3710/runs/preprocessing_check_job<job_id>/{input_review,pilot,replay}`.
The root `summary.json` is written only after all three stages and replay checks
succeed. Logs are under `runs/slurm_logs/preprocessing_check_<job_id>.out/.err`.
No calibration/final-test scoring or independent accuracy estimate is produced;
this is a runtime check, not a completed preprocessing comparison. A failed job
must be inspected before submitting formal experiments.

## Configurable inner-development GPU training

Use the new launcher for multi-epoch experiments; the older
`submit_preprocessing_check.sh` remains an exactly-one-epoch runtime check.
This launcher submits one model/profile/fold per allocation. It does not start
all four controls automatically. Its defaults extend the validated pilot:

| Control | Default |
|---|---|
| Model / preprocessing | Scratch `convnext_lite` / `scan_intensity_crop` |
| Epoch cap / patience / min delta | 30 / 5 / 0.0001 |
| Optimizer / LR / weight decay | AdamW / 0.0001 / 0.05 |
| Batch size / base seed | 32 / 3710 (effective fold-1 seed 3711) |
| Schedule / augmentation / sampler | Constant / none / slice-uniform |
| GPU / CPUs / workers / Torch threads | 1 / 4 / 2 / 2 |
| Wall-clock allocation | 02:00:00 |
| Selection / evaluation | Minimum early-stop scan log loss / inner early-stop only |
| Precision / loss | FP32 / training-class-weighted BCE, no label smoothing |

Thirty is a proposed cap; patience can stop training earlier. An earlier best
epoch is valid. For a fixed thirty-epoch comparison without patience stopping,
set `--epochs 30 --patience 30` consistently in every branch. Existing training
semantics, models and checkpoint formats are unchanged; AMP/label smoothing are
not added by this launcher. Outer validation, calibration and final-test scoring
are not available in this launcher.

First inspect **all** effective commands from the Rangpur login node:

```bash
cd "$HOME/comp3710/comp3710-adni"
bash slurm/submit_preprocessing_train.sh \
  --model convnext_lite --fold 1 \
  --preprocessing scan_intensity_crop --augmentation none \
  --epochs 30 --patience 5 --batch-size 32 \
  --lr 0.0001 --weight-decay 0.05 --lr-schedule constant \
  --seed 3710 --workers 2 --threads 2 --time-limit 02:00:00 \
  --dry-run
```

Dry-run prints the submission, training and replay commands, with the actual job
number represented by `<SLURM_JOB_ID>`. It submits no job, reads no dataset and
creates no output/log directories. Remove only `--dry-run` to submit the same
settings. Do not copy the printed `Train:` command onto the login node; it belongs
inside the allocated GPU job. `bash slurm/submit_preprocessing_train.sh --help`
lists every supported option. Unknown options or invalid warmup/CPU/optimizer/
crop settings are rejected before submission.

Examples that change one explicit control (preview before submitting):

```bash
bash slurm/submit_preprocessing_train.sh --lr 0.0003 --dry-run
bash slurm/submit_preprocessing_train.sh --epochs 30 --patience 30 --dry-run
bash slurm/submit_preprocessing_train.sh --lr-schedule warmup_cosine --warmup-epochs 2 --dry-run
for profile in none scan_intensity scan_crop scan_intensity_crop; do
  bash slurm/submit_preprocessing_train.sh --preprocessing "$profile" --dry-run
done
```

These are syntax examples, not claims that any alternative improves accuracy.
Only training receives optional augmentation. Crop profiles fit geometry on
training scans; `--image-height 240 --image-width 256` describe noncrop canvases,
not a forced resize of cropped inputs. The validated real fold-1 crop resolved
224x192; other folds/settings must log their actual resolved dimensions. Keep
optimizer/epoch/sampler/augmentation/seed/selection settings identical for a
preprocessing comparison. Crop and noncrop geometry/canvas differences remain
part of the intervention and must be reported.

Each allocation activates the configured Conda environment and calls
`slurm/preprocessing_train.py`. Training performs the existing mandatory source/
manifest audit, exports input QA, saves `best.pt` and all classification/confidence/
resource reports. Prediction reloads the selected checkpoint with exactly the
same batch/worker/thread settings and automatically uses its stored preprocessing.
The verifier accepts `1 <= best_epoch <= epochs_completed <= epochs_limit`,
checks the selection loss against all recorded epochs and checks exact slice/scan/
patient predictions, requested controls, source identities and frozen-file seals.
Patient aggregation is explicitly unavailable if a role has mixed diagnoses.
There is no single-epoch or last-epoch assumption.

Outputs are kept on Rangpur:

```text
$HOME/comp3710/runs/preprocessing_train_<profile>_fold<NN>_job<id>/
  launch.json      # Requested options, exact commands, launcher hashes and GPU
  train/          # config.json, metrics.json, history.csv, best.pt and plots
  replay/         # Reloaded predictions, metrics and plots
  summary.json    # Written last, only if training and verification pass
```

The directory is fresh per Slurm job, so completed experiments are never
reused/overwritten. `summary.json` records the completed and best epochs separately;
`stopped_before_epoch_cap` reports a shorter run. A failed stage stops execution
and leaves no successful root summary. Logs are
`runs/slurm_logs/preprocessing_train_<id>.out/.err`. Check them with:

```bash
squeue --me
# Replace JOB_ID with the number printed by the submit command.
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
 tail -f "$HOME/comp3710/runs/slurm_logs/preprocessing_train_JOB_ID.out"
```

`ADNI_DATA_ROOT`, `ADNI_SPLITS_DIR` and `ADNI_RUNS_ROOT` supply path defaults;
explicit `--data-root`, `--splits-dir` and `--runs-root` override them. All must
be absolute, with run storage separate from source/data/manifests.
`ADNI_CONDA_ROOT`/`ADNI_CONDA_ENV` default to `$HOME/miniconda3`/`torch`.
`ADNI_PYTHON` optionally sets the actual Python executable in both submission
and execution. For a local preview, point it at the canonical project `.venv`;
actual training requires a real Slurm allocation and available CUDA.

Launcher verification is synthetic: previews without side effects, exact option
forwarding, invalid-option rejection, and real CPU train/replay under both
historical and scan-standardized checkpoint formats. The multi-epoch fixture
forces early stopping after two epochs with first-epoch weights selected, checking
the failure mode that the original one-epoch summary could not handle. This does
not constitute a new real-data training result.
