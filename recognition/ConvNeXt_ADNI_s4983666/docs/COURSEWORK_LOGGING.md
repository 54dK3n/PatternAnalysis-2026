# Coursework-aligned experiment logging

The [v2.1 assignment](https://edstem.org/au/courses/38242/discussion/3600790)
and [staff ADNI/ConvNeXt clarification](https://edstem.org/au/courses/38242/discussion/3615608?answer=7856998)
were checked on 3 October 2026. ConvNeXt starts from random weights. A smaller
ConvNeXt retains Hard eligibility when its core design is retained and explained.
No fixed 15- or 30-epoch requirement was found for this project.

## Models and budgets

`--model convnext_lite` constructs the separate `convnext_lite_v1` architecture:
depths `(2, 2, 6, 2)` and channels `(48, 96, 192, 384)`. It inherits the local
ConvNeXt block, stem, downsampling and head. A real fold-1 inner pilot is recorded in private local outputs; its development
results do not establish final-test performance. Existing `cnn`, `small_cnn`, `convnext` and `convnext_tiny` aliases
retain their original architecture, initialization, names and state-dict shapes.

The original default of 30 epochs and scan-log-loss checkpoint selection remains
unchanged for compatibility. `--epochs` is an experiment maximum; `epochs_completed`
and `best_epoch` record what actually happened. Declare the cap, early-stopping
policy and other hyperparameters before the comparison. Training uses only the
frozen fold's training slices; selection uses its early-stop patients.

Use `--inner-only` while exploring implementations or hyperparameters. It reloads
the chosen checkpoint and scores early-stop data, without constructing an outer
validation loader. Logs explicitly flag that the scored patients were also used
for checkpoint selection. Those scores are exploratory and optimistic. Omit the
flag for the predeclared outer-fold comparison, scored once after selection.
Calibration and final-test images are never scored by either development mode.
Source integrity verification still checks all original sources.

## Required metric records

New result records use `metrics_format_version: 3`. Checkpoint format remains 2;
formats 1 and 2 remain readable. Old experiment artifacts are never rewritten.

| File | Contents |
|---|---|
| `config.json` | Model/version, random initialization, actual epoch cap, early-stopping rule, frozen-manifest digest, transformation, seed, device and package versions, confidence rule and profiling settings |
| `history.csv` | Weighted train BCE, online train accuracy/F1/AUROC, early-stop slice and scan metrics, elapsed epoch/training times, selected checkpoint flag |
| `epoch_metrics.json` | Full per-epoch train-online and early-stop metrics, including per-class precision/recall/F1 and confusion matrices |
| `metrics.json` | Selected-checkpoint classification, raw confidence/rejection, patient aggregation status, resource measurements and failure-example count |
| `val_*_predictions.csv` | Auditable outer-validation slice/scan/patient probabilities; `early_stop_*` instead in inner-only mode |
| `*_reliability.csv`, `*_risk_coverage.csv` | Reconstructable calibration bins and empirical risk--coverage points at each observed confidence tie |
| `learning_curves.png`, `*_confidence.png` | Training/early-stop curves, slice reliability, confidence by correctness and descriptive risk--coverage |
| `*_failure_cases.json`, `*_failure_cases.csv`, `*_failures.png` | Up to five actual wrong slice predictions, including FP/FN when present, with patient/scan/slice IDs and probabilities |

Accuracy, per-class precision/recall, macro-F1, AUROC, balanced accuracy, log loss
and confusion matrices are written under `metrics.slice` and `metrics.scan`.
Staff recommend slice-primary evaluation for our 2D task. Patient separation
remains mandatory regardless of evaluation unit. `metrics.patient` averages all
slices of each patient only if each patient has a consistent diagnosis. If any
longitudinal patient has mixed diagnoses, patient metrics are `null` and the
reason is logged, rather than silently assigning a diagnosis or dropping cases.

Training metrics use the already-computed training-mode logits, including
augmentation/dropout when configured and parameter updates across the epoch.
They are labelled `online_training_mode_augmented_when_configured`; they are
not an extra deterministic evaluation pass over the training set. No additional
training forward pass or random draw is introduced by metric logging.

## Confidence and rejection

`coursework_report.confidence.<unit>` stores predicted-class ECE, AD Brier score,
15-bin reliability data, correct/incorrect confidence counts and risk--coverage.
Confidence is `max(p(AD), 1-p(AD))`; accuracy uses the fixed AD threshold 0.5.
ECE uses equal-width bins on [0, 1] weighted by sample counts. Confidence 1 falls
in the final bin. Risk--coverage accepts whole confidence-tie groups together,
avoiding an arbitrary ordering of equally confident examples. An empty accepted
set has coverage zero and accuracy/risk `null`.

`--reject-threshold` defaults to 0.8 and is a declared **raw-confidence** rule,
not a fitted/calibrated clinical operating point. It is unrelated to the course's
0.80 accuracy requirement. The record includes accepted count, referred count,
coverage, accepted accuracy/risk and full-coverage accuracy. Descriptive curves
must not be used to choose a threshold on outer validation or final test. Both
models should use a predeclared decision protocol and identical patient splits.

No temperature fitting or empirical threshold optimization is implemented by
this logging update. `calibration: not_fitted` remains accurate. Dedicated
non-test calibration, final refit and one final-test evaluation remain future
work. `final_test_target.status` is explicitly unassessed on development data;
accepted-only accuracy cannot establish the full-coverage test target.

Failure examples are selected from real mistakes, prioritizing confident FP/FN
and then distinct patients. If fewer than three errors exist, only the actual
errors are exported; examples are never invented. Source images are rechecked
against their frozen file digests and use deterministic evaluation preprocessing.
Clinical interpretation and the engineering recommendation remain the owner's
responsibility. Failure images/patient identifiers belong in private outputs.

## Resource measurements

On CUDA, the selected model is benchmarked separately at batches 1 and 64 after
10 warmups and for 100 timed forwards. Inputs are synthetic tensors resident on
the actual device at the configured image shape; they contain no patient labels.
Each timed forward is synchronized. Logs contain batch latency mean/population
standard deviation, amortized milliseconds per slice, throughput and peak CUDA
allocated memory per batch. Optimizer state is released before this benchmark.
Forward-only latency excludes loading and host/device transfer; batch-64 per-slice
time must not be called single-request latency. Allocation is PyTorch allocated
memory, not total system VRAM usage or reserved memory.

Training and early-stop evaluation peaks are tracked separately and include the
optimizer state present in those phases. GPU identity and actual software versions
are recorded. CPU runs write CUDA memory as `null` and normally mark the separate
benchmark `not_measured_cpu`; use `--profile-on-cpu` to measure CPU latency.
`--skip-inference-profile` records an explicit skip. Reduced warmup/repeat values
are flagged as an incomplete publication protocol. CUDA OOM at either batch is
recorded as a resource limitation, with no invented latency.

## Controlled pilot example

This is a proposed experiment configuration, not a coursework epoch rule:

```bash
python3 train.py \
  --data-root /home/groups/comp3710/ADNI \
  --splits-dir "$HOME/comp3710/adni_splits_v1" \
  --output "$HOME/comp3710/runs/convnext_lite_inner_fold01" \
  --model convnext_lite --inner-only --fold 1 \
  --epochs 30 --patience 5 --lr 0.0001 --weight-decay 0.05 \
  --augmentation none --reject-threshold 0.8
```

Run the baseline on the same frozen fold with its predeclared optimizer settings
and a fresh output directory. After inner exploration, freeze choices before
outer-fold evaluation. `predict.py` defaults to the checkpoint's evaluation mode;
`--role early_stop` or `--role val` can explicitly select a development role.
Legacy checkpoints default to their original outer-validation behavior.

The five named coursework files now exist. `modules.py` exports the PyTorch
classes/factory; `dataset.py` is a compatibility facade, while normal Python
imports resolve the `dataset/` package's lazy loader interface. Implementations
remain in their established packages. Confirm at check-off whether the teaching
team wants component definitions literally inside the facades.

## Controlled experiment extensions

Optional class/patient-balanced training, isolated pixel shifts/gamma and a warmup/cosine schedule are now available. Historical defaults and none/light checkpoint transform contracts are preserved. The [predeclared suite](EXPERIMENT_SUITE.md) records exact case settings and remaining feasibility acceptance work. LR is logged each epoch; sampling expectations and algorithm versions are recorded in config. Formal calibration/refit/final-test support is still future work.
