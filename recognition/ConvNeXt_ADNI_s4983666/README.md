# COMP3710: ADNI AD/NC Classification

Patient-isolated MRI classification comparing Alzheimer's disease (AD) with normal controls (NC).

The implementation audits the course JPEG dataset, creates five-fold cross-validation manifests, trains either a small CNN or a ConvNeXt-Tiny on one development fold, and reproduces held-out predictions from a saved checkpoint. Both architectures are implemented in PyTorch and initialized from random weights. Three complete five-fold CNN repetitions have been reviewed locally. An initial unaugmented ConvNeXt fold has also completed; the optional training-augmentation variant remains to be evaluated on ADNI. Final refitting, confidence calibration, and final-test evaluation remain future work.

This project lives at `recognition/ConvNeXt_ADNI_s4983666/` in the course fork, [54dK3n/PatternAnalysis-2026](https://github.com/54dK3n/PatternAnalysis-2026), on `topic-recognition`. Its package layout and current implementation were migrated from the standalone development project, [54dK3n/comp3710-adni](https://github.com/54dK3n/comp3710-adni). Final coursework submission still requires the course pull request and accompanying report; this migration does not claim completion of the remaining report or evaluation requirements.

## Repository layout

The root contains executable entry points, coursework-facing model/dataset interfaces and essential repository metadata (`README.md` and `.gitignore`). Implementations, dependencies, documentation, and tests have separate directories:

```text
ConvNeXt_ADNI_s4983666/
├── modules.py               # PyTorch model interface backed by models/
├── dataset.py               # Coursework compatibility facade
├── train.py                 # Training CLI entry
├── predict.py               # Checkpoint prediction CLI entry
├── adni_splits.py           # Audit/prepare/verify CLI entry
├── models/
│   ├── cnn.py               # Original small CNN
│   ├── convnext.py          # Grayscale ConvNeXt-Tiny and separate Lite preset
│   └── registry.py          # CLI aliases and versioned checkpoint architectures
├── dataset/
│   ├── splits.py            # Source audit and patient-level split implementation
│   ├── manifests.py         # Mandatory verification and frozen-fold loading
│   ├── slices.py            # Digest-checked grayscale slice dataset
│   ├── augmentation.py      # Versioned, training-only geometric transforms
│   └── loaders.py           # Role-specific loaders and worker seeds
├── engine/
│   ├── training.py          # Optimization, early stopping, and outer validation
│   └── prediction.py        # Checkpoint validation and reproduced predictions
├── evaluation/
│   ├── metrics.py           # Scan aggregation and classification metrics
│   └── inference.py         # Evaluation-mode inference and timing
├── utils/
│   ├── runtime.py           # Devices, reproducibility, and environment records
│   └── artifacts.py         # Fingerprints, CSV/JSON, protected outputs, and plots
├── config/
│   ├── requirements.txt
│   └── requirements-train.txt
├── docs/                    # Data protocol, smoke records, and local references
├── tests/                   # Synthetic unit, integration, and CLI tests
└── outputs/                 # Ignored local data/manifests/experiment artifacts
```

Each Python package also has an `__init__.py`. Component implementations remain in these packages; `modules.py` and `dataset.py` now expose coursework-facing interfaces without copying implementations. Normal `dataset` imports resolve the package, which exposes its loader interface lazily so source audits do not require PyTorch. Existing shell commands using the three entry scripts still work. Python callers should import implementations from their new packages, such as `models.create_model` and `dataset.manifests.load_fold`.

The existing split and verification algorithms are unchanged, so frozen manifests remain usable. The target's M0 source audit is retained in the same package. Both models preserve their parameter names and computation. Version 1 CNN/ConvNeXt checkpoints remain supported; new version 2 checkpoints also record the complete training-augmentation configuration. Prediction always uses fixed validation processing.

## Setup

Use Python 3.10 or newer in your project environment (required by the pinned Pillow version):

```bash
git clone --branch topic-recognition https://github.com/54dK3n/PatternAnalysis-2026.git
cd PatternAnalysis-2026/recognition/ConvNeXt_ADNI_s4983666
python3 -m pip install -r config/requirements.txt
```

In an existing checkout, run the commands below from `recognition/ConvNeXt_ADNI_s4983666/`, which contains `train.py` and `README.md`.

For training, install the additional dependencies in your project environment:

```bash
python3 -m pip install -r config/requirements-train.txt
```

The requirements pin Pillow 12.3.0 in `config/requirements.txt`, with PyTorch 2.6.0 and Matplotlib 3.9.4 added by `config/requirements-train.txt`. Local CPU verification used Python 3.12.14 and Pillow 12.3.0. GPU training requires a PyTorch CUDA build compatible with the server's driver; follow the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) or the course environment instructions. The code does not require torchvision or scikit-learn. Every training run records its actual package versions, CUDA build, and device information in `config.json`.

The completed server baseline used Python 3.11.15, PyTorch 2.13.0, Pillow 12.3.0 and Matplotlib 3.11.1 on an NVIDIA A100-PCIE-40GB, with CUDA build 13.0. These recorded server versions differ from the local CPU reference environment above.

## Local workspace files

Local experiment outputs, datasets, model weights, the coursework PDF and `.venv/` remain outside version control. Outputs include private split manifests, prediction records, checkpoints and local review reports; keep them outside all public commits. Historical paths inside experiment configurations remain unchanged because they record where those experiments actually ran. Create a new virtual environment on each machine and install its dependencies locally.

The dataset must be obtained separately through the course's authorised access. Expected layout:

```text
ADNI/
├── meta_data_with_label.json
└── AD_NC/
    ├── train/
    │   ├── AD/
    │   └── NC/
    └── test/
        ├── AD/
        └── NC/
```

Source folder names record the supplied split, not the split to use for experiments. The supplied train/test folders share patients; the generated manifests replace those assignments without changing source images.

## Source audit and preserved patient splits

The existing M0 source audit remains available after moving its implementation
into `dataset/splits.py`:

```bash
python3 adni_splits.py audit \
  --data-root /home/groups/comp3710/ADNI \
  --output runs/m0_audit
```

It prints the observed tree, supplied split/class counts, image sizes/channels,
and filename-to-metadata patient examples, then writes `config.json` and
`metrics.json` with `report_type: data_audit`. Existing output directories are
never overwritten. `--expected-slices` defaults to 20; `--tree-depth` and
`--examples` control the displayed inventory. An exit code of 1 means an input
or source validation failure. Exit code 2 and `status: blocked` describe overlap
in the **original supplied folders only**; they do not invalidate the separately
verified patient manifests used by training. Exit code 0 means the original
folder boundary also passed the audit.

The current training protocol already isolates patients using frozen
`adni_splits_v1` manifests. Migration preserves those assignments, seeds, image
hashes, and verification logic; it does not regenerate a split. Continue to use
`verify` on the existing server manifests before training. Images remain in
their original folders, while the manifest rows determine their experimental
roles. JPEG filenames encode scan and slice IDs; patient IDs come from the
matching metadata filename.

This migration transfers the existing CNN, ConvNeXt-Tiny, optional training
augmentation, prediction, documentation, and tests into the package layout.
It preserves the target's local outputs and environment. It does not perform
new ADNI training or final-test evaluation, or implement the remaining model,
calibration, report, or Slurm milestones. Historical experiment records retain
the paths where those experiments actually ran.

## Prepare and verify the split

Run from this repository on the course server:

```bash
python3 adni_splits.py prepare \
  --data-root /home/groups/comp3710/ADNI \
  --output "$HOME/comp3710/adni_splits_v1" \
  --folds 5 \
  --seed 3710

python3 adni_splits.py verify \
  --data-root /home/groups/comp3710/ADNI \
  --output "$HOME/comp3710/adni_splits_v1"
```

`prepare` refuses to overwrite an existing split or write into the source dataset. If the split already exists, use `verify`; documentation and message translations do not require a new split.

The fixed default allocation is 70% development, 10% calibration, and 20% final test, measured by patients. Development patients are assigned to five folds. Each fold has separate `train.csv`, `early_stop.csv`, and `val.csv` files. The early-stopping subset contains approximately 10% of the four non-validation folds. Calibration and final-test patients never enter cross-validation.

Image paths in each CSV are relative to `--data-root`. Training code must read the manifests instead of inferring experimental roles from the original directories. The model label is AD=1 / NC=0; `metadata_label` preserves the source AD=2 / NC=0 encoding.

## Dataset audit reported from the course server

The project owner ran both commands and supplied their output. The reported verification result was `PASS`:

| Partition | Patients | Scans | JPEG slices |
|---|---:|---:|---:|
| Development | 476 | 1,050 | 21,000 |
| Calibration | 68 | 170 | 3,400 |
| Final test | 136 | 306 | 6,120 |
| Total | 680 | 1,526 | 30,520 |

The original folders shared 216 patients. The new manifests passed patient, scan, path, exact-file, and exact-decoded-pixel separation checks at the required boundaries. Source data matched the recorded manifests, and each development sample appeared in exactly one outer validation fold.

The complete server-generated manifests were subsequently copied into the ignored local `outputs/adni_splits_v1/` folder. Their checksums, frozen assignments, recorded identity/hash boundaries and correspondence with the five baseline runs were checked locally. This is not a local rerun against the original MRI pixels or metadata; those source files remain on the server. Local execution tests use synthetic data. The real images, metadata, generated patient manifests, and model weights are excluded from version control.

## Initial five-fold baseline results

These are development outer-validation results, using the fixed patient splits, seed base 3710, scan-mean aggregation and decision threshold 0.5. The one-epoch smoke check is excluded.

| Fold | Scan accuracy | Macro F1 | AUROC |
|---|---:|---:|---:|
| 1 | 83.41% | 0.8339 | 0.9136 |
| 2 | 56.50% | 0.5557 | 0.6737 |
| 3 | 87.77% | 0.8775 | 0.9241 |
| 4 | 82.33% | 0.8216 | 0.9064 |
| 5 | 75.38% | 0.7538 | 0.8167 |
| Mean | 77.08% | 0.7685 | 0.8469 |

The sample standard deviation of fold accuracy is 12.33 percentage points, showing substantial variation. Validation covers 476 patients, 1,050 scans and 21,000 slices. Independent calculations reproduce the reported metrics, and the predictions match the corresponding frozen manifests. These scores do not measure final-test performance. The additional repetitions with training seed bases 4710 and 5710 have also completed on the same frozen split. Their mean fold accuracies are 77.39% and 76.28%; the mean across all 15 runs is 76.92% (macro F1 0.7634, AUROC 0.8607). These repeated runs reuse the same patients and are not independent cohorts. All repetitions are retained in the comparison; no best seed is selected.

## Baseline model and evaluation

The baseline receives one grayscale slice and outputs an AD logit. Its four convolution blocks have 16, 32, 64, and 128 channels; each contains a 3×3 convolution, group normalization, ReLU, and 2×2 max pooling. A global spatial mean, dropout of 0.2, and a linear layer produce one logit. The model uses only PyTorch operations and starts from random weights.

Default input size is height 240 × width 256, matching the supplied images. Pixel values use the fixed transform `(pixel / 255 - 0.5) / 0.5`; no population mean or standard deviation is estimated. The initial baseline used no augmentation; the default remains `--augmentation none`. Each image is checked against its recorded file digest when loaded.

Training uses AdamW with learning rate 0.001, weight decay 0.0001, and binary cross-entropy with logits. AD's positive loss weight is `NC training slices / AD training slices`, computed only from the current fold's training manifest. Sampling is uniform over training slices: patients with more scans contribute more examples. This weighting choice should be discussed when comparing models.

For each scan, average its 20 slice-level AD probabilities and classify it as AD when the mean is at least 0.5. This aggregation and threshold are fixed before evaluation. These are uncalibrated scores; they should not be interpreted as clinical confidence estimates. Longitudinal scans retain their own diagnoses.

After each epoch, evaluate only the early-stopping patients. Save every strict minimum of their **scan-level log loss**. Stop after 5 epochs without an improvement greater than 0.0001 relative to the last patience-reset value, up to 30 epochs by default (an experiment choice, not a coursework requirement). Reload the selected checkpoint and evaluate the complete outer-validation set once. Report scan-level accuracy, balanced accuracy, per-class precision/recall/F1, macro F1, AUROC, log loss, and a confusion matrix. Historical tables above use scan-level results. New logs mark slice-level evaluation as primary for this 2D task following staff clarification, and retain scan/patient summaries as secondary. Slice-level examples are not independent patients.

## Select a model and training augmentation

Run `python3 train.py --help` to see all options. Model selection and augmentation are independent:

| CLI option | Meaning |
|---|---|
| `--model cnn` or `--model small_cnn` | Original small CNN; default is `small_cnn` |
| `--model convnext` or `--model convnext_tiny` | Complete ConvNeXt-Tiny |
| `--model convnext_lite` | Separate smaller scratch ConvNeXt; no real ADNI performance claim yet |
| `--inner-only` | Exploratory early-stop evaluation; never score outer validation |
| `--augmentation none` | Original deterministic images; default |
| `--augmentation light` | Random rotation and translation on training slices only |
| `--rotation-degrees 5` | Maximum absolute angle for the light profile; accepted range 0–15 degrees |
| `--translation-fraction 0.03` | Maximum translation along each axis as a fraction of that dimension; accepted range 0–0.1 |

The light profile samples angles uniformly in ±5 degrees and horizontal/vertical translations independently within ±3% by default. Bounds are fixed configuration values, not estimated using any patient data. Rotation and translation use one bilinear Pillow operation after fixed resizing and before intensity normalization. The image canvas stays the same size; transformed pixels outside it are discarded and exposed areas are filled with black. These conservative settings are an experimental choice, not a demonstrated improvement on ADNI. See [Pillow's transform reference](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.rotate).

Only a loader explicitly marked `train`, using development manifests, accepts nontrivial augmentation. Early-stop, outer-validation, calibration, test, and prediction roles reject it. Source bytes are checked before transformation, and source JPEGs are never modified. Each slice gets independently sampled transforms; no augmented files are created or redistributed across partitions.

Augmentation uses a private random generator. Identical seeds, worker counts, package versions, and execution settings reproduce its stream; different epochs get fresh draws. It does not consume the model's dropout/stochastic-depth random stream. `config.json` records the profile, bounds, interpolation, fill, scope, and algorithm version. A narrowly scoped worker setup also prevents a known macOS/Python 3.12+ shared-memory shutdown issue; it does not change Linux worker behavior or random seeds. See [PyTorch issue #153050](https://github.com/pytorch/pytorch/issues/153050). Workers must also remain fixed when comparing exact run reproduction; see [PyTorch data-loader seeding](https://docs.pytorch.org/docs/2.6/data.html#randomness-in-multi-process-data-loading).

For the next controlled ConvNeXt experiment, add the three augmentation options to the previous configuration and use a fresh output directory:

```bash
python3 train.py \
  --model convnext --augmentation light \
  --rotation-degrees 5 --translation-fraction 0.03 \
  --data-root /home/groups/comp3710/ADNI \
  --splits-dir "$HOME/comp3710/adni_splits_v1" \
  --output "$HOME/comp3710/runs/convnext_aug_fold01" \
  --fold 1 --epochs 30 --patience 5 --batch-size 16 --workers 4 --device cuda \
  --lr 1e-4 --weight-decay 0.05 --seed 3710
```

For an initial server smoke run, use `--epochs 1` and a distinct output such as `convnext_aug_fold01_check`. The original unaugmented run is preserved for comparison. All new results remain development results; the frozen calibration and final-test sets are not used for this experiment.

## ConvNeXt-Tiny

`--model convnext_tiny` selects the full Tiny architecture described by [Liu et al., A ConvNet for the 2020s (CVPR 2022)](https://arxiv.org/abs/2201.03545), with the block and downsampling design cross-checked against the [official PyTorch implementation](https://docs.pytorch.org/vision/0.21/_modules/torchvision/models/convnext.html). The project implements the components directly in PyTorch; torchvision and pretrained downloads are not required.

The four stages have depths **3, 3, 9, 3** and channels **96, 192, 384, 768**. Blocks use a 7×7 depthwise convolution, channel-wise LayerNorm, a four-times-expanded linear/GELU projection, and a residual connection with LayerScale initialized to 0.000001. Per-example stochastic depth increases linearly from 0 to 0.1 across the 18 blocks and is disabled for evaluation. A stride-4 convolutional stem, three stride-2 downsampling layers, spatial averaging, LayerNorm, and a binary head complete the model. The MRI adaptation uses **one input channel and one output logit**, with 27,817,825 trainable parameters; the small CNN has 97,521.

Both models receive the same fixed grayscale scaling and native 240×256 input; optional geometric augmentation applies only during training. ConvNeXt needs at least 32 pixels in each dimension; dimensions need not be multiples of 32. No center crop, RGB conversion, data-dependent normalization, or external pretraining is used. Both models share the frozen manifests, training-only loss weights, early-stop selection, complete-scan averaging, and fixed 0.5 threshold. Every fold starts a new model and optimizer. `config.json` records the versioned architecture, initialization, hyperparameters, split fingerprint, and code fingerprints; checkpoint loading dispatches to the saved architecture and checks preprocessing.

To reproduce the original unaugmented configuration, begin with one full-fold GPU smoke run on the server, using a new output directory:

```bash
python3 train.py \
  --model convnext_tiny \
  --data-root /home/groups/comp3710/ADNI \
  --splits-dir "$HOME/comp3710/adni_splits_v1" \
  --output "$HOME/comp3710/runs/convnext_fold01_check" \
  --fold 1 --epochs 1 --batch-size 16 --workers 4 --device cuda \
  --lr 1e-4 --weight-decay 0.05 --seed 3710
```

For the initial full development experiment, use a fresh output directory such as `convnext_fold01`, `--epochs 30 --patience 5`, and otherwise keep these settings. These are starting settings, not a validated optimal training recipe. The explicitly specified learning rate and weight decay differ from the CNN baseline and must be reported in comparisons. CLI defaults remain the original CNN settings, so include these options for ConvNeXt. GPU memory use and runtime must be measured on the server before scheduling all five folds; changing batch size creates a new documented configuration.

Reload a ConvNeXt checkpoint with the same prediction command used for the CNN, replacing only the checkpoint and output paths. No `--model` option is needed for prediction: the checkpoint identifies the architecture. Older `small_cnn_v1` checkpoints remain supported. Neither training nor inference offers calibration/final-test scoring at this development stage.

The first unaugmented real-data ConvNeXt run (fold 1, seed base 3710) selected epoch 4 and stopped after epoch 9. Its scan accuracy was 66.82%, macro F1 0.6627, and AUROC 0.7429 on the same frozen fold where the initial CNN achieved 83.41%, 0.8339, and 0.9136. Training loss fell from 0.683 to 0.031 while early-stop scan loss had its minimum of 0.718 at epoch 4. These trends motivate testing augmentation; the losses use different evaluation units and cannot be directly subtracted. This single fold does not establish an overall architecture ranking.

The [ConvNeXt experiment plan](docs/CONVNEXT_EXPERIMENT_PLAN.md) explains the implemented architecture, the subsequently reported augmented fold-1 result, supported observations versus untested explanations, and proposed controlled comparisons. Proposed model variants and tuning features in that plan are not yet implemented.

## Run the first fold

Use an allocated GPU compute node on the course server. The manifests already generated for this project do not need to be recreated. Start with a one-epoch end-to-end check:

```bash
python3 train.py \
  --data-root /home/groups/comp3710/ADNI \
  --splits-dir "$HOME/comp3710/adni_splits_v1" \
  --output "$HOME/comp3710/runs/cnn_fold01_check" \
  --fold 1 --epochs 1 --batch-size 32 --workers 4 --device cuda
```

This processes the complete selected fold at the native image size. Its purpose is to check the pipeline; one epoch is not the planned full baseline. A formal run uses a new output directory:

```bash
python3 train.py \
  --data-root /home/groups/comp3710/ADNI \
  --splits-dir "$HOME/comp3710/adni_splits_v1" \
  --output "$HOME/comp3710/runs/cnn_fold01" \
  --fold 1 --epochs 30 --patience 5 --batch-size 32 --workers 4 --device cuda
```

Use `--device cpu --workers 0` for a CPU run. `--device cuda` fails explicitly if CUDA is unavailable. Existing output directories are never overwritten, and there is no resume option that could accidentally carry a previous fold's weights into another fold. To run later folds, change both `--fold` (1–5) and the output directory. Keep the configuration fixed across folds and report all fold results rather than selecting the best fold.

The full source and manifest audit runs automatically before every training or prediction invocation, including data-integrity checks on the protected holdouts. Calibration/final-test images are never passed to either model or used to calculate training weights or model-selection metrics. There is no option to skip this audit.

| Run artifact | Purpose |
|---|---|
| `config.json` | Arguments, training-only class counts, manifest fingerprint, source-code fingerprints, and environment |
| `history.csv` | Training loss and early-stop metrics for each epoch |
| `best.pt` | Selected model weights and their fold/configuration provenance |
| `learning_curves.png` | Training and early-stop curves; no outer-validation curve is used for selection |
| `val_slice_predictions.csv` | Every outer-validation slice and its AD score |
| `val_scan_predictions.csv` | One aggregated prediction per outer-validation scan |
| `metrics.json` | Completed-run marker, held-out metrics, selected epoch, and resource measurements |

Training loss is class-weighted at slice level; early-stop loss is unweighted at scan level, so the two curves use different objectives. Resource output includes parameter count, total training/evaluation time, and peak allocated CUDA memory when available. Forward timing excludes loading and transfer; milliseconds per slice describe the configured batch size, not single-request latency. CPU runs report CUDA memory as `null`. Patient-clustered confidence intervals and automatic five-fold summary generation are not yet implemented.

All run artifacts stay outside Git. Incomplete runs have no final `metrics.json`; inspect the error and use a new output directory when retrying.

## Reload a checkpoint

```bash
python3 predict.py \
  --checkpoint "$HOME/comp3710/runs/cnn_fold01/best.pt" \
  --data-root /home/groups/comp3710/ADNI \
  --splits-dir "$HOME/comp3710/adni_splits_v1" \
  --output "$HOME/comp3710/runs/cnn_fold01_recheck" \
  --batch-size 32 --workers 4 --device cuda
```

The checkpoint determines its fold and preprocessing. A different manifest fingerprint is rejected. This version of `predict.py` evaluates that fold's outer validation only; it has no final-test or calibration mode. Reproducing a prediction does not create a new independent experiment.

## Inspect inputs and frozen features

`diagnose.py` inspects an existing CNN or ConvNeXt checkpoint on training and
inner early-stop patients. It exports training-only transform comparisons,
per-stage activations, learned LayerScale/residual contributions, and integer
shift sensitivity. Optional fixed-budget linear probes fit new diagnostic heads
to frozen patient-mean embeddings; they do not update the backbone or select a
checkpoint using outer validation. Use a new output outside the original run.

```bash
python3 diagnose.py \
  --checkpoint "$HOME/comp3710/runs/convnext_aug_fold01/best.pt" \
  --data-root /home/groups/comp3710/ADNI \
  --splits-dir "$HOME/comp3710/adni_splits_v1" \
  --output "$HOME/comp3710/diagnostics/convnext_aug_fold01_v1" \
  --device cuda --batch-size 16 --workers 4 --probes
```

Run in the existing training environment on an allocated compute node. See
[feature diagnostics](docs/FEATURE_DIAGNOSTICS.md) for matching all three runs,
sampling limits, interpretation and evidence status. These outputs are diagnostic
subsets, not new outer-validation or final-test scores.

## Sampling boundary and probe controls

`diagnose_followup.py` implements the first A/B/D extension: exact 1-8 pixel
shifts with intermediate features, round-trip boundary-loss controls, and
native-dimension versus training-only PCA probes on replay and expanded patient
cohorts. It preserves the prior early-stop selection and original checkpoints.
See [follow-up usage](docs/FEATURE_DIAGNOSTIC_FOLLOWUP.md) for the command,
server transfer helpers, measured units and interpretation limits. These
controls do not train a backbone or score outer/reserved partitions.

## Reproducibility and remaining safeguards

- Freeze this split before model experiments; do not choose a split seed using model scores.
- Initialise model and training state independently for each fold. Fit preprocessing statistics and class weights using only that fold's training subset.
- Seed Python and PyTorch, use a seeded data loader, and request deterministic algorithms. Reproducibility across different hardware, PyTorch versions, or CPU/GPU execution is not guaranteed; see the [PyTorch reproducibility notes](https://docs.pytorch.org/docs/2.6/notes/randomness.html).
- Use `early_stop.csv` to choose the stopping epoch, and `val.csv` for cross-validation comparison. Cross-validation results used to select a method are development results.
- Refit the selected model on development data using a previously specified training rule. Then freeze it, fit confidence/decision thresholds on calibration data, and freeze the complete pipeline before final testing.
- Exact duplicate checks do not exhaustively detect near-duplicates, incorrect source patient identities, or leakage from upstream preprocessing. Manifest verification does not enforce future training-code behaviour.

See [the full data protocol](docs/DATA_PROTOCOL.md) for evaluation units, mixed longitudinal diagnoses, out-of-fold predictions, and uncertainty reporting.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Install the training dependencies before running the complete suite. Audit-only tests can be run with `python3 -m unittest discover -s tests -p test_adni_splits.py -v`.

ConvNeXt-specific tests also check native-resolution forward/backward execution, training-only stochastic depth, fresh initialization, model dispatch, checkpoint selection, and saved-model prediction reproduction.

Tests cover patient isolation, fold coverage, reproducibility, missing or corrupt inputs, conflicting labels, duplicate content, modified manifests, modified sources, safe image loading, known metric values and ties, full-scan aggregation, checkpoint selection, and actual CPU training followed by identical checkpoint predictions. They use generated images and synthetic patients, not the ADNI dataset. They do not establish real-data model performance or GPU compatibility on the course server.

The [ConvNeXt smoke-test record](docs/CONVNEXT_SMOKE_TEST.md) documents the 57 passing tests and the separate 240×256 command-line training/reload check. Smoke-test scores on generated images are not ADNI performance results.

The package layout and training-only augmentation are covered by the additional [refactor smoke-test record](docs/REFACTOR_SMOKE_TEST.md).

## Artificial Intelligence Usage Disclosure

OpenAI Codex assisted with the data-audit script, baseline CNN and ConvNeXt-Tiny implementations, training/inference code, package restructuring, training-only augmentation, metrics, synthetic tests, code comments, protocol documentation, result review and repository documentation updates. Validation includes source review, synthetic integrity tests, known metric examples, actual CPU training with checkpoint-reload comparisons, and independent review of uploaded baseline predictions and frozen manifests. The project owner executed the real-data audit and three five-fold CNN repetitions on the course server. This development note should be incorporated into the final course-required AI-use disclosure; it does not replace that disclosure.

## Coursework-aligned metric logging

Following the v2.1 task sheet and staff clarification checked on 3 October 2026, new runs log slice-primary classification, raw confidence/ECE/Brier, fixed-rule rejection and risk--coverage, patient aggregation when diagnoses are consistent, factual failure examples and separate resource profiles. Per-epoch metrics include training and early-stop accuracy/AUROC. Scratch initialization and frozen patient roles are retained; the smaller ConvNeXt is optional, and the 30-epoch default is an experiment maximum. Final calibration/refit/test evaluation remains future work. See [the logging protocol and pilot command](docs/COURSEWORK_LOGGING.md) for exact files, definitions, limitations and profiling settings.

## Rangpur GPU pilot submission

The current server checkout is `$HOME/comp3710/comp3710-adni`. Submit a new scratch-trained inner-only Lite pilot from the login node using `bash "$HOME/comp3710/comp3710-adni/slurm/submit_pilot.sh" convnext_lite`. Use `cnn` for a control with the new logging. See [source update, submission and output instructions](docs/SLURM_PILOTS.md) for the complete source bundle, paths, budget and environment overrides. These runs retain the frozen patient roles and do not score outer validation, calibration or final test.

## Predeclared inner-only experiment suite

Submit `bash slurm/submit_suite.sh` to run the 17 fixed CNN/Lite controls sequentially within one GPU allocation. The plan isolates learning rate, weight decay, batch size, optional warmup/cosine, class/patient sampling and geometric/intensity augmentation. Completed cases can be reused under the same plan/code after an interruption. All cases remain development-only. See [acceptance status, exact settings, budget, submission and resume](docs/EXPERIMENT_SUITE.md). No final accuracy or fitted calibration is claimed.

### Prospective feature controls (v2)

The separate `run_feature_experiments.py` runner implements the six-case
[feature/selection investigation](docs/EXPERIMENT_PLAN_V2.md). It retains the
legacy training defaults and frozen patient roles, and scores only train and
inner early-stop patients. See [submission and artifacts](docs/FEATURE_SUITE_USAGE.md).
No new real-data result is claimed until the GPU jobs complete.

## Optional scan-consistent input standardization

Use `--preprocessing scan_intensity_crop` with `train.py` for label-free per-scan
foreground percentile mapping and native-pixel crop/black padding. Run
`audit_preprocessing.py` first to inspect original/processed inner-development
inputs and the training-fitted crop window. Prediction restores the checkpoint's
exact transform; defaults and older checkpoints keep their original behavior.
See [algorithm, QA commands, four controls, logging and limits](docs/INPUT_PREPROCESSING.md).
No real-data accuracy gain or final-test performance has been established.
