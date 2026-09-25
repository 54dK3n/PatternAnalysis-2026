# COMP3710: ADNI AD/NC Classification

Patient-isolated MRI classification comparing Alzheimer's disease (AD) with normal controls (NC).

The implementation audits the course JPEG dataset, creates five-fold cross-validation manifests, trains either a small CNN or a ConvNeXt-Tiny on one development fold, and reproduces held-out predictions from a saved checkpoint. Both architectures are implemented in PyTorch and initialized from random weights. Three complete five-fold CNN repetitions have been reviewed locally. ConvNeXt real-data experiments, final refitting, confidence calibration, and final-test evaluation remain future work.

This project lives at `recognition/ADNI_s4983666/` on the course fork's `topic-recognition` branch. The original three development commits were imported without squashing their history. Final coursework submission still requires a pull request to `shakes76/PatternAnalysis-2026`, targeting `topic-recognition`, and the accompanying report submission.

## Repository contents

- `adni_splits.py`: source-data audit, patient-level partitioning, and independent manifest verification.
- `dataset.py`: mandatory manifest verification and deterministic grayscale image loading.
- `modules.py`: the small CNN, complete grayscale ConvNeXt-Tiny, and versioned model factory.
- `train.py`: one-fold training, early stopping, checkpoint saving, and outer validation.
- `predict.py`: reload a checkpoint and reproduce that fold's outer-validation predictions.
- `metrics.py`: scan aggregation, per-class metrics, macro F1, AUROC, and log loss.
- `training_utils.py`: reproducibility, inference, artifact output, and plotting helpers.
- `tests/`: synthetic tests for splitting, loading, metrics, and actual CPU training/inference.
- `DATA_PROTOCOL.md`: the full experimental protocol, including training-time safeguards.
- `CONVNEXT_SMOKE_TEST.md`: local CPU test results and remaining server validation.
- `requirements.txt`: the image-reading dependency for audit-only use.
- `requirements-train.txt`: the reference training dependencies.

## Setup

Use Python 3.9 or newer in your project environment:

```bash
git clone --branch topic-recognition https://github.com/54dK3n/PatternAnalysis-2026.git
cd PatternAnalysis-2026/recognition/ADNI_s4983666
python3 -m pip install -r requirements.txt
```

In an existing local checkout, enter `recognition/ADNI_s4983666/` before running the commands below.

For training, install the additional dependencies in your project environment:

```bash
python3 -m pip install -r requirements-train.txt
```

The training requirements pin PyTorch 2.6.0 and Matplotlib 3.9.4. Local CPU verification used Python 3.12.14 and Pillow 12.3.0. GPU training requires a PyTorch CUDA build compatible with the server's driver; follow the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) or the course environment instructions. The code does not require torchvision or scikit-learn. Every training run records its actual package versions, CUDA build, and device information in `config.json`.

The completed server baseline used Python 3.11.15, PyTorch 2.13.0, Pillow 12.3.0 and Matplotlib 3.11.1 on an NVIDIA A100-PCIE-40GB, with CUDA build 13.0. These recorded server versions differ from the local CPU reference environment above.

## Local workspace files

The local migration also copies `outputs/`, the coursework PDF and the local `.venv/` into this project folder. They remain ignored by Git. Outputs include private split manifests, prediction records, checkpoints and local review reports; keep them outside all public commits. Historical paths inside experiment configurations remain unchanged because they record where those experiments actually ran. Recreate `.venv/` on a different machine rather than transferring its platform-specific packages.

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

Default input size is height 240 × width 256, matching the supplied images. Pixel values use the fixed transform `(pixel / 255 - 0.5) / 0.5`; no population mean or standard deviation is estimated. There is no augmentation in this initial baseline. Each image is checked against its recorded file digest when loaded.

Training uses AdamW with learning rate 0.001, weight decay 0.0001, and binary cross-entropy with logits. AD's positive loss weight is `NC training slices / AD training slices`, computed only from the current fold's training manifest. Sampling is uniform over training slices: patients with more scans contribute more examples. This weighting choice should be discussed when comparing models.

For each scan, average its 20 slice-level AD probabilities and classify it as AD when the mean is at least 0.5. This aggregation and threshold are fixed before evaluation. These are uncalibrated scores; they should not be interpreted as clinical confidence estimates. Longitudinal scans retain their own diagnoses.

After each epoch, evaluate only the early-stopping patients. Save every strict minimum of their **scan-level log loss**. Stop after 5 epochs without an improvement greater than 0.0001 relative to the last patience-reset value, up to 30 epochs by default. Reload the selected checkpoint and evaluate the complete outer-validation set once. Report scan-level accuracy, balanced accuracy, per-class precision/recall/F1, macro F1, AUROC, log loss, and a confusion matrix. Slice-level results are supplemental; they are not independent-patient results.

## ConvNeXt-Tiny

`--model convnext_tiny` selects the full Tiny architecture described by [Liu et al., A ConvNet for the 2020s (CVPR 2022)](https://arxiv.org/abs/2201.03545), with the block and downsampling design cross-checked against the [official PyTorch implementation](https://docs.pytorch.org/vision/0.21/_modules/torchvision/models/convnext.html). The project implements the components directly in PyTorch; torchvision and pretrained downloads are not required.

The four stages have depths **3, 3, 9, 3** and channels **96, 192, 384, 768**. Blocks use a 7×7 depthwise convolution, channel-wise LayerNorm, a four-times-expanded linear/GELU projection, and a residual connection with LayerScale initialized to 0.000001. Per-example stochastic depth increases linearly from 0 to 0.1 across the 18 blocks and is disabled for evaluation. A stride-4 convolutional stem, three stride-2 downsampling layers, spatial averaging, LayerNorm, and a binary head complete the model. The MRI adaptation uses **one input channel and one output logit**, with 27,817,825 trainable parameters; the small CNN has 97,521.

Both models receive the same fixed grayscale scaling and native 240×256 input. ConvNeXt needs at least 32 pixels in each dimension; dimensions need not be multiples of 32. No center crop, RGB conversion, data-dependent normalization, or external pretraining is used. Both models share the frozen manifests, training-only loss weights, early-stop selection, complete-scan averaging, and fixed 0.5 threshold. Every fold starts a new model and optimizer. `config.json` records the versioned architecture, initialization, hyperparameters, split fingerprint, and code fingerprints; checkpoint loading dispatches to the saved architecture and checks preprocessing.

Begin with one full-fold GPU smoke run on the server, using a new output directory:

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

Training loss is class-weighted at slice level; early-stop loss is unweighted at scan level, so the two curves use different objectives. Resource output includes parameter count, total training/evaluation time, and peak allocated CUDA memory when available. Forward timing excludes loading and transfer; milliseconds per slice describe the configured batch size, not single-request latency. CPU runs report CUDA memory as `null`. Patient-clustered confidence intervals and a five-fold summary are not yet implemented.

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

## Reproducibility and remaining safeguards

- Freeze this split before model experiments; do not choose a split seed using model scores.
- Initialise model and training state independently for each fold. Fit preprocessing statistics and class weights using only that fold's training subset.
- Seed Python and PyTorch, use a seeded data loader, and request deterministic algorithms. Reproducibility across different hardware, PyTorch versions, or CPU/GPU execution is not guaranteed; see the [PyTorch reproducibility notes](https://docs.pytorch.org/docs/2.6/notes/randomness.html).
- Use `early_stop.csv` to choose the stopping epoch, and `val.csv` for cross-validation comparison. Cross-validation results used to select a method are development results.
- Refit the selected model on development data using a previously specified training rule. Then freeze it, fit confidence/decision thresholds on calibration data, and freeze the complete pipeline before final testing.
- Exact duplicate checks do not exhaustively detect near-duplicates, incorrect source patient identities, or leakage from upstream preprocessing. Manifest verification does not enforce future training-code behaviour.

See [the full data protocol](DATA_PROTOCOL.md) for evaluation units, mixed longitudinal diagnoses, out-of-fold predictions, and uncertainty reporting.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Install the training dependencies before running the complete suite. Audit-only tests can be run with `python3 -m unittest discover -s tests -p test_adni_splits.py -v`.

ConvNeXt-specific tests also check native-resolution forward/backward execution, training-only stochastic depth, fresh initialization, model dispatch, checkpoint selection, and saved-model prediction reproduction.

Tests cover patient isolation, fold coverage, reproducibility, missing or corrupt inputs, conflicting labels, duplicate content, modified manifests, modified sources, safe image loading, known metric values and ties, full-scan aggregation, checkpoint selection, and actual CPU training followed by identical checkpoint predictions. They use generated images and synthetic patients, not the ADNI dataset. They do not establish real-data model performance or GPU compatibility on the course server.

The [ConvNeXt smoke-test record](CONVNEXT_SMOKE_TEST.md) documents the 57 passing tests and the separate 240×256 command-line training/reload check. Smoke-test scores on generated images are not ADNI performance results.

## Artificial Intelligence Usage Disclosure

OpenAI Codex assisted with the data-audit script, baseline CNN and ConvNeXt-Tiny implementations, training/inference code, metrics, synthetic tests, code comments, protocol documentation, result review and repository migration. Validation includes source review, synthetic integrity tests, known metric examples, actual CPU training with checkpoint-reload comparisons, and independent review of uploaded baseline predictions and frozen manifests. The project owner executed the real-data audit and three five-fold CNN repetitions on the course server. This development note should be incorporated into the final course-required AI-use disclosure; it does not replace that disclosure.
