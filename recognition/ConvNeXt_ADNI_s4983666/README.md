# ConvNeXt for Alzheimer's Disease Classification on ADNI Brain MRI

COMP3710 Pattern Recognition, recognition task: ConvNeXt (hard difficulty) on the
ADNI dataset, AD versus cognitively normal (NC). Student s4983666.

## Problem and algorithm

Alzheimer's disease (AD) causes progressive hippocampal and cortical atrophy and
ventricular enlargement that are visible on structural T1-weighted MRI. The task
is to classify 2D sagittal MRI slices of **previously unseen patients** as AD or NC,
while avoiding the main pitfall of this dataset: slices of one patient leaking
into both training and evaluation. The model is ConvNeXt (Liu et al., 2022), a
convolutional network that adopts design choices of vision transformers
(patchify stem, large 7x7 depthwise kernels, inverted bottlenecks, LayerNorm,
GELU and few activation/normalization layers). It is implemented from scratch in
PyTorch and trained either from random initialization or, as a cited
comparison, starting from the authors' ImageNet-1k ConvNeXt-Tiny weights
(allowed by the course staff when referenced). A four-block CNN is the baseline. Beyond accuracy, the project asks
whether the model's confidence can be trusted: a calibrated referral rule sends
low-confidence patients to a specialist instead of automating them.

## How it works

![Pipeline: three context slices pass through the ConvNeXt-Tiny stem, four stages and the head; slice logits are averaged per scan or patient, Platt-scaled and compared with the referral threshold tau](figures/architecture.png)

*Figure 1. Pipeline overview: three neighbouring sagittal slices enter the ConvNeXt-Tiny stem and four stages; slice logits are averaged per scan or patient, calibrated by Platt scaling and compared against the referral threshold τ.*

Each input is one slice stacked with its two neighbouring slices from the same
scan (three channels), so the 2D network sees some through-plane anatomy. The
stem cuts the image into 4×4 patches; four stages (Lite: 2-2-6-2 blocks with
48-384 channels; Tiny: 3-3-9-3 blocks with 96-192-384-768 channels) are separated by
LayerNorm + 2×2 stride-2 convolutions. Each block applies a 7×7 depthwise
convolution, LayerNorm, a 4×-expanded MLP with GELU, a learnable layer scale
(initialized to 1e-6) and stochastic depth [2] on the residual branch. The model
outputs one AD logit per slice. The baseline (`SmallCNN` in `models/cnn.py`) uses
four sequential stages of `Conv2d(3x3, padding=1, bias=False) -> GroupNorm(4 groups) -> ReLU -> MaxPool2d(2x2)`
with channel widths `16 -> 32 -> 64 -> 128`, followed by global average pooling,
`Dropout(0.2)` and a `Linear(128, 1)` head (`97,809` trainable parameters for
three-slice input). Training uses class-weighted binary
cross-entropy, AdamW with warm-up and cosine decay, strong augmentation, and an
exponential moving average (EMA) of the weights [5]. Training runs for a fixed
number of epochs and the final weights are kept, so validation patients are
only monitored, never used to pick a checkpoint. Slice probabilities are
averaged per scan and per patient; Platt scaling [4] and a referral threshold are
fitted on calibration patients before the test patients are scored once.

## Feasibility review

One-page review for the project manager. Tutor check-off completed on
Wednesday, 14 October 2026; the tutor noted that training did not need to use a
fixed epoch count and that validation performance could be used to decide early
stopping (tested via an inner validation subset in Rounds 1-2 before adopting
EMA weight averaging over a fixed schedule so all 381 non-validation patients per
fold could be used for training).

**1. User need, scope and acceptance criteria.** The user is a research
reviewer deciding whether an AD vs normal-control (NC) classifier for course
ADNI T1 MRI slices can automate confident cases and route uncertain ones to a
specialist. Clinical deployment is out of scope. Acceptance criteria:
(1) zero patient overlap between development, calibration and test patients,
and checkpoints that replay their predictions exactly; (2) test accuracy
≥ 0.80 at slice level (the unit recommended by staff for 2D models, Ed #137),
with scan and patient level also reported; (3) ConvNeXt and the CNN baseline
compared on identical patients with per-class precision/recall, macro-F1, AUROC
and ECE before/after calibration; (4) a referral rule frozen on calibration
patients before the test set is scored once, reported as coverage, accuracy of
automated cases and number referred, plus 3-5 real failure cases; (5) training
and inference on one A100 GPU with parameter count, peak memory and latency
measured.

**2. Model choice and course concepts.** Hard band: ConvNeXt-Tiny [1]
implemented locally in PyTorch (4×4 patchify stem, 7×7 depthwise convolutions,
LayerNorm, inverted bottleneck, layer scale, stochastic depth [2]), with the
slice and its two neighbours as three input channels. The baseline is a
four-block convolutional network (`SmallCNN`) trained by the identical pipeline. Course
concepts: convolutional inductive bias, overfitting and regularisation
(augmentation, weight decay, EMA [5]), patient-level validation, and
probability calibration [4]. Data: frozen patient-level manifests (476
development patients in five folds, 68 calibration, 136 test). Models train
for a fixed number of epochs and keep the final weights, so validation
patients are only used to compare configurations, never to pick a checkpoint.
The project started from random initialisation as advised in Ed #137; after a
later staff answer allowing properly cited external resources, an
ImageNet-1k-initialised ConvNeXt-Tiny (official weights, checksum recorded)
was added as a cited comparison, and the scratch model is still reported.

**3. Feasibility evidence.** The source audit found 680 patients (1,050
development, 170 calibration and 306 test scans of 20 slices, 240×256
grayscale), with 21 manifest hashes verified and no patient in two roles.
Five-fold development results (validation patients, not test): scratch
ConvNeXt-Tiny slice accuracy 0.715, CNN 0.690, ImageNet-initialised
ConvNeXt-Tiny 0.798 (patient level 0.782 / 0.750 / 0.840). ConvNeXt-Tiny has
27.8 M parameters, a 4,656 MiB training peak and 5.19 ms single-image latency,
against 97.8 k, 727 MiB and 1.16 ms for the CNN.

**4. Risks, budget and fallback.** (a) Acquisition confound: every NC scan was
acquired 2017-2021, while 417 of 503 AD scans are older, so models can learn
scanner style instead of atrophy; results are therefore also reported on the
2017-2021 scans, and Grad-CAM [10] is used to check where the evidence lies.
(b) About 95 patients per validation fold, so differences below roughly 0.04
are within noise; comparisons are paired by fold. (c) Overfitting (training
accuracy 0.95-1.00). (d) The slice-level target is marginal on development
data. Budget: one Rangpur A100 job at a time; a five-fold run costs about
5 GPU hours (pretrained) to 8.5 GPU hours (scratch). Next step: freeze the
final pipeline, fit Platt scaling and the referral threshold on calibration
patients, then score the test patients once. Fallback: if the test target is
missed, report the measured result honestly, keep the CNN comparison and the
referral rule, and recommend external validation on data where acquisition
style does not predict the diagnosis before any deployment.

## Repository layout

```text
ConvNeXt_ADNI_s4983666/
|-- modules.py            # Model interface (PyTorch only): models/convnext.py, models/cnn.py
|-- dataset.py            # Data interface: dataset/ (splits, manifests, slices, augmentation)
|-- train.py              # Training entry point (engine/training.py)
|-- predict.py            # Inference, calibration and test entry point (engine/prediction.py)
|-- adni_splits.py        # Audit / prepare / verify the patient-level split
|-- audit_preprocessing.py
|-- probe_acquisition.py  # Acquisition-style probe (docs/ACQUISITION_CONFOUND.md)
|-- explain_gradcam.py    # Grad-CAM figures on validation patients
|-- engine/               # Training loop, EMA, LR schedule, prediction
|-- evaluation/           # Metrics, inference, calibration, reporting, resource profiling
|-- utils/                # Runtime/seeding, output files, plots, training controls
|-- slurm/                # Rangpur batch scripts and the experiment-plan runner
|-- config/               # Requirements and experiment plans
|-- docs/                 # Data protocol, preprocessing notes, development experiments
|-- figures/              # README figures (plots of saved results)
|   '-- adni/             # De-identified ADNI slices: test examples, failures, Grad-CAM
'-- tests/                # Unit and end-to-end tests on synthetic data
```

## Dependencies and reproducibility

Rangpur runs used Python 3.11.15, PyTorch 2.13.0 (CUDA build 13.0), Pillow
12.3.0 and Matplotlib 3.11.1 on an NVIDIA A100-PCIE-40GB. The code needs no
torchvision, NumPy or scikit-learn; `config/requirements*.txt` list the minimal
packages (install a PyTorch build that matches your GPU driver).

```bash
python3 -m pip install -r config/requirements-train.txt
python3 -m unittest discover -s tests          # synthetic data only
```

**Optional pretrained weights.** `--model convnext_tiny_in1k` starts from the
authors' ImageNet-1k ConvNeXt-Tiny weights (Liu et al., 2022;
https://github.com/facebookresearch/ConvNeXt). The code never downloads them;
fetch the file once and pass its path:

```bash
curl -L -o convnext_tiny_1k_224_ema.pth https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth
sha256sum convnext_tiny_1k_224_ema.pth   # 14f3164e3ea6ac32ab3f574f528ce817696c9176fad4221e0a77a905a7360595
python train.py --model convnext_tiny_in1k --pretrained-weights convnext_tiny_1k_224_ema.pth --context-slices 3 ...
```

`models/pretrained.py` maps the official parameter names onto this
implementation, requires every backbone tensor to match, and leaves the
one-logit classifier random. A fixed input layer converts the [-1, 1] inputs to
ImageNet mean/std; the three context slices take the place of the RGB
channels. The file's SHA-256 and source are written to the run's
`config.json`. The weights are not committed.

Every run seeds Python and PyTorch (`seed + fold`), requests deterministic
algorithms [7] and writes `config.json` (all options, package versions, GPU,
manifest and source-code hashes). Bit-exact results are only expected on the
same hardware and software versions.

## Data

The course copy at `/home/groups/comp3710/ADNI` has 680 patients (221 AD, 459 NC),
1,526 scans (743 AD, 783 NC) and 30,520 grayscale 256×240 JPEG slices (20 consecutive
slices per scan). The course's own train/test folders share 216 patients, so patients
are re-split once (seed 3710), stratified by patient diagnosis (~32.5% AD across all
partitions and across all five development folds) while keeping all longitudinal scans
of a patient together:

| Partition | Patients (AD / NC) | Scans (AD / NC) | Slices (AD / NC) | Used for |
|---|---:|---:|---:|---|
| development (5 folds) | 476 (155 / 321, 32.6% AD) | 1,050 (503 / 547) | 21,000 (10,060 / 10,940) | training and validation (model comparison) |
| calibration | 68 (22 / 46, 32.4% AD) | 170 (91 / 79) | 3,400 (1,820 / 1,580) | Platt scaling and referral threshold of the frozen model |
| test | 136 (44 / 92, 32.4% AD) | 306 (149 / 157) | 6,120 (2,980 / 3,140) | one final evaluation |

Each of the five development folds holds out 95-96 validation patients (31 AD and
64-65 NC; 32.3-32.6% AD); scan and slice counts have a higher AD share (47.9-53.5%)
because AD patients contribute more longitudinal scans per subject on average (3.2-4.1
vs 1.7 for NC). Within fold 1, 381 patients train the model (the fold's `train` and small
`early_stop` manifests) and 95 validation patients are scored after every
epoch; the checkpoint is always the last epoch. `adni_splits.py verify` (run automatically by `train.py` and `predict.py`)
fails if any patient, scan, file or pixel hash crosses a role boundary. See
[docs/DATA_PROTOCOL.md](docs/DATA_PROTOCOL.md).

**Preprocessing.** Slices keep their native 240×256 size and are mapped to
[-1, 1] with fixed constants, so no statistic is estimated from any split.
Optional per-scan intensity/crop standardisation is described in
[docs/INPUT_PREPROCESSING.md](docs/INPUT_PREPROCESSING.md); it did not help in
development and is off by default.

**Augmentation (training slices only).** Horizontal flip (on these sagittal
slices a front-to-back mirror: it keeps the atrophy pattern but produces an
orientation that does not occur in the data, so it acts as a generic
regulariser), random-resized crop covering 85-100% of the area [6], rotation up to
10 degrees, translation up to 5%, and intensity gain/gamma within ±10%/±15%. All three context
channels receive the same transform. Vertical flips and large rotations are not
used because they distort the anatomy further.

## Usage

```bash
SPLITS="$HOME/comp3710/adni_splits_v1"
WEIGHTS="$HOME/comp3710/weights/convnext_tiny_1k_224_ema.pth"

# 1. Train the primary five-fold ImageNet-1k-initialised ConvNeXt-Tiny ensemble
# (40 epochs, 3-slice context, strong augmentation, EMA 0.999, AdamW 1e-4 / wd 0.05).
for f in 1 2 3 4 5; do
    python train.py --splits-dir "$SPLITS" --output "runs/pt_cv/fold${f}" --fold "$f" \
        --model convnext_tiny_in1k --pretrained-weights "$WEIGHTS" --epochs 40
done

# 2. Train the five-fold scratch ConvNeXt-Tiny and SmallCNN baselines (80 epochs).
for f in 1 2 3 4 5; do
    python train.py --splits-dir "$SPLITS" --output "runs/scratch_cv/fold${f}" --fold "$f" \
        --model convnext_tiny --epochs 80
    python train.py --splits-dir "$SPLITS" --output "runs/cnn_cv/fold${f}" --fold "$f" \
        --model cnn --lr 1e-3 --weight-decay 1e-4 --epochs 80
done

# 3. Reproduce the validation result of any single fold checkpoint.
python predict.py --checkpoint runs/pt_cv/fold1/final.pt --splits-dir "$SPLITS" \
    --role val --output runs/pt_cv/fold1_val

# 4. Final evaluation of the frozen five-fold ensemble (calibration first, then test once).
python predict.py --checkpoint runs/pt_cv/fold{1,2,3,4,5}/final.pt --splits-dir "$SPLITS" \
    --role calibration --output runs/pt_cv_ensemble/calibration
python predict.py --checkpoint runs/pt_cv/fold{1,2,3,4,5}/final.pt --splits-dir "$SPLITS" \
    --role test --calibration-file runs/pt_cv_ensemble/calibration/calibration.json \
    --output runs/pt_cv_ensemble/test

# 5. Grad-CAM of a checkpoint on its fold's validation patients (never calibration/test).
python explain_gradcam.py --checkpoint runs/pt_cv/fold5/final.pt --splits-dir "$SPLITS" \
    --output runs/pt_cv/fold5/gradcam

# Note: omitting --model defaults to ConvNeXt-Lite (convnext_lite) for fast single-fold smoke runs:
# python train.py --splits-dir "$SPLITS" --output runs/lite_fold1 --fold 1
```

When the metadata file names carry acquisition dates, `predict.py` also reports
scan/patient metrics and referral counts separately for scans acquired before
and from 2017 (`by_acquisition_period` in `metrics.json`; see
[docs/ACQUISITION_CONFOUND.md](docs/ACQUISITION_CONFOUND.md)).
`explain_gradcam.py` (Selvaraju et al., 2017) writes the most confident correct
and wrong cases per diagnosis × acquisition period, the mean map of each group
and the share of each map that falls outside the head.

On Rangpur, `sbatch slurm/run_plan.sbatch config/experiments_round3_cv.json --smoke`
checks every planned case for one epoch, `--cases <id> ...` runs selected cases,
and `sbatch slurm/final_eval.sbatch <output dir> <run dir> [<run dir> ...]` performs the
two final steps for one model or a fold ensemble.
After every case the runner reloads `final.pt` and checks that the re-scored
accuracy and AUROC equal the training run's (`replay_check.json`).
`python slurm/summarize_runs.py <plan dirs> --output <prefix> --baseline <group> --candidate <group>`
collects every completed run into `<prefix>.md`/`.csv`, averages seeds or folds,
and pairs ConvNeXt with the CNN baseline by fold and seed.

A training run writes `config.json`, `history.csv` (per-epoch losses, metrics,
time and peak GPU memory), `learning_curves.png`,
`final.pt`, per-slice and per-scan predictions, confidence plots and
`metrics.json`. The test step prints one line per patient followed by the
calibrated summary (actual summary lines from the primary model's test run,
Slurm 659813):

```text
patient       true    p(AD)  decision
<patient_1>   NC      0.081  NC
<patient_2>   AD      0.552  REFER
...
Platt slope 0.246, intercept 0.193; referral threshold 0.6255970686222304 (fitted on calibration patients, target accuracy 0.9).
Referred to a specialist: 17/136 patients.
test slice: accuracy=0.8667 AUROC=0.9429123028256317 (n=6120)
test scan: accuracy=0.8889 AUROC=0.9636215962039927 (n=306)
test patient: accuracy=0.8824 AUROC=0.9439229249011858 (n=136)
```

and saves `test_examples.png` (input slice, true label, p(AD), decision).
Example input and output of the final test (primary ensemble): the middle
slice of the patient's first scan, with the patient-level calibrated p(AD)
and the decision. Top row: the four most confident automated patients (all
correct); bottom row: the four least confident referred patients (patient and
scan IDs removed).

![Test examples](figures/adni/test_examples.png)

*Figure 2. Example test predictions (patient and scan IDs removed).*

## Results

### Earlier development screens (fold 1)

Development results before the current recipe (from `metrics.json`; full table
in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)), fold-1 outer validation, 95 patients:

| Model | Augmentation | Slice accuracy | Scan accuracy | Slice AUROC |
|---|---|---:|---:|---:|
| SmallCNN | none | 0.809 | 0.834 | 0.882 |
| ConvNeXt-Tiny | none | 0.634 | 0.668 | 0.693 |
| ConvNeXt-Tiny | light | 0.726 | 0.768 | 0.801 |

First run of the current recipe (`config/experiments_strong_aug.json`, two seeds,
same 95 patients, checkpoint chosen on the 38-patient inner `early_stop`
subset): ConvNeXt-Lite with three-slice context reached slice accuracy
0.717 / 0.740 and scan accuracy 0.754 / 0.810; details in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). Slice accuracy is still below 0.80.

Second fold-1 screen (`config/experiments_round2.json`, 80 epochs, mean ± sd
over two seeds, same 95 patients):

| Model | Slice accuracy | Scan accuracy | Patient accuracy | Slice AUROC |
|---|---:|---:|---:|---:|
| SmallCNN, three-slice context | 0.735 ± 0.008 | 0.791 ± 0.020 | 0.779 ± 0.030 | 0.841 ± 0.048 |
| ConvNeXt-Lite, single slice | 0.714 ± 0.004 | 0.780 ± 0.010 | 0.753 ± 0.007 | 0.783 ± 0.004 |
| ConvNeXt-Lite, three-slice context | 0.728 ± 0.019 | 0.780 ± 0.030 | 0.742 ± 0.022 | 0.799 ± 0.022 |
| ConvNeXt-Tiny, three-slice context | 0.742 ± 0.001 | 0.799 ± 0.023 | 0.768 ± 0.030 | 0.812 ± 0.008 |

The scratch ConvNeXt memorised the training slices within a few epochs
(training accuracy 1.00) and optimizer/regularisation tweaks (including mixup [3])
stayed within seed noise; augmentation was the only clear improvement. Because the 38-patient
`early_stop` subset proved noisy as a checkpoint selector (scan AUROC 0.64-0.73
vs 0.84-0.87 on outer validation), subsequent rounds merged `early_stop` into
training and kept the final-epoch EMA weights.

### Five-fold comparison from scratch (development patients)

[`config/experiments_round3_cv.json`](config/experiments_round3_cv.json), Slurm
jobs 649955 (ConvNeXt) and 649956 (CNN): both models use the identical pipeline
(three-slice context, strong augmentation, EMA 0.999, 80 epochs, final-epoch
weights, seed 3710) on the same five folds. Each fold is scored on its own ~95
validation patients, so together the folds cover all 476 development patients.
Mean ± sd over the five folds (`slurm/summarize_runs.py`):

| Model | Slice acc. | Scan acc. | Patient acc. | Slice AUROC | Scan AUROC | Patient AUROC |
|---|---:|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny | 0.715 ± 0.028 | 0.748 ± 0.054 | 0.782 ± 0.032 | 0.799 ± 0.028 | 0.847 ± 0.035 | 0.851 ± 0.015 |
| SmallCNN (baseline) | 0.690 ± 0.032 | 0.701 ± 0.054 | 0.750 ± 0.032 | 0.800 ± 0.068 | 0.853 ± 0.072 | 0.840 ± 0.068 |

Per fold (ConvNeXt / CNN accuracy):

| Fold | Slice | Scan | Patient | Patient FP / FN (ConvNeXt; CNN) |
|---:|---|---|---|---|
| 1 | 0.739 / 0.665 | 0.815 / 0.630 | 0.789 / 0.737 | 11 / 9; 0 / 25 |
| 2 | 0.708 / 0.671 | 0.765 / 0.685 | 0.832 / 0.747 | 3 / 13; 6 / 18 |
| 3 | 0.693 / 0.716 | 0.721 / 0.699 | 0.768 / 0.789 | 12 / 10; 2 / 18 |
| 4 | 0.749 / 0.733 | 0.767 / 0.781 | 0.771 / 0.771 | 13 / 9; 5 / 17 |
| 5 | 0.685 / 0.664 | 0.672 / 0.708 | 0.747 / 0.705 | 6 / 18; 16 / 12 |

* Neither model reaches 0.80 on average at any level; ConvNeXt is closest at
  patient level (0.782).
* Paired by fold, ConvNeXt is ahead by +0.025 slice (4 of 5 folds), +0.047 scan
  and +0.032 patient accuracy, but a paired t-test over five folds does not
  reach significance (|t| = 1.2-1.7 against 2.78 at the 5% level).
* Discrimination is the same (slice AUROC 0.799 vs 0.800). The accuracy gap
  comes from where the 0.5 threshold falls: over all folds the CNN misses 90 AD
  patients and over-calls 29 NC patients, ConvNeXt 59 and 45.
* ConvNeXt overfits: final training slice accuracy 0.95-0.96 against 0.69-0.75
  on validation (CNN: 0.87-0.89).

![Fold-1 training and validation slice loss and accuracy of the three models](figures/learning_curves.png)

*Figure 3. Fold-1 training (dashed) and validation (solid) slice loss and accuracy of the three models (`history.csv`).*

Fold 1 (dashed = training). The ImageNet-initialised model fits the training
slices within about five epochs; its validation accuracy then stays flat while
the validation loss rises, i.e. its wrong predictions become more confident.
As the tutor noted during the 14 October check-off, early stopping on validation
loss (or a shorter fixed epoch schedule) would curb this progressive logit
inflation; our fixed-schedule + EMA design instead keeps the outer validation
fold strictly out of checkpoint selection and corrects the resulting epoch-40
over-confidence post-hoc via Platt scaling (`slope a = 0.246`) on the
independent calibration set.

### Acquisition confound

The metadata file names show that acquisition period and scanner are entangled
with the diagnosis: all 417 development scans acquired 2011-2016 are AD, and
all 547 NC scans were acquired 2017-2021 with accelerated sequences. AD scans
acquired 2017-2021 are misclassified far more often (ConvNeXt scan error 0.67
vs 0.30 for earlier AD scans; CNN 0.84 vs 0.45). On the 633 scans acquired
2017-2021, where both classes share the same period, scan AUROC falls from
0.850 to 0.729 for ConvNeXt and from 0.839 to 0.654 for the CNN. A probe
network recognises the scanner vendor from pixels almost perfectly (scan AUROC
0.986 pooled over five folds, on NC scans of one period). Part of the development accuracy
therefore reflects acquisition style rather than anatomy. Details, probe
results and possible mitigations: [docs/ACQUISITION_CONFOUND.md](docs/ACQUISITION_CONFOUND.md).

### Fold 4-5 screens and five-fold ImageNet initialisation

Two attempts to weaken the acquisition shortcut were screened on folds 4 and 5
(one seed each; details in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)).
Validation accuracy slice / scan / patient, and scan AUROC on the 2017-2021
validation scans:

| Run | Fold 4 | Fold 5 | 2017-21 scan AUROC (f4, f5) |
|---|---|---|---|
| ConvNeXt-Tiny scratch, strong aug. (round 3) | 0.749 / 0.767 / 0.771 | 0.685 / 0.672 / 0.747 | 0.818, 0.696 |
| SmallCNN, strong aug. (round 3) | 0.733 / 0.781 / 0.771 | 0.664 / 0.708 / 0.705 | 0.718, 0.560 |
| ConvNeXt-Tiny scratch, scanner aug. | 0.738 / 0.763 / 0.729 | 0.720 / 0.774 / 0.789 | 0.779, 0.725 |
| SmallCNN, scanner aug. | 0.758 / 0.828 / 0.823 | 0.756 / 0.764 / 0.768 | 0.752, 0.710 |
| ConvNeXt-Tiny ImageNet-1k, strong aug. | 0.817 / 0.847 / 0.844 | 0.773 / 0.774 / 0.821 | 0.770, 0.740 |
| ConvNeXt-Tiny ImageNet-1k, scanner aug. | 0.801 / 0.814 / 0.802 | 0.773 / 0.785 / 0.789 | 0.745, 0.719 |

* ImageNet initialisation (cited weights, 40 epochs) gives the best overall
  accuracy so far, but discrimination within 2017-2021 stays at the level of
  the scratch model: the gain comes mostly from scans whose acquisition style
  already identifies the class.
* Scanner-style augmentation helps the CNN on plain accuracy and 2017-2021 scan
  AUROC (though on fold 5 it misses 26/26 `2017-2021 AD` scans, reducing fold-5
  2017-2021 scan balanced accuracy to 0.452), is mixed for scratch ConvNeXt and
  adds nothing to the pretrained model. We retained `strong` augmentation across
  all three pre-declared five-fold ensembles for a controlled single-variable
  comparison, and later ran a post-hoc five-fold `SmallCNN + scanner aug.`
  control after the locked test run (see *Post-hoc control: SmallCNN with
  scanner augmentation* below).
* Five folds (round 6, [`config/experiments_round6_pretrained_cv.json`](config/experiments_round6_pretrained_cv.json),
  Slurm 656072), same patients as the table above:

| Model | Slice acc. | Scan acc. | Patient acc. | Scan AUROC | 2017-21 scan AUROC |
|---|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny, ImageNet-1k init. | **0.798 ± 0.028** | **0.828 ± 0.039** | **0.840 ± 0.019** | 0.920 ± 0.014 | 0.778 ± 0.047 |
| ConvNeXt-Tiny, scratch | 0.715 ± 0.028 | 0.748 ± 0.054 | 0.782 ± 0.032 | 0.847 ± 0.035 | 0.736 ± 0.048 |
| SmallCNN | 0.690 ± 0.032 | 0.701 ± 0.054 | 0.750 ± 0.032 | 0.853 ± 0.072 | 0.606 ± 0.081 |

  Paired differences across the five folds (paired t, 4 df; 2.78 = 5% two-sided threshold):

| Comparison | Slice acc. | Scan acc. | Patient acc. | 2017-21 scan AUROC |
|---|---|---|---|---|
| pretrained - scratch ConvNeXt-Tiny | +0.083 (t = 14.3) | +0.080 (t = 6.4) | +0.059 (t = 3.4) | +0.042 (t = 1.5) |
| pretrained - SmallCNN | +0.108 (t = 6.1) | +0.128 (t = 3.9) | +0.090 (t = 7.1) | +0.172 (t = 5.1) |
| scratch ConvNeXt-Tiny - SmallCNN | +0.025 (t = 1.6) | +0.047 (t = 1.2) | +0.032 (t = 1.7) | +0.130 (t = 5.4) |

  Within 2017-2021, the pretrained model's gain over the scratch model (+0.042
  AUROC, t = 1.5) is not significant and it still misses 63 of 86 AD scans
  acquired then. Details: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

**Why the folds differ.** Per-fold slice / scan / patient accuracy of the
pretrained model, and the number of 2017-2021 AD scans in each validation fold:

| Fold | Slice acc. | Scan acc. | Patient acc. | 2017-21 AD scans in fold (missed) |
|---:|---:|---:|---:|---:|
| 1 | 0.836 | 0.877 | 0.863 | 10 (7) |
| 2 | 0.778 | 0.810 | 0.821 | 17 (15) |
| 3 | 0.783 | 0.834 | 0.853 | 9 (5) |
| 4 | 0.817 | 0.847 | 0.844 | 24 (14) |
| 5 | 0.773 | 0.774 | 0.821 | 26 (22) |

Fold-to-fold variance stems from (1) ~95 validation patients per fold (±0.04
sampling error), (2) an uneven share of the hard 2017-2021 AD group (9 to 26
scans per fold), and (3) seed-to-seed training variance: in Round 2 (two seeds
on fold 1 for the scratch model), slice-level seed sd was tiny (0.001) but scan
(0.023) and patient (0.030) seed sd were comparable to across-fold patient sd
(0.019-0.032) because a few borderline patients flipping across the 0.5
threshold move patient accuracy by ~1 pp each (the five-fold comparisons use one
deterministic seed `3710 + fold` per fold). Pooled out-of-fold metrics across
all 476 development patients (`pooled_out_of_fold_round3_round6.json`) match the
fold means: pretrained ConvNeXt-Tiny `0.798` slice / `0.830` scan / `0.840`
patient accuracy, scratch ConvNeXt-Tiny `0.715 / 0.749 / 0.782`, and SmallCNN
`0.691 / 0.701 / 0.750`.

### Grad-CAM (folds 4 and 5, validation)

Share of the Grad-CAM mass outside the head on the middle slice of each
validation scan (background is 0.72 of the slice, so a uniform map scores
about 0.72): scratch ConvNeXt-Tiny 0.72-0.83, ImageNet-initialised
ConvNeXt-Tiny 0.36-0.46, SmallCNN 0.22-0.31, with little difference between
diagnosis/period groups. The scratch ConvNeXt's evidence is not localised to the
head. Tables and caveats:
[docs/ACQUISITION_CONFOUND.md](docs/ACQUISITION_CONFOUND.md#grad-cam-where-the-evidence-lies).

Mean input (top) and mean Grad-CAM (bottom) per diagnosis and acquisition
group, fold 5 validation scans:

| ConvNeXt-Tiny, scratch | ConvNeXt-Tiny, ImageNet init. | SmallCNN |
|---|---|---|
| ![](figures/adni/gradcam_means_scratch_fold5.png) | ![](figures/adni/gradcam_means_pretrained_fold5.png) | ![](figures/adni/gradcam_means_cnn_fold5.png) |

*Figure 4. Mean mid-sagittal input slice (top) and mean Grad-CAM heatmap (bottom) per diagnosis × acquisition period on fold-5 validation scans for scratch ConvNeXt-Tiny, ImageNet-1k ConvNeXt-Tiny, and SmallCNN.*

Individual scans for the ImageNet-initialised model (two most confident
correct and two most confident wrong per group; input slice, then Grad-CAM):

![Grad-CAM cases](figures/adni/gradcam_cases_pretrained_fold5.png)

*Figure 5. Grad-CAM of the ImageNet-initialised model on single fold-5 validation scans: two most confident correct and two most confident wrong scans per subgroup (input slice above, heatmap below; IDs removed).*

* **Scratch ConvNeXt-Tiny** (`gradcam_means_scratch_fold5.png`) scatters its
  attribution across the black corners and frame borders (`0.72-0.83` of map
  mass outside the head; only 1% of validation scans have more than 50% of the
  map inside the skull), reflecting coarse 7×8 final-stage cells driven
  by field-of-view boundary cues rather than brain morphology. Note that
  outside-head attribution alone does not explain why a model scores lower
  (SmallCNN keeps `98-99%` of scans' majority attribution inside the head yet
  scores similarly to or below scratch ConvNeXt in single-fold development);
  rather, it shows that a high-capacity 7×7 depthwise network trained from
  scratch on 381 patients latches onto background framing shortcuts.
* **ImageNet-1k–initialised ConvNeXt-Tiny** (`gradcam_means_pretrained_fold5.png`,
  `gradcam_cases_pretrained_fold5.png`) concentrates attribution inside the
  cranium for 82% / 72% of fold 4 / fold 5 scans (`0.36-0.46` outside-head
  share). Given the unregistered 2D slices and coarse 7×8 final-stage grid,
  this indoor focus should be read conservatively as a broad central
  intracranial region (roughly overlapping the mid-sagittal ventricular/callosal
  and deep parenchymal zone) rather than fine hippocampal localisation. Both
  confident correct scans and confident errors in `AD 2017-21` (p(AD) = 0.00)
  and `NC 2017-21` (p(AD) = 0.86–0.96) attend to this same central region;
  paired with the `vendor_within_nc` probe (`0.986` scan AUROC), this is
  consistent with the network reading global intracranial contrast/protocol
  signatures alongside gross ventricular morphology.
* **SmallCNN** (`gradcam_means_cnn_fold5.png`) keeps `98-99%` of scans' majority
  attribution inside the head (`0.22-0.31` outside), tracing high-contrast
  cortical and cerebellar boundaries.

### Final test (136 patients, scored once)

The model, the comparisons and the calibration procedure were declared before
the test job (Slurm 659813, `PREDECLARATION.txt` 2026-10-14T14:57+10:00):
primary model = the five round-6 fold models averaged at the logit level;
the scratch ConvNeXt-Tiny and CNN fold ensembles are reported alongside. Each
ensemble fits Platt scaling and its referral threshold on the 68 calibration
patients, then scores the test patients once.

To separate the effect of five-fold logit ensembling from Platt threshold
calibration, the table below reports test accuracy **before → after** Platt
scaling alongside AUROC (`final_eval_20261014/*/test/metrics.json`):

| Five-fold ensemble | Slice acc. (uncal → cal) | Scan acc. (uncal → cal) | Patient acc. (uncal → cal) | Patient bal. acc. (uncal → cal) | Slice AUROC | Scan AUROC | Patient AUROC |
|---|---|---|---|---|---:|---:|---:|
| ConvNeXt-Tiny, ImageNet-1k init. (primary) | 0.862 → **0.867** | 0.879 → **0.889** | 0.882 → **0.882** | 0.854 → **0.866** | 0.943 | 0.964 | 0.944 |
| ConvNeXt-Tiny, scratch | 0.791 → 0.793 | 0.824 → 0.817 | 0.809 → 0.801 | 0.782 → 0.806 | 0.882 | 0.910 | 0.876 |
| SmallCNN | 0.783 → 0.826 | 0.814 → 0.873 | 0.824 → 0.875 | 0.751 → 0.860 | 0.908 | 0.947 | 0.927 |

Patient level, per class (after Platt scaling):

| Five-fold ensemble | AD precision / recall | NC precision / recall | Macro-F1 | Confusion [TN FP; FN TP] |
|---|---|---|---:|---|
| ConvNeXt-Tiny, ImageNet-1k init. | 0.818 / 0.818 | 0.913 / 0.913 | 0.866 | [84 8; 8 36] |
| ConvNeXt-Tiny, scratch | 0.655 / 0.818 | 0.901 / 0.793 | 0.786 | [73 19; 8 36] |
| SmallCNN | 0.800 / 0.818 | 0.912 / 0.902 | 0.858 | [83 9; 8 36] |

Scans by acquisition period (test: 122 AD scans 2011-2016 across 28 AD patients;
27 AD and 157 NC scans 2017-2021 across 16 AD and 92 NC patients):

| Five-fold ensemble | 2011-16 scan acc. | 2017-21 scan acc. | 2017-21 scan AUROC | 2017-21 AD scans missed |
|---|---:|---:|---:|---:|
| ConvNeXt-Tiny, ImageNet-1k init. | 0.951 (116/122) | 0.848 | 0.878 | 14/27 |
| ConvNeXt-Tiny, scratch | 0.893 (109/122) | 0.766 | 0.777 | 9/27 |
| SmallCNN | 0.926 (113/122) | 0.837 | 0.866 | 11/27 |

**Summary of feasibility acceptance criteria:**

| Criterion | Target | Measured result (primary ensemble) | Status |
|---|---|---|---|
| 1. Leakage-free split & replay | 0 shared patients; exact replay | 0 overlap across 21 manifests; 15/15 checkpoints replay bit-for-bit | Met |
| 2. Held-out test accuracy | ≥ 0.80 at slice level (scan/patient reported) | Slice **0.867**, scan **0.889**, patient **0.882** (AUROC 0.943 / 0.964 / 0.944) | Met on this test split |
| 3. Fair baseline & calibration metrics | Paired ConvNeXt vs SmallCNN + ECE | 5-fold paired CV + test ensemble per-class P/R, macro-F1, AUROC, ECE before/after Platt | Met |
| 4. Frozen referral rule & failure autopsy | τ frozen on calibration; ≥ 90% automated acc. | τ = 0.626 → **87.5%** test coverage (`119/136`), **92.4%** automated accuracy, `17` referred | Met on this test split (not guaranteed for future cohorts) |
| 5. Single-A100 resource profile | Params, VRAM (train/infer), latency | 27.8M params, 4,677 MiB train / 586 MiB batch-1 infer VRAM, 5.37 ms/slice (~0.54 s/scan ensemble) | Met |

* **Disentangling ensembling vs Platt calibration vs split composition:**
  1. *Five-fold ensembling + test split (before Platt scaling):* Averaging logits
     across the five fold checkpoints lifts uncalibrated slice accuracy above the
     single-fold development means for all three architectures (`0.798 -> 0.862`
     pretrained ConvNeXt, `0.715 -> 0.791` scratch ConvNeXt, `0.690 -> 0.783`
     SmallCNN). Because `PREDECLARATION.txt` committed to scoring the test set
     once per five-fold ensemble rather than evaluating individual fold
     checkpoints on test, this jump reflects the combined effect of five-fold
     logit ensembling and split-to-split variation between the 476 development
     patients and 136 test patients. Crucially, at this uncalibrated ensemble
     stage on the exact same 136 test patients, pretrained ConvNeXt still leads
     SmallCNN by **+0.079** slice (`0.862` vs `0.783`), **+0.065** scan (`0.879`
     vs `0.814`), **+0.059** patient accuracy (`0.882` vs `0.824`) and **+0.103**
     patient balanced accuracy (`0.854` vs `0.751`), because uncalibrated SmallCNN
     misses 20 of the 44 AD test patients at the raw 0.5 threshold.
  2. *Platt intercept calibration:* Fitting Platt scaling on the 68 calibration
     patients leaves pretrained ConvNeXt's test patient accuracy unchanged
     (`0.882 -> 0.882`, balanced accuracy `0.854 -> 0.866`), whereas SmallCNN's
     large positive intercept (`b = +0.813`) shifts its operating threshold down
     in raw logit space, recovering 12 of its 20 missed AD patients and jumping
     its test patient accuracy by **+0.051** (`0.824 -> 0.875`) and slice accuracy
     by **+0.043** (`0.783 -> 0.826`).
* **2017-2021 subgroup on test vs development:** Although pretrained ConvNeXt
  outperformed SmallCNN by +0.172 scan AUROC on 2017-2021 scans in single-fold
  development (`0.778` vs `0.606`), on the 184 test scans from 2017-2021 their
  ensemble scan AUROCs are nearly tied (`0.878` vs `0.866`), and at the
  calibrated threshold the pretrained ensemble misses `14/27` 2017-2021 AD scans
  (versus `11/27` for SmallCNN and `9/27` for scratch ConvNeXt, which achieves
  higher AD recall only by over-calling `34/157` NC scans as AD).

![Per-fold validation accuracy of the single fold models and test accuracy of the fold ensembles](figures/fold_accuracy.png)

*Figure 6. Per-fold validation accuracy of single-fold models (bars) and test accuracy of the five-fold ensembles after Platt scaling (stars); dotted line marks the 0.80 coursework target.*

**How certain are these test differences?** 136 test patients is a small
sample (1 patient = 0.74 percentage points). Patient-level 95% intervals from
10,000 stratified bootstrap resamples (AD and NC patients resampled separately),
paired differences on the same resamples, and an exact McNemar test on
per-patient correctness (computed after the test run from the saved predictions;
`posthoc_patient_analysis.json`):

| Five-fold ensemble | Patient acc. [95% CI] | Balanced acc. [95% CI] | Patient AUROC [95% CI] |
|---|---|---|---|
| ConvNeXt-Tiny, ImageNet-1k init. | 0.882 [0.824, 0.934] | 0.866 [0.798, 0.927] | 0.944 [0.902, 0.977] |
| ConvNeXt-Tiny, scratch | 0.801 [0.728, 0.868] | 0.806 [0.733, 0.873] | 0.876 [0.812, 0.932] |
| SmallCNN | 0.875 [0.816, 0.926] | 0.860 [0.792, 0.921] | 0.927 [0.875, 0.968] |

| Comparison | Δ patient acc. [95% CI] | Δ AUROC [95% CI] | McNemar (only A right / only B right, p) |
|---|---|---|---|
| pretrained - SmallCNN | +0.007 [-0.029, +0.044] | +0.017 [-0.013, +0.047] | 4 / 3, p = 1.00 |
| pretrained - scratch ConvNeXt | +0.081 [+0.022, +0.140] | +0.067 [+0.029, +0.111] | 14 / 3, p = 0.013 |
| SmallCNN - scratch ConvNeXt | +0.074 [+0.015, +0.140] | +0.051 [+0.001, +0.106] | 15 / 5, p = 0.041 |

After five-fold ensembling and Platt threshold calibration, the patient-level
difference between the pretrained ConvNeXt and `strong`-augmented SmallCNN
ensembles is not statistically significant on the 136 test subjects (`120/136`
vs `119/136`, disagreeing on 7 of 136 patients, McNemar p = 1.00; with n = 136
patients, non-significance does not prove equivalence), while pretrained
ConvNeXt retains a +0.041 slice-accuracy lead (`0.867` vs `0.826`) and an
advantage in the 476-patient paired single-fold development comparison
(`+0.090` patient accuracy vs `strong` SmallCNN, t = 7.1; `+0.029` vs post-hoc
`scanner` SmallCNN, t = 1.5) and uncalibrated test ensemble comparison (`0.882`
vs `0.824` `strong` / `0.838` `scanner`). Both pre-declared ensembles
significantly outperform the scratch ConvNeXt ensemble on this test split.

## Resource profiling

Development measurements (A100, FP32, batch 32, 224×224 single-slice input,
30 epochs on 14,900 training slices; forward latency over 100 runs after 10
warm-up runs, excluding data loading):

| Model | Parameters | Peak training memory | Training time | Latency batch 1 | Latency batch 64 |
|---|---:|---:|---:|---:|---:|
| SmallCNN | 97,521 | 594 MiB | 328 s | 1.02 ± 0.02 ms | 3.90 ± 0.13 ms |
| ConvNeXt-Tiny | 27,817,825 | 3,756 MiB | 1,629 s | 4.70 ± 0.14 ms | 53.56 ± 0.16 ms |

Five-fold runs and final test benchmark (A100-PCIE-40GB, FP32, batch 32, 240×256
three-slice input on 16,420-17,100 training slices per fold):

| Model | Parameters | Peak train VRAM | Peak eval VRAM (batch 32) | Peak infer VRAM (batch 1 / 64, dev; test) | Training time per fold | Latency batch 1 (dev / test) | Latency batch 64 (dev / test) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SmallCNN (80 epochs) | 97,809 | 727 MiB | 328 MiB | 73 / 591 MiB; 42 / 559 MiB | 3,078 s (3,024-3,143) | 1.16 ± 0.01 / 1.14 ± 0.01 ms | 4.83 ± 0.42 / 4.74 ± 0.01 ms |
| ConvNeXt-Tiny, scratch (80 epochs) | 27,820,897 | 4,656 MiB | 1,137 MiB | 415 / 1,433 MiB; 585 / 1,603 MiB | 6,038 s (5,897-6,138) | 5.19 ± 0.04 / 5.27 ± 0.07 ms | 64.37 ± 0.17 / 64.30 ± 0.26 ms |
| ConvNeXt-Tiny, ImageNet-1k (40 epochs) | 27,820,897 | 4,677 MiB | 1,160 MiB | 416 / 1,479 MiB; 586 / 1,649 MiB | ~3,000 s (~75 s/epoch) | 4.93 ± 0.11 / 5.37 ± 0.05 ms | 64.28 ± 0.23 / 64.29 ± 0.23 ms |

ConvNeXt-Tiny has 284× the parameters (`27,820,897 / 97,809`), 6.4× the peak
training memory and 4.3×–4.7× the single-slice forward latency of SmallCNN (4.3×
for the pretrained model in fold-1 development, `4.93 vs 1.16 ms`; 4.5× for the
scratch model in fold-1 development, `5.19 vs 1.16 ms`; and 4.7× in the final
test benchmark, `5.37 vs 1.14 ms`). Trained from scratch (80 epochs), that extra
cost yields a non-significant accuracy gain; initialised from ImageNet-1k
(40 epochs), its per-fold training time matches 80-epoch SmallCNN (~50 min/fold),
and the five-fold ensemble scores a 20-slice scan in ~0.54 s on one A100.

## Calibration and referral

Raw network confidence is not trusted. On the 68 calibration patients,
`predict.py --role calibration` fits Platt scaling, p = sigmoid(a * logit + b)
(Platt, 1999; Guo et al., 2017 [4]), by minimising the slice log loss. The slope a
corrects over- or under-confidence; the intercept b moves the decision boundary,
which matters when uncalibrated models have a shifted operating threshold. It then picks the lowest patient
confidence threshold τ at which the automated calibration patients reach 90%
accuracy (`--target-accuracy`). Both values are frozen in `calibration.json`
and applied unchanged to the test patients, whose confidence below τ is
reported as `REFER`. The reports include ECE (15 bins), reliability diagrams,
confidence histograms of correct/incorrect predictions and risk-coverage curves.

Expected calibration error (15 bins, predicted-class confidence) on the test
patients before / after Platt scaling. The ECE before scaling was computed
afterwards from the saved test logits with the same function
(`ece_before_after_platt.json`):

| Five-fold ensemble | Platt slope a / intercept b | Slice ECE | Scan ECE | Patient ECE |
|---|---|---|---|---|
| ConvNeXt-Tiny, ImageNet-1k init. | 0.246 / 0.193 | 0.112 → 0.036 | 0.043 → 0.059 | 0.062 → 0.049 |
| ConvNeXt-Tiny, scratch | 0.354 / 0.500 | 0.144 → 0.022 | 0.076 → 0.057 | 0.076 → 0.070 |
| SmallCNN (`strong`, pre-declared) | 0.957 / 0.813 | 0.033 → 0.029 | 0.041 → 0.109 | 0.073 → 0.112 |
| SmallCNN (`scanner`, post-hoc) | 1.071 / -0.063 | — → 0.022 | — → 0.064 | — → 0.078 |

Referral rule (τ fitted on the 68 calibration patients for 90% accuracy of
automated patients, applied unchanged to the 136 test patients):

| Five-fold ensemble | τ | Calibration coverage | Test coverage | Test accuracy of automated patients | Test patients referred | AD 2017-21 patients automated as NC |
|---|---:|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny, ImageNet-1k init. | 0.626 | 0.941 (64/68) | 0.875 (119/136) | 0.924 (110/119) | 17 | 5 |
| ConvNeXt-Tiny, scratch | 0.836 | 0.397 (27/68) | 0.346 (47/136) | 0.915 (43/47) | 89 | 0 |
| SmallCNN (`strong`) | 0.563 | 0.897 (61/68) | 0.912 (124/136) | 0.903 (112/124) | 12 | 5 |

* All three pre-declared ensembles meet the 90% target on this test split, though fitting τ on 68
  calibration patients carries finite-sample variance (so ≥ 90% is not guaranteed on future cohorts, as the post-hoc `scanner` SmallCNN control illustrates below): for the primary ensemble,
  coverage shifts by 6.6 percentage points from calibration (`64/68 = 0.941` at
  `0.906` accuracy) to test (`119/136 = 0.875` at `0.924` accuracy).
* The ImageNet-initialised ensemble is strongly over-confident at slice level
  (slope 0.246), and slice-level Platt scaling cuts its slice ECE threefold
  (`0.112 -> 0.036`) and lowers patient ECE (`0.062 -> 0.049`), while scan ECE
  rises slightly (`0.043 -> 0.059`). Because patient probabilities average
  20-160 slices (1-8 scans) and face a `32.4%` patient AD prior (whereas AD
  accounts for `53.5%` of the 3,400 calibration slices because AD patients have
  more longitudinal scans on average: `4.1` vs `1.7`), applying slice-fitted
  Platt parameters (`b = +0.813`) directly to patient averages increases
  `strong` SmallCNN's scan (`0.041 -> 0.109`) and patient (`0.073 -> 0.112`)
  ECE.
* The scratch ConvNeXt needs a high threshold (`tau = 0.836`) and refers 89 of 136
  patients (65.4%).
* Under the symmetric confidence rule `max(p, 1-p) >= tau`, both deployable
  ensembles still automate 5 of the 16 `AD 2017-2021` test patients as NC.

**Post-hoc comparison: Slice-level vs Patient-level Platt calibration (not in `PREDECLARATION.txt`).**
Because final referral decisions are made per patient, we directly test whether
`strong` SmallCNN's elevated post-Platt patient ECE (`0.112`) stems from
slice-level Platt fitting by fitting Platt scaling `(a_pat, b_pat)` and the
90%-accuracy referral threshold `tau_pat` **directly on the 68 calibration
patients' aggregated scores** (evaluating both each patient's mean slice logit
and the logit of each patient's mean uncalibrated slice probability, then
applying `(a_pat, b_pat, tau_pat)` unchanged to the 136 test patients):

| Five-fold ensemble & calibration unit | Fitted `a / b` | `tau` | Calibration cov. (auto. acc.) | Test patient acc. (bal. acc.) | Test patient ECE | Test cov. (`accepted/136`) | Test auto. acc. |
|---|---|---:|---|---|---|---|---|
| **ConvNeXt-Tiny, ImageNet-1k** — Uncalibrated | `1.000 / 0.000` | — | — | 0.882 (0.854) | 0.062 | — | — |
| — Slice-fitted Platt (pre-declared) | `0.246 / +0.193` | 0.626 | 0.941 (0.906) | **0.882 (0.866)** | 0.049 | 0.875 (`119/136`) | **0.924** (`110/119`) |
| — Patient-mean-logit Platt (post-hoc) | `0.309 / -0.519` | 0.611 | 0.956 (0.908) | **0.882 (0.842)** | **0.043** | 0.934 (`127/136`) | 0.890 (`113/127`) |
| — Patient-mean-prob Platt (post-hoc) | `0.549 / -0.241` | 0.561 | 0.956 (0.908) | 0.875 (0.836) | **0.041** | 0.941 (`128/136`) | 0.898 (`115/128`) |
| **ConvNeXt-Tiny, scratch** — Uncalibrated | `1.000 / 0.000` | — | — | 0.809 (0.782) | 0.076 | — | — |
| — Slice-fitted Platt (pre-declared) | `0.354 / +0.500` | 0.836 | 0.397 (0.926) | 0.801 (0.806) | 0.070 | 0.346 (`47/136`) | 0.915 (`43/47`) |
| — Patient-mean-logit Platt (post-hoc) | `0.449 / -0.324` | 0.682 | 0.735 (0.900) | 0.801 (0.752) | **0.047** | 0.831 (`113/136`) | 0.858 (`97/113`) |
| **SmallCNN (`strong`)** — Uncalibrated | `1.000 / 0.000` | — | — | 0.824 (0.751) | 0.073 | — | — |
| — Slice-fitted Platt (pre-declared) | `0.957 / +0.813` | 0.563 | 0.897 (0.902) | 0.875 (0.860) | 0.112 | 0.912 (`124/136`) | 0.903 (`112/124`) |
| — Patient-mean-logit Platt (post-hoc) | `1.083 / +0.215` | 0.638 | 0.897 (0.902) | 0.846 (0.785) | **0.038** | 0.868 (`118/136`) | 0.898 (`106/118`) |
| — Patient-mean-prob Platt (post-hoc) | `1.212 / +0.174` | 0.632 | 0.853 (0.914) | 0.838 (0.774) | **0.055** | 0.890 (`121/136`) | 0.884 (`107/121`) |
| **SmallCNN (`scanner`, post-hoc)** — Slice-fitted Platt | `1.071 / -0.063` | 0.508 | 1.000 (0.912) | 0.838 (0.821) | 0.078 | 0.993 (`135/136`) | 0.837 (`113/135`) |
| — Patient-mean-prob Platt (post-hoc) | `1.403 / -0.753` | 0.579 | 0.941 (0.906) | 0.838 (0.786) | **0.057** | 0.926 (`126/136`) | 0.865 (`109/126`) |

* **Direct verification of the ECE explanation:** Fitting Platt scaling at the
  patient level cuts test patient ECE across all four ensembles (`0.049 -> 0.043/0.041`
  for pretrained ConvNeXt, `0.070 -> 0.047` for scratch ConvNeXt, `0.078 -> 0.057`
  for `scanner` SmallCNN, and **`0.112 -> 0.038`** for `strong` SmallCNN),
  confirming empirically that `strong` SmallCNN's elevated post-Platt patient
  ECE (`0.112`) was caused by applying slice-fitted Platt parameters
  (`b = +0.813`) to variance-compressed patient averages rather than by an
  intrinsic inability of SmallCNN to calibrate at the patient level.
* **Threshold stability of pretrained ConvNeXt vs `strong` SmallCNN:** Without
  the `53.5%` slice-prior intercept boost (`b = +0.813`), `strong` SmallCNN's
  test patient accuracy settles at `0.846` (`115/136`, balanced accuracy `0.785`;
  `0.838` under prob-logit), whereas pretrained ConvNeXt holds `0.882` (`120/136`)
  under both slice-fitted and patient-mean-logit Platt calibration.
* **Finite-sample trade-off on 68 calibration patients:** Fitting all three
  patient-level parameters (`a_pat, b_pat, tau_pat`) on the same 68 calibration
  patients yields slightly optimistic `tau_pat` values on calibration, leaving
  automated test accuracy at `89.0%-89.8%` for pretrained ConvNeXt and `strong`
  SmallCNN (while improving `scanner` SmallCNN's automated test accuracy from
  `83.7%` to `86.5%`), reinforcing the need for a larger calibration cohort.

**Post-hoc sensitivity analysis: asymmetric referral (`tau_NC > tau_AD`, not in `PREDECLARATION.txt`).**
The locked test evaluation above used the pre-declared symmetric rule `tau_AD = tau_NC = 0.626`.
Because automating a true AD patient as NC (false negative) is clinically more
harmful than automating an NC patient as AD (which triggers specialist follow-up),
we also inspect an exploratory **post-hoc sensitivity curve** for the asymmetric rule
`REFER if p(AD) < tau_AD and p(NC) < tau_NC` with `tau_AD = 0.626` fixed and four
candidate values of `tau_NC`:

| Exploratory rule on primary ensemble (`tau_AD = 0.626`) | `tau_NC` | Calibration coverage (accepted acc.) | Calibration auto-FN / auto-FP | Test coverage (accepted acc.) | Test auto-FN (`2017-21 AD` missed & automated) / auto-FP |
|---|---:|---|---|---|---:|
| Pre-declared symmetric rule (`tau_NC = tau_AD`) | 0.626 | 0.941 (64/68, acc. 0.906) | 2 / 4 | 0.875 (119/136, acc. 0.924) | 5 / 4 |
| Post-hoc sensitivity point 1 (above cal. FN `p(NC)=0.662`) | 0.750 | 0.824 (56/68, acc. 0.911) | 1 / 4 | 0.801 (109/136, acc. 0.936) | 3 / 4 |
| Post-hoc sensitivity point 2 | 0.800 | 0.794 (54/68, acc. 0.907) | 1 / 4 | 0.779 (106/136, acc. 0.943) | 2 / 4 |
| Post-hoc sensitivity point 3 (above cal. FN `p(NC)=0.910`) | 0.920 | 0.588 (40/68, acc. 0.900) | 0 / 4 | 0.566 (77/136, acc. 0.948) | 0 / 4 |

On the 68-patient calibration set there are only 22 AD patients and only 2 AD
false negatives—at `p(NC) = 0.662` (`p(AD) = 0.338`) and `p(NC) = 0.910`
(`p(AD) = 0.090`)—with zero calibration AD patients in between. Consequently,
calibration data alone cannot distinguish `tau_NC = 0.750` from `tau_NC = 0.800`
(in fact, `0.750` has slightly higher calibration accuracy, `0.911` vs `0.907`,
because moving from `0.750` to `0.800` only refers 2 true NC calibration
patients). We therefore report the table strictly as a post-hoc trade-off curve
rather than claiming a single calibration-selected `tau_NC`: raising `tau_NC`
into the `0.750-0.800` range reduces automated `2017-2021 AD` test false
negatives from `5` to `2-3` at `77.9-80.1%` test coverage (`93.6-94.3%`
accuracy), whereas eliminating all automated AD false negatives (`0/5`,
`tau_NC = 0.920`) cuts test coverage to `56.6%` (`77/136`). Mitigating the
acquisition shortcut more reliably therefore requires protocol-aware referral
and a larger calibration cohort rather than post-hoc confidence thresholding
alone.

![Left: slice reliability after Platt scaling. Middle and right: patient confidence of correct and wrong test predictions with the frozen tau](figures/test_calibration.png)

*Figure 7. Left: slice reliability on test after Platt scaling. Middle and right: patient confidence distributions for correct and wrong test predictions of the primary ConvNeXt and SmallCNN ensembles with the frozen threshold τ.*

**Does confidence separate right from wrong?** Patient level, after Platt
scaling (`posthoc_patient_analysis.json`). AUROC of confidence = probability
that a correct patient receives higher confidence than a wrong one; AURC =
area under the risk-coverage curve (lower is better):

| Five-fold ensemble | Total wrong patients (before referral) | Mean confidence correct / wrong | Wrong with confidence ≥ 0.9 | AUROC of confidence | AURC [95% CI] |
|---|---:|---|---:|---:|---|
| ConvNeXt-Tiny, ImageNet-1k init. | 16 (8 FP, 8 FN) | 0.854 / 0.682 | 1 | 0.838 | 0.030 [0.012, 0.053] |
| ConvNeXt-Tiny, scratch | 27 (19 FP, 8 FN) | 0.772 / 0.678 | 1 | 0.702 | 0.101 [0.054, 0.159] |
| SmallCNN | 17 (9 FP, 8 FN) | 0.783 / 0.645 | 1 | 0.783 | 0.044 [0.018, 0.080] |

![Patient risk-coverage curves on test with the operating point of the frozen tau](figures/test_risk_coverage.png)

*Figure 8. Patient risk-coverage curves on test with the operating point of the frozen threshold τ for each five-fold ensemble.*

* The pretrained ensemble ranks its own errors best: most of its 16 unreferred
  patient errors (8 FP and 8 FN across the 136 test patients) sit at low
  confidence, so 7 of those 16 errors fall below τ = 0.626 and are referred. Its
  AURC (`0.030` vs `0.044`) is lower than SmallCNN's in 93% of bootstrap
  resamples, though the 95% confidence interval of the difference (`-0.015`,
  95% CI `[-0.037, +0.004]`) crosses zero and should be treated as a directional
  rather than statistically conclusive advantage on 136 patients.
* Note the three distinct `16`s in the test-set breakdown: (i) `16/136` total
  patients misclassified by the primary model at the 0.5 threshold before
  referral (`8 FP` and `8 FN`, comprising `7` FN from 2017-2021 and `1` FN from
  2011-2016); (ii) `16` total test AD patients scanned in 2017-2021 (of whom `7`
  are missed at 0.5: `5` automated as NC and `2` referred); and (iii) `16` of
  the `17` referred test patients (`9 NC` and `7 AD`) come from 2017-2021 scans
  (the 17th referred patient is the single missed `2011-2016 AD` patient, Case 10
  below).

### Post-hoc control: SmallCNN with scanner augmentation (not in `PREDECLARATION.txt`)

Added on 2026-10-21, after the locked test run, in response to review. Because
`scanner` augmentation helped `SmallCNN` on folds 4-5 in Round 4 while the
pre-declared five-fold baseline kept `strong` augmentation, folds 1-3 of
`SmallCNN + scanner aug.` were trained with the identical Round-4 recipe (80
epochs, LR 1e-3, weight decay 1e-4, EMA 0.999, final-epoch weights, seed
`3710 + fold`; Slurm 681962, all five checkpoints replaying bit-for-bit) and
evaluated through the identical five-fold ensemble calibration and test pipeline
(Slurm 681963; `results/posthoc_cnn_scanner_20261021/`). This control was not
used to select or alter the primary model or its threshold.

Development, five folds, same validation patients (mean ± sd):

| Model | Slice acc. | Scan acc. | Patient acc. | Slice / scan AUROC | Patient AUROC | 2017-21 scan AUROC | Patient FP / FN (all folds) |
|---|---:|---:|---:|---|---:|---:|---:|
| ConvNeXt-Tiny, ImageNet-1k | 0.798 ± 0.028 | 0.828 ± 0.039 | 0.840 ± 0.019 | 0.885 ± 0.021 / 0.920 ± 0.014 | 0.905 ± 0.025 | 0.778 ± 0.047 | 24 / 52 |
| SmallCNN, `scanner` aug. (post-hoc) | 0.769 ± 0.016 | 0.808 ± 0.026 | 0.811 ± 0.042 | 0.850 ± 0.020 / 0.910 ± 0.015 | 0.893 ± 0.037 | 0.726 ± 0.090 | 41 / 49 |
| SmallCNN, `strong` aug. (pre-declared) | 0.690 ± 0.032 | 0.701 ± 0.054 | 0.750 ± 0.032 | 0.800 ± 0.068 / 0.853 ± 0.072 | 0.840 ± 0.068 | 0.606 ± 0.081 | 29 / 90 |

| Paired difference (t, 4 df; 2.78 = 5% threshold) | Slice acc. | Scan acc. | Patient acc. | 2017-21 scan AUROC |
|---|---|---|---|---|
| `scanner` - `strong` SmallCNN | +0.079 (t = 4.6) | +0.107 (t = 4.2) | +0.061 (t = 6.7) | +0.121 (t = 4.0) |
| pretrained - `scanner` SmallCNN | +0.029 (t = 3.0) | +0.021 (t = 2.0) | +0.029 (t = 1.5) | +0.052 (t = 2.1) |

Test, 136 patients, same pipeline (calibrated unless marked):

| Five-fold ensemble | Slice / scan / patient acc. (uncal → cal) | Patient bal. acc. (uncal → cal) | Patient AUROC | Platt a / b | τ | Test coverage (automated acc.) | Referred | AD 2017-21 patients automated as NC |
|---|---|---|---:|---|---:|---|---:|---:|
| ConvNeXt-Tiny, ImageNet-1k | 0.862 → 0.867 / 0.879 → 0.889 / 0.882 → 0.882 | 0.854 → 0.866 | 0.944 | 0.246 / 0.193 | 0.626 | 0.875 (0.924) | 17 | 5 |
| SmallCNN, `strong` aug. | 0.783 → 0.826 / 0.814 → 0.873 / 0.824 → 0.875 | 0.751 → 0.860 | 0.927 | 0.957 / 0.813 | 0.563 | 0.912 (0.903) | 12 | 5 |
| SmallCNN, `scanner` aug. (post-hoc) | 0.812 → 0.812 / 0.846 → 0.853 / 0.838 → 0.838 | 0.821 → 0.821 | 0.914 | 1.071 / -0.063 | 0.508 | 0.993 (0.837) | 1 | 9 |

* In five-fold development, `scanner` augmentation improves `SmallCNN` on all
  five folds and narrows its patient-accuracy gap to pretrained ConvNeXt from
  `+0.090 (t = 7.1)` to `+0.029 (t = 1.5, not significant, 3/5 folds)`, with
  pretrained ConvNeXt remaining ahead on all five folds at slice (`+0.029, t = 3.0`)
  and scan (`+0.021, t = 2.0`) levels.
* On the 136 test patients, however, `SmallCNN + scanner` reaches `0.838 [0.772, 0.897]`
  patient accuracy (`114/136`, 12 FP, 10 FN; AD precision / recall `0.739 / 0.773`,
  NC precision / recall `0.889 / 0.870`, macro-F1 `0.817`, confusion `[80 12; 10 34]`;
  slice / scan / patient AUROC `0.894 / 0.931 / 0.914 [0.859, 0.958]`; post-Platt
  slice / scan / patient ECE `0.022 / 0.064 / 0.078`, or `0.057` under
  patient-fitted Platt; 2017-2021 scan acc. / AUROC `0.804 / 0.805`, `14/27` AD
  scans missed; confidence AUROC `0.821`, AURC `0.050`), trailing pretrained
  ConvNeXt by `+0.044 [+0.007, +0.088]` (McNemar `7 / 1, p = 0.070`) and
  calibrated `strong` SmallCNN by `+0.037 [-0.015, +0.088]` (McNemar
  `9 / 4, p = 0.267`). Because all 68 calibration patients already reach `91.2%`
  accuracy (`62/68` at coverage `1.000`), τ drops to `0.508`, referring only
  `1/136` test patients and leaving automated test accuracy at `83.7%`
  (`113/135`, below the 90% target, with `9/16` `2017-2021 AD` patients
  automated as NC).

## Failure case autopsy

The five most confident slice-level errors of the primary ensemble on the
test set, as exported by `predict.py` (one slice per patient; titles give the
slice probability and the patient-level probability and decision; IDs removed).
All five come from scans acquired 2017-2021. The three false-positive slices
belong to NC patients that are also automated as AD at patient level; the two
false-negative slices belong to AD patients automated as NC, i.e. to the
confident misses in the AD 2017-2021 group described above.

![Most confident test failures](figures/adni/test_failures.png)

*Figure 9. Five most confident slice-level test failures of the primary ensemble (IDs removed). Titles report slice p(AD), patient-level p(AD) and decision.*

Cross-model comparison of the five most confident slice-level errors (Cases 1-5,
*Figure 9*), the four borderline referred patients (Cases 6-9, bottom row of
*Figure 2*), and the single misclassified `2011-2016 AD` patient (Case 10), from
`test_failure_cases.csv`, `scan_predictions.csv` and `test_patient_predictions.csv`
(thresholds: pretrained τ = 0.626, scratch τ = 0.836, SmallCNN τ = 0.563):

| Case | Period & true label | Scans | Slice p(AD) (pretrained) | Pretrained: patient p(AD), decision | Scratch: patient p(AD), decision | SmallCNN: patient p(AD), decision |
|---|---|---:|---:|---|---|---|
| 1 | 2017-21 NC | 1 | 0.971 | 0.84, AD (wrong) | 0.87, AD (wrong) | 0.54, REFER |
| 2 | 2017-21 AD | 3 | 0.040 | 0.31, NC (wrong) | 0.25, REFER | 0.29, NC (wrong) |
| 3 | 2017-21 NC | 2 | 0.961 | 0.63, AD (wrong) | 0.69, REFER | 0.68, AD (wrong) |
| 4 | 2017-21 AD | 1 | 0.047 | 0.09, NC (wrong) | 0.35, REFER | 0.10, NC (wrong) |
| 5 | 2017-21 NC | 1 | 0.953 | 0.87, AD (wrong) | 0.83, REFER | 0.77, AD (wrong) |
| 6 (Ex. 5) | 2017-21 AD | 1 | — | 0.50, REFER | 0.54, REFER | 0.53, REFER |
| 7 (Ex. 6) | 2017-21 AD | 1 | — | 0.51, REFER | 0.64, REFER | 0.46, REFER |
| 8 (Ex. 7) | 2017-21 NC | 2 | — | 0.49, REFER | 0.61, REFER | 0.27, NC (correct) |
| 9 (Ex. 8) | 2017-21 NC | 1 | — | 0.53, REFER | 0.51, REFER | 0.51, REFER |
| 10 (pre-2017 FN) | 2011-16 AD | 6 | — (scan p: 0.29-0.89) | 0.47, REFER | 0.45, REFER | 0.31, NC (wrong) |

Qualitative notes below are based on **non-clinician visual inspection** of the
mid-sagittal slices alongside the recorded model probabilities:

* **Cases 1 and 5 — Ventricular enlargement in 2017-2021 NC patients
  (confident false positives).** Visually, both single-scan NC subjects exhibit
  marked lateral-ventricle dilatation, a thinned corpus callosum and widened
  parieto-occipital sulci that resemble gross AD atrophy. Even though their
  2017-2021 acquisition period biases the model toward NC, this macroscopic
  atrophy-like appearance seems to dominate: the primary ensemble assigns slice
  p(AD) = 0.971 and 0.953 and patient p(AD) = 0.84 and 0.87, automating both as
  AD. SmallCNN refers Case 1 (p(AD) = 0.54 < 0.563) but also automates Case 5 as
  AD (p(AD) = 0.77).
* **Case 3 — Multi-scan borderline false positive on a 2017-2021 NC patient.**
  A single sagittal slice with widened sulci and periventricular hypointensity
  scores p(AD) = 0.961, but averaging across all 40 slices of the patient's two
  scans pulls the patient probability down to p(AD) = 0.630—just 0.004 above
  τ = 0.626, so the referral rule narrowly misses catching it.
* **Cases 2 and 4 — Confounded 2017-2021 AD patients with mild midline atrophy
  (confident false negatives).** Both are true AD patients scanned in 2017-2021
  whose mid-sagittal slices visually preserve a relatively narrow lateral
  ventricle and intact cortical ribbon while exhibiting global gray-white
  contrast visually similar to the 2017-2021 accelerated-MPRAGE NC training
  cohort. With acquisition style and subtle midline morphology both pointing to
  NC, the primary ensemble assigns slice p(AD) = 0.040 and 0.047 and patient
  p(AD) = 0.31 (3 scans, confidence 0.69 > 0.626) and 0.09 (1 scan, confidence
  0.91 > 0.626), automating both as NC. SmallCNN makes the same automated
  errors (p(AD) = 0.29 and 0.10). Under a stricter negative confidence threshold
  (`tau_NC = 0.750` or `0.800` in the post-hoc sensitivity analysis above),
  Case 2 (`p(NC) = 0.69`) would be routed to specialist review.
* **Cases 6-9 (Examples 5-8) — Borderline 2017-2021 patients caught by the
  referral rule.** When moderate structural atrophy or degraded scan contrast
  (e.g. Case 7 / Example 6) competes with the 2017-2021 acquisition prior,
  calibrated patient probabilities cluster near 0.49–0.53 (confidence
  0.50–0.53 < τ = 0.626), and all three ensembles route these ambiguous
  patients to a specialist.
* **Case 10 — The single misclassified 2011-2016 AD patient (caught by referral
  in ConvNeXt, automated as NC by SmallCNN).** Of the 122 `2011-2016 AD` test
  scans across 28 patients, the primary ensemble misclassifies only 6 scans: 1
  borderline scan (`p(AD) = 0.48`) from a multi-scan AD patient whose overall
  patient prediction remains correctly AD, and 5 of the 6 longitudinal scans of
  Case 10 (`scan p(AD) = 0.29, 0.35, 0.37, 0.42, 0.49`, while its sixth scan
  scores `0.89`). Averaged across all 120 slices, Case 10 receives `p(AD) = 0.47`
  (`confidence 0.53 < 0.626`) and is safely referred by the primary ConvNeXt
  ensemble (`0` automated AD misses in 2011-2016), whereas SmallCNN assigns
  `p(AD) = 0.31` (`confidence 0.69 > 0.563`) and wrongly automates this patient
  as NC.

## Recommendation to the project manager

1. **Decision: Advance to a controlled shadow study with conditions — do not
   deploy for autonomous clinical triage yet.** On held-out test patients the
   primary pipeline meets the internal coursework benchmarks on this test split
   (slice / scan / patient accuracy **0.867 / 0.889 / 0.882**, AUROC **0.943 /
   0.964 / 0.944**, and **92.4%** accuracy on the **87.5%** of patients
   automated at τ = 0.626), but the dataset's acquisition-period confound causes
   systematic false negatives on 2017-2021 AD scans (`14/27` scans missed,
   `5/16` patients confidently automated as NC under symmetric thresholding).
2. **Recommended operating configuration.** Use the five-fold **ImageNet-1k–
   initialised ConvNeXt-Tiny ensemble** with three-slice context and Platt
   scaling (`a = 0.246, b = 0.193`). Under the pre-declared symmetric threshold
   `tau = 0.626` (`REFER` when `max(p, 1-p) < 0.626`), the pipeline automates 119
   of 136 test patients (87.5% coverage, 92.4% accuracy, 9 automated errors) and
   refers 17 patients (12.5% specialist load, catching 7 of the 16 unreferred
   patient errors including the single 2011-2016 AD miss); with τ fitted on 68
   calibration patients, ≥ 90% automated accuracy is not guaranteed on future
   cohorts and should be re-verified. For future prospective
   calibration on a larger cohort where false negatives carry higher clinical
   cost than specialist referrals, our post-hoc sensitivity analysis shows that
   an asymmetric rule (`tau_AD = 0.626` with a stricter `tau_NC` in `0.750-0.800`)
   can reduce automated 2017-2021 AD false negatives while keeping roughly 78-80%
   coverage, though selecting `tau_NC` reliably requires more than the 22 AD
   patients available in our 68-patient calibration split.
3. **Is ConvNeXt worth its compute overhead over SmallCNN?**
   * *From scratch:* **No.** Scratch ConvNeXt-Tiny requires 284× the parameters
     (`27.8 M` vs `97.8 k`), 6.4× the training memory (`4,656 MiB` vs `727 MiB`)
     and 4.5×–4.6× the single-slice latency (`5.19-5.27 ms` vs `1.14-1.16 ms`),
     yet attends heavily to background pixels (Grad-CAM outside-head share
     `0.72-0.83`), does not significantly beat SmallCNN in single-fold
     development (`+0.032` patient accuracy, `t = 1.7`), trails the calibrated
     SmallCNN ensemble on test (`0.801` vs `0.875` patient accuracy, McNemar
     `p = 0.041`), and must refer 65.4% of test patients to maintain 90% accuracy.
   * *With ImageNet-1k initialisation:* **The two perform closely on this test
     split; ConvNeXt has advantages on some metrics** (uncalibrated accuracy,
     single-fold development accuracy, patient-fitted Platt calibration and
     slice-level accuracy), while patient accuracy is essentially tied once a
     five-fold `strong`-augmented SmallCNN ensemble is re-centred by slice-level
     Platt scaling. This conclusion applies to the specific configurations
     tested here: we describe pretrained ConvNeXt as performing similarly to, or
     slightly better than, the best SmallCNN tried here, not as categorically
     better. The post-hoc `SmallCNN + scanner aug.` control narrows the
     five-fold development gap (`+0.029` patient accuracy, `t = 1.5`, not
     significant) but reaches `0.838` test patient accuracy (`114/136`) and
     misses the 90% referral target (`83.7%` automated accuracy at `tau = 0.508`).
     Before Platt scaling, the five-fold pretrained ConvNeXt ensemble already
     achieves **0.882** test patient accuracy and **0.854** balanced accuracy
     (`0.862` slice accuracy), whereas the uncalibrated `strong` SmallCNN
     ensemble achieves **0.824** patient accuracy and **0.751** balanced
     accuracy (`0.783` slice accuracy, missing `20/44` AD test patients) and
     depends on a `+0.813` slice-fitted Platt intercept shift fitted on 68
     calibration patients to reach `0.875` (`McNemar p = 1.00`, `0.878` vs
     `0.866` on 2017-2021 scan AUROC)—an offset whose transferability to a new
     hospital or scanner distribution is uncertain, and which settles at `0.846`
     test patient accuracy when Platt scaling is fitted directly at the patient
     level (`0.882` vs `0.846`). Thus, for strict CPU/edge environments a
     Platt-calibrated SmallCNN ensemble is a strong lightweight choice (284×
     fewer parameters and 4.3×–4.7× lower single-slice latency, `1.14-1.16 ms`
     vs `4.93-5.37 ms`); for offline or hospital-server GPU triage (~27 ms/slice,
     ~0.54 s per 20-slice scan on one A100), we lean towards the pretrained
     ConvNeXt ensemble primarily for its lower dependence on threshold
     calibration (`0.882 -> 0.882`), supported by its 476-patient single-fold
     development lead (`+0.090` patient accuracy vs `strong` SmallCNN, `t = 7.1`;
     `+0.029` vs `scanner` SmallCNN, `t = 1.5`), higher slice accuracy (`0.867`
     vs `0.826` / `0.812`), and referral of the single borderline `2011-2016 AD`
     miss (Case 10).
4. **Conditions before clinical or multi-site progression.**
   * **External protocol-matched validation:** Re-evaluate on an independent
     cohort where AD and NC patients are balanced within each scanner vendor,
     field strength and acquisition era, verifying that subgroup AD sensitivity on
     modern accelerated sequences meets a pre-deployment benchmark of ≥ 0.85
     (a proposed target chosen to match the primary ensemble's overall patient
     balanced accuracy of `0.866` and AD recall of `0.818`).
   * **Subgroup-aware and patient-level calibration:** Fit Platt scaling at the
     patient/scan level (and with asymmetric thresholds `tau_NC > tau_AD`), and
     enforce mandatory `REFER` whenever a scan comes from an underrepresented
     protocol-by-diagnosis cell.
   * **Anatomically constrained pooling:** Replace full-frame global average
     pooling with skull-masked pooling and 3D/multi-slab coronal+sagittal views
     so mild medial-temporal atrophy in 2017-2021 AD scans is not washed out by
     whole-slice acquisition contrast.

## Limitations and future work

* **Acquisition confound:** No NC scans were acquired before 2017 in the
  supplied course dataset; overall accuracy here is therefore a likely
  optimistic estimate for a clinical site where AD and NC share the same scanner
  and protocol (though training without the 2011-2016 AD vs 2017-2021 NC
  imbalance would also remove the shortcut that currently depresses 2017-2021 AD
  recall).
* **Small validation and calibration cohorts:** Each development fold has ~95
  validation patients (±0.04 binomial standard error), calibration uses 68
  patients, and multi-seed repeats were run only on fold 1 for the scratch
  model (where patient-level seed sd was 0.030).
* **2D slice representation and coarse Grad-CAM grid:** Slices are not
  spatially registered across patients, and the final ConvNeXt stage has a 7×8
  spatial grid.
* **Scope of the model comparison:** Each model used its own learning rate,
  epoch budget and weight decay, and only two `SmallCNN` augmentation recipes
  were tried (`strong`, pre-declared; `scanner`, post-hoc). Conclusions about
  ConvNeXt versus `SmallCNN` therefore apply to these configurations rather than
  to CNNs in general; on test the pretrained ConvNeXt is close to the `strong`
  SmallCNN (one patient apart) and ahead of the `scanner` SmallCNN (six
  patients), with only partial statistical support on 136 patients.
* **Horizontal flip orientation:** The horizontal flip mirrors sagittal slices
  front-to-back, an orientation absent from natural brain MRI; its standalone
  effect was not ablated separately from the rest of the strong augmentation
  profile.
* **Future extensions:** Skull-masked global pooling (restricting feature
  aggregation to intracranial pixels), background randomisation during training,
  within-mask intensity normalisation (CLAHE or z-score), folds stratified
  jointly by diagnosis and acquisition period, higher-resolution Grad-CAM from
  stage 3, and external validation on protocol-matched multi-site data.

## Artificial Intelligence Usage Disclosure

AI coding assistants (OpenAI ChatGPT/Codex and Anthropic Claude) were used as
programming, debugging, plotting and language-polishing utilities during
development, in accordance with course policy.

**Scope of AI assistance.**

* *Boilerplate and tooling:* Assisting with CLI argument parsing, configuration
  serialisation, Slurm batch-script templates, and Matplotlib figure formatting
  (`figures/*.png`).
* *Testing and refactoring:* Generating synthetic-data unit test scaffolding
  under `tests/` and assisting with modularising the codebase into the package
  layout behind `modules.py`, `dataset.py`, `train.py` and `predict.py`.
* *Documentation polish:* Proofreading English grammar and formatting Markdown
  tables from the saved `metrics.json` and summary files.

**Author's work and engineering decisions.**

* Problem formulation, selection of the ConvNeXt-Tiny and SmallCNN architectures,
  and design of the three-slice 2.5D input representation.
* Identifying the 216-patient overlap in the course's default directories and
  establishing the frozen patient-level split (`development`, `calibration`,
  `test`).
* Designing the experimental progression (six pre-declared rounds plus the
  post-hoc control), discovering the 2011-2016 vs 2017-2021 acquisition-period
  confound in `meta_data_with_label.json`, and formulating the non-diagnostic
  scanner/period probe, Grad-CAM attribution check, five-fold logit ensemble and
  Platt-calibrated referral rule.
* Running all training, calibration and evaluation jobs on the Rangpur A100
  cluster, and writing the feasibility review, Grad-CAM interpretation, failure
  case autopsy and project-manager recommendation.

**Verification.** Every piece of code was reviewed and executed against the
synthetic test suite (`python3 -m unittest discover -s tests`) and checked for
zero patient leakage (`adni_splits.py verify`) before use. Every trained
checkpoint was reloaded and verified to reproduce its own validation metrics
(`replay_check.json`), uncalibrated and calibrated ECE values were cross-checked
directly from raw logits, and every number in this report was verified against
the corresponding `metrics.json` output. Initial AI suggestions that contained
errors—such as assuming a fixed number of slices per patient, describing the
slices as axial rather than sagittal, or clipping figure labels—were identified
during code and visual review and corrected.

## Data acknowledgement and references

**ADNI data acknowledgement.** Magnetic resonance imaging data used in this
coursework were obtained from the Alzheimer's Disease Neuroimaging Initiative
(ADNI) database (`adni.loni.usc.edu`) via the COMP3710 course dataset
(`/home/groups/comp3710/ADNI`). As such, the investigators within the ADNI
contributed to the design and implementation of ADNI and/or provided data but
did not participate in the analysis or writing of this report.

1. Z. Liu et al., "A ConvNet for the 2020s", CVPR 2022. https://arxiv.org/abs/2201.03545 ;
   reference implementation and ImageNet-1k weights (`convnext_tiny_1k_224_ema.pth`)
   https://github.com/facebookresearch/ConvNeXt
2. G. Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016.
3. H. Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018. https://arxiv.org/abs/1710.09412
4. C. Guo et al., "On Calibration of Modern Neural Networks", ICML 2017. https://arxiv.org/abs/1706.04599 ;
   J. Platt, "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods", 1999.
5. B. Polyak and A. Juditsky, "Acceleration of Stochastic Approximation by Averaging", SIAM J. Control Optim., 1992.
6. C. Szegedy et al., "Going Deeper with Convolutions", CVPR 2015 (random-resized crop).
7. PyTorch reproducibility notes, https://docs.pytorch.org/docs/stable/notes/randomness.html
8. R. Geirhos et al., "Shortcut learning in deep neural networks", Nature Machine Intelligence, 2020. https://arxiv.org/abs/2004.07780
9. J. R. Zech et al., "Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study", PLOS Medicine, 2018.
10. R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017. https://arxiv.org/abs/1610.02391
