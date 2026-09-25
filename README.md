# COMP3710: ADNI AD/NC Classification

Patient-isolated data preparation for a planned MRI classification project comparing Alzheimer's disease (AD) with normal controls (NC).

The current implementation audits the course JPEG dataset and creates five-fold cross-validation manifests. Model training, confidence calibration, and final evaluation are not implemented yet.

This is the working repository. Final coursework submission still requires the prescribed pull request to the course repository and the accompanying report.

## Repository contents

- `adni_splits.py`: source-data audit, patient-level partitioning, and independent manifest verification.
- `tests/test_adni_splits.py`: 17 integration tests using synthetic patients and generated images.
- `DATA_PROTOCOL.md`: the full experimental protocol, including training-time safeguards, in Chinese.
- `requirements.txt`: the image-reading dependency.

## Setup

Use Python 3.9 or newer in your project environment:

```bash
python3 -m pip install -r requirements.txt
```

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

`prepare` refuses to overwrite an existing split or write into the source dataset. If the split already exists, use `verify`; a comment-only code update does not require a new split.

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

These are user-supplied server results, not a local rerun of the real dataset. Local tests use synthetic data. The real images, metadata, generated patient manifests, and model weights are excluded from version control.

## Reproducibility and remaining safeguards

- Freeze this split before model experiments; do not choose a split seed using model scores.
- Initialise model and training state independently for each fold. Fit preprocessing statistics and class weights using only that fold's training subset.
- Use `early_stop.csv` to choose the stopping epoch, and `val.csv` for cross-validation comparison. Cross-validation results used to select a method are development results.
- Refit the selected model on development data using a previously specified training rule. Then freeze it, fit confidence/decision thresholds on calibration data, and freeze the complete pipeline before final testing.
- Exact duplicate checks do not exhaustively detect near-duplicates, incorrect source patient identities, or leakage from upstream preprocessing. Manifest verification does not enforce future training-code behaviour.

See [the full data protocol](DATA_PROTOCOL.md) for evaluation units, mixed longitudinal diagnoses, out-of-fold predictions, and uncertainty reporting.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Tests cover patient isolation, fold coverage, reproducibility, missing or corrupt inputs, conflicting labels, duplicate content, modified manifests, modified sources, and refusing to overwrite an existing split.

## Artificial Intelligence Usage Disclosure

OpenAI Codex assisted with the data-audit script, synthetic tests, code comments, and protocol documentation. The project owner executed the real-data audit on the course server. This development note should be incorporated into the final course-required AI-use disclosure; it does not replace that disclosure.
