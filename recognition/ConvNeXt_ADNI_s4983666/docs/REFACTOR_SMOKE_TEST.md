# Package Refactor and Training Augmentation Smoke Test

Date: 2026-09-25. Result: **PASS**.

This check validates source reorganization, optional training augmentation, checkpoint compatibility, and local CPU execution. It uses synthetic patients and generated JPEGs, not real ADNI MRI images.

## Structure and preserved behavior

The root Python files are the three ten-line CLI entry points: `train.py`, `predict.py`, and `adni_splits.py`. Models are separated into `models/cnn.py` and `models/convnext.py`; dataset code, engine code, evaluation, utilities, dependencies, documentation, and tests are in their own directories. README and Git configuration remain at the root.

The split implementation in `dataset/splits.py` and metrics implementation in `evaluation/metrics.py` are byte-for-byte copies of their previous implementations. Both model implementations retain their original parameter names, initialization, random-number consumption, forward computation, and gradients. Fixed preprocessing remains the default, and existing frozen manifests remain valid.

## Automated validation

```bash
python3 -m unittest discover -s tests -v
```

All **76 tests passed** in 103.277 seconds, and the complete test process exited with **code 0**. Coverage includes:

- Original patient-isolation, source-audit, manifest, image-loading, metric, and training checks.
- Both models, CLI aliases, native-size forward/backward execution, and checkpoint dispatch.
- Light augmentation bounds, source immutability, metadata preservation, strict profile serialization, and exclusion from every evaluation role.
- Private random streams, repeatable augmented training, global model RNG preservation, and actual two-worker reproducibility across epochs.
- Version 1 and version 2 checkpoint support, deterministic validation, and exact reloaded predictions.
- Entry-point help, source fingerprints across every implementation package, and prepare/verify commands with training-library imports blocked.

## Separate command-line checks

| Check | Model input | Training workers | Result |
|---|---|---:|---|
| Full ConvNeXt epoch with light augmentation | 240×256 | 2 | Completed; all 22 slice and 11 scan predictions reproduced exactly |
| Final-code ConvNeXt epoch after worker compatibility fix | 32×32 | 2 | Completed and exited normally; all predictions reproduced exactly with a separate process using 0 workers |
| Pre-existing CNN checkpoint from before the refactor | 240×256 | Inference only | Original slice and scan predictions reproduced exactly |

The synthetic fixture contains 84 patients, 94 scans, and 188 images, with two slices per scan. The selected fold uses 43 training patients, 5 early-stop patients, and 11 outer-validation patients. The native model inputs were resized from generated images. Real ADNI retains its frozen 20-slice scan contract.

The first native-size run and test suite completed their result files but initially stalled during interpreter shutdown. Investigation identified an inherited resource-tracker pipe held by PyTorch's shared-memory manager, matching [PyTorch issue #153050](https://github.com/pytorch/pytorch/issues/153050). The worker setup now marks that descriptor non-inheritable on macOS with Python 3.12 or newer. It does not close the worker's tracker pipe or alter seeds; the Linux path is unchanged. A focused two-worker run, the final full suite, the final training CLI, and both ConvNeXt prediction CLIs all subsequently exited normally. The initial native-size checkpoint was evaluated again with the corrected loader and reproduced exactly.

Current implementation fingerprints were checked against the final training run. New version 2 checkpoints record the complete light profile and validate it during loading, while prediction uses unaugmented validation images. Local logs and summaries are retained under the ignored `outputs/layout_smoke_v1/` directory. The two newly generated smoke checkpoints were removed only after their checksums and reproduced predictions were verified; existing experiments were preserved.

## Remaining validation

This record does not establish improved ADNI accuracy or validate the new augmentation run on server CUDA. Run the new one-epoch server check before the full development experiment, preserving the existing patient split and early-stop rule. The augmentation remains an experimental choice. Calibration and final-test scoring are not part of this implementation stage.
