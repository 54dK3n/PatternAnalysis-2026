# ConvNeXt-Tiny Local Smoke Test

Date: 2026-09-25. Result: **PASS**.

This is the historical record for the initial ConvNeXt implementation. The later package refactor and optional augmentation are documented separately in [REFACTOR_SMOKE_TEST.md](REFACTOR_SMOKE_TEST.md).

The complete grayscale ConvNeXt-Tiny has 27,817,825 trainable parameters. This check validates local CPU execution and checkpoint compatibility; it does not measure ADNI classification performance. The model starts from random weights and downloads no external weights.

## Automated tests

```bash
python3 -m unittest discover -s tests -v
```

All **57 tests passed** in 28.358 seconds in the local Python 3.12.14 / PyTorch 2.6.0 environment. The suite includes the original CNN, split, dataset, and metric tests, plus:

- Real forward and backward execution at 240×256, including finite gradients through every ConvNeXt parameter.
- Training-only stochastic depth, channel normalization, minimum input sizes, and independent model initialization.
- Two epochs of actual ConvNeXt optimization at 32×32 with a controlled early-stop loss trajectory, confirming that the first checkpoint is selected and restored before outer validation.
- Exact slice and scan predictions after checkpoint reloading; training/early-stop/validation patient separation and exclusion of calibration/test patients from model loaders.
- Rejection of changed manifests, unsupported preprocessing, invalid input sizes, unknown models, and model weights that disagree with the checkpoint architecture.

The controlled loss values affect checkpoint selection in one unit test only. The command-line smoke below uses the actual observed losses without mocks.

## Command-line check at the production input size

A separate `train.py --model convnext_tiny` run completed one full epoch using 240×256 model inputs, batch size 2, two CPU threads, seed base 3710, learning rate 0.0001, and weight decay 0.05. These inputs were resized from generated noise JPEGs; no real MRI pixels were used. The synthetic fixture contains 84 patients, 94 scans, and 188 JPEGs, with two slices per scan to keep this execution check small. Production ADNI manifests still require 20 slices per scan.

| Fold 1 role | Synthetic patients | Scans | Slices |
|---|---:|---:|---:|
| Train | 43 | 50 | 100 |
| Early stop | 5 | 6 | 12 |
| Outer validation | 11 | 11 | 22 |

The mandatory source/manifest audit passed. Training, checkpoint saving, best-checkpoint reloading, outer validation, CSV/JSON export, and learning-curve generation completed successfully. The recorded training-and-evaluation duration was 88.56 seconds; this local smoke timing is not a hardware benchmark.

A separate `predict.py` process reloaded the saved ConvNeXt checkpoint. All **22 slice predictions and 11 scan predictions matched exactly**, and every reported scan/slice classification metric was reproduced. A pre-existing `small_cnn_v1` checkpoint from the earlier synthetic smoke run also reproduced all of its original predictions exactly through the updated prediction entry point.

Raw logs and generated outputs are retained locally under `outputs/convnext_smoke_v1/`, which is ignored by Git. Only the newly generated ConvNeXt smoke checkpoint was removed after its reload and checksum were verified; existing experiment files were left intact. Synthetic test checkpoints created by the automated suite are cleaned up with their temporary fixtures.

## Remaining verification

CUDA execution, GPU memory use, runtime on the course server, and real ADNI ConvNeXt accuracy remain to be measured. Run the README's one-epoch GPU command before the formal five-fold experiment. This check does not fit confidence calibration or score the locked final test set.
