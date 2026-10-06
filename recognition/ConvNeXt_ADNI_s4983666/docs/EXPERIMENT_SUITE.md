# Predeclared inner-only experiment suite (5 October 2026)

This suite addresses weak transfer, longitudinal sampling weights, geometric
sampling and raw overconfidence observed in the real Lite pilot, job 631647.
It does not complete formal final fitting, calibration or final-test evaluation.
No performance improvement is claimed before the actual runs finish.

## Feasibility acceptance status

| Criterion from the consultation draft | Implementation/evidence | Remaining acceptance work |
|---|---|---|
| Patient isolation and reproducible checkpoint predictions | Frozen patient manifests, mandatory source verification, source fingerprints, CPU synthetic checkpoint reload tests; real Lite CSV arithmetic and membership verified | Reload selected real CNN/ConvNeXt checkpoints under the matched final protocol; no claim of cross-hardware bitwise equality |
| Minimum 0.80 full-coverage slice test accuracy | Classification logger and explicit development/final-target scope exist | Target not achieved or assessed on the untouched final test; final fitting/testing entrypoint is still future work |
| Per-class precision/recall, macro-F1, AUROC and confidence/calibration measurements | Slice/scan/patient metrics, confusion, ECE/Brier/reliability/confidence histograms are implemented | Complete the matched selected-model comparison; actual probability calibration fitting is not implemented |
| Frozen rejection rule, coverage/accepted error/referred counts and 3-5 failures | Predeclared raw threshold, risk-coverage, counts and actual FP/FN exports exist; real Lite exported five errors | Fit/freeze the final operating rule on designated non-test data, then test once; owner writes clinical interpretation |
| One A100 GPU, parameter/memory/latency measurements | Actual Lite A100-PCIE-40GB log: 4,831,633 parameters; 1,549.877 MiB training peak; 10 warmups/100 repeats at batches 1 and 64 | Profile the selected CNN and advanced model with a matched protocol; batch-64 amortized time is not single-request latency |

The check-off handout is a consultation draft. Its single-pilot budget and Tiny
wording predate this expanded suite; the owner should explain the Lite option and
new declared budget to the tutor. No final accuracy/clinical claim should be added.
AMP and label smoothing remain unimplemented; they are not silently enabled.
The optional warmup/cosine schedule below is newly implemented and off by default.

## Fixed experiments

Shared settings: fold 1, scratch initialization, native 240x256 grayscale,
maximum 30 epochs, patience 5, scan-log-loss checkpoint selection, raw reject
threshold 0.8, inner-only scoring, workers 2 and PyTorch threads 2. Every training
run performs the mandatory original source/frozen-manifest audit. Calibration and
final-test images are not model-scored. Outer validation is not constructed.
The unchanged default is slice-uniform loading and constant learning rate.

Reference Lite: AdamW LR 0.0001, weight decay 0.05, batch 32, no augmentation,
seed base 3710 (actual training seed 3711). Each row changes only the stated
setting relative to the reference unless labelled interaction or seed repeat.

| ID | Change / purpose |
|---|---|
| E00_cnn | CNN control; existing CNN LR 0.001, decay 0.0001 |
| E01_lite_reference | Fresh Lite reference; also checks reproducibility of the earlier pilot |
| E02_lr_low | LR 0.00003 |
| E03_lr_high | LR 0.0003 |
| E04_decay_low | Weight decay 0.01 |
| E05_decay_high | Weight decay 0.2 |
| E06_batch16 | Batch 16; epoch draw count unchanged, optimizer updates increase |
| E07_warmup_cosine | Two-epoch linear warmup and cosine to 1% of base LR at epoch 30 |
| E08_patient_sampling | Class/patient-balanced sampling |
| E09_integer_shift | Independent x/y integer shifts in [-4,4] pixels; crop/paste, zero fill |
| E10_subpixel_shift | Independent x/y continuous shifts in [-4,4] pixels; bilinear affine, zero fill |
| E11_rotation_only | Rotation in [-3,3] degrees; translation zero; bilinear |
| E12_gamma | Gamma sampled uniformly in [0.9,1.1]; 8-bit LUT |
| E13_sampling_integer | Predeclared interaction: balanced sampling + integer shifts |
| E14_sampling_integer_gamma | Predeclared interaction: sampling + integer shifts + gamma |
| E15_combined_seed4710 | Repeat E14 with seed base 4710 |
| E16_combined_seed5710 | Repeat E14 with seed base 5710 |

The interactions are fixed proposals, not combinations selected after looking at
this suite's results. E14/E15/E16 provide three seeds for that one proposal; they
do not establish seed robustness for whichever other single-factor run wins.
A later selected candidate must be independently checked across seeds/folds.
Many trials on 38 already-used patients can overfit the inner cohort; freeze the
next comparison before scoring outer validation. Never choose settings using
calibration/final-test model scores.

## Sampling and augmentation contracts

`--sampling class_patient_balanced` uses weight
`1 / (2 * patients_in_class * slices_of_this_patient)` per training slice,
with replacement and exactly the original training slice count drawn per epoch.
It gives equal expected mass to each class and to each patient within that class.
Class imbalance is corrected by the sampler, so BCE `pos_weight` becomes 1;
retaining the old slice-count correction would double-correct the sampling prior.
No validation/test sample is reweighted. A mixed-diagnosis patient is rejected
rather than silently assigned a class. Sampled class supports appear in the
existing online-training per-class records. Expected mass is not an exact
per-epoch count, and an epoch may repeat or omit individual slices.

New augmentation profiles apply only to development training. `none` and `light`
retain their old algorithms/serialized fields. New profiles record their algorithm,
interpolation, pixel bound or gamma range in the checkpoint config. Integer
shifts do not interpolate or wrap, but can clip boundary pixels. Continuous
shifts have the same bounds but different offset distributions as well as
interpolation; their comparison is not a pure interpolation-only causal test.
Gamma preserves black/white endpoints and spatial coordinates but changes
intensities. These are experimental perturbations, not established clinical
invariances. Flips, random anatomical crops, MixUp/CutMix and RoPE are not added.

The implementations use the official [PyTorch sampler API](https://docs.pytorch.org/docs/2.6/data.html)
and [Pillow transform/LUT API](https://pillow.readthedocs.io/en/stable/reference/Image.html).
Model components remain local and PyTorch-only; no pretrained download occurs.

## Submit once from the login node

After installing and checking the complete new source archive:

```bash
cd "$HOME/comp3710/comp3710-adni"
bash slurm/submit_suite.sh --dry-run
bash slurm/submit_suite.sh
squeue --me
```

This requests **one GPU for six hours**, then runs all 17 cases sequentially in
separate Python processes. It is not a 17-GPU array. The account/partition remain
`comp3710`. The Conda root/environment remain `$HOME/miniconda3` / `torch`.
The six-hour request is a proposed allocation, not a verified cluster limit.
If Slurm rejects that duration, use an allowed `ADNI_SUITE_TIME_LIMIT`; resumable
completed-case reuse allows later allocations to continue the same suite.

The measured Lite reference used about 25 seconds per training epoch and stopped
after nine. Seventeen similar runs imply roughly 70 minutes of training/evaluation;
30-epoch runs imply roughly 3.6 hours at the same per-epoch cost. These estimates
exclude repeated startup audits, augmentation overhead, hardware variation and
queue delay. The suite wall time is not measured yet; six hours is a request
with margin, not a guarantee that every run will complete.

Output defaults to `$HOME/comp3710/runs/convnext_suite_job<jobid>/`.
Each case has its own attempt folder, selected checkpoint, full config/history,
confidence/rejection/failure outputs and resource profile. `suite_plan.json`
freezes all flags, code hashes and the manifest marker. `suite_summary.csv` and
`.json` update after each run in plan order and do not automatically select a winner.
Separate case logs preserve errors; failure of one case does not cancel later cases.
Slurm standard logs are in `runs/slurm_logs/suite_<jobid>.out/.err`.

## Smoke, resume and result transfer

Run just the two references with a one-epoch cap as an optional server smoke:

```bash
ADNI_SUITE_EPOCHS=1 ADNI_SUITE_IDS=E00_cnn,E01_lite_reference \
  bash slurm/submit_suite.sh
```

This is a different plan and must not share an output root with the 30-epoch suite.
To continue an interrupted 30-epoch suite, keep the original code/settings and
provide its exact output root:

```bash
ADNI_SUITE_OUTPUT="$HOME/comp3710/runs/convnext_suite_job<original_jobid>" \
  bash slurm/submit_suite.sh
```

Completed cases are skipped after checking the case binding, code and manifest.
Partial/failed attempts are retained; a retry trains from scratch in a new attempt
folder. There is no mid-epoch or optimizer-state resume. A changed plan or source
is rejected for an existing suite. Do not update source while a suite is running.

For summary review, download `suite_plan.json`, `suite_summary.csv` and
`suite_summary.json` first. Full plots/predictions can be transferred afterwards,
excluding `best.pt`. Keep all raw results labelled exploratory. Formal refit,
probability calibration, threshold fitting and one final-test evaluation remain
separate later work after selecting and freezing the protocol.
