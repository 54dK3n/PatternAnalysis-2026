# Feature experiment suite v2

This opt-in suite implements `config/experiment_plan_v2.json`. It does not change
`train.py` defaults, old aliases, frozen manifests or the legacy scan-selected
`best.pt` convention. Models are implemented locally and initialized from scratch.
No final-test, calibration or outer-validation loader is constructed by this suite.
Full source-integrity auditing still runs before each training/diagnostic child.

## Submit on Rangpur

From the login node, submit one GPU allocation:

```bash
cd "$HOME/comp3710/comp3710-adni"
bash slurm/submit_feature_suite.sh --dry-run
bash slurm/submit_feature_suite.sh
```

The default allocation requests one GPU, four CPUs and six hours. On 2026-10-06,
Rangpur reported partition `comp3710` MaxTime=UNLIMITED and QoS `comp3710`
MaxWall=12:00:00; the six-hour request fits those observed limits. Its first phase
replays the existing CNN, Lite-reference and integer-shift checkpoints in
`$HOME/comp3710/runs/convnext_suite_job633000`. It then performs one-epoch native-size
resource checks for R02/R03/R05, six full-budget trajectories, and eligible paired
repeats. Smoke outputs are never promoted into the progression gate. Failed new-model
smokes block that model's full run without reducing its batch size.

The run root is `$HOME/comp3710/runs/feature_suite_v2_job<JOBID>`; Slurm output is in
`$HOME/comp3710/runs/slurm_logs/feature_v2_<JOBID>.out` and `.err`.

To run only the six controls and gated repeats after separately reviewing existing
checkpoint diagnostics:

```bash
ADNI_FEATURE_PHASE=core bash slurm/submit_feature_suite.sh
```

To run diagnostics only, set `ADNI_FEATURE_PHASE=diagnostics`. To stop after the
six controls, set `ADNI_FEATURE_NO_REPEATS=1`. Paths are configurable through
`ADNI_DATA_ROOT`, `ADNI_SPLITS_DIR`, `ADNI_RUNS_ROOT`, `ADNI_FEATURE_REFERENCE_SUITE`,
`ADNI_CONDA_ROOT` and `ADNI_CONDA_ENV`. `ADNI_FEATURE_TIME_LIMIT` controls the requested
allocation duration; it must respect the actual partition limit.

A timed-out/failed suite can retry with the exact same source/settings and root:

```bash
ADNI_FEATURE_OUTPUT="$HOME/comp3710/runs/feature_suite_v2_job<ORIGINAL_JOBID>" \
  bash slurm/submit_feature_suite.sh
```

Completed cases are reused only under matching source/manifest/case bindings.
Incomplete trajectories start from scratch in a new numbered attempt. No optimizer
resume or silent source-change reuse is supported. Never edit source while it runs.

## Six controlled cases

| ID | Intervention | Versioned model |
|---|---|---|
| R00 | Reference: LR 1e-4, WD .05, original stem/DropPath | convnext_lite_v1 |
| R01 | Two-epoch warmup plus cosine, final LR 1e-6 | convnext_lite_v1 |
| R02 | Conv7 stride4 padding3 overlapping stem | convnext_lite_overlap_v1 |
| R03 | Conv4 stride2 padding1 stem | convnext_lite_stride2_v1 |
| R04 | Zero WD for all one-dimensional parameters | convnext_lite_v1 |
| R05 | DropPath disabled | convnext_lite_nodrop_v1 |

All full cases use fold 1, 30 complete epochs, batch 32, native 240x256 grayscale,
no augmentation, slice-uniform shuffle, float32 and the original class-weighted BCE.
Reference-shaped initial tensors are identical per paired seed. The differently
shaped overlap weight uses seed + 2000003. Training RNG is reset afterward.
Per-epoch batch/order hashes must agree across matched cases.

Two checkpoint rules operate on the same training trajectory: strict minimum
unweighted early-stop slice log loss (primary) and scan log loss (shadow). Equal
losses retain the earliest epoch. Both candidates are exported on the same cohort.
The gate requires at least 2 percentage points more slice accuracy than R00,
nondecreasing AUROC and no more than 2 percentage points less AD recall. Only the
highest ranked eligible candidate and R00 repeat at seed bases 4710 and 5710.
Incomplete core cases or default-phase diagnostic failures block this gate.

## Inspect the outputs

| Artifact | Meaning |
|---|---|
| suite_plan.json / suite_records.json / suite_summary.json | Exact bindings, progress, failures and completed attempts |
| repeat_gate.json | Prospective eligibility decision; no least-bad promotion |
| config.json / optimizer_parameter_groups.json | Actual geometry, random-state sharing hashes and exhaustive WD membership |
| best_slice_loss.pt / best_scan_loss.pt | Two distinct selection rules; no ambiguous best.pt alias |
| epoch_01/05/10/20/30.pt | Declared milestone states |
| history.csv / epoch_metrics.json | Online train versus clean fixed-weight scores, full early-stop slice/scan/patient metrics, LR, optimizer steps, order hashes and phase timings |
| gradient_batches.csv / gradient_flow.png | Read-only per-block/head gradients at selected epochs, first ten batches; zero/missing counters |
| milestone_features.json / residual_layerscale.png | Clean training activations, post-LayerScale residual/skip ratios, zero skips and LayerScale values |
| primary_slice/ and shadow_scan/ | Both candidates' classification, reliability, risk-coverage, confidence, failure predictions/grids and forward-only resource benchmarks |
| feature_learning_curves.png / selection_comparison.png | Comparable unweighted loss trajectories and same-patient selection error differences |
| D*_features*/stage_probe_auroc.png | Train-fitted frozen GAP patient-mean probe AUROC; diagnostic units |
| D*_spatial*/spatial/shift_boundary_curves.png | Equal-patient shift/round-trip changes and fixed source-margin subsets |
| D*_spatial*/spatial/stage_feature_changes.png | Absolute/relative GAP changes, reference norms/zero fractions; gray undefined alignment |
| feature_suite_comparison.png | Primary-selected accuracy/AUROC/AD recall/ECE by case and seed |
| paired_seed_differences.csv / paired_seed_summary.json | Paired per-seed differences, mean/sample SD and conditional patient-cluster accuracy intervals |

Training allocated/reserved CUDA peaks include optimizer state and gradient
instrumentation. Early-stop memory is measured separately; milestone feature
collection is outside those peaks. Diagnostic time is logged separately. Both
selected candidates have separate forward benchmarks after optimizer release:
warmup 10, repeats 100, batch sizes 1 and 64. OOM is recorded; it does not become
an invented measurement. Timing excludes clinical workflow and queue time.

Raw confidence rejection remains fixed at .8; classification remains at .5/full
coverage. Probe scores and paired-seed intervals are diagnostic development
statistics. Clinical interpretation and final recommendations belong to Ken.

## Validation boundary and storage

`tests/test_feature_suite.py` uses generated disjoint patients and CPU tensors.
Its synthetic smoke mode is explicitly one epoch at 32x32; it refuses the real
frozen manifest identity and cannot enter the repeat gate. Resource smokes retain
native real-data settings but are explicitly not full experiments.

Real ADNI accuracy remains `TODO(result)`. CUDA resources and full 30-epoch results
must be measured on Rangpur. Keep data/checkpoints there; archive actual JSON/CSV,
figures and experiment logs to the private Google Drive after verification, without
retained local evidence copies. Source specifications/deployment bundles are code.

## Recover report exports after completed training

If the original job completed every child but failed during final figure export,
do not rerun training or modify historical bindings. Use the report-only tool:

```bash
cd "$HOME/comp3710/comp3710-adni"
python recover_feature_report.py \
  --suite-root "$HOME/comp3710/runs/feature_suite_v2_job634456"
```

Run in the `torch` environment for Matplotlib. This reads saved metrics and checks
historical case/source/manifest bindings, then writes a fresh
`report_recovery_v1/feature_suite_comparison.png` and `suite_summary.json`. It never
loads model weights, scores images, submits jobs or rewrites training artifacts.
The original Slurm failure status remains unchanged; the recovered report records
postprocessing success separately. Existing destinations are refused; pass
`--output` with another fresh path for an explicitly requested repeat export.

The plotting filter admits completed R00-R05 cases and their paired seeds;
diagnostic summary-only directories and resource smokes are excluded. Missing
training metrics remain errors. A source patch changes future suite hashes;
full-suite retries against an old root still refuse changed source. Report-only
recovery is the intended route for completed historical results.
