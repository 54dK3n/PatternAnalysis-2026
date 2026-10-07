# Selectable scratch training recipes

All commands use the existing frozen patient assignments. No recipe downloads
weights, repartitions patients, scores calibration/test, or changes checkpoint
selection: the strict minimum early-stop scan log loss selects the checkpoint.
The Slurm launcher remains inner-only. Direct `train.py` supports the existing
outer-validation mode after selection. Slice metrics remain coursework-primary.

## Recipe defaults

Explicit CLI options always override a recipe, regardless of their order.
`custom` preserves the defaults of the entry point you call; the train CLI and
Slurm pilot historically have different defaults. Named recipes share resolved
settings across both entry points. Every named recipe defaults patience to its
resolved epoch cap. A smaller explicit `--patience` enables earlier stopping.

| Recipe | Architecture / geometry | Training | Execution / patient summary |
|---|---|---|---|
| `lite_reference` | Lite; 240 x 256; original input rule | AdamW LR 1e-4, decay 0.05, batch 32, constant LR, 30 epochs, no augmentation | 1 channel, weighted BCE, FP32, mean probability |
| `lite_augmented` | Lite; training-fitted scan intensity/crop | Same optimizer; integer shifts + gamma | Same execution as reference; fitted crop can change the canvas size |
| `peer_tiny` | Tiny; 224 x 224; original resize/normalization | AdamW LR 3e-4, decay 1e-4, batch 64, cosine, 30 epochs, no augmentation | 3 repeated grayscale channels, unweighted CE, BF16 AMP, mean logit, max DropPath 0.1 |

`peer_tiny` approximates Ken's peer screenshots. BF16 rather than FP16, DropPath
0.1, no augmentation, unweighted CE, no warmup, and a zero cosine floor are
**our declared assumptions** where the screenshots do not specify details.
Resize uses this project's existing image interpolation and normalization to
[-1, 1]; the peer's exact transform is unknown. Seed base remains 3710 and the
fold is added, so fold 1 actually uses 3711. Patient manifests, training cohort,
and checkpoint-selection rule also differ from the peer. This is not an exact
reproduction or evidence that its reported accuracy transfers to our pipeline.

## Independent switches

| Option | Values |
|---|---|
| `--recipe` | `custom`, `lite_reference`, `lite_augmented`, `peer_tiny` |
| `--input-channels` | `1`, `3` (three identical normalized grayscale channels) |
| `--loss` | `bce`, `weighted_bce`, `cross_entropy`, `weighted_cross_entropy` |
| `--precision` | `fp32`, `amp_fp16`, `amp_bf16` |
| `--patient-aggregation` | `mean_probability`, `mean_logit` |
| `--drop-path` | Float in [0, 1); zero disables it; omitted uses architecture default; ConvNeXt only |
| `--lr-schedule` | `constant`, `cosine`, `warmup_cosine` |

Existing switches select architecture, preprocessing, crop geometry, resize,
augmentation and its bounds, sampler, LR, decay, batch, epochs, patience, fold,
seed, workers and profiling. For `cosine`, warmup is always zero; use
`warmup_cosine` to set `--warmup-epochs`. The resolved schedule is logged.
No label smoothing is implemented. Balanced sampling uses unit loss prior;
uniform sampling uses the training NC/AD ratio only for weighted objectives.
CE class order is NC=0 / AD=1. Slice AD probability is sigmoid(AD logit minus
NC logit); the one-logit BCE head uses sigmoid of that logit.

Patient logit averaging exports the original FP32 AD margins before sigmoid.
It does not reconstruct logits from rounded or saturated probability CSVs.
Patient probability averaging remains the historical default. Scan averaging
and selection remain means of slice probabilities in both modes. Mixed-label
longitudinal patients make patient summaries unavailable under the existing rule.
Changing only patient aggregation cannot change slice accuracy.

## Review before submission

From the project source directory on the server:

```bash
bash slurm/submit_preprocessing_train.sh --recipe peer_tiny --dry-run
```

Example explicit overrides (no job is submitted):

```bash
bash slurm/submit_preprocessing_train.sh \
  --recipe peer_tiny --model convnext_lite \
  --precision fp32 --input-channels 1 --loss weighted_bce \
  --patient-aggregation mean_probability --drop-path 0 \
  --epochs 60 --patience 60 --dry-run
```

Remove `--dry-run` only after inspecting the displayed train/replay commands and
setting server `--data-root`, `--splits-dir`, `--runs-root` if needed. A100 supports
BF16; actual GPU execution must still be checked on the course server. AMP is
explicitly CUDA-only. FP16 uses a persistent GradScaler; BF16 does not. CPU
checks require `--precision fp32`.

JSON suite plans can override each individual execution option alongside the
existing training options. Select the base recipe on the suite CLI; per-case
`recipe` overrides are rejected to avoid silently reinterpreting resolved values.
Each case still needs patience equal to its epoch budget. The historical
overnight plan is retained; it is not automatically replaced by a new experiment.

## Logs and compatibility

`config.json`, checkpoint configuration, and final metrics record the effective
`execution_controls` and `training_recipe`. The latter is the starting recipe
name, not a claim that no overrides were used. The displayed/recorded command
and resolved config are authoritative. Resource profiles record actual channel
shape and precision. Replay inherits controls from the checkpoint and checks
identical slice, scan and patient prediction bytes in the suite verifier.

Historical default runs retain checkpoint formats 2 (original input) and 3
(scan preprocessing), state-dict shapes, and prediction CSV columns. New
execution controls use format 4, including raw `ad_logit` exports. `predict.py`
reads formats 1 through 4; older diagnostic tools only support their original
formats and reject format 4. Extend those tools before analyzing new-format
checkpoints with them. No existing run is rewritten.

## Work still required for final acceptance

1. Complete the fixed real GPU comparison and repeat/outer validation. Three
   actual one-epoch FP32/BF16/FP16 train/replay checks passed on A100, with
   exact predictions and resource measurements; this verifies runtime only.
   Final target performance remains unassessed.
2. Freeze a short list of methods, then perform the predeclared five-fold outer
   comparison against a baseline using the same patient roles and metric units.
   The existing CNN repeats are historical development evidence; the overnight
   suite is inner exploration and cannot replace this comparison.
3. Implement/freeze final development refit, calibration/threshold fitting and a
   one-time final-test scorer. Current ECE/reliability plots describe raw
   confidence; they do not fit calibration. Temperature scaling is optional;
   a defensible declared rejection/threshold policy is still needed. Current
   `predict.py` intentionally has no final-test/calibration scoring mode.
4. Confirm tutor checkoff completion and prepare a reproducible demonstration
   of patient separation, scratch architecture, run configuration, required
   metrics, curves, resources and failure examples. The repo has no recorded
   completed checkoff evidence.
5. Update report/README using real output references. Ken writes failure-case
   interpretation, project recommendation and AI disclosure. All unmeasured
   final results remain `TODO(result)`.

Implementing switches alone does not validate performance. Actual authorized
GPU submissions and private archives are described in the
[comparison workflow](RECIPE_GPU_EXPERIMENTS.md). The >=0.80 final-test target
remains unassessed.

## Local verification (7 October 2026)

The initial execution-control regression suite passed 210 tests. After adding
the fixed-operation/archive checks, the submission review reran
`python -m unittest discover -s tests -q`: all 214 tests passed in 117.547s. Nine new tests cover recipe precedence/forwarding,
side-effect-free previews, CE margins, shapes/parameter counts, DropPath, patient
aggregation, checkpoint contracts, suite controls, and actual CPU train/replay
under both original and fitted scan preprocessing. GPU AMP and real-data results
are not covered. These counts are verification evidence, not accuracy metrics.
