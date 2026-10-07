# Authorized GPU recipe comparison and completion workflow

Ken authorized code upload, actual GPU checks, a bounded experiment batch,
monitoring, private Drive archiving and visual analysis on 7 October 2026.
The frozen patient roles and scratch initialization are retained. Git operations
require Ken's explicit authorization; no pretraining, resplit, final-test scoring
or adaptive overnight search is performed. The later authorized source push
is documented in the [submission review](SUBMISSION_STATUS.md).

`config/recipe_gpu_smoke_20261007.json` contains three one-epoch train/replay
checks: Lite FP32, Tiny three-channel CE BF16, and Lite three-channel CE FP16.
`config/recipe_comparison_20261007.json` contains twelve full 30-epoch cases
(360 epochs total) at the same fold-1, 224 x 224 resized geometry. Case options
explicitly specify all model/execution/optimizer/input controls. Patience equals
30; checkpoint selection remains minimum early-stop scan log loss. Two cases
repeat Tiny reference and approximate peer recipe at seed base 4710; the fold is
added to the declared base seed. The comparison is inner exploration, not CV.

Four cases form Lite/Tiny x reference/peer-optimizer comparisons. The optimizer
bundle jointly changes LR, decay, batch and constant/cosine schedule. Three
further Tiny controls isolate channel repetition, CE head/objective, and BF16.
CE changes output parameterization and class weighting together. Full peer-style
settings, integer/gamma augmentation, brightness-only preprocessing and paired
second-seed runs complete the plan. There is no automatic candidate promotion.

Submit a one-GPU/four-CPU smoke allocation with a two-hour cap, then submit the
one-GPU/four-CPU comparison with a ten-hour cap and `--dependency afterok:JOBID`.
The latter starts only if all smoke training/replay checks succeed. The wall cap
is a bound, not a duration promise. A failed smoke dependency requires inspection
and cancellation of only the matching blocked comparison job before a corrected
submission; do not rerun completed successful cases silently.

```bash
cd "$HOME/comp3710/comp3710-adni"
bash slurm/submit_preprocessing_suite.sh \
  --plan "$PWD/config/recipe_gpu_smoke_20261007.json" --time-limit 02:00:00
# Replace JOBID with the successful submission's actual numeric ID.
bash slurm/submit_preprocessing_suite.sh \
  --plan "$PWD/config/recipe_comparison_20261007.json" \
  --time-limit 10:00:00 --dependency afterok:JOBID
```

Every case generates learning curves, confidence/reliability/risk-coverage plots
and factual failure panels. The suite generates aggregate comparison/trajectory
figures even when no candidate reaches 0.80. The dashed 0.80 line is a development
reference. The archive assessment reports each slice/scan/patient unit separately
and labels any threshold crossing as selection-cohort evidence, never a final-test
claim. A threshold-crossing case also receives an aggregate confusion matrix
visualization. MRI failure panels and per-patient records remain on Rangpur.

Completion monitoring uses a thread heartbeat, bounded SSH checks and actual
Slurm accounting plus summary/marker/replay verification. Notify on completion,
failure, substantive progress or required intervention; unchanged running states
remain quiet. The authenticated multiplexed SSH socket is temporary and contains
no saved password. If it expires, request renewed authentication without storing
credentials in code, prompts or archives. Accepted Slurm jobs continue while the
laptop sleeps, but local heartbeat/Drive work requires Codex and network access.

Archive aggregate training/configuration summaries, history/epoch metrics,
resource measures, calibration tables and non-MRI figures into the established
private Drive analysis folder. The archive inventory hashes payloads; verify the
local transfer and cloud metadata before removing temporary local result copies.
Weights, raw MRI, patient/slice prediction rows, failure grids, full source-bound
configs and credentials are excluded. Completed archives are not duplicated by
later heartbeat checks. Logs and weights remain on Rangpur.
