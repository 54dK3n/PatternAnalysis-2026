# Train-only regularization controls and fixed comparison

Ken requested additional Mixup and label-smoothing controls on 2026-10-07.
This prospective eight-case comparison follows the existing fixed recipe jobs;
it does not modify their source, plans, patient assignments or outputs.

## Executable switches

```bash
python train.py --data-root /home/groups/comp3710/ADNI \
  --splits-dir "$HOME/comp3710/adni_splits_v1" --output /absolute/new/run \
  --model convnext_tiny --inner-only --epochs 30 --patience 30 \
  --label-smoothing 0.05 --mixup-alpha 0.1
```

Both switches default to zero. Disabled controls preserve the version-one
execution dictionary, loss and RNG behavior. Nonzero controls use the closed
`scratch_execution_v2` dictionary within checkpoint format four; old version-one
checkpoints still load with zero regularization. Inference never mixes inputs or
labels and never smooths model outputs.

Smoothing uses `y_s = (1-epsilon)*y + epsilon/2`. Binary BCE uses that fractional
target before the positive-class weight; CE uses PyTorch's equivalent uniform
mixture. Class counts and weights come from training only.

Mixup samples one `lambda ~ Beta(alpha, alpha)` per training minibatch and a
permutation from that minibatch, then computes `x_m = lambda*x_a + (1-lambda)*x_b`.
Loss is `lambda*criterion(logits,y_a) + (1-lambda)*criterion(logits,y_b)`, applying
the same declared smoothing/class weights to both labels. For weighted CE this
is a mixture of the two independently weight-normalized mean CE losses; it is
not presented as a single probability-target weighted CE reduction. The fixed
comparison below uses weighted BCE, whose loss is linear in the target.
Singleton batches are unchanged; no examples enter from early_stop/val/calibration/test.

Mixed inputs do not have an ordinary hard class label. For Mixup runs, online
training accuracy/F1/AUROC therefore use a separate no-gradient eval-mode forward
on the original minibatch inputs before the optimizer update. This does not
update BatchNorm or sample DropPath. History marks this scope as
`online_unmixed_eval_mode_for_mixup`; disabled runs retain their historical
training-mode scope. Timing includes the extra metric forward and is not a fair
measure of Mixup training cost alone. Training objective losses with smoothing
are not directly comparable to unsmoothed losses or held-out log loss.

## Fixed cases

All eight cases use the Tiny reference: 224x224, one grayscale channel, FP32,
weighted BCE, AdamW lr=1e-4, weight_decay=0.05, batch=32, constant learning rate,
DropPath=0.1, slice-uniform sampling, no other augmentation/preprocessing.
All run 30 epochs; scan log loss on early_stop selects the saved checkpoint.

| Case | Smoothing epsilon | Mixup alpha | Base seed |
|---|---:|---:|---:|
| G00_reference | 0 | 0 | 3710 |
| G01_smoothing_005 | 0.05 | 0 | 3710 |
| G02_smoothing_010 | 0.10 | 0 | 3710 |
| G03_mixup_010 | 0 | 0.10 | 3710 |
| G04_mixup_020 | 0 | 0.20 | 3710 |
| G05_mixup_smoothing | 0.05 | 0.10 | 3710 |
| G06_reference_seed4710 | 0 | 0 | 4710 |
| G07_combined_seed4710 | 0.05 | 0.10 | 4710 |

The fold number is added to the base seed. These are paired repeats for the
predeclared combination only, not an adaptive winner repeat. G00 checks the
implementation with both new controls disabled, against the existing reference.
This eight-case suite is 240 training epochs; three GPU prechecks add three.
The prechecks cover disabled FP32, mixed/smoothed weighted BCE FP32 and
mixed/smoothed three-channel CE BF16, including checkpoint prediction replay.

## Deployment and assessment

Use a new, immutable source snapshot under a distinct Rangpur source directory.
Do not deploy into the source used by running/queued recipe jobs. GPU smoke is
afterok-dependent on the registered existing follow-up job; formal comparison
is afterok-dependent on its smoke. Read job IDs from the regularization
submission record, never infer or submit duplicate jobs.

Plans: `config/regularization_gpu_smoke_20261007.json` and
`config/regularization_comparison_20261007.json`. Preview using
`bash slurm/submit_preprocessing_suite.sh --plan /absolute/plan.json --dry-run`
with explicit shared data/split/run paths for an isolated checkout.

All evaluation remains on the inner checkpoint-selection cohort (38 patients
on fold one). Slice >=0.80 is a development reference, not final-test success.
Scan/patient scores, AUROC, calibration, loss trajectories and seed variation
remain separate. Do not choose new parameters on calibration or final test.
Mixed MRI inputs are artificial training examples, not anatomical images for
clinical interpretation. Ken writes the clinical and course-report analysis.

Aggregate archives and figures use `slurm/archive_recipe_results.py`; exclude
weights, raw MRI, patient-level prediction rows and failure-image grids.
GPU execution and performance are `TODO(result)` until actually completed.

## References

- Zhang et al., [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412), ICLR 2018.
- PyTorch, [CrossEntropyLoss uniform-mixture label smoothing](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

## Local verification

After adding these controls, all **222 synthetic tests passed in 114.720s**
on 7 October 2026. Eight new tests cover target formulas/gradients, RNG and
version compatibility, unmixed metric inputs, CLI forwarding and BCE/CE
checkpoint prediction replay. The earlier sandbox-only full-suite attempt
failed because OpenMP/DataLoader subprocesses could not access system shared
memory; the unrestricted local full-suite rerun passed. No real MRI is used
by these tests. Python syntax, plan resolution and diff whitespace checks passed.
