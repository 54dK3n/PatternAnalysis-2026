# Fixed follow-up recipe contrasts (7 October 2026)

Ken authorized additional trials after reviewing historical recipe overlap and
the real fold-1 sample count. This prospective plan adds eight cases of 30 epochs
(240 epochs), one GPU/four CPUs, serial execution with a six-hour wall cap.
It uses existing execution switches only. No implementation or current plan is
modified while job 638789 runs. There is no outcome-dependent case addition.

All cases start from the existing X04_peer_full recipe: scratch ConvNeXt-Tiny,
224x224, three repeated grayscale channels, unweighted two-class CE, BF16,
AdamW LR 3e-4 / decay 1e-4 / batch 64, cosine without warmup to zero, DropPath 0.1,
no augmentation/preprocessing, slice-uniform sampling and mean-logit patient
aggregation. Fold 1 and the minimum inner early-stop scan-loss selection rule
remain unchanged. All cases run 30 epochs with patience 30. Base seed 3710 is used
except the declared 5710 repeat; training adds the fold number.

| Case | Change from X04 | Question |
|---|---|---|
| N01_peer_one_channel | One grayscale channel | Does repeating channels affect this CE/BF16 recipe? |
| N02_peer_lr_low | LR 1e-4 | Is the smaller learning rate preferable? |
| N03_peer_decay_01 | Weight decay 0.01 | Does stronger parameter regularization help? |
| N04_peer_warmup | Two warmup epochs before cosine | Does the warmup/cosine trajectory help? |
| N05_peer_drop_02 | Maximum DropPath 0.2 | Does stronger stochastic depth help? |
| N06_peer_patient_balance | Class/patient-balanced replacement draws | Does equal patient mass help given unequal scan counts? |
| N07_peer_native_geometry | 240x256 input | Does the historical rectangular geometry help? |
| N08_peer_seed5710 | Base seed 5710 | Does the full recipe transfer to another initialization? |

Compare each intervention with X04 from the first suite, not with an adaptive
winner. N04 changes the scheduler name and its warmup duration as one trajectory
intervention. N07 changes both dimensions and resource costs; it does not isolate
aspect ratio from pixel count. N06 retains 14900 draws per epoch but replacement
means unique-slice coverage is not guaranteed, unlike slice-uniform loading.
N08 does not add a third matched reference seed, so it checks full-recipe
sensitivity rather than a three-seed paired treatment effect. Channel changes
also change stem parameterization/initialization. More inner trials increase
selection reuse; none is independent validation or proof of final accuracy.

## Upload and submit without changing active source

Keep the authoritative plan in config/recipe_followup_20261007.json. Place an
identical immutable copy outside the active server source under
$HOME/comp3710/experiment_plans/recipe_followup_20261007.json. Use the existing
launcher; do not deploy a full release or alter current source/plan files.

```bash
cd "$HOME/comp3710/comp3710-adni"
bash slurm/submit_preprocessing_suite.sh \
  --plan "$HOME/comp3710/experiment_plans/recipe_followup_20261007.json" \
  --time-limit 06:00:00 --dependency afterok:638789 --dry-run
```

After checking the preview, omit --dry-run to submit:

```bash
bash slurm/submit_preprocessing_suite.sh \
  --plan "$HOME/comp3710/experiment_plans/recipe_followup_20261007.json" \
  --time-limit 06:00:00 --dependency afterok:638789
```

The existing Python launcher invokes sbatch; the allocation script requests
--gres=gpu:1 on partition comp3710 / account comp3710. Successful completion of 638789
is required before this follow-up starts. Actual accepted job: **639159**, currently pending on `afterok:638789`.
Its job ID/output/plan hash is saved in
`runs/recipe_submission_followup_20261007.json` on Rangpur. Never repeat submission
if the outcome is unknown; inspect that record and Slurm first.

Existing per-case figures, replay checks, source identity and 22 frozen seals
apply. Archive each completed suite using archive_recipe_results.py and the
aggregate allowlist, then verify and upload to the established owner-only
20261007_Recipe_Comparison Drive folder. Temporary downloads are deleted only
after cloud metadata verification. No weights, MRI failure grids or prediction
rows are included. Monitor all registered jobs and pause only after every suite
has a terminal state and its result/failure evidence is archived.
