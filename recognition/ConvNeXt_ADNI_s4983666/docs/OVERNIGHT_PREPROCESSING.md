# Overnight preprocessing experiments

This prospective suite uses scratch ConvNeXt Lite, frozen fold-1 train/early-stop
patients and one GPU serially. It does not score outer validation, calibration,
or final test. Comparing cases on early-stop patients is development tuning,
not independent five-fold validation or evidence that the course target is met.

The default plan is `config/preprocessing_overnight.json`: 18 fresh cases,
17 with 30 epochs and one reference with 60 (570 epochs total). Patience equals
the full epoch budget. Every checkpoint remains selected by minimum early-stop
scan log loss, rather than maximum slice accuracy or the final epoch. No previous
weights are resumed. AdamW, weighted slice BCE and FP32 retain their meanings.

The first four cases separate intensity normalization and native centering/crop.
Historical/noncrop canvases are 240 x 256; the cropped canvas is fitted using
training scans only and recorded independently for each case. Compare intensity
within each geometry pair; do not attribute a canvas change solely to brightness.

Other cases compare integer shifts (+/-4 pixels, no interpolation), gamma LUT
(0.9 to 1.1), both augmentations, constant LR 3e-5/1e-4/3e-4, weight decay
0.05/0.2, two-epoch warmup/cosine, and class/patient balancing. Balanced sampling
also uses the sampler's declared unit BCE class prior; it is not purely an ordering
change. `C01_combined` prospectively combines integer/gamma, cosine and balancing.
Reference/combined are paired at base seeds 3710, 4710, 5710; effective seed adds
the fold number. The combination is not chosen adaptively by overnight results.

```bash
cd "$HOME/comp3710/comp3710-adni"
bash slurm/submit_preprocessing_suite.sh --dry-run
bash slurm/submit_preprocessing_suite.sh
```

The default allocation requests one GPU, four CPUs and eight hours. Based on
job635874's 532.145 seconds of training/evaluation for 30 epochs, the training
portion alone is about 2.8 hours across 570 epochs; scans are audited and each
checkpoint replayed as additional work. Queue time and total duration are not
promised. Slurm wall time is a cap, not an instruction to stay idle until eight
hours. Disconnecting SSH or sleeping the laptop does not stop an accepted job.
Do not edit the server source/plan/manifests while this job runs.

Results stay under `$HOME/comp3710/runs/preprocessing_overnight_job<JOBID>/`:

- `launch.json`: resolved plan, actual settings, source/launcher/plan hashes,
  GPU/node, and all frozen file seals.
- Each case: `execution.log`, command record, `train/` including the best
  checkpoint/history/metrics/curves/resources/calibration/rejection/failures and
  preprocessing QA; `replay/` plus verified `summary.json` on success.
- `case_result.json` marks failures with the stage/error; cases use separate
  fresh directories and cannot overwrite earlier experiments.
- Suite `summary.json`/`summary.csv` update after every case. Failed cases remain
  explicit and have blank metrics. Source/seal changes abort the entire suite.
- `comparison.png` and `trajectories.png` are generated after all cases finish;
  selected metrics and full histories are kept separately. No MRI thumbnails are
  added to these aggregate figures.

```bash
squeue --me
cat "$HOME/comp3710/runs/preprocessing_overnight_job<JOBID>/summary.json"
tail -n 30 "$HOME/comp3710/runs/slurm_logs/preprocessing_overnight_<JOBID>.out"
```

A `running` summary is partial. `complete` requires all 18 training/replay checks
to pass; `completed_with_failures` has a nonzero exit status and failed cases to
inspect. A killed job leaves its last atomic progress summary and case log; an
unfinished case must not be counted as successful. No automatic result uploading
or deletion is performed by the GPU job. Outputs remain on Rangpur.
