# Sampling boundary and probe followup

This command implements the first A/B/D extension in the [diagnostic plan](FEATURE_DIAGNOSTIC_PLAN.md). It scores existing checkpoints on inner data, records intermediate responses to exact integer shifts, controls input edge loss and fits native-dimension or training-only PCA linear probes. It does not train a backbone, alter original checkpoints or evaluate outer-validation, calibration or final-test predictions.

## Execute on an allocated compute node

Use the same activated environment as the original experiments. The prior completed diagnosis is an input for exact cohort replay; it is never overwritten. Run the three models with identical settings and separate new outputs.

```bash
python3 diagnose_followup.py \
  --checkpoint "$HOME/comp3710/runs/convnext_aug_fold01/best.pt" \
  --data-root /home/groups/comp3710/ADNI \
  --splits-dir "$HOME/comp3710/adni_splits_v1" \
  --reference-dir "$HOME/comp3710/diagnostics_20261001_130255/convnext_aug_fold01" \
  --output "$HOME/comp3710/diagnostic_followup/convnext_aug_fold01_v1" \
  --device cuda --batch-size 16 --workers 4 --threads 4 \
  --seed 3710 --max-shift 8 --bootstrap-samples 1000 \
  --max-scans-per-patient 2 --pca-components 16 --probe-epochs 200
```

Change the checkpoint/reference/output run name for `cnn_fold01` and `convnext_fold01`. The exact prior train and early-stop CSV rows must match the verified role manifests, complete-scan coverage, checkpoint bytes and configuration. The early-stop list stays fixed across all four probe comparisons. Original replay patients retain exactly their selected scans in the expanded training cohort; additional training patients use independent patient-derived scan sampling. The expanded cohort therefore contains all eligible patients, but is not a full longitudinal history evaluation.

`--routes spatial` runs A and B together; `--routes probes` runs D only. The default is `all`. Without `--reference-dir`, selection falls back to the original deterministic sampling policy; that is explicitly recorded as a newly generated selection rather than a verified prior-artifact replay. Default reference patient/scan caps match the earlier tool. A smaller maximum shift may be used for a prespecified runtime pilot; it must not be chosen after inspecting scores. Zero bootstrap samples disables intervals, not the measurements.

On the Mac, the prepared isolated bundle and interactive upload/download helpers are in `outputs/diagnostic_followup_transfer/`. Run `upload_diagnostic_followup.py` in a local terminal; SSH asks for authentication directly and the helper prints the new server runner command. The runner defaults to the prior `diagnostics_20261001_130255` directory, checks bundle hashes and requires an allocated CUDA compute session. It does not replace remote project code or submit a job. Its arguments are Python executable, project directory, reference directory and fresh output directory, in that order. Use `download_diagnostic_followup.py` after all three top-level run summaries are complete. The fetch excludes source MRI images and checkpoints; it permits only bounded JSON/CSV/PNG diagnostic artifacts.

## Spatial and boundary outputs

The default sweep performs one original forward, 32 translated forwards and 32 round-trip forwards per slice, for **65 forwards**. Sampling alone uses 33; boundary controls account for the additional 32. Measure a short pilot for real runtime rather than assuming the previous diagnostic duration. Only early-stop slices are swept. Dense maps are streamed by batch and not stacked for the full cohort; maps and MRI images are not exported. Transient model workspaces can add to peak memory.

| Output under `spatial/` | Meaning |
|---|---|
| `original_slice_predictions.csv` and `original_scan_predictions.csv` | Unmodified checkpoint predictions on the declared early-stop subset |
| `shift_slice_predictions.csv` | Per-slice original/translated/round-trip outputs, logit changes, flips, exact input pixel loss, three intensity-proxy margins and fixed confidence strata |
| `stage_slice_changes.csv` | Stem/downsampler/stage responses with effective stride, absolute/relative GAP changes and norms, optional exact aligned error, overlap size and reference zero flags |
| `shift_patient_means.csv` | Equal-patient summaries for all slices, confidence strata and the margin-greater-than-eight subgroup |
| `stage_patient_means.csv` | Per-patient mean feature responses for each observer and displacement |
| `summary.json` | Patient-weighted groups, conditional patient-bootstrap intervals, class coverage, layer summaries, original slice/scan metrics and limitations |
| `shift_boundary_curves.png` | Prediction and round-trip response by direction and displacement |

Intensity thresholds are fixed at gray levels 8, 16 and 32. Empty proxies have margin -1 and do not enter the margin subgroup. These proxies are not anatomical masks. The pixel MAE is in normalized input units [-1, 1]. Round-trip changes isolate image-edge loss but do not remove internal convolution-padding effects.

Exact spatial alignment is reported only when the input displacement is divisible by the observed layer's effective stride and a common grid overlap exists. Relative error is the L2 norm of the aligned difference divided by the reference overlap norm, clamped at 1e-12. Other displacements have an empty aligned value, never an interpolated or zero substitute. Common overlap does not imply a boundary-free receptive field. GAP cosine distance is empty if either vector has zero norm. Near-zero denominators can amplify relative measures, so inspect original logits, probabilities and zero flags as well.

Every patient group averages its available selected slices, then gives patients equal weight. Undefined feature distances are omitted, with per-metric valid slice and patient counts recorded. Its scan/slice counts and NC/AD slice counts are explicit; patients with mixed selected labels can contribute to both class-coverage counts. Confidence strata use the original checkpoint probability margins and can contain different patients across models. Bootstrap intervals resample patients within a fixed group and are conditional on the checkpoint and already-used early-stop patients. They do not include training, checkpoint selection, seed or fold uncertainty. Cross-model paired comparisons require matching the exported patient IDs and populations; interval overlap alone is not a test of a paired difference.

## Probe outputs

Four comparisons share the same early-stop embeddings: `reference/native`, `reference/pca_16`, `expanded/native` and `expanded/pca_16`, under `probes/`. Mixed longitudinal labels are excluded using each patient's full verified role, even if only one scan was selected. The top-level summary records exclusions and actual cohort coverage.

Every probe has 200 fixed full-batch AdamW steps by default, learning rate 0.01, weight decay 0.01 on weights and zero on bias, seed 3710 and threshold 0.5. Each patient supplies one mean GAP embedding. Training patients alone determine class weights and feature mean/std. The PCA control then centers those standardized training rows, fits an unwhitened SVD basis and applies it to early-stop rows. Components use deterministic loading signs; the basis, center, singular values and numerical rank are recorded. PCA component count is prescribed, not chosen using early-stop scores. Rank deficiency is reported, not hidden by a component search.

Probe dimensions must not exceed the stage width or training-patient count minus one. Sixteen is appropriate for the real prescribed cohort; smaller synthetic smoke cohorts need an explicitly smaller `--pca-components`. A failed PCA/probe run has no completed top-level summary. Probe classifiers and dense embeddings are not saved as model weights. The historical aggregation field is retained for compatibility, with an added scope field explicitly limiting it to supplied selected scans.

## Validation and evidence status

Synthetic validation: 140 full-suite cases passed. A separate native-resolution smoke exercised all 18 standard Tiny blocks, eight spatial observers, all 65 default forwards on a synthetic 240x256 canvas, unchanged backbone state and five 16-component probes with the 200-step budget. The probe part used independently generated patient feature matrices matching the Tiny stage widths. These are implementation checks, not ADNI feature or performance evidence. Real A/B/D follow-up results remain `TODO(result)` until execution on Rangpur. This extension does not implement interpolation retraining, an overlapping stem, antialiasing, an inner-only backbone training mode or attention/RoPE.
