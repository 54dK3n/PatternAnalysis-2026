# Input and frozen-feature diagnostics

Use this diagnostic to investigate whether geometric transforms, spatial
compression, or ineffective residual branches contribute to weak development
performance. It does not change an architecture or optimize the backbone.
Poor generalization alone cannot identify which mechanism is responsible.

## Run on Rangpur

Use the same Python environment and an allocated compute session used for the
existing experiments. Do not run the full diagnostic on a login node. All three
models should use the same sampling seed and limits. Existing checkpoints and
run folders remain read-only; use a new, separate output directory for each run.

```bash
python3 diagnose.py \
  --checkpoint "$HOME/comp3710/runs/convnext_aug_fold01/best.pt" \
  --data-root /home/groups/comp3710/ADNI \
  --splits-dir "$HOME/comp3710/adni_splits_v1" \
  --output "$HOME/comp3710/diagnostics/convnext_aug_fold01_v1" \
  --device cuda --batch-size 16 --workers 4 --threads 4 \
  --seed 3710 --max-train-patients 64 --max-early-patients 0 \
  --max-scans-per-patient 2 --max-transform-images 12 --shift-pixels 2 \
  --probes --probe-epochs 200
```

Repeat with `cnn_fold01/best.pt` and `convnext_fold01/best.pt`, changing the output
name. Zero patient or scan limits keep all available records. Patient sampling
is stratified by NC-only, AD-only, or mixed longitudinal labels; scan sampling
within each selected patient does not inspect predictions. Every selected scan
retains all its slices. A manifest records the exact selected identities.

The diagnostic always verifies the source images and frozen manifests first.
That integrity audit covers all partitions, as in existing training commands;
model inference and feature probes use only `train` and `early_stop`. There is
no option to select `val`, `calibration`, or `test` for model scoring.

Early format-one `small_cnn_v1` checkpoints omitted both `initialization` and
`pretrained_weights`. This exact legacy format is accepted with
`initialization_metadata_status=legacy_fields_not_recorded`; the original config
and checkpoint remain unchanged. This status does not claim that weight tensors
prove initialization. Other supported checkpoints require both fields explicitly
recording random initialization and no pretrained weights. Missing modern or
ConvNeXt provenance and explicit pretrained records are rejected.

## What the outputs measure

- `config.json`: the checkpoint SHA-256, original configuration, frozen manifest
  hash, diagnostic source fingerprints, environment and sampling seed.
- `transforms/transform_grid.png`: an EXIF-corrected grayscale source, the
  unaugmented input, one fixed random light transform and one boundary case for
  each sampled training patient. The source panel is display-scaled only when
  needed; source dimensions and preprocessing differences remain in the CSV.
- `transforms/transform_samples.csv`: actual dimensions, image mode, EXIF,
  resizing and foreground margins at gray-level thresholds 8, 16 and 32.
- `transforms/transform_draws.csv`: eight fixed random and four boundary draws
  per image, comparing the production fixed canvas with a padded reference
  subjected to the same single interpolation operation. A thresholded intensity
  proxy is not a brain mask, and its retention is not clinical information loss.
- `train_*_predictions.csv` and `early_stop_*_predictions.csv`: unchanged model
  outputs on the declared diagnostic subsets. These are not new outer-fold or
  final-test performance estimates.
- `summary.json`: per-stage shapes and activation moments; per-block learned
  LayerScale and residual/skip norm ratios for ConvNeXt; and slice probability
  changes after exact two-pixel translations with black fill. Translation can
  clip boundaries and differs from the bilinear training augmentation.
- `probes/feature_probes.json` and its patient prediction CSV, when `--probes`
  is set: fixed-budget linear heads on patient-mean frozen GAP embeddings.

For a light-augmented checkpoint the transform audit uses its recorded rotation
and translation bounds. For an unaugmented checkpoint it examines the default
light profile as a proposed control, not as a reconstruction of training inputs.
No diagnostic RNG stream claims to reproduce the exact historical worker draws.

Linear heads fit only training patients; their feature standardization and class
weights also use training patients only. Each stage starts with the same seed
and runs exactly the declared number of full-batch steps. Early-stop scores do
not choose a head checkpoint or threshold. Patients with longitudinally mixed
labels stay in scan diagnostics but are explicitly excluded from patient-mean
probes using their full-role label history, even when only one scan was sampled.

## Interpret the evidence

1. Inspect foreground margins and the actual image grid before changing input
   normalization or crop policy. If content leaves the canvas, distinguish
   canvas cropping from interpolation smoothing using the padded reference.
2. Inspect residual contributions, rather than treating the initial LayerScale
   of `1e-6` as proof that the blocks never learned. A small residual can be
   appropriate; its magnitude alone does not identify a failed layer.
3. Compare shallow and late probe results as evidence about linear separability.
   Feature dimension and the patient-mean aggregation differ between stages and
   from the original classifier. Scores do not prove anatomical relevance or
   establish that spatial information was destroyed.
4. Compare shift sensitivity on the same selected identities for CNN and both
   ConvNeXt runs. Large changes identify a robustness question; they do not
   identify the cause without a controlled follow-up experiment.

If spatial processing remains implicated, a separate experiment can compare
the current `4x4/stride4` stem with an overlapping `4x4/stride2/padding1` stem,
holding widths, depths and the training recipe fixed. Crossing those two stems
with `none` and `light` gives a controlled interaction comparison. This changes
overlap, receptive fields and computational cost, so it is not a pure proof of
information recovery. **No such architecture or training experiment is
implemented by this diagnostic.** Further tuning requires an inner-only training
path; the existing outer-scoring training default is unchanged.

## Validation and evidence status

Synthetic checks cover complete-scan sampling, role separation, source and
checkpoint fingerprints, model-state preservation, hook cleanup, known residual
ratios, padded transform references and actual fixed-budget linear probes. They
verify implementation behavior, not ADNI feature quality. The synthetic image
grid must never be presented as real MRI evidence.

Real fold-1 diagnostics for the three existing checkpoints completed on Rangpur
on 2026-10-01. The imported outputs are in
`outputs/server_imports/features_20261001_131402/diagnostics_20261001_130255/`;
the verified comparison is in `outputs/review/feature_diagnosis_20261001/`.
These are internal development diagnostics, with subset and probe limitations,
not independent final performance estimates. Local synthetic validation remains
a software check only.

The [feature and position diagnostic plan](FEATURE_DIAGNOSTIC_PLAN.md) specifies
follow-up controls for sampling phase, boundaries, interpolation, probe
reliability, training dynamics and optional attention with 2D RoPE. The
first A/B/D extension is implemented in `diagnose_followup.py`; see
[follow-up usage](FEATURE_DIAGNOSTIC_FOLLOWUP.md). Interpolation retraining,
optimizer controls, architecture variants and attention/RoPE remain proposals.

References: [ConvNeXt model implementation](https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py),
[optimizer recipe](https://github.com/facebookresearch/ConvNeXt/blob/main/optim_factory.py),
and [Pillow rotation semantics](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.rotate).
