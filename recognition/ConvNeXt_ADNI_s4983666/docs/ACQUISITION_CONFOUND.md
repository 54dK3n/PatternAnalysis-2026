# Acquisition period and scanner are entangled with the diagnosis

The five-fold runs left two questions: why accuracy varies so much between
folds, and why most errors are AD patients called NC. Fold composition does not
explain either. The folds have the same AD share (0.32-0.33), number of scans
per patient (2.05-2.41) and mean JPEG size (6.08-6.20 KB), and JPEG size is
unrelated to the error rate. The supplied metadata does explain it.

## What the metadata contains

`meta_data_with_label.json` has 2,189 records with the same fields: `label`
(0 CN, 1 MCI, 2 AD), `raw`, `masked` and `c1`-`c5` (paths of the original
NIfTI, an N4-corrected skull-stripped volume and SPM tissue maps). There is
no age, sex, site or scanner field. The `raw` file name, however, contains the
**sequence name** and the **acquisition date**, for example
`ADNI_<NNN_S_NNNN>_MR_Accelerated_Sagittal_MPRAGE__br_raw_<YYYYMMDD...>_I<scan ID>.nii`.
All 1,050 development scans could be matched to a parsed record.

## The confound

Development scans by acquisition period and diagnosis:

| Acquired | AD scans | NC scans |
|---|---:|---:|
| 2011-2016 | 417 | 0 |
| 2017-2021 | 86 | 547 |

By sequence family (IR-SPGR/FSPGR = GE, SENSE = Philips, other MPRAGE mostly
Siemens; "accelerated" = GRAPPA/SENSE/"Accelerated" in the name):

| Sequence family | AD scans | NC scans |
|---|---:|---:|
| Siemens/other, non-accelerated | 144 | 0 |
| GE, non-accelerated | 63 | 0 |
| Philips, accelerated | 47 | 0 |
| Siemens/other, accelerated | 159 | 438 |
| GE, accelerated | 90 | 109 |

In the development split, every NC scan was acquired in 2017-2021 with an
accelerated sequence, and any scan acquired before 2017, or with a
non-accelerated or Philips sequence, is AD. A classifier that recognises the
acquisition style therefore separates more than 80% of the development AD scans
from all NC scans without looking at atrophy.

## Errors follow the confound

Round-3 out-of-fold predictions (each scan scored by the fold model that never
trained on its patient), scan error rate:

| Group | ConvNeXt-Tiny | SmallCNN |
|---|---:|---:|
| AD acquired 2011-2016 (style unique to AD) | 0.30 (n = 417) | 0.45 |
| AD acquired 2017-2021 (same style as NC) | **0.67** (n = 86) | **0.84** |
| NC (all 2017-2021) | 0.15 (n = 547) | 0.10 |

Restricting the evaluation to the 633 scans acquired in 2017-2021, where both
classes share the same acquisition period, shows how much of the
discrimination remains:

| Scans | Model | AUROC | Balanced accuracy | Accuracy |
|---|---|---:|---:|---:|
| all 1,050 | ConvNeXt-Tiny | 0.850 | 0.744 | 0.749 |
| 2017-2021 only (86 AD, 547 NC) | ConvNeXt-Tiny | **0.729** | 0.589 | 0.780 |
| all 1,050 | SmallCNN | 0.839 | 0.692 | 0.701 |
| 2017-2021 only (86 AD, 547 NC) | SmallCNN | **0.654** | 0.532 | 0.801 |

(Plain accuracy rises on the subset only because 86% of it is NC.) Part of the
headline AUROC comes from the acquisition period. Under the Round-3
`strong`-augmentation recipe, scratch ConvNeXt keeps higher discrimination on
this subset than SmallCNN (0.729 vs 0.654 pooled scan AUROC), though adding
`scanner` augmentation to SmallCNN narrows that gap in five-fold development
(0.726 ± 0.090 vs 0.736 ± 0.048 for scratch ConvNeXt and 0.778 ± 0.047 for
pretrained ConvNeXt; see [EXPERIMENTS.md](EXPERIMENTS.md)).

The fold-to-fold spread follows the same pattern: the folds differ mainly in
how many AD patients acquired in 2017-2021 land in validation (5 to 14 of 31),
and fold 5, the weakest for ConvNeXt, missed 12 of its 13 such patients.
(Those counts use the subject number, >= 6000 for ADNI-3 enrolment, which
coincides with this group: all 2017-2021 AD scans belong to subject
numbers >= 6000 except one.)

NC patients with older subject numbers (< 2000, enrolled in ADNI-1 but scanned
in 2017-2021) are also called AD more often (scan error 0.35 against 0.11 for
NC with subject numbers >= 6000). Their acquisition period is the same as the
other NC scans, so age-related atrophy is a more likely explanation there than
acquisition style.

## Probe: can the network see the acquisition style?

The model input never contains metadata, but sequence, vendor and acquisition
period change image contrast, noise texture and intensity inhomogeneity.
[`probe_acquisition.py`](../probe_acquisition.py) trains the same SmallCNN
(three-slice context, strong augmentation, EMA, 30 epochs, final-epoch
weights, same five folds) on a non-diagnostic target **with the diagnosis held
fixed**, and scores each fold once on its validation patients (Slurm job
652758):

* `era_within_ad`: AD scans only, acquired 2017-2021 vs 2011-2016.
* `vendor_within_nc`: NC scans only (all 2017-2021), GE IR-SPGR vs other sequences.

| Fold | era_within_ad scan AUROC (2017-21 / 2011-16 val scans) | vendor_within_nc scan AUROC (GE / other val scans) |
|---:|---|---|
| 1 | 0.956 (10 / 89) | 1.000 (24 / 88) |
| 2 | 0.966 (17 / 70) | 0.979 (23 / 90) |
| 3 | 0.626 (9 / 114) | 1.000 (14 / 92) |
| 4 | 0.598 (24 / 80) | 0.972 (27 / 84) |
| 5 | 0.936 (26 / 64) | 0.973 (21 / 84) |
| pooled out-of-fold | 0.807 (slice 0.754) | **0.986** (slice 0.931) |

* With the diagnosis and period fixed, the vendor/sequence is recognised with
  very high accuracy from pixels (pooled scan AUROC 0.986): the scanner
  signature is more easily separable from pixels than the disease signal itself
  (diagnosis scan AUROC 0.85).
* The acquisition period within AD patients is recognised with pooled AUROC
  0.81, but unevenly (0.60-0.97 by fold). This probe is itself partly
  confounded: 2017-2021 AD patients are newer ADNI-3 enrolments and may differ
  in disease stage, not only in acquisition.

## Grad-CAM: where the evidence lies

[`explain_gradcam.py`](../explain_gradcam.py) (Selvaraju et al. 2017; Slurm
job 656034) explains the middle slice of every validation scan of folds 4 and 5
for three checkpoints, using the last feature stage and the predicted class.
"Outside head" is the share of the map's mass on background pixels. As a
reference, background covers 0.72 of these slices on average (pixels exactly 0
in the source JPEGs, 410 slices), so a map spread uniformly over the image
would score about 0.72 and a map confined to the head would score near 0.

| Checkpoint (fold 4 / fold 5) | Outside head: AD 2011-16 | AD 2017-21 | NC 2017-21 |
|---|---|---|---|
| ConvNeXt-Tiny scratch (round 3) | 0.716 / 0.720 | 0.823 / 0.776 | 0.830 / 0.771 |
| ConvNeXt-Tiny ImageNet-1k (round 5, strong aug.) | 0.378 / 0.447 | 0.398 / 0.464 | 0.357 / 0.404 |
| SmallCNN (round 3) | 0.230 / 0.272 | 0.222 / 0.267 | 0.308 / 0.273 |

Scan accuracy and mean p(AD) of the same groups (fold 4 / fold 5):

| Checkpoint | AD 2011-16 acc. (mean p) | AD 2017-21 acc. (mean p) | NC 2017-21 acc. (mean p) |
|---|---|---|---|
| ConvNeXt-Tiny scratch | 0.762 (0.74) / 0.625 (0.64) | 0.583 (0.54) / 0.077 (0.24) | 0.811 (0.21) / 0.848 (0.18) |
| ConvNeXt-Tiny ImageNet-1k | 0.887 (0.85) / 0.797 (0.79) | 0.417 (0.39) / 0.154 (0.23) | 0.910 (0.12) / 0.914 (0.11) |
| SmallCNN | 0.775 (0.63) / 0.812 (0.71) | 0.083 (0.26) / 0.385 (0.40) | 0.937 (0.15) / 0.724 (0.36) |

Observations (descriptive only):

* The scratch ConvNeXt-Tiny maps are not concentrated on the head: their
  outside share is at or above the uniform reference, and the group means show
  a block pattern at the 1/32 resolution of the last stage rather than a
  head-shaped region.
* The ImageNet-initialised model and the CNN put most of the map inside the
  head (0.36-0.46 and 0.22-0.31). The CNN maps follow the head outline; the
  pretrained maps form one central region.
* Within each checkpoint the outside share hardly differs between the groups,
  including the AD 2017-2021 scans that are mostly missed, so the maps alone do
  not show *which* cue separates the acquisition periods.
* Caveats: the last ConvNeXt stage has a 7x8 grid, so its maps are coarse; a
  large receptive field can carry head information into background cells;
  middle slices are not registered, so group means are indicative only.

Anatomical and spatial interpretation of the maps (`gradcam_group_means.png` and
`gradcam_cases.png` in `outputs/experiment_logs/gradcam_20261013/` and
`figures/adni/`):

* **Scratch ConvNeXt-Tiny (`gradcam_means_scratch_fold5.png`)**: Across all three
  cohort groups (`AD 2011-16`, `AD 2017-21`, `NC 2017-21`), the mean heatmaps
  exhibit a coarse 7×8 checkerboard/border pattern concentrated along the
  top, bottom-left and bottom-right black background margins (`0.716-0.830` of
  attribution mass outside the skull, with only 1% of validation scans placing
  more than half of their attribution inside the head). While outside-head
  attribution alone does not explain lower accuracy (SmallCNN places `98-99%` of
  scans' majority attribution inside the head yet scores similarly to or below
  scratch ConvNeXt in single-fold development), it shows that training a
  27.8M-parameter network from scratch on 381 patients allows global
  field-of-view border artifacts and background padding statistics carried by
  large 7×7 depthwise receptive fields to dominate the final-stage features.
* **ImageNet-1k–initialised ConvNeXt-Tiny (`gradcam_means_pretrained_fold5.png`,
  `gradcam_cases_pretrained_fold5.png`)**: Pretrained initialisation pulls the
  attribution into the intracranial vault for 82% (fold 4) and 72% (fold 5) of
  validation scans (`0.357-0.464` outside-head mass). On these unregistered
  mid-sagittal slices at 7×8 final-stage resolution, the group-mean and
  individual-case maps form a broad central intracranial region (roughly
  overlapping the mid-sagittal ventricular/callosal and deep parenchymal zone).
  However, in `gradcam_cases_pretrained_fold5.png`, confident false negatives in
  `AD 2017-21` (scan p(AD) = 0.00) and confident false positives in `NC 2017-21`
  (scan p(AD) = 0.86–0.96) highlight this same central brain region as correct
  predictions. Paired with the `vendor_within_nc` probe (`0.986` scan AUROC),
  this is consistent with the classifier reading global intracranial contrast
  and protocol signatures alongside gross ventricular morphology.
* **SmallCNN (`gradcam_means_cnn_fold5.png`)**: SmallCNN confines `98-99%` of
  scans' majority attribution inside the head (`0.222-0.308` outside), with its
  smaller receptive field distributing weight along high-contrast tissue
  boundaries—the superior frontoparietal cortical rim, occipital/cerebellar
  contour and brainstem edge.

## Consequences

* Development accuracy and AUROC overstate how well the models detect
  AD-related anatomy. The 2017-2021 subset is the least confounded estimate
  available.
* The test patients come from the same pool, so the test score inherits the
  same confound. Test results should be reported overall and for the
  2017-2021 subset.
* A deployed model would likely degrade on sites, scanners or protocols where
  acquisition style does not predict diagnosis.

## Possible mitigations

| Option | Idea | Cost / caveat |
|---|---|---|
| Report by acquisition period | Overall and 2017-2021-only metrics, here and on test | Free; does not change the model |
| Scanner-robust augmentation | Random bias field, noise, blur/sharpening, stronger contrast/gamma so texture and contrast stop identifying the scanner | Implemented as the `scanner` profile. Screened on folds 4-5 (Slurm 653053): CNN 2017-21 scan AUROC 0.718 / 0.560 -> 0.752 / 0.710; scratch ConvNeXt 0.818 / 0.696 -> 0.779 / 0.725; no gain with ImageNet initialisation. Completed across all five folds for SmallCNN post-hoc (Round 7, Slurm 681962/681963): development 2017-21 scan AUROC rises from 0.606 ± 0.081 to 0.726 ± 0.090 (patient acc. 0.750 -> 0.811), though on test its 2017-21 scan AUROC is 0.805 (vs 0.866 for `strong` SmallCNN and 0.878 for pretrained ConvNeXt) ([EXPERIMENTS.md](EXPERIMENTS.md)) |
| ImageNet initialisation | Generic pretrained features instead of features learnt only from this confounded data (round 5) | Overall accuracy up (fold 4 slice 0.817), 2017-21 scan AUROC unchanged (0.770 / 0.740); does not remove the shortcut |
| Train only on 2017-2021 scans | Removes the period cue completely | Only 86 AD scans left; likely too few |
| Re-weight by period x diagnosis | Up-weight 2017-2021 AD scans | Cannot balance cells that are empty (no NC before 2017) |
| Adversarial removal of the scanner signal | Gradient reversal on a vendor/period head, conditioned on diagnosis | More complex; risk of removing disease signal |
| Harmonised inputs | Use the N4-corrected, skull-stripped `masked` volumes | Not part of the supplied JPEG data |

Scripts used for the tables: `probe_acquisition.py` and the per-scan
predictions in each run's `val_scan_predictions.csv`; acquisition year and
sequence were parsed from the metadata `raw` paths.

## References

* R. Geirhos et al., "Shortcut learning in deep neural networks", Nature Machine Intelligence, 2020 [8].
* J. R. Zech et al., "Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study", PLOS Medicine, 2018 [9].
