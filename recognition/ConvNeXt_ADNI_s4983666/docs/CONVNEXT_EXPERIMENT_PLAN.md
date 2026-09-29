# ConvNeXt Architecture, Diagnosis, and Experiment Plan

Status: proposal, 2026-09-25. This document changes no training behavior. New model variants, tuning controls, and schedules described below are not implemented.

## Question and current conclusion

Why does the current randomly initialized ConvNeXt-Tiny perform worse than SmallCNN on development fold 1, and which controlled experiments could explain or reduce the difference?

The evidence suggests a generalization problem under the current data and training recipe. It does not establish that ConvNeXt is inherently unsuitable for AD/NC classification, that parameter count alone caused the difference, or that longer training would fix it. ConvNeXt is itself a convolutional neural network; the comparison is between two CNN architectures.

## Implemented architecture

The implementation in [models/convnext.py](../models/convnext.py) follows the Tiny stage depths and widths in the [authors' reference code](https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py). This project uses one grayscale input channel, one binary output logit, and random initialization. Feature dimensions below are channels x height x width; source images are 256 pixels wide and 240 pixels high.

| Component | Operation | Output shape |
|---|---|---|
| Input | One grayscale slice | 1 x 240 x 256 |
| Stem | 4 x 4 convolution, stride 4, LayerNorm | 96 x 60 x 64 |
| Stage 1 | 3 ConvNeXt blocks | 96 x 60 x 64 |
| Downsample + stage 2 | LayerNorm, 2 x 2 stride-2 convolution, 3 blocks | 192 x 30 x 32 |
| Downsample + stage 3 | LayerNorm, 2 x 2 stride-2 convolution, 9 blocks | 384 x 15 x 16 |
| Downsample + stage 4 | LayerNorm, 2 x 2 stride-2 convolution, 3 blocks | 768 x 7 x 8 |
| Head | Spatial mean, LayerNorm, linear layer | 1 logit |

Each of the 18 residual blocks applies a 7 x 7 depthwise convolution, channel normalization, channel expansion from C to 4C, GELU, projection back to C, learned LayerScale, and stochastic depth before adding the input. The depthwise convolution processes spatial patterns separately in each channel; the linear layers mix channels at each spatial location. LayerScale starts at 1e-6. Stochastic-depth probability increases from 0 to 0.1 across the blocks. There is no self-attention.

For evaluation, sigmoid converts each slice logit into an AD probability. The arithmetic mean of all 20 slice probabilities supplies the scan probability; the decision threshold is 0.5. Both models use this same aggregation. Neither model jointly processes the slices as a 3D volume.

| Property | SmallCNN | ConvNeXt-Tiny |
|---|---|---|
| Trainable parameters | 97,521 | 27,817,825 (285.25 times as many) |
| Feature channels | 16, 32, 64, 128 | 96, 192, 384, 768 |
| Feature extractor | Four 3 x 3 convolution / GroupNorm / ReLU / max-pool groups | Four stages containing 18 residual blocks |
| Final spatial map | 15 x 16 | 7 x 8 |
| Regularization | Classifier dropout 0.2 | Stochastic depth up to 0.1 |
| Initialization | Random | Random |

Actual forward passes at the native resolution reproduced these shapes and parameter counts. Source review found no obvious stage, shape, or block-construction mismatch. This is not a claim that all possible implementation defects have been excluded.

## Evidence from the project

The frozen fold-1 manifests contain:

| Role | Patients | Scans | Slices |
|---|---:|---:|---:|
| Training | 343 | 745 | 14,900 |
| Early stopping | 38 | 94 | 1,880 |
| Outer validation | 95 | 211 | 4,220 |

Multiple slices from one scan and repeated scans from one patient are correlated observations. The 14,900 training slices are not 14,900 independent patients. Training patients contribute between one and eight scans each. Slice shuffling gives patients with more scans greater weight; the loss's class weighting does not make patient contributions equal. This shared property is a possible issue to investigate, not an established explanation for the architecture gap.

| Fold-1 run | Scan accuracy | Scan macro F1 | Scan AUROC | Selected epoch |
|---|---:|---:|---:|---:|
| SmallCNN, no augmentation | 83.41% | 0.8339 | 0.9136 | 14 |
| ConvNeXt-Tiny, no augmentation | 66.82% | 0.6627 | 0.7429 | 4 |
| ConvNeXt-Tiny, light augmentation | 76.78% | 0.7677 | 0.8588 | 8 |

The CNN result and configuration are available in local server-run artifacts. ConvNeXt results here were supplied by the project owner in the conversation: history, metrics, and the unaugmented configuration. The augmented run's full configuration and predictions still need collection before declaring a fully verified matched comparison. The two ConvNeXt metric files report the same manifest fingerprint as the CNN.

Augmentation increased fold-1 accuracy by 9.95 percentage points. AD scans predicted as NC decreased from 42 to 20, while NC scans predicted as AD changed from 28 to 29. This is promising single-fold evidence, not a significance test or a general improvement claim.

The unaugmented ConvNeXt training loss fell from 0.683 to 0.031 over nine epochs, while early-stop loss reached its minimum at epoch 4. With augmentation, training loss decreased from 0.695 to 0.320 over thirteen epochs; early-stop loss reached its minimum at epoch 8. Continued training fit without consistent improvement on unseen patients is consistent with overfitting. Training uses class-weighted slice loss, while selection uses unweighted scan log loss, so their absolute values cannot be subtracted to quantify a generalization gap. Augmentation also changes the training inputs, making cross-run training losses different objectives.

## Explanations, with confidence limits

1. **Capacity relative to independent data is the leading hypothesis.** Tiny is small relative to other ConvNeXt variants, but is much larger than this project's baseline. Learning 27.8 million parameters from scratch using 343 training patients may allow a strong training fit without reliable transfer to new patients. Augmentation helping is consistent with this explanation; it does not prove it.
2. **The training recipe is a separate possible contributor.** The CNN used learning rate 1e-3, weight decay 1e-4, and batch size 32. The unaugmented ConvNeXt used 1e-4, 0.05, and 16 respectively; the augmented command requested the same ConvNeXt settings. Both use AdamW. Current code uses a constant learning rate and no warmup or moving-average model. Thus the existing comparison changes more than architecture. These settings are not necessarily wrong, but are not established optimal settings for either model.
3. **The architecture and published benchmark recipe are different things.** The original paper studies ConvNeXt on large image benchmarks. Its official training implementation includes learning-rate warmup and cosine decay; the ImageNet commands also enable a moving-average model. Copying those large-scale settings directly is not justified for this dataset. Random initialization is a valid experiment, but it does not import knowledge from a pretrained image model. See the [paper](https://arxiv.org/abs/2201.03545), [official training implementation](https://github.com/facebookresearch/ConvNeXt/blob/main/main.py), and [training commands](https://github.com/facebookresearch/ConvNeXt/blob/main/TRAINING.md).
4. **Spatial compression is a secondary hypothesis.** ConvNeXt starts with stride 4 and ends at 7 x 8, while SmallCNN downsamples gradually and ends at 15 x 16. A different treatment of spatial detail may matter, but there is no evidence yet that this explains the observed errors. Both models discard spatial arrangement at final global pooling and both classify slices independently, so neither fact alone explains their relative performance.
5. **Fold and seed variability remain unresolved for ConvNeXt.** Its two runs cover only one outer fold and one seed. The existing CNN five-fold results already vary substantially. Neither a single favorable nor unfavorable fold settles the architecture ranking. The 38-patient early-stop subset is also a limited basis for checkpoint selection; it is different from outer validation, so their scores need not match.

## Proposed sequence

The first objective is to identify a reproducible, efficient method for unseen patients, rather than to require ConvNeXt to beat the smaller baseline.

1. **Complete the evidence.** Collect the augmented configuration and both ConvNeXt prediction files, preserving all original experiments. Verify manifest identity, patient/scan coverage, seeds, environment, preprocessing, and optimizer settings. Recompute the reported metrics from predictions. Keep the three existing results as historical baselines.
2. **Make future diagnostic tuning use only inner data.** Add an explicit tuning mode before running parameter searches: it must not construct or score outer validation. The current training entry point always scores outer validation after checkpoint selection, so this restriction is not currently available as a CLI option. Compare deterministic, unaugmented training and early-stop predictions using the same scan-level loss if a direct generalization gap is needed. Such training metrics are diagnostic, not held-out results.
3. **Complete the augmentation control.** Add SmallCNN with the same light augmentation, keeping its original optimizer settings. Together with the three existing configurations, this gives a 2 x 2 model/augmentation comparison. Within-model comparisons isolate the augmentation change; between-model comparisons still represent complete training recipes. This establishes whether the augmentation benefit is specific to ConvNeXt or also helps the baseline.
4. **Test model capacity first.** Propose one compact ConvNeXt variant with unchanged stage depths (3, 3, 9, 3), block operations, downsampling, and maximum stochastic depth, but halve the channels to (48, 96, 192, 384). Keep the full-Tiny augmented run's optimizer, batch size, preprocessing, and selection rule. Name it explicitly as a custom compact variant, not standard Tiny, and measure its actual parameters. Better inner validation with comparable training behavior would support a capacity explanation, not prove it in isolation.
5. **Test training schedule second, only if needed.** On a fixed architecture, compare the existing constant learning rate with one specified warmup/cosine schedule. Specify its length, minimum learning rate, budget, and early-stopping interaction before execution. Do not simultaneously change width, learning rate, augmentation, sampling, and pretrained initialization. This schedule is a hypothesis to test, not a promised improvement.
6. **Freeze candidates, then compare across five folds.** Give competing recipes a comparable tuning budget. Fix candidate definitions before additional outer-fold comparisons, use the same frozen patient folds and paired seed bases, and retain every run. Reuse old runs only when the recorded configuration matches. A complete claim about augmentation requires all four cells of the 2 x 2 comparison across folds. Broader repeats should use the same predefined seed set for all finalists, not each model's best seed.

Pretraining, changing spatial downsampling, patient-balanced sampling, and joint multi-slice models remain separate later experiments. Pretraining would require checking course constraints and weight provenance and specifying a grayscale adaptation. It should not be added silently to a from-scratch architecture comparison. None of these extensions is necessary to explain the current results.

## Evaluation commitments

- Keep the sealed patient partitions and duplicate-group boundaries. All transforms remain training-only. Calibration and final-test performance remain unavailable during method development.
- Continue choosing checkpoints by minimum early-stop scan log loss, with the stopping rule recorded in advance. Do not retrospectively select an epoch using outer-validation accuracy.
- For the prospective comparison, nominate scan macro F1 at threshold 0.5 as the primary ranking metric before additional experiments; report accuracy, balanced accuracy, AUROC, both class recalls, log loss, and confusion matrices alongside it. This is a proposed prospective choice, not a claim that historical runs preregistered a primary ranking metric.
- Report paired fold results and mean/sample standard deviation, plus training time, parameters, and consistently measured inference cost. Fold variability is not an independent-patient confidence interval. If uncertainty intervals are computed, resample patients together with all their scans and preserve pairing between methods.
- Do not label the 211 scan predictions as 211 independent patients or average longitudinal patient diagnoses without a predefined clinical target.
- Fold 1 has already informed changes. Development cross-validation is therefore model-development evidence, not a fully independent final estimate. Final refitting, calibration, and one locked test evaluation follow the existing [data protocol](DATA_PROTOCOL.md) after the method is fixed; those execution stages remain to be implemented.

Success can be a well-supported finding that SmallCNN gives the best accuracy/resource tradeoff. A scientifically useful result does not require the larger architecture to win.
