# ConvNeXt feature and position diagnostic plan

Status: updated on 2026-10-01. The first A/B/D diagnostic extension is now implemented; see [follow-up usage](FEATURE_DIAGNOSTIC_FOLLOWUP.md). Real follow-up execution is pending. This plan specifies the remaining controls and launches no backbone training. Its objective is to distinguish sampling sensitivity, boundary effects, interpolation, probe overfitting, training dynamics and spatial relationship modeling before choosing the next architecture. It updates the investigation order in the earlier [experiment plan](CONVNEXT_EXPERIMENT_PLAN.md) using the completed real diagnostics.

## Evidence and limits

The three existing fold 1 checkpoints were inspected on the same 64 training patients with 99 complete scans and 38 early-stop patients with 66 complete scans. The frozen full roles contain 343 training patients with 745 scans and 38 early-stop patients with 94 scans. Each scan contains 20 slices. Early-stop patients already informed checkpoint selection, so these investigations remain development analyses.

Four-direction mean slice decision changes after exact two-pixel shifts were 4.96% for SmallCNN, 21.78% for ConvNeXt without augmentation and 9.85% for ConvNeXt with light augmentation. Corresponding mean absolute probability changes were 0.03518, 0.19442 and 0.07000. Actual ConvNeXt residual contributions were nonzero. The same 12 training-image transform samples across the three audits showed no detected foreground clipping or unexpected resizing. These observations do not identify a responsible layer, establish that interpolation is harmless across the dataset or prove preservation of classification-relevant information.

Source evidence: [raw diagnostic outputs](../outputs/server_imports/features_20261001_131402/diagnostics_20261001_130255/), [verified analysis](../outputs/review/feature_diagnosis_20261001/summary.json) and [independent verification](../outputs/review/feature_diagnosis_20261001/verification.json). These ignored local artifacts are not intended for Git submission.

## Shared protocol

Keep the existing checkpoints, source images and sealed patient assignments unchanged. Diagnostic inference may use only fold 1 train and early_stop. Integrity verification may read all manifest roles, but it must not construct or score outer-validation, calibration or final-test loaders. Freeze each identity list before examining predictions and reuse it across models.

For the first matched shift comparison, preserve the existing 66 early-stop scans and report every included identity. For broader feature probes, preserve this evaluation list initially; expanding it to all 94 scans is a separately labeled coverage check. Training cohorts must also be identical across compared models. Patients with mixed longitudinal labels remain eligible for scan diagnostics but are excluded from patient-mean probes using their full-role labels.

Use evaluation mode and no backbone gradients for existing-checkpoint tests. Record checkpoint hashes before and after execution, exact source fingerprints, environment, selected identities, diagnostic settings and timings. Use a fresh output directory. Export per-slice observations and summarize them per scan and per patient before comparison; retain the historical pooled slice summaries separately. Uncertainty calculations resample patients with all their scans and paired model outputs, not individual slices.

Probability threshold remains 0.5. Report logit changes and absolute probability changes alongside decision flips. Predeclare confidence strata by original absolute distance from 0.5: [0, 0.1), [0.1, 0.3), and [0.3, 0.5]. This checks whether a large flip rate mainly reflects predictions near the threshold; it does not remove every confidence-related confound.

The diagnostic sampling and probe seed is 3710. Historical fold 1 training used seed base 3710 and actual seed 3711 because training adds the fold index. The original `diagnose.py` retains its native-dimension probes and one shift magnitude. The new `diagnose_followup.py` implements shifted intermediate-feature recording, boundary controls and common-dimension PCA probes. Attention variants and new backbone training controls below remain proposals, not available CLI options.

## Investigation routes

| Route | Question | Initial method | Backbone training | Priority |
|---|---|---|---|---|
| A Sampling phase | Do strided layers amplify small translations? | Integer shifts and intermediate features | No | First |
| B Boundaries | Does clipping or finite padding explain the response? | Shift round trips and margin controls | No | With A |
| C Interpolation | Does resampling change usable features? | Matched geometry and expanded training-image audit | No initially | After A and B |
| D Probe reliability | Are weak late probes an artifact of small sample size and high dimension? | More training patients and matched readout dimension | Only fresh linear heads | With A |
| E Training dynamics | Does the optimization recipe contribute to unstable features? | Historical curves, then one optimizer control if needed | Only if initial diagnostics justify it | Conditional |
| F Spatial relationships and RoPE | Would explicit spatial interactions improve the representation? | Attention with and without 2D RoPE | Yes | After cheaper controls |

## Route A Sampling phase

Run exact integer translations of 1 through 8 pixels in each of four axial directions, plus one unchanged baseline. Use black fill at the image-input level, equivalent to -1 after current normalization, with no interpolation. This requires 33 forward passes per slice per checkpoint, compared with five passes for the earlier original-plus-two-pixel diagnostic. Measure a small fixed pilot first to estimate memory and wall time; no runtime estimate is assumed.

Record the stem output, each downsampler output, each complete stage output and final logits. Save online feature summaries rather than all dense maps. Plot patient-averaged probability and logit changes against displacement. For spatial feature alignment, compare only shifts divisible by that layer's effective stride: 4 at the stem/stage 1 and 8 at stage 2 are available in this sweep. Translate the reference feature map by the corresponding integer cell offset and compare the valid common region. Do not interpolate feature maps to manufacture exact equivariance for other displacements. For those shifts and later stages, report GAP-embedding changes and unaligned summaries with explicit limitations.

A repeated modulo-four response that appears at the stem would support involvement of sampling phase. A marked increase immediately after another downsampler would identify an additional candidate. Four-pixel input shifts need not yield stable final predictions because subsequent stride-2 operations and boundaries remain. Absence of a clean periodic pattern weakens a simple stem-only explanation without excluding more complex sampling effects. Any common-interior measurement must report how much spatial support remains; late stages may have no region free from boundary influence.

If A implicates the stem, a later scratch-trained control may compare standard 4x4/stride4 with overlapping 4x4/stride2/padding1, retaining Tiny widths and depths. Alternatively test a separately specified antialias design. Do not run both changes together. The overlapping variant changes output stride, overlap, effective receptive fields and compute; an improvement would not isolate one of these properties. Antialias filtering can also suppress useful variation and must be assessed for classification as well as stability.

## Route B Boundaries

For each displacement in A, compute the exact round trip T(-d)T(d)x. Integer shifts introduce no interpolation, so differences from x identify pixels lost at the canvas edge. Compare original, translated and round-trip model outputs. Round-trip prediction changes provide a boundary-loss control, although they do not reproduce every effect of translating content through the network.

Measure input foreground-proxy margins at fixed gray thresholds 8, 16 and 32. Report results for all images and for a prespecified margin-greater-than-8 subgroup, with its patient and class counts. This is an intensity proxy, not an anatomical segmentation. The subgroup and round-trip controls must be identical across models. If translated predictions change strongly when round trips preserve pixels and predictions, direct image clipping becomes a weaker explanation. Internal convolution padding can still contribute even when no image content is clipped.

Do not silently replace zero padding or pad images into a different resolution when testing an existing checkpoint: these change its evaluated function. A later padding experiment needs a named variant and scratch-trained matched control. Routes A and B together narrow the mechanism; neither gives a unique causal attribution by itself.

## Route C Interpolation

Expand the training-only image audit to one complete, deterministically selected scan per training patient and one predefined middle slice per selected scan. Keep the eight random and four extreme geometry draws per image. Also report the lowest-margin training cases as a separately identified stress group rather than treating them as random samples. Inspect canvas loss, padded-reference differences and technical image grids without clinical interpretation.

For the previously selected 64 training patients, use one deterministically selected complete scan and its predefined middle slice per patient. On this fixed subset, isolate translation-only cases at 0.5, 1.5 and 2 pixels horizontally and vertically. Compare bilinear and bicubic outputs at identical geometry; compare the two-pixel case with the exact integer operation as an implementation check. Record pixel differences, gradient-energy changes, stage-embedding changes, original classifier outputs and class-conditional summaries. Gradient energy measures texture variation, not disease information. Applying alternative transforms to frozen checkpoints is a sensitivity test, not a fair estimate of retraining benefit.

If interpolation choices have a consistent consequential effect, a future bilinear-versus-bicubic training comparison must replay the same sampled angles/translations and alter only the interpolation kernel. Keep rotation bounds, translation bounds, canvas, fill and input size unchanged. Integer-only augmentation is a separate geometry experiment and must not be labeled a pure interpolation ablation. If matched transforms have similar model responses and the expanded audit finds little clipping, move this route below sampling and representation controls.

## Route D Probe reliability

First reproduce the 64-patient probe on the exact existing 99 training scans and 66 early-stop scans. Then expand fitting to all eligible training patients while retaining a maximum of two complete scans per patient and the same frozen early-stop evaluation list. This keeps the scan-cap policy aligned with the initial diagnostic. Report actual eligible counts and labels; do not assume that every patient has a consistent longitudinal label.

Fit native-dimension probes and one prespecified common-dimension control: 16 PCA components at every stage, using only training-patient embeddings. Centering, standardization and PCA fitting are training-only; no whitening or component-count search is planned. Sixteen matches the smallest baseline stage width and is below the rank limit of the initial training cohort. Refit training-only preprocessing for each declared training cohort and record it. PCA can itself discard discriminative directions, so a poor PCA probe does not prove that the source stage lacks information.

Keep the existing fixed probe settings: 200 full-batch AdamW steps, learning rate 0.01, weight decay 0.01 on weights and zero on bias, seed 3710 and threshold 0.5. Class weights use training-patient counts only. Each patient contributes one mean embedding; retain complete scans. Report train and early-stop AUROC, log loss, accuracy and class recall. No early-stop head selection or threshold fitting is allowed. A subsequent all-scan coverage check may use the full training scans and all 94 early-stop scans, but must be reported separately from this matched test.

If late-stage performance improves materially with more training patients or the 16-component control, readout overfitting becomes a stronger explanation for the earlier poor probes. If native and common-dimension probes remain weak while early-stage probes retain better separation, investigate representation learning and spatial processing further. Linear separability of GAP features does not establish that a layer preserved or destroyed all information, and these already-used early-stop patients cannot provide an independent final estimate.

## Route E Training dynamics

Review existing histories using comparable objectives: deterministic, unaugmented inference on training and early-stop scans, with the same scan-level log loss. Do not subtract the class-weighted augmented slice training loss from unweighted scan evaluation loss. Existing final-checkpoint branch activity does not show how quickly branches learned or identify a defective training phase.

If this route remains relevant, run one paired control that changes only AdamW parameter grouping: current all-parameter decay versus no decay on one-dimensional parameters and biases. Keep the published project recipe otherwise fixed. Measure group definitions, learning rate, sampled gradient/update norms, LayerScale evolution, train and early-stop predictions, selected epoch and shift stability. This grouping control affects LayerScale, normalization and biases together; it is not a LayerScale-only test.

Do not simultaneously add warmup, cosine scheduling, new widths or new augmentation. A schedule comparison would need a later separate prescription and matched training budget. Better training fit alone is not evidence of better generalization. If original-model generalization and robustness remain weak after probe and sampling checks, a width reduction can be evaluated separately; it does not directly test the positional-encoding hypothesis.

## Route F Spatial relationships and RoPE

The current backbone already has offset-specific local convolution weights and a two-dimensional feature grid; it has no attention Q/K. Missing an explicit position code is therefore not established as a defect. Conventional RoPE requires an attention-style operation, and it cannot reconstruct detail removed upstream.

If routes A through E leave spatial relationship modeling as a useful candidate, prescribe a small bottleneck self-attention head on the final 7x8 feature grid before pooling. For an initial controlled proposal, project 768 channels to 128, use four heads, one pre-normalized residual attention block and zero attention dropout. Apply the same spatial mean, final normalization and one-logit classifier after this block; add no feed-forward block. Keep these choices identical in the following three scratch-trained arms:

- R0: current ConvNeXt and GAP head.
- R1: ConvNeXt with the bottleneck attention head and no explicit positional encoding.
- R2: the identical attention head with fixed axial 2D RoPE applied to Q/K using row and column coordinates; V is unchanged. Each head has 32 dimensions, split equally between row and column rotations. Use base 10000 and, in each 16-dimensional axis group, frequencies 10000^(-2j/16) for j from 0 through 7. Frequencies are fixed, not learned; do not add absolute position embeddings.

R1 versus R2 isolates the position-encoding addition within the same attention architecture. R0 versus R1 measures a larger head/interaction change and is confounded by additional parameters and computation. R2 improving on R1 would support a benefit from RoPE in this head, not prove that missing position encoding caused the original shift sensitivity. A final-grid head cannot recover details already absent from the grid. If low resolution appears decisive, test a higher-resolution head in a later separate experiment, not simultaneously.

Twenty-slice probability averaging also ignores slice order. Attention with slice-index encoding would test a separate cross-slice question; do not combine that change with the within-slice 2D RoPE comparison.

## Gates for new training

Complete and review A, B and D first. Run C when the initial evidence or expanded margins justify it. Record a short decision for every route: supported, weakened or unresolved, with the relevant outputs and alternatives. An inconclusive result should remain inconclusive rather than trigger every possible architecture search.

Before any new backbone training, implement and verify an inner-only path that never constructs or scores an outer-validation loader. The current training command always evaluates outer validation after selection; it is not suitable unchanged for these tuning experiments. Freeze the candidate definition, output name, initialization, seed base, augmentation, optimizer, budget, selection rule and metrics before running it. Give every architecture a distinct name and leave existing checkpoint meanings unchanged.

For a first architecture/optimizer comparison, use a fresh matching Tiny/light control and one candidate, fold 1, seed base 3710 (actual seed 3711), learning rate 1e-4, weight decay 0.05, batch size 16, maximum 30 epochs, patience 5 and min_delta 1e-4. Preserve the constant learning rate, scan-level checkpoint selection and recorded light profile (5 degrees and 0.03 translation, bilinear) unless that factor is the declared intervention. Historical checkpoints remain diagnostic references rather than automatically serving as a prospective controlled training arm. Use an augmentation RNG independent of model initialization so changes in architecture do not silently change geometry draws.

Primary development selection remains minimum early-stop scan log loss. At the selected checkpoint, report scan AUROC, macro F1, balanced accuracy, accuracy, both class recalls and patient-clustered shift summaries, plus parameters, runtime and peak memory. Improved stability with worse classification is a tradeoff, not an automatic success. A single-seed improvement is a pilot signal only. Any finalist comparison should use the same additional seed bases 4710 and 5710 for both arms (actual fold 1 seeds 4711 and 5711), then freeze the method before broader development-fold evaluation. No training time or gain is promised by this plan.

## Deliverables and review

The first execution deliverable is an unchanged-checkpoint report with shift curves, layer summaries, boundary controls and native/common-dimension probe comparisons. It must include exact cohorts and all planned runs, including unfavorable results. Source hashes, checkpoint hashes and patient-pairing checks must pass. Synthetic checks should test feature alignment and boundary controls before real-data execution; synthetic outcomes are not ADNI evidence.

The later training deliverable, if warranted and authorized, is one named intervention with a matched fresh control and the prescribed inner-only evaluation. Ken reviews each completed extension before another implementation milestone. This document itself launches no experiment, modifies no source code or frozen data and makes no commit or PR.

## References

Sampling and antialiasing rationale: [Zhang, Making Convolutional Networks Shift-Invariant Again](https://proceedings.mlr.press/v97/zhang19a.html). The published improvements do not establish benefit on this dataset.

Architecture and position mechanisms: [Liu et al., A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545), [Su et al., RoFormer](https://arxiv.org/abs/2104.09864), and [Heo et al., Rotary Position Embedding for Vision Transformer](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01584.pdf). These papers motivate controls, not a claim that RoPE fixes this project's trained ConvNeXt.
