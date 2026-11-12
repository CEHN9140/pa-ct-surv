# CT representation audit after test15 QKNorm

This audit reads data, code and saved checkpoints only. No new training, BN updates or checkpoint changes were performed.

The current evidence supports weak use of patient-specific CT in the teacher, rather than a demonstrated CT extraction bug. QKNorm makes the predictor query-sensitive, but query sensitivity alone does not establish useful pathology-to-CT correspondence.

## Fixed-training-CT control

Each fold uses three predetermined random training anchors separately; in each inference pass, all validation patients share one anchor CT while retaining their own pathology. No anchor is selected by outcome/performance. Normal five-fold mean C-index is 0.752375; the shared-CT mean is 0.744399 (drop 0.007976). The model retains most ranking performance even with no between-patient CT variation within each pass. This supports a pathology-dominated query-conditioned computation that can operate on common CT feature patterns. It does not establish that original CT has no complementary DFS information. This is an intervention on a fixed predictor, not a retrained clinical model.

## Checked implementation and data

- ROI64 cohort: all 721 images have finite float32-compatible [1,64,64,64] values in [0,1]. Every cohort case's preprocessing record reports full binary-mask tumor retention. Median tumor-mask fraction of ROI is 2.96%; surrounding tissue is not assumed irrelevant.
- CT/PA CSV pairing, labels and saved splits agree between data/eagle/top25 and data/trident/features_uni_v1 for ROI64.
- Preprocessing: LPS orientation, mask/image geometry alignment, 1-mm spacing, linear CT interpolation / nearest mask interpolation, full-mask bounding-box centered 64-mm crop; HU [-1000,400] clipped and linearly scaled to [0,1]. No obvious crop/loader logic defect was found in this review. Raw-scan annotation correctness and biological labels were not re-established here.
- ResNet18 basic-block counts, shortcut A, dilation2/4 and feature extraction through layer4 agree with the official MedicalNet encoder. Backbone loading rejects missing backbone or unexpected keys; public weights initialize the model, and saved convolution weights demonstrably differ after joint training. Trainable CT parameters are not frozen.
- All 34 BN running mean/variance buffers remain identical to public pretraining in every QKNorm fold; this follows freeze_bn_stats=True, while affine parameters and convolutions remain trainable.
- Test15 extracts layer4 [B,512,8,8,8], correctly flattens spatial dimensions into [B,512 positions,512 channels], and projects to 256-D K/V. No pre-KV global pooling or channel/spatial transpose defect was found. The diagnostic CT risk head is outside the fusion path and has lambda_ct=0; its logged auxiliary Cox loss is not a trained-unimodal CT quality measure.

## Representation suitability concerns, not established bugs

1. Spatial-token support: this encoder has theoretical maximal receptive field 483 voxels with output stride 8 at layer4, much larger than the 64-voxel ROI. The 512 tokens are overlapping deep contexts, not 512 independent 8-mm crops. A layer4 3x3 dilation4 kernel on an 8x8x8 map samples only 8/27 non-padding kernel sites per output position. This is the official architecture, not a coding error; its suitability for local cross-modal retrieval is unproven. Original teacher diagnostics and the new probe show spatial variation rather than all tokens being identical.
2. Input/BN adaptation: current HU-window [0,1] input differs from Med3D paper section3.1 percentile clipping and per-volume mean/std normalization. The 23-dataset checkpoint's entire original preprocessing recipe is not established by the older 8-dataset paper or MRBrainS example. In a 32-case probe, first-BN per-volume observed SD/stored running SD is about 0.09, including public, unimodal and teacher models. This supports a distribution-adaptation question, not proof that current windowing or fixed BN caused poor DFS performance. No global feature collapse/complete inactivity was observed.
3. Learning asymmetry: pathology uses fixed foundation features and EAGLE selection, while CT must adapt a public encoder to DFS from patient-level supervision. Fusion-only Cox need not learn a uniquely CT-dependent predictor. This is consistent with the shared-CT control; it does not show which signal is absent from raw images.

## Existing CT controls

Matched ROI64/v7 CT without pooled-feature L2 has Train C-index 0.7340±0.0141, Val 0.7157±0.0508; pooled-feature L2 gives Train 0.7143±0.0181, Val 0.7096±0.0455. This shows nontrivial CT prediction, without evidence that merely matching L2 scales improves performance. Different ROI comparisons use different cohorts (721/789/799), so their averages do not isolate ROI size alone.

## Priority

The most actionable representation-adaptation question is CT intensity preprocessing together with fixed pretraining BN statistics; it needs a controlled test before changing either. The next question is whether these wide-context layer4 tokens are suitable for pathology-conditioned regional retrieval. Neither issue is established as the unique cause. There is no current evidence that MedicalNet is categorically invalid because it was pretrained with segmentation, or that raw CT carries no additional information. Do not change preprocessing, BN behavior, token layers and losses simultaneously.

Official sources:
- Encoder / shortcut specification: https://github.com/Tencent/MedicalNet and https://github.com/Tencent/MedicalNet/blob/master/models/resnet.py
- Med3D paper (section3.1 intensity normalization; section3.3 pulmonary-nodule classification transfer): https://arxiv.org/html/1904.00625
- Official MRBrainS transfer example (not a universal CT specification): https://github.com/Tencent/MedicalNet/blob/master/datasets/brains18.py
