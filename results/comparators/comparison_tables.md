## Table A1. Whole-model comparison: VinDr-Mammo official test set (image-level AUC with 95% CI)

| Model | Family | Screening AUC | Screening AP | Mass AUC | Mass AP | Calcification AUC | Calcification AP |
| --- | --- | --- | ---: | --- | ---: | --- | ---: |
| Baseline (DINOv3-B/16) | Ours, without GRAIN and CUSP | 0.8123 (0.7810–0.8442) | 0.526 | 0.8215 (0.7826–0.8570) | 0.291 | 0.9176 (0.8688–0.9596) | 0.686 |
| ResNet-50 | CNN | 0.7898 (0.7581–0.8216) | 0.491 | 0.7946 (0.7551–0.8333) | 0.359 | 0.9287 (0.8917–0.9616) | 0.686 |
| EfficientNet-B3 | CNN | 0.8088 (0.7749–0.8411) | 0.531 | 0.8110 (0.7700–0.8504) | 0.399 | 0.9347 (0.8838–0.9710) | 0.683 |
| ConvNeXt-T | CNN | 0.7973 (0.7664–0.8281) | 0.473 | 0.8002 (0.7606–0.8386) | 0.339 | 0.9163 (0.8812–0.9509) | 0.518 |
| MobileNetV4-Conv-M | CNN | 0.7848 (0.7522–0.8178) | 0.459 | 0.8110 (0.7689–0.8508) | 0.360 | 0.8792 (0.8238–0.9309) | 0.519 |
| RoPE-ViT-B/16 | Transformer | 0.8253 (0.7937–0.8557) | 0.528 | 0.8369 (0.7970–0.8747) | 0.449 | 0.9364 (0.9058–0.9670) | 0.683 |
| MambaVision-T | Hybrid | 0.8007 (0.7666–0.8327) | 0.503 | 0.8136 (0.7695–0.8522) | 0.371 | 0.9257 (0.8864–0.9631) | 0.691 |
| DINOv2-B/14 | Foundation model | 0.7800 (0.7463–0.8117) | 0.409 | 0.7779 (0.7313–0.8186) | 0.235 | 0.9075 (0.8618–0.9486) | 0.376 |
| RAD-DINO | Medical foundation model | 0.7759 (0.7451–0.8067) | 0.432 | 0.7924 (0.7537–0.8310) | 0.354 | 0.8800 (0.8314–0.9260) | 0.523 |
| Mammo-CLIP (EfficientNet-B2) | Mammography foundation model | 0.8226 (0.7877–0.8558) | 0.526 | 0.8444 (0.8048–0.8809) | 0.419 | 0.9155 (0.8684–0.9576) | 0.629 |
| LSA-SSViT | Ours | 0.8348 (0.8058–0.8632) | 0.548 | 0.8496 (0.8147–0.8825) | 0.464 | 0.9371 (0.8945–0.9726) | 0.723 |

Paired AUC difference, LSA-SSViT minus the comparator (exams resampled; * 95% CI excludes 0):

| Comparator | Screening ΔAUC | Mass ΔAUC | Calcification ΔAUC |
| --- | --- | --- | --- |
| Baseline (DINOv3-B/16) | +0.0225 (+0.0063, +0.0390)\* | +0.0282 (+0.0060, +0.0506)\* | +0.0194 (+0.0009, +0.0401)\* |
| ResNet-50 | +0.0450 (+0.0208, +0.0692)\* | +0.0550 (+0.0244, +0.0869)\* | +0.0084 (−0.0231, +0.0383) |
| EfficientNet-B3 | +0.0260 (+0.0013, +0.0501)\* | +0.0386 (+0.0133, +0.0651)\* | +0.0024 (−0.0208, +0.0241) |
| ConvNeXt-T | +0.0375 (+0.0167, +0.0574)\* | +0.0494 (+0.0279, +0.0722)\* | +0.0208 (−0.0099, +0.0471) |
| MobileNetV4-Conv-M | +0.0500 (+0.0257, +0.0740)\* | +0.0387 (+0.0104, +0.0684)\* | +0.0578 (+0.0195, +0.1016)\* |
| RoPE-ViT-B/16 | +0.0095 (−0.0115, +0.0308) | +0.0127 (−0.0135, +0.0408) | +0.0006 (−0.0268, +0.0213) |
| MambaVision-T | +0.0341 (+0.0113, +0.0582)\* | +0.0361 (+0.0061, +0.0662)\* | +0.0114 (−0.0156, +0.0346) |
| DINOv2-B/14 | +0.0548 (+0.0305, +0.0807)\* | +0.0718 (+0.0399, +0.1077)\* | +0.0295 (−0.0035, +0.0646) |
| RAD-DINO | +0.0589 (+0.0348, +0.0837)\* | +0.0572 (+0.0283, +0.0866)\* | +0.0570 (+0.0211, +0.0980)\* |
| Mammo-CLIP (EfficientNet-B2) | +0.0122 (−0.0077, +0.0330) | +0.0052 (−0.0181, +0.0323) | +0.0215 (−0.0028, +0.0480) |

Training of the comparators (checkpoint with the best validation screening AUC):

- ResNet-50: best epoch 16, maximum number of epochs reached
- EfficientNet-B3: best epoch 8, early stopping: validation screening AUC did not improve for 6 epochs
- ConvNeXt-T: best epoch 12, early stopping: validation screening AUC did not improve for 6 epochs
- MobileNetV4-Conv-M: best epoch 19, maximum number of epochs reached
- RoPE-ViT-B/16: best epoch 13, early stopping: validation screening AUC did not improve for 6 epochs
- MambaVision-T: best epoch 12, early stopping: validation screening AUC did not improve for 6 epochs
- DINOv2-B/14: best epoch 14, early stopping: validation screening AUC did not improve for 6 epochs
- RAD-DINO: best epoch 11, early stopping: validation screening AUC did not improve for 6 epochs
- Mammo-CLIP (EfficientNet-B2): best epoch 3, early stopping: validation screening AUC did not improve for 6 epochs
