## Table 1. Datasets (VinDr-Mammo)

| Split | Exams | Images | Screening + | Mass + | Calcification + |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 3,200 | 12,800 | 1,232 | 716 | 270 |
| Validation | 400 | 1,600 | 150 | 88 | 33 |
| Held-out | 400 | 1,600 | 152 | 90 | 34 |
| Official test | 1,000 | 4,000 | 384 | 219 | 105 |

Screening-positive test images by BI-RADS 3/4/5: 186/152/46.

## Table 2. Image-level results on the official VinDr-Mammo test set (AUC with 95% CI)

| Model | Screening AUC | Screening AP | Mass AUC | Mass AP | Calcification AUC | Calcification AP |
| --- | --- | ---: | --- | ---: | --- | ---: |
| Baseline | 0.8123 (0.7810–0.8442) | 0.526 | 0.8215 (0.7826–0.8570) | 0.291 | 0.9176 (0.8688–0.9596) | 0.686 |
| +GRAIN | 0.8331 (0.8033–0.8624) | 0.552 | 0.8442 (0.8091–0.8788) | 0.448 | 0.9272 (0.8796–0.9672) | 0.705 |
| +CUSP | 0.8286 (0.7987–0.8583) | 0.542 | 0.8332 (0.7961–0.8677) | 0.347 | 0.9163 (0.8692–0.9577) | 0.642 |
| LSA-SSViT | 0.8348 (0.8058–0.8632) | 0.548 | 0.8496 (0.8147–0.8825) | 0.464 | 0.9371 (0.8945–0.9726) | 0.723 |

## Table 3. Paired differences in image-level AUC (* 95% CI excludes 0)

| Comparison | Screening ΔAUC | Mass ΔAUC | Calcification ΔAUC |
| --- | --- | --- | --- |
| +GRAIN vs Baseline | +0.0208 (+0.0025, +0.0377)\* | +0.0228 (+0.0019, +0.0444)\* | +0.0096 (−0.0062, +0.0286) |
| +CUSP vs Baseline | +0.0163 (+0.0032, +0.0299)\* | +0.0117 (−0.0044, +0.0286) | −0.0013 (−0.0197, +0.0176) |
| LSA-SSViT vs Baseline | +0.0225 (+0.0063, +0.0390)\* | +0.0282 (+0.0060, +0.0506)\* | +0.0194 (+0.0009, +0.0401)\* |

## Table 4. Operating points (thresholds selected on the validation set with Youden's index)

| Task | Model | Sensitivity / specificity | Sensitivity at 90% specificity | Specificity at 90% sensitivity |
| --- | --- | --- | ---: | ---: |
| Screening | Baseline | 63.0% / 83.4% | 56.0% | 39.1% |
| Screening | +GRAIN | 56.0% / 90.7% | 57.8% | 45.3% |
| Screening | +CUSP | 56.8% / 90.5% | 57.3% | 45.3% |
| Screening | LSA-SSViT | 65.9% / 83.8% | 58.9% | 49.6% |
| Mass | Baseline | 60.3% / 87.9% | 57.1% | 45.6% |
| Mass | +GRAIN | 53.0% / 93.8% | 58.9% | 48.1% |
| Mass | +CUSP | 64.8% / 86.3% | 59.4% | 50.8% |
| Mass | LSA-SSViT | 61.2% / 90.3% | 61.6% | 49.6% |
| Calcification | Baseline | 80.0% / 92.6% | 82.9% | 70.4% |
| Calcification | +GRAIN | 75.2% / 97.4% | 83.8% | 69.0% |
| Calcification | +CUSP | 68.6% / 97.2% | 79.0% | 66.6% |
| Calcification | LSA-SSViT | 80.0% / 95.7% | 84.8% | 73.6% |

## Table 5. Exam-level AUC (maximum score over the views of an exam; 95% CI)

Positive exams: screening 187, mass 115, calcification 54.

| Model | Screening AUC | Mass AUC | Calcification AUC |
| --- | --- | --- | --- |
| Baseline | 0.8190 (0.7814–0.8538) | 0.7933 (0.7456–0.8395) | 0.9292 (0.8871–0.9642) |
| +GRAIN | 0.8352 (0.8000–0.8674) | 0.8190 (0.7728–0.8631) | 0.9351 (0.8928–0.9706) |
| +CUSP | 0.8394 (0.8051–0.8705) | 0.8149 (0.7714–0.8550) | 0.9301 (0.8910–0.9632) |
| LSA-SSViT | 0.8365 (0.8045–0.8684) | 0.8355 (0.7922–0.8763) | 0.9219 (0.8709–0.9656) |

## Table 6. External test on INbreast without fine-tuning

| Model | Screening AUC (95% CI) | Screening AP | Mass AUC | Mass AP | Calcification AUC | Calcification AP |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.8174 (0.7484–0.8834) | 0.698 | 0.7577 | 0.512 | 0.6988 | 0.863 |
| +GRAIN | 0.8628 (0.7982–0.9219) | 0.765 | 0.8735 | 0.797 | 0.7070 | 0.871 |
| +CUSP | 0.8409 (0.7740–0.9039) | 0.712 | 0.7940 | 0.601 | 0.6825 | 0.854 |
| LSA-SSViT | 0.8605 (0.7970–0.9179) | 0.757 | 0.8757 | 0.813 | 0.7078 | 0.874 |

Paired difference in screening AUC versus the baseline (patients resampled):

- +GRAIN: +0.0454 (+0.0169, +0.0780) *
- +CUSP: +0.0235 (+0.0034, +0.0432) *
- LSA-SSViT: +0.0431 (+0.0185, +0.0698) *
