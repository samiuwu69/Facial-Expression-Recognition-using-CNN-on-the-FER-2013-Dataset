# Facial Expression Recognition on FER-2013: Complete Project Summary

**Course:** CSE366 (Artificial Intelligence)  
**Group:** 01  
**Date:** 2026-11-13

| Name | ID |
|---|---|
| Rafid Rahman Sami | 2023-1-60-116 |
| Abdullah Al Sami | 2023-2-60-147 |
| Sumya Kawsar | 2023-3-60-168 |

---

## Executive Summary

This project trains and compares multiple CNN architectures for facial expression recognition on the FER-2013 dataset (7 emotions: angry, disgust, fear, happy, neutral, sad, surprise). We follow a three-stage pipeline:

1. **Submission 1 (EDA):** Explore the dataset, identify data quality issues, and review related work.
2. **Submission 2 (Pretrained):** Fine-tune three pretrained models (VGG16, ResNet50, MobileNetV2) with and without unfreezing the backbone.
3. **Submission 3 (Custom CNN):** Train a custom CNN from scratch and compare against pretrained baselines.

**Best Result:** ResNet50 (fine-tuned) achieved **64.6% accuracy** and **0.638 macro-F1** on the official test set.

---

## 1. Dataset Overview (Submission 1)

### Dataset Statistics
| Item | Details |
|---|---|
| **Name** | FER-2013 (Facial Expression Recognition 2013) |
| **Source** | Kaggle (Goodfellow et al., 2013) |
| **Images** | 35,887 total (train: 28,709, test: 7,178) |
| **Format** | 48×48 grayscale JPEG images |
| **Classes** | 7 emotions: angry, disgust, fear, happy, neutral, sad, surprise |

### Key Findings from EDA

#### 1. **Severe Class Imbalance**
| Class | Train | Test | Weight |
|---|---|---|---|
| Angry | 3,995 (13.9%) | 958 | 1.02 |
| Disgust | 436 (1.5%) | 111 | 9.40 |
| Fear | 4,097 (14.3%) | 1,024 | 1.00 |
| Happy | 7,215 (25.1%) | 1,774 | 0.57 |
| Neutral | 4,965 (17.3%) | 1,233 | 0.83 |
| Sad | 4,830 (16.8%) | 1,247 | 0.85 |
| Surprise | 3,171 (11.0%) | 831 | 1.29 |

**Imbalance ratio:** 16.5:1 (happy vs disgust). A model always predicting "happy" achieves 24.7% accuracy → **must use class weights and per-class metrics**.

#### 2. **Data Quality Issues**
- **1,236 duplicate images** (extra copies of identical faces)
- **39 groups with conflicting labels** (same image labeled with different emotions)
- **568 test images leaked into training** (7.9% of test set; creates data leakage)
- **13 low-contrast images** (almost blank; likely corrupted)
- **Human ceiling:** ~65% accuracy (Goodfellow et al., 2013)

#### 3. **Data Characteristics**
- Pixel range: 0–255, mean ≈ 107, std ≈ 56
- Images vary in: age, pose, lighting, occlusion
- Similar emotions visually: fear/sad/surprise, angry/disgust/sad
- All images are 48×48 pixels (low resolution)

---

## 2. Data Preparation (Submissions 2 & 3)

### Cleaning Steps
1. **Remove duplicates:** Keep only the first copy of identical images.
2. **Remove conflicting labels:** Drop all copies of images with multiple labels.
3. **Stratified split:** 85% train / 15% validation (maintains class distribution).
4. **After cleaning:** 27,434 training images (removed 1,275 problematic images)

### Final Split
| Set | Count |
|---|---|
| Training | 23,318 images (after 85/15 split) |
| Validation | 4,116 images |
| Test | 7,178 images (official, unchanged) |
| Test leak-free subset | 6,610 images (excluding the 568 leaked) |

### Preprocessing
- **Pixel scaling:** 0–255 → 0–1 (or model-specific normalization)
- **Class weights:** Inverse of class frequency (disgust: 10.3×, happy: 0.6×)
- **Augmentation** (training only): horizontal flip, rotation ±8–15°, shift ±10–15%, zoom ±10–15%
- **Label smoothing:** 0.10–0.15 (during training)

---

## 3. Submission 2: Pretrained Models with Fine-Tuning

### Methodology

**Models Tested:**
| Model | Backbone Params | Why Chosen |
|---|---|---|
| **VGG16** | 14.7 M | Best single-network result in literature (73.28%) |
| **ResNet50** | 23.6 M | Skip connections enable deep networks; part of SOTA ensembles |
| **MobileNetV2** | 2.3 M | Lightweight for real-time demo applications |

**Modes:**
1. **Frozen backbone:** Only train the new classifier head (5 epochs equivalent).
2. **Fine-tuning:** Warm-up head for 5 epochs, then unfreeze entire backbone for 25 epochs at lower learning rate.

**Hyperparameters:**
- Batch size: 64
- Head learning rate: 1e-3
- Fine-tune learning rate: VGG16 (1e-5), ResNet50 (5e-5), MobileNetV2 (1e-4)
- Epochs: 30 per run
- Regularization levels: 1 (main), 2 (if overfitting detected)

### Results Summary

#### Test Accuracy & Macro-F1
| Model | Mode | Test Acc | Leak-Free | Macro-F1 | Train Time |
|---|---|---|---|---|---|
| VGG16 | Frozen | 0.405 | 0.402 | 0.339 | 11.6 min |
| VGG16 | Fine-tuned | 0.617 | 0.606 | 0.598 | 30.4 min |
| **ResNet50** | Frozen | 0.462 | 0.449 | 0.405 | 12.7 min |
| **ResNet50** | **Fine-tuned** | **0.646** | **0.630** | **0.638** | **43.4 min** |
| MobileNetV2 | Frozen | 0.395 | 0.389 | 0.338 | 5.5 min |
| MobileNetV2 | Fine-tuned | 0.608 | 0.595 | 0.578 | 22.0 min |

#### Ensemble (Average of 3 Fine-Tuned Models)
- **Test Accuracy:** 0.664
- **Macro-F1:** 0.656
- **Leak-free Accuracy:** 0.648

### Key Insights

1. **Fine-tuning is crucial:** Frozen backbone vs fine-tuned gains +18–21 percentage points.
   - VGG16: 0.405 → 0.617 (+21.2 pp)
   - ResNet50: 0.462 → 0.646 (+18.4 pp)
   - MobileNetV2: 0.395 → 0.608 (+21.4 pp)

2. **ResNet50 is best:** Validation macro-F1 = 0.599 (best of 6 runs).

3. **MobileNetV2 is fastest:** 1/10 the parameters, 30 min training time vs 40+ min.

4. **Hardest emotions:** Fear (F1 0.46) confused with sad/neutral.  
   **Easiest emotion:** Happy (F1 0.86).

5. **Data leakage impact:** Official test accuracy 0.646 vs leak-free 0.630 (−1.6 pp).

### Inference Time (NEW in Submission 2)

| Model | Mode | Total (s) | Per Image (ms) | Speed |
|---|---|---|---|---|
| VGG16 | Frozen | — | — | — |
| VGG16 | Fine-tuned | — | — | — |
| ResNet50 | Frozen | — | — | — |
| ResNet50 | Fine-tuned | — | — | — |
| MobileNetV2 | Frozen | — | — | — |
| MobileNetV2 | Fine-tuned | — | — | — |

*Timing to be measured when Submission 2 is re-run with the new H section.*

---

## 4. Submission 3: Custom CNN from Scratch

### Architecture

| Stage | Layers | Output |
|---|---|---|
| Input | 48×48×1 | 48×48×1 |
| Augmentation | Flip, rotation, shift, zoom | 48×48×1 |
| Rescaling | 1/255 | 48×48×1 |
| **Block 1** | Conv(32) → BN → ReLU ×2, MaxPool(2), Dropout(0.25) | 24×24×32 |
| **Block 2** | Conv(64) → BN → ReLU ×2, MaxPool(2), Dropout(0.25) | 12×12×64 |
| **Block 3** | Conv(128) → BN → ReLU ×2, MaxPool(2), Dropout(0.25) | 6×6×128 |
| **Block 4** | Conv(256) → BN → ReLU ×2, MaxPool(2), Dropout(0.25) | 3×3×256 |
| Classifier | Flatten, Dense(256) → BN → ReLU → Dropout(0.5), Dense(7) softmax | 7 |

**Total Parameters:** ~1.77 million (no pretrained weights).

### Training Setup

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam, learning rate 1e-3 |
| Batch size | 64 |
| Loss | Categorical cross-entropy (label smoothing 0.10) |
| Max epochs | 60 |
| Early stopping | Patience 10, minimum 5 epochs |
| Learning rate schedule | ReduceLROnPlateau (×0.5, patience 4) |
| Best model | Lowest validation loss |

### Results

| Metric | Value |
|---|---|
| **Test Accuracy** | 0.586 |
| **Leak-free Accuracy** | 0.572 |
| **Macro Precision** | 0.530 |
| **Macro Recall** | 0.588 |
| **Macro-F1** | 0.541 |
| **Weighted-F1** | 0.586 |
| **Training Time** | 127.0 min |
| **Epochs Run** | 60 (max) |
| **Best Epoch** | 51 |
| **Diagnosis** | Good fit (gap 0.019) |

### Inference Time

| Metric | Value |
|---|---|
| **Total test time (7,178 images)** | 8.49 seconds |
| **Per-image inference time** | 1.183 ms |
| **Images per second** | 845 |
| **Hardware** | CPU |

### Per-Class Performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Angry | 0.584 | 0.578 | 0.581 |
| Disgust | 0.500 | 0.378 | 0.431 |
| Fear | 0.436 | 0.618 | 0.512 |
| Happy | 0.748 | 0.750 | 0.749 |
| Neutral | 0.534 | 0.531 | 0.533 |
| Sad | 0.420 | 0.426 | 0.423 |
| Surprise | 0.576 | 0.557 | 0.566 |

---

## 5. Full Comparison: Submission 2 vs Submission 3

### Test-Set Performance

| Model | Type | Mode | Test Acc | Macro-F1 | Train (min) | Test (ms/img) |
|---|---|---|---|---|---|---|
| VGG16 | Pretrained | Frozen | 0.405 | 0.339 | 11.6 | — |
| VGG16 | Pretrained | Fine-tuned | 0.617 | 0.598 | 30.4 | — |
| ResNet50 | Pretrained | Frozen | 0.462 | 0.405 | 12.7 | — |
| **ResNet50** | **Pretrained** | **Fine-tuned** | **0.646** | **0.638** | **43.4** | **—** |
| MobileNetV2 | Pretrained | Frozen | 0.395 | 0.338 | 5.5 | — |
| MobileNetV2 | Pretrained | Fine-tuned | 0.608 | 0.578 | 22.0 | — |
| Ensemble (S2) | Pretrained | 3×FT avg | 0.664 | 0.656 | — | — |
| **Custom CNN** | **Scratch** | N/A | **0.586** | **0.541** | **127.0** | **1.183** |

### Key Comparisons

1. **Best Pretrained vs Custom CNN:**
   - ResNet50 (fine-tuned): 64.6% accuracy, 0.638 F1
   - Custom CNN: 58.6% accuracy, 0.541 F1
   - **Gap:** −6.0 pp (pretrained is 10% better)

2. **Training Time:**
   - Fastest: MobileNetV2 frozen (5.5 min)
   - Slowest: Custom CNN (127.0 min)
   - ResNet50 fine-tuned: 43.4 min (good balance)

3. **Inference Speed:**
   - Custom CNN: 1.183 ms/image (845 img/s) on CPU
   - Pretrained models: Not yet measured (to be added in Submission 2)
   - Expected: Pretrained faster due to optimized implementations

4. **Why Custom CNN is Slower to Train:**
   - Needs to learn all features from scratch (no ImageNet prior)
   - Ran full 60 epochs (early stopping didn't kick in)
   - ResNet50 uses pre-learned filters, only fine-tunes

---

## 6. Data Leakage Impact

**Leak-Free Test Subset:** 6,610 images (excluding 568 leaked).

| Model | Official | Leak-Free | Drop |
|---|---|---|---|
| ResNet50 FT | 0.646 | 0.630 | −1.6 pp |
| Custom CNN | 0.586 | 0.572 | −1.4 pp |

**Conclusion:** Leakage inflates accuracy by ~1–2 percentage points. Leak-free numbers are more honest.

---

## 7. Confusion Matrix Insights

### Most Confused Emotion Pairs

**ResNet50 (best model):**
1. Fear → Sad: 20.8% of fear faces predicted as sad
2. Sad → Neutral: 17.7% of sad faces predicted as neutral
3. Neutral → Sad: 16.8% of neutral faces predicted as sad

**Custom CNN:**
1. Disgust → Angry: 31.9%
2. Sad → Neutral: 21.6%
3. Neutral → Sad: 19.6%

**Pattern:** Similar low-resolution emotions (fear/sad, sad/neutral) are hardest to distinguish.

---

## 8. Related Work Context

### Published Results on FER-2013

| Paper | Year | Model(s) | Result |
|---|---|---|---|
| Goodfellow et al. | 2013 | CNN contest | Human: 65%, Best team: 71% |
| Tang | 2013 | CNN + SVM | 71.2% |
| Pramerdorfer & Kampel | 2016 | VGG/Inception/ResNet ensemble | 75.2% |
| Khaireddin & Chen | 2021 | VGGNet (fine-tuned) | 73.28% |
| Minaee et al. | 2021 | CNN + attention | 70% |

### Our Results in Context
- **ResNet50 fine-tuned: 64.6%** (solid, below SOTA but reasonable for fresh fine-tuning)
- **Ensemble: 66.4%** (closer to published baselines)
- **Custom CNN: 58.6%** (expected; no ImageNet prior, learned from scratch)

---

## 9. Recommendations

### For Improving Performance

1. **Ensemble predictions** → already done in Submission 2 (66.4%)
2. **Longer fine-tuning** → try 50–100 epochs instead of 30
3. **Attention mechanisms** → focus on eyes/mouth regions
4. **Grad-CAM visualization** → explain model predictions
5. **Larger input size** → resize 48×48 to 96×96 (already done in Submission 2)
6. **Heavy augmentation** → increase rotation/shift/zoom (regularization level 2)

### For Production Deployment

1. **Use MobileNetV2 fine-tuned** (22 min training, small size, reasonable accuracy)
2. **Measure inference time** on target hardware (GPU/CPU/edge device)
3. **Deploy with ensemble** or single best model (ResNet50) depending on latency tolerance
4. **Handle edge cases:** low-contrast faces, extreme poses, occlusions
5. **Monitor leakage:** ensure test images never appear in future training

### For Next Phase

- [ ] Add Grad-CAM to show where model looks
- [ ] Build live webcam demo (MobileNetV2 or ResNet50)
- [ ] Test on other datasets (CK+, JAFFE) for generalization
- [ ] Experiment with attention and self-attention mechanisms
- [ ] Explore knowledge distillation (compress ResNet50 to MobileNetV2-like size)

---

## 10. File Organization

```
project/
├── Submission1_EDA_RelatedWork.ipynb
│   └── Dataset exploration, class imbalance, data quality checks
├── Submission2_Pretrained_FineTuning.ipynb
│   ├── VGG16, ResNet50, MobileNetV2 fine-tuning
│   ├── Overfitting/underfitting diagnosis and fixes
│   ├── Comparison table and charts
│   └── NEW: Inference timing for all 6 model variants
├── Submission3_Custom_CNN_Scratch.ipynb
│   ├── Custom architecture from scratch
│   ├── Early stopping and regularization
│   ├── Comparison with Submission 2
│   └── Inference time measurement
├── PROJECT_SUMMARY.md (this file)
│   └── Consolidated results, insights, recommendations
└── results/
    ├── submission2_results.pkl (model metrics)
    ├── submission2_testing_times.json (NEW: inference times)
    ├── submission3_results.pkl (model metrics)
    └── submission3_testing_time.json (inference times)
```

---

## 11. Conclusion

This project demonstrates a complete pipeline for facial expression recognition:

1. **Exploratory phase** identifies class imbalance (16.5:1), data quality issues (duplicates, leakage), and sets realistic targets (65–75% published, ~65% human).

2. **Transfer learning phase** fine-tunes three pretrained models. ResNet50 achieves **64.6% accuracy**, beating custom CNN by 6 percentage points despite learning from scratch.

3. **Custom CNN phase** trains 1.77M-parameter network from scratch, reaching 58.6% accuracy. Slower to train (127 min vs 43 min for best pretrained) but demonstrates architectural understanding.

4. **Ensemble** of three fine-tuned models reaches **66.4%**, approaching published baselines (70–75%).

5. **Data leakage** inflates test accuracy by ~1–2 pp; leak-free numbers are more honest.

**Recommendation:** Use **ResNet50 fine-tuned** for production (good accuracy, reasonable size/speed) or **MobileNetV2** for edge devices (faster training, lighter).

---

## References

1. I. J. Goodfellow et al., "Challenges in Representation Learning," *ICML*, 2013. https://arxiv.org/abs/1307.0414
2. Y. Tang, "Deep Learning using Linear Support Vector Machines," 2013. https://arxiv.org/abs/1306.0239
3. C. Pramerdorfer and M. Kampel, "Facial Expression Recognition using CNNs," 2016. https://arxiv.org/abs/1612.02903
4. Y. Khaireddin and Z. Chen, "Facial Emotion Recognition: SOTA on FER2013," 2021. https://arxiv.org/abs/2105.03588
5. S. Minaee et al., "Deep-Emotion," *Sensors*, 2021. https://arxiv.org/abs/1902.01019
6. FER-2013 Dataset: https://www.kaggle.com/datasets/msambare/fer2013

---
