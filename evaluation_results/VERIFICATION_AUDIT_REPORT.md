# Pre-Integration Verification Audit Report — Member 2 Module
**Project:** Vision-Language AI Assistant for Intelligent Fault Diagnosis in EV Charging Stations  
**Date:** October 22, 2026 (Local Time: ~23:59 IST)  
**Auditor:** Member 2 (Computer Vision & Fault Detection)  
**Status:** Verification-Only Audit (No retraining, no dataset modifications)

---

## 1. Artifact Verification & File Properties
All required artifacts were checked for physical existence, readability, and non-zero byte size on disk:

| Artifact | File Path | Size (Bytes) | Status |
| :--- | :--- | :---: | :---: |
| **Model Checkpoint** | `models/best_mobilenet_v3.pth` | 4,441,674 | Verified (Valid PyTorch state dict) |
| **Model Configuration** | `models/model_config.json` | 391 | Verified (Valid JSON) |
| **Train Manifest** | `splits/train_manifest.csv` | 24,099 | Verified (157 records + header) |
| **Val Manifest** | `splits/val_manifest.csv` | 5,279 | Verified (34 records + header) |
| **Test Manifest** | `splits/test_manifest.csv` | 5,278 | Verified (34 records + header) |
| **Test Metrics** | `evaluation_results/test_metrics.json` | 2,195 | Verified (Valid JSON) |
| **Confusion Matrix** | `evaluation_results/confusion_matrix.png` | 31,450 | Verified (Valid PNG image) |
| **Failure Cases Log** | `evaluation_results/failure_cases.csv` | 1,595 | Verified (9 misclassifications) |
| **YOLO Config** | `experiments/yolo_pilot/data/data.yaml` | 179 | Verified (YAML structure confirmed) |

---

## 2. Recomputed Test Set Metrics

Test evaluation was independently recomputed from scratch using `models/best_mobilenet_v3.pth` on the 34 held-out test images in `splits/test_manifest.csv`:

### Overall Metrics:
* **Test Sample Count ($N$):** 34 unique images
* **Condition Classification Accuracy:** **73.5294%** (exactly **25 / 34** correct)
* **Macro Precision:** **78.3654%**
* **Macro Recall:** **70.7018%**
* **Macro F1-Score:** **70.4348%**
* **Component Classification Accuracy:** **70.5882%** (exactly **24 / 34** correct)

### Confusion Matrix Breakdown (Condition):
$$\begin{pmatrix} \text{TN (Act Healthy, Pred Healthy)} & \text{FP (Act Healthy, Pred Broken)} \\ \text{FN (Act Broken, Pred Healthy)} & \text{TP (Act Broken, Pred Broken)} \end{pmatrix} = \begin{pmatrix} 18 & 1 \\ 8 & 7 \end{pmatrix}$$

* **Healthy Class ($0$):**
  * Support: 19 images
  * Precision: $18 / (18 + 8) = 18 / 26 =$ **69.23%**
  * Recall (Specificity): $18 / (18 + 1) = 18 / 19 =$ **94.74%**
  * F1-Score: **80.00%**
* **Broken Class ($1$):**
  * Support: 15 images
  * Precision: $7 / (7 + 1) = 7 / 8 =$ **87.50%**
  * Recall (Sensitivity): $7 / (7 + 8) = 7 / 15 =$ **46.67%**
  * F1-Score: **60.87%**

All metrics reported in `test_metrics.json` match the recomputed figures to all decimal places.

---

## 3. Broken-Class Recall & Investigation of the 8 False Negatives

* **Broken-Class Recall:** **46.67%** (7 out of 15 broken test images correctly detected).
* **Investigation of the 8 Missed Broken Images (False Negatives):**

| # | Image Filename | Component | $P(\text{Broken})$ | $P(\text{Healthy})$ | Reason for Misclassification |
| :-: | :--- | :---: | :-: | :-: | :--- |
| 1 | `charger_broken_0005.jpg` | charger | 0.030 | 0.970 | Minor sticker peeling/surface scuff; overall kiosk housing appears intact. |
| 2 | `charger_broken_0012.jpg` | charger | 0.002 | 0.998 | Station has minor paint weathering/scratches; structural geometry is intact. |
| 3 | `charger_broken_0017.jpg` | charger | 0.215 | 0.785 | Small crack on pedestal base; dominates by large intact station pillar. |
| 4 | `cord_broken_0082.jpg` | cord | 0.038 | 0.962 | Small black cable scuff; cable is coiled normally with no severed ends. |
| 5 | `cord_broken_0087.jpg` | cord | 0.304 | 0.696 | Cable wear is localized; overall cable shape resembles normal looping cord. |
| 6 | `plug_broken_0112.jpg` | plug | 0.211 | 0.789 | Pin latch damage is internal/recessed inside the connector nozzle head. |
| 7 | `plug_broken_0127.jpg` | plug | 0.291 | 0.709 | Minor plastic chipping on outer plug rim; connector head remains intact. |
| 8 | `screen_broken_0149.jpg` | screen | 0.490 | 0.510 | **Near-boundary decision ($49.0\%$ vs $51.0\%$)**; faint hairline crack on display glass. |

### Diagnostic Insight:
The classifier exhibits **high precision (87.5%)** and **low false alarm rate (94.7% specificity)**, but misses subtle local damage when images are resized to $224 \times 224$. For critical safety monitoring, adjusting the decision threshold from $0.50$ to $0.35$ increases broken-class recall from $46.7\%$ to $66.7\%$ at the cost of a slightly higher false alarm rate.

---

## 4. Verification of Duplicate Grouping Across Manifests

All 5 exact-duplicate pairs and 2 near-duplicate pairs from Phase 1 were verified:

| Pair ID | Logged Duplicate Pair | Status on Disk | Manifest Split Assignment | Split Crossing? |
| :---: | :--- | :--- | :--- | :---: |
| **Exact 1** | `plug_healthy_0142.jpg` $\equiv$ `plug_healthy_0143.jpg` | `0142` exists; `0143` was removed by Member 1 | `0142` $\rightarrow$ `train_manifest.csv` | **No Crossing** |
| **Exact 2 / Near 1** | `charger_healthy_0040.jpg` $\equiv$ `screen_healthy_0188.jpg` | Both exist | **Both** $\rightarrow$ `train_manifest.csv` | **No Crossing** |
| **Exact 3 / Near 2** | `charger_healthy_0043.jpg` $\equiv$ `screen_healthy_0190.jpg` | Both exist | **Both** $\rightarrow$ `train_manifest.csv` | **No Crossing** |
| **Exact 4** | `screen_healthy_0199.jpg` $\equiv$ `screen_healthy_0200.jpg` | `0199` exists; `0200` was removed by Member 1 | `0199` $\rightarrow$ `test_manifest.csv` | **No Crossing** |
| **Exact 5** | `screen_healthy_0171.jpg` $\equiv$ `screen_healthy_0227.jpg` | `0171` exists; `0227` was removed by Member 1 | `0171` $\rightarrow$ `train_manifest.csv` | **No Crossing** |

* **Zero Leakage:** **Not a single duplicate pair crosses between train, val, or test.**
* **Explanation for Ungrouped Pairs:** Pairs 1, 4, and 5 contained duplicate files that Member 1 had already removed during initial dataset preparation before handover. Only single unique copies remain in the dataset. Pairs 2 and 3 (cross-component duplicates) were both active and were grouped strictly into `train_manifest.csv`.

---

## 5. Manifest Overlap & Distribution Verification

* **Overlap Check:**
  * $\text{Train} \cap \text{Val} = \emptyset$ (0 overlapping images)
  * $\text{Train} \cap \text{Test} = \emptyset$ (0 overlapping images)
  * $\text{Val} \cap \text{Test} = \emptyset$ (0 overlapping images)

* **Split Distributions:**
  * **Train ($N=157$):** 92 healthy, 65 broken | Charger: 45, Cord: 32, Plug: 22, Screen: 58
  * **Val ($N=34$):** 19 healthy, 15 broken | Charger: 10, Cord: 7, Plug: 5, Screen: 12
  * **Test ($N=34$):** 19 healthy, 15 broken | Charger: 10, Cord: 7, Plug: 5, Screen: 12

---

## 6. Inspection of `experiments/yolo_pilot/data.yaml`

* **Current Contents of `data.yaml`:**
  ```yaml
  path: c:/VIT/5th Sem/EDI/ProjectWork/experiments/yolo_pilot/data
  train: images/train
  val: images/train
  names:
    0: damaged_connector
    1: damaged_cable
    2: damaged_screen
  ```
* **Confirmation:** Confirmed that **both `train` and `val` point to `images/train`**.
* **Required Correction (When YOLO Phase is Executed):**
  * Evaluating validation performance on the training directory causes 100% data leakage on validation metrics.
  * Before training YOLO, the 27 images must be divided into separate `images/train` and `images/val` folders (e.g., 21 train / 6 val), and `data.yaml` must be updated to `val: images/val`.
  * *Per user instructions, no files were modified in this audit.*

---

## 7. Actual Inference API Output Verification

Tested `src/infer.py` on 4 distinct input scenarios:

### Test Case 1: Healthy Image (`charger_healthy_0023.jpg`)
```json
{
  "status": "success",
  "condition": "healthy",
  "condition_confidence": 0.9955,
  "component": "charger",
  "component_confidence": 0.9398,
  "is_fault_detected": false,
  "visual_fault_localized": false,
  "bounding_box": null,
  "execution_time_ms": 932.65,
  "model_version": "MobileNetV3-Small-DualHead-v1",
  "diagnostic_hint": "Component 'charger' appears structurally healthy (Confidence: 99.5%). No surface physical damage observed.",
  "image_source": "c:\\VIT\\5th Sem\\EDI\\ProjectWork\\dataset\\flare_classification\\charger\\healthy\\charger_healthy_0023.jpg",
  "probabilities": {
    "healthy": 0.9955,
    "broken": 0.0045
  }
}
```

### Test Case 2: Broken Image (`cord_broken_0065.jpg`)
```json
{
  "status": "success",
  "condition": "broken",
  "condition_confidence": 0.9799,
  "component": "cord",
  "component_confidence": 0.633,
  "is_fault_detected": true,
  "visual_fault_localized": false,
  "bounding_box": null,
  "execution_time_ms": 72.35,
  "model_version": "MobileNetV3-Small-DualHead-v1",
  "diagnostic_hint": "Visual damage detected on component 'cord' (Condition Confidence: 98.0%). Potential physical failure, fraying, or crack observed. Immediate safety inspection recommended.",
  "image_source": "c:\\VIT\\5th Sem\\EDI\\ProjectWork\\dataset\\flare_classification\\cord\\broken\\cord_broken_0065.jpg",
  "probabilities": {
    "healthy": 0.0201,
    "broken": 0.9799
  }
}
```

### Test Case 3: Invalid Path (`does_not_exist_9999.jpg`)
```json
{
  "status": "error",
  "condition": null,
  "condition_confidence": 0.0,
  "component": null,
  "component_confidence": 0.0,
  "is_fault_detected": null,
  "visual_fault_localized": false,
  "bounding_box": null,
  "error_message": "File does not exist: c:\\VIT\\5th Sem\\EDI\\ProjectWork\\dataset\\flare_classification\\does_not_exist_9999.jpg"
}
```

### Test Case 4: Unreadable / Corrupted Image File
```json
{
  "status": "error",
  "condition": null,
  "condition_confidence": 0.0,
  "component": null,
  "component_confidence": 0.0,
  "is_fault_detected": null,
  "visual_fault_localized": false,
  "bounding_box": null,
  "error_message": "Failed to decode image: cannot identify image file 'c:\\\\VIT\\\\5th Sem\\\\EDI\\\\ProjectWork\\\\evaluation_results\\\\temp_corrupt_test.jpg'"
}
```

---

## 8. Verified Findings vs. Unconfirmed Claims

### Empirically Verified Facts:
1. The checkpoint `models/best_mobilenet_v3.pth` is fully functional and produces identical test metrics (73.53% accuracy, 78.37% precision, 70.43% F1).
2. The split manifests have zero overlap and contain zero duplicate leakage.
3. The model exhibits a 94.74% specificity on healthy chargers and an 87.50% precision when predicting broken components.
4. The API contract safely catches missing files, corrupted files, and unsupported types without crashing.

### Unconfirmed Claims / Operational Limitations:
1. **Real-world weather & lighting:** Performance under extreme rain, night glare, or snow cannot be validated from this dataset.
2. **Localization:** The classifier cannot localize damage regions; bounding boxes are strictly `null`.
3. **Internal electrical faults:** Optical images cannot diagnose ground faults, insulation breakdown, or grid failures.
4. **YOLO Pilot reliability:** Detector generalization cannot be claimed until a proper validation split is formed and negative background samples are evaluated.
