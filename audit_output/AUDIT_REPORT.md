# EV Charging Station AI — Dataset Audit & Validation Report
**Phase 1 Deliverable — Member 2 (Computer Vision and Fault Detection)**
**Date:** October 22, 2026

---

## 1. Executive Summary

This report documents the Phase 1 audit of the EV charging station dataset handed over by Member 1. The original archive `EV_Charging_AI_Dataset_Handover.zip` was safely extracted into `c:\VIT\5th Sem\EDI\ProjectWork\dataset` without modifying the original zip.

All 225 classification images, 27 fault-annotation images, 115 candidate images, and 167 segmentation image-mask pairs were verified for existence, readability, dimensions, and label correspondence. All 27 YOLO bounding boxes were visually inspected and overlaid previews were saved to `audit_output/bbox_previews/`.

---

## 2. Handover Archive & Extraction Verification

- **Archive Location:** `c:\VIT\5th Sem\EDI\ProjectWork\EV_Charging_AI_Dataset_Handover.zip`
- **Archive Size:** 297,521,727 bytes (283.74 MB)
- **Status:** Verified present, readable, and completely untouched.
- **Working Extraction Folder:** `c:\VIT\5th Sem\EDI\ProjectWork\dataset/`
- **Extracted Subfolders:**
  1. `flare_classification/`
  2. `fault_annotations/`
  3. `fault_candidates/`
  4. `flare_segmentation/`
  5. `README_HANDOVER.txt`

---

## 3. Dataset Component Audits

### 3.1 FLARE Classification Dataset
- **Total Records:** 225 in `metadata.csv`
- **Image Readability Check:** 225 of 225 readable (100%), 0 missing, 0 corrupted.
- **Image Dimensions:** Varied from minimum `259 x 194` to maximum `4032 x 3024`. All images have 3 RGB color channels.
- **Label Mapping:**
  - `0`: Healthy (Structurally intact, normal appearance) — **130 images**
  - `1`: Broken (Visible physical damage/defect) — **95 images**
- **Breakdown by Component & Condition:**
  | Component | Healthy (0) | Broken (1) | Total |
  | :--- | :---: | :---: | :---: |
  | **charger** (body/kiosk) | 42 | 23 | 65 |
  | **cord** (charging cable) | 17 | 29 | 46 |
  | **plug** (connector head) | 15 | 17 | 32 |
  | **screen** (display/glass) | 56 | 26 | 82 |
  | **Total** | **130** | **95** | **225** |

### 3.2 Duplicates and Data Leakage Audit
- **Exact Duplicates:** 5 pairs identified in `exact_duplicates.csv` (all in Healthy class):
  1. `plug/healthy/plug_healthy_0142.jpg` == `plug/healthy/plug_healthy_0143.jpg`
  2. `charger/healthy/charger_healthy_0040.jpg` == `screen/healthy/screen_healthy_0188.jpg`
  3. `charger/healthy/charger_healthy_0043.jpg` == `screen/healthy/screen_healthy_0190.jpg`
  4. `screen/healthy/screen_healthy_0199.jpg` == `screen/healthy/screen_healthy_0200.jpg`
  5. `screen/healthy/screen_healthy_0171.jpg` == `screen/healthy/screen_healthy_0227.jpg`
- **Near Duplicates:** Pairs 2 and 3 above were also logged with perceptual hash distance = 0.
- **Key Finding:** Pairs 2 and 3 represent cross-component duplicates (the identical image is categorized as `charger` in one record and `screen` in another).
- **Unique Image Count:** 220 unique images (125 unique healthy, 95 unique broken).

### 3.3 YOLO Fault Annotations Audit (27 Images)
- **Directory:** `dataset/fault_annotations/EV_Charging_Fault_Annotations_Final (1)/`
- **Total Images:** 27 JPEG images (all readable).
- **Total Labels:** 27 `.txt` files in YOLO normalized format (`class xc yc w h`).
- **Coordinate Integrity:** 100% valid. All coordinates are normalized in $(0, 1)$ range. No syntax or numerical parsing errors. Exactly 1 bounding box per file.
- **Class Mapping Defined in `classes.txt` / `data.yaml`:**
  - `0`: `damaged_connector`
  - `1`: `damaged_cable`
  - `2`: `damaged_screen`
- **Actual Class Distribution in YOLO Label Files:**
  - Class `0` (`damaged_connector`): **19 boxes**
  - Class `1` (`damaged_cable`): **4 boxes**
  - Class `2` (`damaged_screen`): **4 boxes**

#### Critical Findings on Fault Annotations:
1. **Class Discrepancy on `cord_broken_0068.txt` and `cord_broken_0088.txt`:**
   - Both files have `cord_broken_` prefixes and were listed under cable faults in Member 1's candidate list.
   - However, their YOLO label files assign them to **Class `0` (`damaged_connector`)**.
   - *Visual verification of `cord_broken_0068.jpg`:* Shows a connector plug torn off from the cable with exposed internal wires. The bounding box targets the damaged plug head. Hence, class `0` is semantically defensible for the box.
   - *Visual verification of `cord_broken_0088.jpg`:* Shows a severed charging cable stub dangling from a ChargePoint station. The box is drawn around the severed cable, yet labeled class `0` (`damaged_connector`). This appears to be a mislabel (should be class `1: damaged_cable`).
2. **Oversized Box on `screen_broken_0168.txt`:**
   - Bounding box covers **95.7%** of the entire image (`w=0.995, h=0.962`).
   - *Visual verification:* The photo is a full-frame macro close-up of a completely shattered LCD display. Because the entire frame consists of cracked screen glass, the full-frame box correctly represents the visual damage area.
3. **No Negative (Normal) Images in YOLO Annotations:**
   - All 27 images contain faults. There are zero background or healthy images in the YOLO training folder.

### 3.4 Fault Candidates Audit (115 Images)
- **Directory:** `dataset/fault_candidates/`
- **Total Records:** 115 images in `metadata.csv` (all 115 readable, 0 corrupt).
- **Breakdown:**
  - `damaged_connector`: 17 images
  - `damaged_cable`: 6 images
  - `damaged_screen`: 4 images
  - `normal`: 88 images
- **YOLO Bounding Box Availability:** Confirmed **"No"** for all 115 records. These are candidate images with image-level labels only.

### 3.5 FLARE Segmentation Audit (167 Pairs)
- **Directory:** `dataset/flare_segmentation/`
- **Total Records:** 167 image-mask pairs in `metadata.csv`.
- **Purpose:** Component segmentation (`1`=screen, `2`=body, `3`=cable, `4`=plug, `5`=background).
- **Dimension Check:** 166 of 167 pairs match perfectly.
  - Known mismatch: `flare_151.jpg` (`1280 x 720`) vs `flare_151_mask.png` (`720 x 720 / 720 x 1280`).
- **Nature of Data:** Segmenting normal components, **not** fault locations.

---

## 4. Leakage-Free Splitting Recommendation

To prevent data leakage caused by the 5 exact duplicate pairs:

1. **Grouped Stratified Split Strategy:**
   - Assign each unique image a `group_id`. Exact duplicate pairs share the same `group_id`.
   - Perform a stratified split on `(component, condition)` while constraining that all samples sharing a `group_id` are assigned strictly to the **same split** (e.g. Training).
2. **Recommended Split Ratio (70% Train / 15% Val / 15% Test):**
   - **Train:** ~154–157 images (~88–91 healthy, 66 broken)
   - **Validation:** ~34 images (~20 healthy, 14 broken)
   - **Test:** ~34 images (~19 healthy, 15 broken)
   - Both Validation and Test sets will contain strictly unique images with zero duplicate leakage into Training.

---

## 5. Summary of Audit Artifacts Created

1. Extracted Dataset: `c:\VIT\5th Sem\EDI\ProjectWork\dataset\`
2. Machine-Readable Audit Report: `c:\VIT\5th Sem\EDI\ProjectWork\audit_output\dataset_audit_report.json`
3. Visual BBox Previews (27 images): `c:\VIT\5th Sem\EDI\ProjectWork\audit_output\bbox_previews\`
4. Markdown Audit Report: `c:\VIT\5th Sem\EDI\ProjectWork\audit_output\AUDIT_REPORT.md`
