# YOLO Pilot Working-Copy Annotation Changelog

**Date:** October 22, 2026  
**Auditor / Member:** Member 2 (Computer Vision & Fault Detection)  
**Status:** Working Copy Patch (Original Handover Files in `dataset/fault_annotations/` Preserved Untouched)

---

## 1. Scope & Policy
Member 1's handed-over annotations in `dataset/fault_annotations/` remain preserved and unmodified. This changelog documents the exact modifications made in the isolated working directory `experiments/yolo_pilot/data/` for experimental pilot object detection.

---

## 2. Documented Modifications

### Case 1: `cord_broken_0088.txt`
* **Original Label in Handover:**
  ```
  0 0.711220 0.396120 0.160613 0.095710
  ```
  *(Class `0`: `damaged_connector`)*
* **Updated Label in Working Copy:**
  ```
  1 0.711220 0.396120 0.160613 0.095710
  ```
  *(Class `1`: `damaged_cable`)*
* **Visual Verification Justification:**
  Image `cord_broken_0088.jpg` shows a ChargePoint EV charging pedestal whose charging cord was severed. The severed stub of the thick black charging cable hangs from the upper cradle with copper wiring exposed. The bounding box tightly bounds the severed cable stub. Class `0` (`damaged_connector`) was an annotator class-index typo; class `1` (`damaged_cable`) accurately reflects the physical object enclosed by the bounding box.

---

### Case 2: `cord_broken_0068.txt`
* **Label in Handover:**
  ```
  0 0.695819 0.354783 0.228828 0.458750
  ```
  *(Class `0`: `damaged_connector`)*
* **Action:** **Kept as Class `0` (No change).**
* **Visual Verification Justification:**
  Image `cord_broken_0068.jpg` shows an EV charging connector plug head that has been torn open at the cable-plug junction, exposing internal wiring. Although the filename begins with `cord_broken_`, the bounding box is positioned specifically around the damaged connector plug head. Class `0` is semantically correct for this localized box.

---

### Case 3: `screen_broken_0168.txt`
* **Label in Handover:**
  ```
  2 0.497510 0.480844 0.995020 0.961687
  ```
  *(Class `2`: `damaged_screen`, area = 95.7%)*
* **Action:** **Kept as Class `2` (No change).**
* **Visual Verification Justification:**
  Macro close-up photo of a completely shattered LCD display. The cracked glass and fractured screen elements span the entire photographic frame, justifying the full-frame bounding box.

---

## 3. Working-Copy Class Distribution Summary
* Class `0` (`damaged_connector`): **18 images**
* Class `1` (`damaged_cable`): **5 images**
* Class `2` (`damaged_screen`): **4 images**
* **Total:** **27 images**
