EV CHARGING AI DATASET HANDOVER
================================

Purpose
-------
Dataset prepared for the Vision-Language AI Assistant for Intelligent
Fault Diagnosis in EV Charging Stations.

1. FLARE CLASSIFICATION
-----------------------
225 metadata records.
Image paths verified.
130 healthy and 95 broken images.

2. FLARE SEGMENTATION
---------------------
167 images and 167 masks extracted.
Official labels:
1 = screen
2 = body
3 = cable
4 = plug
5 = void-background

QA limitation:
flare_151.jpg has dimensions 1280x720, while its original mask
has dimensions 720x1280. The original source has the same mismatch.
The original mask is preserved. A rotated candidate exists outside
the handover dataset and is not approved for training.
The meaning of pixel value 0 in this export remains unconfirmed.
Other pairs passed structural and dimension/read checks; this does
not guarantee semantic correctness of every mask.

3. FAULT ANNOTATIONS
--------------------
27 images and 27 YOLO label files.
Structural checks passed: paths and label syntax are valid.
Visual review was limited. The large box on screen_broken_0168
was inspected and appeared reasonable.
Do not interpret structural validation as proof that every box
is semantically correct.

Classes:
0 = damaged_connector
1 = damaged_cable
2 = damaged_screen

4. FAULT CANDIDATES
-------------------
115 readable images; no exact duplicate groups detected.
17 damaged connector candidates
6 damaged cable candidates
4 damaged screen candidates
88 normal/reference candidates

These have image-level candidate labels only. They do not have
YOLO bounding-box annotations and are not a finished YOLO dataset.
Review category suitability and annotate visible faults before use
in object-detection training.

TRAINING AND SPLITS
-------------------
Do not combine charger-type labels with fault-detection classes.
Do not create train/validation/test splits until the responsible
teammate confirms the split plan.
Verify labels and suitability before training.

HANDOVER STATUS
---------------
Prepared for teammate review and initial pipeline setup.
Known limitations are documented; not all annotations have been
fully semantically verified.
