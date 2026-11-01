# EV Charging Fault Candidate Dataset

## Purpose
Preliminary EV charging fault-candidate dataset prepared from the FLARE dataset.

## Dataset Summary
- Total images: 115
- Damaged connector: 17
- Damaged cable: 6
- Damaged screen: 4
- Normal/reference: 88

## Important Label Information
These images have image-level candidate labels only.
They do NOT contain YOLO bounding-box annotations.

Therefore, these images must be manually annotated with bounding boxes before being used for YOLO fault detection.

## Classes
- damaged_connector
- damaged_cable
- damaged_screen
- normal

## Source
FLARE EV charging infrastructure dataset.

## Current Status
Candidate dataset prepared and filtered for EV-relevant fault examples.

## Next Steps
1. Review the candidate images.
2. Create bounding-box annotations for visible faults.
3. Validate annotations.
4. Split into train/validation/test sets.
5. Prepare the final YOLO dataset.

## Important
Do not treat image-level labels as bounding-box ground truth.
Do not assume the entire image is the fault region.
