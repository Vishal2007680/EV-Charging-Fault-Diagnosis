# EV Charging Station Fault Diagnosis — CV Module Integration Guide
**Deliverable for Member 3 (Vision-Language Assistant) and Member 4 (Dashboard)**
**Author:** Member 2 (Computer Vision & Fault Detection)  
**Date:** October 22, 2026

---

## 1. Overview
The Computer Vision module provides automated visual condition assessment of EV charging stations and their components. It uses a trained **MobileNetV3-Small Dual-Head** deep neural network that evaluates an input image and predicts:
1. **Condition:** `healthy` vs. `broken` (with confidence probability).
2. **Component:** `charger` (kiosk/housing), `cord` (cable), `plug` (connector), or `screen` (display).

---

## 2. Quickstart for Python Integration (Member 3 & Member 4)

You can import and run the inference module directly in Python:

```python
from pathlib import Path
from src.infer import predict_fault

# 1. Provide an image path, PIL Image, or numpy array
image_path = "dataset/flare_classification/cord/broken/cord_broken_0065.jpg"

# 2. Run inference (loads weights once and caches model in memory)
result = predict_fault(image_path)

# 3. Access structured diagnosis
print("Status:", result["status"])
print("Condition:", result["condition"])               # 'healthy' or 'broken'
print("Condition Confidence:", result["condition_confidence"])  # Float 0.0 - 1.0
print("Component:", result["component"])               # 'charger', 'cord', 'plug', 'screen'
print("Fault Detected:", result["is_fault_detected"])  # True or False
print("Diagnostic Hint:", result["diagnostic_hint"])   # Natural language context for VLM
```

---

## 3. Command-Line Interface (CLI)

You can also run inference from the command line or subprocess:

```bash
python src/infer.py --image "path/to/station_photo.jpg"
```

Output is printed as clean JSON to `stdout`.

---

## 4. Input & Output Contract

### Successful Response Format
```json
{
  "status": "success",
  "condition": "broken",
  "condition_confidence": 0.9799,
  "component": "cord",
  "component_confidence": 0.6330,
  "is_fault_detected": true,
  "visual_fault_localized": false,
  "bounding_box": null,
  "execution_time_ms": 12.4,
  "model_version": "MobileNetV3-Small-DualHead-v1",
  "diagnostic_hint": "Visual damage detected on component 'cord' (Condition Confidence: 98.0%). Potential physical failure, fraying, or crack observed. Immediate safety inspection recommended.",
  "image_source": "dataset/flare_classification/cord/broken/cord_broken_0065.jpg",
  "probabilities": {
    "healthy": 0.0201,
    "broken": 0.9799
  }
}
```

### Error Response Format
If an invalid file or unsupported image format is supplied:
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
  "error_message": "File does not exist: invalid_path.xyz"
}
```

---

## 5. Guidelines for Integration

### For Member 3 (Vision-Language Assistant)
* **Ingesting Context:** Feed the `diagnostic_hint`, `component`, `condition`, and `condition_confidence` into your LLM prompt alongside the image.
* **Low Confidence Guardrail:** If `condition_confidence < 0.65`, prompt the model to add a disclaimer: *"AI confidence is moderate; technician manual inspection is strongly recommended."*
* **Explaining Faults:** Use the component type to give relevant safety advice (e.g. if `component == 'cord'` and `condition == 'broken'`, advise checking for severed insulation, copper exposure, and electrical leakage risk).

### For Member 4 (Dashboard Integration)
* **Status Badges:** Display a Green Badge (`Healthy`) or Red Badge (`Broken / Fault Detected`) using `result["condition"]`.
* **Confidence Bar:** Display a percentage gauge using `result["condition_confidence"] * 100`.
* **Component Tag:** Display which component was inspected (`Charger Kiosk`, `Cable`, `Connector Plug`, `Screen`).
* **Visual Bounding Boxes:** Note that the classification model outputs `bounding_box: null`. Do not display empty box frames.

---

## 6. System Boundaries & Constraints
1. **Surface Damage Only:** This model detects physical, surface-visible defects. It does not measure grid voltage, internal PCB faults, or insulation resistance.
2. **Resolution:** High-resolution images are resized to $224 \times 224$ for inference; microscopic cracks may occasionally be missed.
