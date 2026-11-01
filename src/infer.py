import os
import sys
import json
import time
from pathlib import Path
from typing import Union, Dict, Any
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

# Resolve project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_WEIGHTS = MODELS_DIR / "best_mobilenet_v3.pth"

# Model definition import
try:
    from model import EVFaultDualHeadModel
except ImportError:
    from src.model import EVFaultDualHeadModel

# Label definitions
CONDITION_NAMES = {0: "healthy", 1: "broken"}
COMPONENT_NAMES = {0: "charger", 1: "cord", 2: "plug", 3: "screen"}

# Singleton model holder for fast repeated inference
_MODEL = None
_DEVICE = None

def _get_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def load_inference_model(weights_path: Union[str, Path] = DEFAULT_WEIGHTS, device: str = None):
    global _MODEL, _DEVICE
    if device is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _DEVICE = torch.device(device)

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Trained model weights not found at: {weights_path}")

    model = EVFaultDualHeadModel(pretrained=False).to(_DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=_DEVICE, weights_only=True))
    model.eval()
    _MODEL = model
    return _MODEL, _DEVICE

def predict_fault(image_input: Union[str, Path, Image.Image, np.ndarray],
                  threshold: float = 0.5,
                  weights_path: Union[str, Path] = DEFAULT_WEIGHTS) -> Dict[str, Any]:
    """
    Main Computer Vision Inference API for Member 3 (VLM Assistant) and Member 4 (Dashboard).

    Args:
        image_input: Filepath (str/Path), PIL Image, or RGB numpy array.
        threshold: Decision threshold for 'broken' condition (default: 0.5).
        weights_path: Optional custom path to model weights.

    Returns:
        Structured JSON-serializable dictionary with diagnosis results.
    """
    t_start = time.perf_counter()

    # 1. Input Validation and Ingestion
    img = None
    input_desc = ""
    try:
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            input_desc = str(p)
            if not p.exists():
                return {
                    "status": "error",
                    "condition": None,
                    "condition_confidence": 0.0,
                    "component": None,
                    "component_confidence": 0.0,
                    "is_fault_detected": None,
                    "visual_fault_localized": False,
                    "bounding_box": None,
                    "error_message": f"File does not exist: {p}"
                }
            img = Image.open(p).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
            input_desc = "PIL.Image"
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input).convert("RGB")
            input_desc = "numpy.ndarray"
        else:
            return {
                "status": "error",
                "condition": None,
                "condition_confidence": 0.0,
                "component": None,
                "component_confidence": 0.0,
                "is_fault_detected": None,
                "visual_fault_localized": False,
                "bounding_box": None,
                "error_message": f"Unsupported input type: {type(image_input)}"
            }
    except Exception as e:
        return {
            "status": "error",
            "condition": None,
            "condition_confidence": 0.0,
            "component": None,
            "component_confidence": 0.0,
            "is_fault_detected": None,
            "visual_fault_localized": False,
            "bounding_box": None,
            "error_message": f"Failed to decode image: {str(e)}"
        }

    # 2. Ensure Model is Loaded
    global _MODEL, _DEVICE
    if _MODEL is None:
        try:
            load_inference_model(weights_path=weights_path)
        except Exception as e:
            return {
                "status": "error",
                "condition": None,
                "condition_confidence": 0.0,
                "component": None,
                "component_confidence": 0.0,
                "is_fault_detected": None,
                "visual_fault_localized": False,
                "bounding_box": None,
                "error_message": f"Model initialization failed: {str(e)}"
            }

    # 3. Preprocess and Forward Pass
    try:
        transform = _get_transform()
        img_tensor = transform(img).unsqueeze(0).to(_DEVICE)

        with torch.no_grad():
            cond_logits, comp_logits = _MODEL(img_tensor)
            cond_probs = torch.softmax(cond_logits, dim=1)[0].cpu().numpy()
            comp_probs = torch.softmax(comp_logits, dim=1)[0].cpu().numpy()

        p_healthy = float(cond_probs[0])
        p_broken = float(cond_probs[1])

        # Binary decision based on threshold
        if p_broken >= threshold:
            condition = "broken"
            cond_conf = p_broken
            is_fault = True
        else:
            condition = "healthy"
            cond_conf = p_healthy
            is_fault = False

        comp_idx = int(np.argmax(comp_probs))
        component = COMPONENT_NAMES.get(comp_idx, "unknown")
        comp_conf = float(comp_probs[comp_idx])

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Construct actionable diagnostic hint for Member 3
        if is_fault:
            diagnostic_hint = (
                f"Visual damage detected on component '{component}' "
                f"(Condition Confidence: {cond_conf*100:.1f}%). "
                f"Potential physical failure, fraying, or crack observed. Immediate safety inspection recommended."
            )
        else:
            diagnostic_hint = (
                f"Component '{component}' appears structurally healthy "
                f"(Confidence: {cond_conf*100:.1f}%). No surface physical damage observed."
            )

        if cond_conf < 0.65:
            diagnostic_hint += " [Warning: Low classification confidence. Manual technician verification strongly advised.]"

        return {
            "status": "success",
            "condition": condition,
            "condition_confidence": round(cond_conf, 4),
            "component": component,
            "component_confidence": round(comp_conf, 4),
            "is_fault_detected": is_fault,
            "visual_fault_localized": False,  # Explicit boundary: classifier does not localize
            "bounding_box": None,             # Explicit boundary: classifier does not return bbox
            "execution_time_ms": round(elapsed_ms, 2),
            "model_version": "MobileNetV3-Small-DualHead-v1",
            "diagnostic_hint": diagnostic_hint,
            "image_source": input_desc,
            "probabilities": {
                "healthy": round(p_healthy, 4),
                "broken": round(p_broken, 4)
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "condition": None,
            "condition_confidence": 0.0,
            "component": None,
            "component_confidence": 0.0,
            "is_fault_detected": None,
            "visual_fault_localized": False,
            "bounding_box": None,
            "error_message": f"Inference execution error: {str(e)}"
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EV Charging Station Fault Diagnosis Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), help="Path to weights file")
    args = parser.parse_args()

    result = predict_fault(args.image, weights_path=args.weights)
    print(json.dumps(result, indent=2))
