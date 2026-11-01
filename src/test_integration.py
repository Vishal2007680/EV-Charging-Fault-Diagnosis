import os
import sys
import json
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from infer import predict_fault

def run_smoke_test():
    print("=== EV Charging AI: Computer Vision Module Smoke Test ===")
    
    # 1. Test Broken Image
    broken_img = PROJECT_ROOT / "dataset" / "flare_classification" / "cord" / "broken" / "cord_broken_0065.jpg"
    print(f"\n[Test 1] Running inference on broken cord sample...")
    res1 = predict_fault(broken_img)
    print("  Status:", res1["status"])
    print("  Condition:", res1["condition"], f"(Conf: {res1['condition_confidence']:.2f})")
    print("  Component:", res1["component"])
    assert res1["status"] == "success", "Test 1 Failed: Status is not success"
    assert res1["condition"] == "broken", "Test 1 Failed: Condition is not broken"
    assert res1["bounding_box"] is None, "Test 1 Failed: Bounding box must be None"
    print("  -> Test 1 PASSED!")

    # 2. Test Healthy Image
    healthy_img = PROJECT_ROOT / "dataset" / "flare_classification" / "charger" / "healthy" / "charger_healthy_0023.jpg"
    print(f"\n[Test 2] Running inference on healthy charger sample...")
    res2 = predict_fault(healthy_img)
    print("  Status:", res2["status"])
    print("  Condition:", res2["condition"], f"(Conf: {res2['condition_confidence']:.2f})")
    print("  Component:", res2["component"])
    assert res2["status"] == "success", "Test 2 Failed: Status is not success"
    assert res2["condition"] == "healthy", "Test 2 Failed: Condition is not healthy"
    print("  -> Test 2 PASSED!")

    # 3. Test Invalid Path (Error handling)
    print(f"\n[Test 3] Testing graceful error handling on missing file...")
    res3 = predict_fault("non_existent_file.jpg")
    print("  Status:", res3["status"])
    print("  Error message:", res3["error_message"])
    assert res3["status"] == "error", "Test 3 Failed: Must return status='error'"
    print("  -> Test 3 PASSED!")

    print("\n==========================================================")
    print("ALL INTEGRATION SMOKE TESTS PASSED SUCCESSFULLY!")
    print("The CV module is fully functional and ready for Members 3 & 4.")
    print("==========================================================")

if __name__ == "__main__":
    run_smoke_test()
