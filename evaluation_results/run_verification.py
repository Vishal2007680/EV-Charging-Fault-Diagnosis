import os
import sys
import json
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from dataset import EVFaultDataset, INV_CONDITION_MAP, INV_COMPONENT_MAP
from model import EVFaultDualHeadModel

test_csv = project_root / "splits" / "test_manifest.csv"
base_dir = project_root / "dataset" / "flare_classification"
weights = project_root / "models" / "best_mobilenet_v3.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = EVFaultDataset(test_csv, base_dir, is_train=False)
loader = DataLoader(ds, batch_size=16, shuffle=False)

model = EVFaultDualHeadModel(pretrained=False).to(device)
model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
model.eval()

preds_cond, targets_cond, probs_cond = [], [], []
preds_comp, targets_comp = [], []
paths = []

with torch.no_grad():
    for batch in loader:
        imgs = batch["image"].to(device)
        c_logits, p_logits = model(imgs)
        c_probs = torch.softmax(c_logits, dim=1)
        
        preds_cond.extend(torch.argmax(c_logits, dim=1).cpu().numpy())
        targets_cond.extend(batch["condition"].numpy())
        probs_cond.extend(c_probs.cpu().numpy())
        
        preds_comp.extend(torch.argmax(p_logits, dim=1).cpu().numpy())
        targets_comp.extend(batch["component"].numpy())
        paths.extend(batch["image_path"])

acc = accuracy_score(targets_cond, preds_cond)
prec_macro = precision_score(targets_cond, preds_cond, average="macro")
rec_macro = recall_score(targets_cond, preds_cond, average="macro")
f1_macro = f1_score(targets_cond, preds_cond, average="macro")
comp_acc = accuracy_score(targets_comp, preds_comp)

# Per-class metrics
cm = confusion_matrix(targets_cond, preds_cond)
# 0: Healthy, 1: Broken
tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

rec_broken = tp / (tp + fn)
prec_broken = tp / (tp + fp) if (tp + fp) > 0 else 0
rec_healthy = tn / (tn + fp)
prec_healthy = tn / (tn + fn) if (tn + fn) > 0 else 0

print("=== RECOMPUTED TEST METRICS ===")
print(f"Total Test Samples: {len(targets_cond)}")
print(f"Accuracy:           {acc:.6f} ({acc*100:.2f}%) -> {int(round(acc*len(targets_cond)))}/{len(targets_cond)}")
print(f"Macro Precision:    {prec_macro:.6f}")
print(f"Macro Recall:       {rec_macro:.6f}")
print(f"Macro F1:           {f1_macro:.6f}")
print(f"Component Accuracy: {comp_acc:.6f} ({comp_acc*100:.2f}%) -> {int(round(comp_acc*len(targets_comp)))}/{len(targets_comp)}")
print()
print(f"Broken-Class Recall:     {rec_broken:.6f} ({rec_broken*100:.2f}%) -> {tp}/{tp+fn}")
print(f"Broken-Class Precision:  {prec_broken:.6f} ({prec_broken*100:.2f}%) -> {tp}/{tp+fp}")
print(f"Healthy-Class Recall:    {rec_healthy:.6f} ({rec_healthy*100:.2f}%) -> {tn}/{tn+fp}")
print(f"Healthy-Class Precision: {prec_healthy:.6f} ({prec_healthy*100:.2f}%) -> {tn}/{tn+fn}")
print()
print(f"Confusion Matrix:\n{cm}")
print(f"  Act Healthy, Pred Healthy: {tn}")
print(f"  Act Healthy, Pred Broken:  {fp} (False Positive)")
print(f"  Act Broken,  Pred Healthy: {fn} (False Negative)")
print(f"  Act Broken,  Pred Broken:  {tp}")

# Check failure cases
failures = []
for i in range(len(targets_cond)):
    if targets_cond[i] != preds_cond[i]:
        failures.append({
            "file": Path(paths[i]).name,
            "full_path": paths[i],
            "act_cond": INV_CONDITION_MAP[targets_cond[i]],
            "pred_cond": INV_CONDITION_MAP[preds_cond[i]],
            "p_broken": float(probs_cond[i][1]),
            "p_healthy": float(probs_cond[i][0]),
            "comp": INV_COMPONENT_MAP[targets_comp[i]]
        })

print(f"\nTotal Misclassifications: {len(failures)}")
broken_to_healthy = [f for f in failures if f["act_cond"] == "broken" and f["pred_cond"] == "healthy"]
print(f"Broken-to-Healthy False Negatives: {len(broken_to_healthy)}")
for f in broken_to_healthy:
    print(f"  - {f['file']:25s} | Comp: {f['comp']:7s} | P(Broken)={f['p_broken']:.3f} | Path: {f['full_path']}")

healthy_to_broken = [f for f in failures if f["act_cond"] == "healthy" and f["pred_cond"] == "broken"]
print(f"Healthy-to-Broken False Positives: {len(healthy_to_broken)}")
for f in healthy_to_broken:
    print(f"  - {f['file']:25s} | Comp: {f['comp']:7s} | P(Broken)={f['p_broken']:.3f} | Path: {f['full_path']}")
