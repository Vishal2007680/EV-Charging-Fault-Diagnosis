
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.model import EVFaultDualHeadModel
from src.dataset import EVFaultDataset

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cpu")

dataset = EVFaultDataset(
    ROOT / "splits" / "val_manifest.csv",
    ROOT / "dataset" / "flare_classification",
    is_train=False,
)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

model = EVFaultDualHeadModel(pretrained=False).to(DEVICE)
weights_path = ROOT / "models" / "best_mobilenet_v3_candidate_A.pth"
state = torch.load(weights_path, map_location=DEVICE, weights_only=True)
model.load_state_dict(state)
model.eval()

actual, probabilities = [], []

with torch.no_grad():
    for batch in loader:
        images = batch["image"].to(DEVICE)
        targets = batch["condition"].cpu().tolist()
        condition_logits, _ = model(images)
        probs = torch.softmax(condition_logits, dim=1)[:, 1].cpu().tolist()
        actual.extend(targets)
        probabilities.extend(probs)

print(f"Validation images: {len(actual)}")
print("Positive class: broken (label 1)")
print("\nThreshold | Accuracy | Broken precision | Broken recall | False-positive rate | Macro F1")
print("-" * 92)

for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
    predicted = [int(p >= threshold) for p in probabilities]
    healthy_count = sum(y == 0 for y in actual)
    false_positives = sum(y == 0 and p == 1 for y, p in zip(actual, predicted))
    fpr = false_positives / healthy_count if healthy_count else 0.0

    print(
        f"{threshold:9.2f} | "
        f"{accuracy_score(actual, predicted):8.3f} | "
        f"{precision_score(actual, predicted, zero_division=0):16.3f} | "
        f"{recall_score(actual, predicted, zero_division=0):13.3f} | "
        f"{fpr:19.3f} | "
        f"{f1_score(actual, predicted, average='macro', zero_division=0):8.3f}"
    )

print("\nAnalysis complete. No training or test-set evaluation was performed.")
