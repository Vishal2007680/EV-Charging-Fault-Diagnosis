
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from src.model import EVFaultDualHeadModel
from src.dataset import EVFaultDataset
from pathlib import Path

root = Path(__file__).resolve().parent
device = torch.device("cpu")

dataset = EVFaultDataset(
    root / "splits" / "val_manifest.csv",
    root / "dataset" / "flare_classification",
    is_train=False
)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

for name in ["candidate_A", "candidate_B"]:
    path = root / "models" / f"best_mobilenet_v3_{name}.pth"
    model = EVFaultDualHeadModel(pretrained=False).to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    actual, predicted = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["condition"].cpu().tolist()
            logits, _ = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            actual.extend(targets)
            predicted.extend(preds)

    print(f"\n=== {name.upper()} ===")
    print("Confusion matrix [healthy, broken]:")
    print(confusion_matrix(actual, predicted, labels=[0, 1]))
    print(classification_report(
        actual, predicted,
        labels=[0, 1],
        target_names=["healthy", "broken"],
        zero_division=0
    ))
