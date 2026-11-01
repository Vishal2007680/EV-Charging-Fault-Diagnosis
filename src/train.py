import os
import json
import random
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from dataset import EVFaultDataset, CONDITION_MAP, COMPONENT_MAP, INV_CONDITION_MAP, INV_COMPONENT_MAP
from model import EVFaultDualHeadModel

# Fix all seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "splits"
DATASET_BASE = PROJECT_ROOT / "dataset" / "flare_classification"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train_epoch(model, loader, optimizer, criterion_cond, criterion_comp, device):
    model.train()
    total_loss = 0.0
    correct_cond = 0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(device)
        cond_targets = batch["condition"].to(device)
        comp_targets = batch["component"].to(device)

        optimizer.zero_grad()
        cond_logits, comp_logits = model(images)

        loss_cond = criterion_cond(cond_logits, cond_targets)
        loss_comp = criterion_comp(comp_logits, comp_targets)
        loss = loss_cond + 0.5 * loss_comp

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds_cond = torch.argmax(cond_logits, dim=1)
        correct_cond += (preds_cond == cond_targets).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, correct_cond / total_samples

def validate(model, loader, criterion_cond, criterion_comp, device):
    model.eval()
    total_loss = 0.0
    all_cond_preds = []
    all_cond_targets = []
    all_comp_preds = []
    all_comp_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            cond_targets = batch["condition"].to(device)
            comp_targets = batch["component"].to(device)

            cond_logits, comp_logits = model(images)
            loss_cond = criterion_cond(cond_logits, cond_targets)
            loss_comp = criterion_comp(comp_logits, comp_targets)
            loss = loss_cond + 0.5 * loss_comp

            total_loss += loss.item() * images.size(0)

            all_cond_preds.extend(torch.argmax(cond_logits, dim=1).cpu().numpy())
            all_cond_targets.extend(cond_targets.cpu().numpy())
            all_comp_preds.extend(torch.argmax(comp_logits, dim=1).cpu().numpy())
            all_comp_targets.extend(comp_targets.cpu().numpy())

    avg_loss = total_loss / len(all_cond_targets)
    cond_acc = accuracy_score(all_cond_targets, all_cond_preds)
    cond_f1 = f1_score(all_cond_targets, all_cond_preds, average="macro")
    comp_acc = accuracy_score(all_comp_targets, all_comp_preds)

    return avg_loss, cond_acc, cond_f1, comp_acc

def main():
    parser = argparse.ArgumentParser(description="Train EV Fault Dual Head Classifier")
    parser.add_argument(
        "--output_weights",
        type=str,
        default=str(MODELS_DIR / "best_mobilenet_v3_candidate_A.pth"),
        help="Path where experimental candidate weights should be saved."
    )
    parser.add_argument(
        "--broken_weight",
        type=float,
        default=1.415,
        help="Class weight for broken condition in CrossEntropyLoss (default derived from 92/65 train ratio)."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Candidate Training on Device: {device} ===")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_ds = EVFaultDataset(SPLITS_DIR / "train_manifest.csv", DATASET_BASE, is_train=True)
    val_ds = EVFaultDataset(SPLITS_DIR / "val_manifest.csv", DATASET_BASE, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
    print(f"Broken class loss weight: {args.broken_weight:.4f}")

    model = EVFaultDualHeadModel(pretrained=True).to(device)

    # Class-weighted condition loss (Experiment A)
    cond_weights = torch.tensor([1.0, args.broken_weight], dtype=torch.float32).to(device)
    criterion_cond = nn.CrossEntropyLoss(weight=cond_weights)
    criterion_comp = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    best_weights_path = Path(args.output_weights)
    best_weights_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # STAGE 1: Warmup - Freeze backbone, train heads only (6 epochs)
    # -------------------------------------------------------------
    print("\n--- Stage 1: Warmup (Frozen Backbone) ---")
    model.freeze_backbone(freeze=True)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=1e-4
    )

    for epoch in range(1, 7):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion_cond, criterion_comp, device)
        val_loss, val_acc, val_f1, val_comp_acc = validate(model, loader=val_loader, criterion_cond=criterion_cond, criterion_comp=criterion_comp, device=device)
        print(f"Warmup Epoch [{epoch}/6] - Train Loss: {train_loss:.4f}, Train Cond Acc: {train_acc:.3f} | Val Cond Acc: {val_acc:.3f}, Val Cond F1: {val_f1:.3f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_weights_path)

    # -------------------------------------------------------------
    # STAGE 2: Fine-Tuning - Unfreeze backbone (20 epochs)
    # -------------------------------------------------------------
    print("\n--- Stage 2: End-to-End Fine-Tuning ---")
    model.freeze_backbone(freeze=False)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": 1e-4},
            {"params": model.shared_fc.parameters(), "lr": 3e-4},
            {"params": model.condition_head.parameters(), "lr": 5e-4},
            {"params": model.component_head.parameters(), "lr": 5e-4},
        ],
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

    for epoch in range(1, 21):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion_cond, criterion_comp, device)
        val_loss, val_acc, val_f1, val_comp_acc = validate(model, val_loader, criterion_cond, criterion_comp, device)
        scheduler.step()

        is_best = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_weights_path)
            is_best = " [BEST SAVED]"

        print(f"FineTune Epoch [{epoch:2d}/20] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f}, Val Cond Acc: {val_acc:.3f}, Val Cond F1: {val_f1:.3f}, Val Comp Acc: {val_comp_acc:.3f}{is_best}")

    print(f"\nTraining Complete! Best Validation Condition F1: {best_val_f1:.4f}")
    print(f"Saved best model weights to: {best_weights_path}")

    # Save model metadata and config for candidate model (do not overwrite production model_config.json)
    config = {
        "model_architecture": "MobileNetV3-Small Dual-Head",
        "random_seed": SEED,
        "input_resolution": [224, 224],
        "condition_classes": INV_CONDITION_MAP,
        "component_classes": INV_COMPONENT_MAP,
        "best_val_condition_f1": round(best_val_f1, 4),
        "weights_file": best_weights_path.name
    }
    candidate_config_path = MODELS_DIR / f"{best_weights_path.stem}_config.json"
    with open(candidate_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved candidate config metadata to: {candidate_config_path}")

if __name__ == "__main__":
    main()
