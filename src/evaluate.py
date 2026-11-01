import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

from dataset import EVFaultDataset, INV_CONDITION_MAP, INV_COMPONENT_MAP
from model import EVFaultDualHeadModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "splits"
DATASET_BASE = PROJECT_ROOT / "dataset" / "flare_classification"
MODELS_DIR = PROJECT_ROOT / "models"
EVAL_DIR = PROJECT_ROOT / "evaluation_results"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Evaluate EV Fault Dual Head Classifier on Test Set")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(MODELS_DIR / "best_mobilenet_v3_candidate_A.pth"),
        help="Path to the model weights to evaluate. Defaults to candidate_A."
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="candidate_A",
        help="Prefix for saved evaluation metrics and plots (e.g., candidate_A, baseline)."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = Path(args.weights)
    print(f"=== Evaluating Checkpoint: {weights_path.name} on Unseen Test Set ({device}) ===")

    test_manifest = SPLITS_DIR / "test_manifest.csv"
    test_ds = EVFaultDataset(test_manifest, DATASET_BASE, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    model = EVFaultDualHeadModel(pretrained=False).to(device)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    all_cond_preds = []
    all_cond_probs = []
    all_cond_targets = []
    all_comp_preds = []
    all_comp_targets = []
    image_paths = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            cond_targets = batch["condition"]
            comp_targets = batch["component"]
            paths = batch["image_path"]

            cond_logits, comp_logits = model(images)
            cond_probs = torch.softmax(cond_logits, dim=1)

            all_cond_probs.extend(cond_probs.cpu().numpy())
            all_cond_preds.extend(torch.argmax(cond_logits, dim=1).cpu().numpy())
            all_cond_targets.extend(cond_targets.numpy())
            all_comp_preds.extend(torch.argmax(comp_logits, dim=1).cpu().numpy())
            all_comp_targets.extend(comp_targets.numpy())
            image_paths.extend(paths)

    # Condition Metrics
    cond_acc = accuracy_score(all_cond_targets, all_cond_preds)
    cond_f1_macro = f1_score(all_cond_targets, all_cond_preds, average="macro")
    cond_prec_macro = precision_score(all_cond_targets, all_cond_preds, average="macro", zero_division=0)
    cond_rec_macro = recall_score(all_cond_targets, all_cond_preds, average="macro", zero_division=0)

    cond_report = classification_report(
        all_cond_targets,
        all_cond_preds,
        target_names=["healthy", "broken"],
        output_dict=True,
        zero_division=0
    )
    cm_cond = confusion_matrix(all_cond_targets, all_cond_preds)

    # Component Metrics
    comp_acc = accuracy_score(all_comp_targets, all_comp_preds)
    comp_report = classification_report(
        all_comp_targets,
        all_comp_preds,
        target_names=["charger", "cord", "plug", "screen"],
        output_dict=True,
        zero_division=0
    )

    print("\n--- TEST EVALUATION RESULTS ---")
    print(f"Condition Accuracy:      {cond_acc:.4f} ({cond_acc*100:.1f}%)")
    print(f"Condition Macro F1:      {cond_f1_macro:.4f}")
    print(f"Condition Macro Precision: {cond_prec_macro:.4f}")
    print(f"Condition Macro Recall:    {cond_rec_macro:.4f}")
    print(f"Component Accuracy:      {comp_acc:.4f} ({comp_acc*100:.1f}%)")

    print("\nConfusion Matrix [Condition]:")
    print(f"            Pred Healthy   Pred Broken")
    print(f"Act Healthy      {cm_cond[0, 0]:2d}             {cm_cond[0, 1]:2d}")
    print(f"Act Broken       {cm_cond[1, 0]:2d}             {cm_cond[1, 1]:2d}")

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_cond, cmap="Blues", interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    classes = ["Healthy (0)", "Broken (1)"]
    ax.set(xticks=np.arange(2), yticks=np.arange(2),
           xticklabels=classes, yticklabels=classes,
           title="Test Set Confusion Matrix (Condition)",
           ylabel="Ground Truth", xlabel="Predicted")

    # Annotate numbers in heatmap
    thresh = cm_cond.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm_cond[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_cond[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    cm_plot_path = EVAL_DIR / f"{args.output_prefix}_confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()
    print(f"\nSaved confusion matrix plot to: {cm_plot_path}")

    # Identify and save failure cases
    failure_cases = []
    for idx in range(len(all_cond_targets)):
        act_c = all_cond_targets[idx]
        pred_c = all_cond_preds[idx]
        if act_c != pred_c:
            failure_cases.append({
                "image_path": Path(image_paths[idx]).name,
                "full_path": image_paths[idx],
                "actual_condition": INV_CONDITION_MAP[act_c],
                "predicted_condition": INV_CONDITION_MAP[pred_c],
                "confidence_healthy": round(float(all_cond_probs[idx][0]), 4),
                "confidence_broken": round(float(all_cond_probs[idx][1]), 4),
                "actual_component": INV_COMPONENT_MAP[all_comp_targets[idx]],
                "predicted_component": INV_COMPONENT_MAP[all_comp_preds[idx]]
            })

    failures_df = pd.DataFrame(failure_cases)
    fail_csv_path = EVAL_DIR / f"{args.output_prefix}_failure_cases.csv"
    failures_df.to_csv(fail_csv_path, index=False)
    print(f"Logged {len(failure_cases)} misclassifications to: {fail_csv_path}")

    # Save complete metrics json
    results = {
        "checkpoint_evaluated": str(weights_path.name),
        "test_sample_count": len(all_cond_targets),
        "condition_metrics": {
            "accuracy": round(cond_acc, 4),
            "macro_f1": round(cond_f1_macro, 4),
            "macro_precision": round(cond_prec_macro, 4),
            "macro_recall": round(cond_rec_macro, 4),
            "confusion_matrix": cm_cond.tolist(),
            "classification_report": cond_report
        },
        "component_metrics": {
            "accuracy": round(comp_acc, 4),
            "classification_report": comp_report
        },
        "misclassification_count": len(failure_cases)
    }

    metrics_json_path = EVAL_DIR / f"{args.output_prefix}_test_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved full test metrics to: {metrics_json_path}")
    print(f"Saved full test metrics to: {metrics_json_path}")

if __name__ == "__main__":
    main()
