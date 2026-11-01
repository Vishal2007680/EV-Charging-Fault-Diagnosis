import os
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

CONDITION_MAP = {"healthy": 0, "broken": 1}
INV_CONDITION_MAP = {0: "healthy", 1: "broken"}

COMPONENT_MAP = {"charger": 0, "cord": 1, "plug": 2, "screen": 3}
INV_COMPONENT_MAP = {0: "charger", 1: "cord", 2: "plug", 3: "screen"}

def get_transforms(is_train: bool = True, img_size: int = 224):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])

class EVFaultDataset(Dataset):
    def __init__(self, manifest_csv: str | Path, base_dir: str | Path, is_train: bool = False, img_size: int = 224):
        self.manifest_path = Path(manifest_csv)
        self.base_dir = Path(base_dir)
        self.df = pd.read_csv(self.manifest_path)
        self.transform = get_transforms(is_train=is_train, img_size=img_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = str(row['image_path']).replace('\\', '/')
        if 'flare_classification/' in rel_path:
            rel_path = rel_path.split('flare_classification/')[-1]

        full_path = self.base_dir / rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")

        image = Image.open(full_path).convert("RGB")
        image_tensor = self.transform(image)

        cond_label = CONDITION_MAP[str(row['condition']).lower().strip()]
        comp_label = COMPONENT_MAP[str(row['component']).lower().strip()]

        return {
            "image": image_tensor,
            "condition": torch.tensor(cond_label, dtype=torch.long),
            "component": torch.tensor(comp_label, dtype=torch.long),
            "image_path": str(full_path)
        }
