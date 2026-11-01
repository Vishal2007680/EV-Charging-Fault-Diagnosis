import os
import csv
import random
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "flare_classification"
SPLITS_DIR = PROJECT_ROOT / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def normalize_rel_path(p: str) -> str:
    p = p.replace('\\', '/')
    if 'flare_classification/' in p:
        p = p.split('flare_classification/')[-1]
    return p.strip('/')

def main():
    print("=== Creating Leakage-Free Stratified Splits ===")
    
    # 1. Load metadata
    meta_df = pd.read_csv(DATASET_DIR / "metadata.csv", encoding="utf-8-sig")
    meta_df['image_path_norm'] = meta_df['image_path'].apply(normalize_rel_path)
    meta_paths = set(meta_df['image_path_norm'])
    
    # 2. Load duplicate logs
    exact_df = pd.read_csv(DATASET_DIR / "exact_duplicates.csv", encoding="utf-8-sig")
    near_df = pd.read_csv(DATASET_DIR / "near_duplicates.csv", encoding="utf-8-sig")
    
    # Collect only active duplicate pairs where both images exist in current dataset
    active_dup_pairs = set()
    for _, r in exact_df.iterrows():
        p1 = normalize_rel_path(r['original'])
        p2 = normalize_rel_path(r['duplicate'])
        if p1 in meta_paths and p2 in meta_paths and p1 != p2:
            pair = tuple(sorted([p1, p2]))
            active_dup_pairs.add(pair)
            
    for _, r in near_df.iterrows():
        p1 = normalize_rel_path(r['image_1'])
        p2 = normalize_rel_path(r['image_2'])
        if p1 in meta_paths and p2 in meta_paths and p1 != p2:
            pair = tuple(sorted([p1, p2]))
            active_dup_pairs.add(pair)
            
    print(f"Active duplicate pairs present in dataset: {len(active_dup_pairs)}")
    for p1, p2 in sorted(active_dup_pairs):
        print(f"  - {p1} <==> {p2}")

    # Build disjoint connected components for groups
    parent = {p: p for p in meta_paths}
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
        
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for p1, p2 in active_dup_pairs:
        union(p1, p2)
            
    meta_df['group_id'] = meta_df['image_path_norm'].apply(find)
    
    group_sizes = meta_df.groupby('group_id').size()
    multi_item_groups = group_sizes[group_sizes > 1].index.tolist()
    print(f"Active multi-image duplicate groups: {len(multi_item_groups)} (totaling {group_sizes[group_sizes > 1].sum()} images).")
    
    # Stratification strata: component_condition
    meta_df['strata'] = meta_df['component'] + "_" + meta_df['condition']
    
    # Assign all duplicate groups to Train
    train_indices = []
    remaining_indices = []
    
    for idx, r in meta_df.iterrows():
        if r['group_id'] in multi_item_groups:
            train_indices.append(idx)
        else:
            remaining_indices.append(idx)
            
    print(f"Assigned {len(train_indices)} duplicate images strictly to Training split.")
    
    # Target: 70% Train (~157), 15% Val (~34), 15% Test (~34)
    # Remaining to split: 221 unique images
    val_indices = []
    test_indices = []
    
    rem_df = meta_df.loc[remaining_indices]
    for strata, grp in rem_df.groupby('strata'):
        items = grp.index.tolist()
        random.shuffle(items)
        n = len(items)
        n_val = max(1, int(round(n * 0.155)))
        n_test = max(1, int(round(n * 0.155)))
        
        if n_val + n_test >= n:
            n_val = 1
            n_test = 1
            
        v_sub = items[:n_val]
        t_sub = items[n_val:n_val + n_test]
        tr_sub = items[n_val + n_test:]
        
        val_indices.extend(v_sub)
        test_indices.extend(t_sub)
        train_indices.extend(tr_sub)
        
    train_df = meta_df.loc[train_indices].copy().reset_index(drop=True)
    val_df = meta_df.loc[val_indices].copy().reset_index(drop=True)
    test_df = meta_df.loc[test_indices].copy().reset_index(drop=True)
    
    print(f"\nFinal Split Counts:")
    print(f"  Train: {len(train_df):3d} ({len(train_df)/len(meta_df):.1%})")
    print(f"  Val:   {len(val_df):3d} ({len(val_df)/len(meta_df):.1%})")
    print(f"  Test:  {len(test_df):3d} ({len(test_df)/len(meta_df):.1%})")
    print(f"  Total: {len(train_df) + len(val_df) + len(test_df)} (Original: {len(meta_df)})")
    
    # 3. Leakage & Group Verification
    train_set = set(train_df['image_path_norm'])
    val_set = set(val_df['image_path_norm'])
    test_set = set(test_df['image_path_norm'])
    
    inter_tr_val = train_set & val_set
    inter_tr_te = train_set & test_set
    inter_val_te = val_set & test_set
    
    assert len(inter_tr_val) == 0, f"LEAKAGE DETECTED between Train and Val: {inter_tr_val}"
    assert len(inter_tr_te) == 0, f"LEAKAGE DETECTED between Train and Test: {inter_tr_te}"
    assert len(inter_val_te) == 0, f"LEAKAGE DETECTED between Val and Test: {inter_val_te}"
    
    # Verify no duplicate pair across splits
    for p1, p2 in active_dup_pairs:
        s1 = 'train' if p1 in train_set else ('val' if p1 in val_set else 'test')
        s2 = 'train' if p2 in train_set else ('val' if p2 in val_set else 'test')
        assert s1 == s2, f"DUPLICATE LEAKAGE: {p1} in {s1} but {p2} in {s2}!"
        
    print("\n[VERIFICATION PASSED] Zero overlap and zero duplicate leakage between splits!")
    
    # 4. Save CSV manifests
    train_path = SPLITS_DIR / "train_manifest.csv"
    val_path = SPLITS_DIR / "val_manifest.csv"
    test_path = SPLITS_DIR / "test_manifest.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Manifests successfully saved:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    
    # Log verification details
    log_path = SPLITS_DIR / "split_verification.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== SPLIT VERIFICATION AUDIT LOG ===\n")
        f.write(f"Random Seed: {RANDOM_SEED}\n")
        f.write(f"Total Records: {len(meta_df)}\n")
        f.write(f"Train Count: {len(train_df)}\n")
        f.write(f"Val Count: {len(val_df)}\n")
        f.write(f"Test Count: {len(test_df)}\n")
        f.write("\nDistribution in Train:\n")
        f.write(train_df.groupby(['component', 'condition']).size().to_string())
        f.write("\n\nDistribution in Val:\n")
        f.write(val_df.groupby(['component', 'condition']).size().to_string())
        f.write("\n\nDistribution in Test:\n")
        f.write(test_df.groupby(['component', 'condition']).size().to_string())
        f.write("\n\nLEAKAGE CHECKS:\n")
        f.write(f"Train & Val intersection: {len(inter_tr_val)}\n")
        f.write(f"Train & Test intersection: {len(inter_tr_te)}\n")
        f.write(f"Val & Test intersection: {len(inter_val_te)}\n")
        f.write(f"Active duplicate pairs checked: {len(active_dup_pairs)}\n")
        f.write("All active duplicate pairs confined strictly to Training set: YES\n")
        f.write("STATUS: VERIFIED LEAKAGE-FREE\n")

if __name__ == "__main__":
    main()
