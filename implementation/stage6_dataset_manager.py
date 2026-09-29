"""
stage6_dataset_manager.py — Stage 6: Real CK+ Dataset Loader, Subject-Disjoint 80:10:10 Split & Batch Feature Extraction
Paper: Quantum Information Entanglement Index for Facial Expression Analysis (ECML PKDD 2026)

This module handles:
- Ingestion of the real CK+48 dataset (extracted from archive.zip)
- Parsing subject IDs (e.g., S010, S011) to enforce strict subject-disjoint train/val/test splits
- Extracting 68 facial landmarks and 24D QIEI descriptors per sample
- Extracting classical baseline landmark coordinate features (136D)
- Persisting processed features into numpy arrays (X_qiei.npy, X_base.npy, y.npy, groups.npy)
- Verification checkpoint ensuring zero subject leakage across train, val, and test splits
"""

import sys
import os
import glob
import json
import numpy as np
import cv2

# Add implementation directory to path
sys.path.insert(0, os.path.dirname(__file__))
from stage2_landmark_detector import detect_landmarks
from stage5_descriptor_builder import build_qiei_descriptor, QIEI_FEATURE_NAMES

EMOTION_LABEL_MAP = {
    "anger": 0,
    "contempt": 1,
    "disgust": 2,
    "fear": 3,
    "happy": 4,
    "sadness": 5,
    "surprise": 6
}

LABEL_EMOTION_MAP = {v: k for k, v in EMOTION_LABEL_MAP.items()}


def scan_ckplus_dataset(data_dir: str) -> list:
    """
    Scans the CK+48 dataset directory and returns a list of sample dictionaries:
    [{'path': ..., 'emotion': ..., 'label': int, 'subject': ...}, ...]
    """
    samples = []
    for emotion_name, label_idx in EMOTION_LABEL_MAP.items():
        folder = os.path.join(data_dir, emotion_name)
        if not os.path.exists(folder):
            continue
        img_files = sorted(glob.glob(os.path.join(folder, "*.png")))
        for fpath in img_files:
            fname = os.path.basename(fpath)
            # Subject ID is the first prefix: S010_004_...
            subject_id = fname.split("_")[0]
            samples.append({
                "path": fpath,
                "emotion": emotion_name,
                "label": label_idx,
                "subject": subject_id
            })
    return samples


def create_subject_disjoint_splits(samples: list, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42) -> tuple:
    """
    Splits samples into Train (80%), Val (10%), Test (10%) sets such that
    NO subject appears in more than one partition (strict subject-disjoint protocol).
    """
    rng = np.random.RandomState(seed)
    
    # Collect unique subjects and their primary emotion distribution
    subjects = sorted(list(set(s["subject"] for s in samples)))
    rng.shuffle(subjects)
    
    num_subs = len(subjects)
    n_train = int(train_ratio * num_subs)
    n_val = int(val_ratio * num_subs)
    
    train_subs = set(subjects[:n_train])
    val_subs = set(subjects[n_train:n_train + n_val])
    test_subs = set(subjects[n_train + n_val:])
    
    train_samples = [s for s in samples if s["subject"] in train_subs]
    val_samples = [s for s in samples if s["subject"] in val_subs]
    test_samples = [s for s in samples if s["subject"] in test_subs]
    
    return train_samples, val_samples, test_samples, {
        "train_subjects": list(train_subs),
        "val_subjects": list(val_subs),
        "test_subjects": list(test_subs)
    }


def extract_features_from_samples(samples: list, max_samples: int = None, verbose: bool = True) -> tuple:
    """
    Extracts 24D QIEI descriptors and classical 136D normalized landmark coordinates
    for all provided samples.
    """
    if max_samples is not None:
        samples = samples[:max_samples]
        
    X_qiei = []
    X_base = []
    y = []
    subjects = []
    
    total = len(samples)
    if verbose:
        print(f"Extracting features from {total} real CK+ images...")
        
    for idx, s in enumerate(samples):
        img_path = s["path"]
        lms = detect_landmarks(img_path)  # shape (68, 2)
        
        # 1. Classical baseline: flattened L2-normalized 68 landmark coordinates (136D)
        lms_flat = lms.flatten().astype(np.float64)
        norm = np.linalg.norm(lms_flat)
        lms_norm = lms_flat / norm if norm > 1e-8 else lms_flat
        
        # 2. QIEI descriptor (24D)
        qiei_vec, _ = build_qiei_descriptor(lms)
        
        X_qiei.append(qiei_vec)
        X_base.append(lms_norm)
        y.append(s["label"])
        subjects.append(s["subject"])
        
        if verbose and ((idx + 1) % 50 == 0 or (idx + 1) == total):
            print(f"  Processed {idx + 1}/{total} samples ({(idx + 1)/total*100:.1f}%)")
            
    return (
        np.array(X_qiei, dtype=np.float64),
        np.array(X_base, dtype=np.float64),
        np.array(y, dtype=np.int64),
        np.array(subjects)
    )


def run_stage6_checkpoint(max_samples_for_test: int = 120):
    """Stage 6 Test & Verification Checkpoint."""
    print("=" * 65)
    print("STAGE 6: CK+ DATASET MANAGER & SUBJECT-DISJOINT SPLIT CHECKPOINT")
    print("=" * 65)
    
    data_dir = os.path.join(os.path.dirname(__file__), "data", "CK+48")
    assert os.path.exists(data_dir), f"CK+ dataset directory not found at {data_dir}"
    
    samples = scan_ckplus_dataset(data_dir)
    print(f"  Found {len(samples)} total CK+ images across all 7 emotion classes")
    assert len(samples) > 0, "No samples found in CK+48 dataset"
    print(f"  [PASS] Test 1: Ingested {len(samples)} real CK+ images successfully")
    
    # 2. Subject-disjoint split test
    train_s, val_s, test_s, sub_dict = create_subject_disjoint_splits(samples, train_ratio=0.8, val_ratio=0.1, seed=42)
    
    set_train = set(sub_dict["train_subjects"])
    set_val = set(sub_dict["val_subjects"])
    set_test = set(sub_dict["test_subjects"])
    
    # Zero subject leakage check
    leak_train_val = set_train.intersection(set_val)
    leak_train_test = set_train.intersection(set_test)
    leak_val_test = set_val.intersection(set_test)
    
    assert len(leak_train_val) == 0, f"Subject leakage train-val: {leak_train_val}"
    assert len(leak_train_test) == 0, f"Subject leakage train-test: {leak_train_test}"
    assert len(leak_val_test) == 0, f"Subject leakage val-test: {leak_val_test}"
    print(f"  [PASS] Test 2: Subject-disjoint 80:10:10 split verified (Zero subject leakage)")
    print(f"         Train subjects: {len(set_train)} | Val subjects: {len(set_val)} | Test subjects: {len(set_test)}")
    
    # 3. Feature extraction verification on a representative subset
    out_dir = os.path.join(os.path.dirname(__file__), "artifacts", "stages")
    os.makedirs(out_dir, exist_ok=True)
    cache_qiei = os.path.join(out_dir, "stage6_X_qiei.npy")
    cache_base = os.path.join(out_dir, "stage6_X_base.npy")
    cache_y = os.path.join(out_dir, "stage6_y.npy")
    cache_subs = os.path.join(out_dir, "stage6_subjects.npy")
    
    # Extract features
    eval_samples = samples[:max_samples_for_test] if max_samples_for_test else samples
    X_qiei, X_base, y, subjects = extract_features_from_samples(eval_samples, verbose=False)
    
    assert X_qiei.shape == (len(eval_samples), 24), f"Unexpected X_qiei shape: {X_qiei.shape}"
    assert X_base.shape == (len(eval_samples), 136), f"Unexpected X_base shape: {X_base.shape}"
    assert len(y) == len(eval_samples), "Label length mismatch"
    print(f"  [PASS] Test 3: X_qiei shape: {X_qiei.shape} (strictly 24-dimensional)")
    print(f"  [PASS] Test 4: X_base shape: {X_base.shape} (strictly 136-dimensional)")
    
    # Save cache arrays
    np.save(cache_qiei, X_qiei)
    np.save(cache_base, X_base)
    np.save(cache_y, y)
    np.save(cache_subs, subjects)
    print(f"  [PASS] Test 5: Persisted feature arrays (X_qiei.npy, X_base.npy, y.npy, subjects.npy)")
    
    print("-" * 65)
    print("[STAGE 6 CHECKPOINT PASSED: DATASET & SPLITS VALIDATED]")
    print("=" * 65)


if __name__ == "__main__":
    run_stage6_checkpoint()
