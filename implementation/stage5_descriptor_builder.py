"""
stage5_descriptor_builder.py — Stage 5: 24D QIEI Descriptor Builder & FACS Action Unit Mapping
Paper: Quantum Information Entanglement Index for Facial Expression Analysis (ECML PKDD 2026)

This module handles:
- Assembling the canonical 24-dimensional QIEI descriptor:
    * 7 intra-region entropies (eyebrows, eyes, nose, jaw, mouth_averaged)
    * 17 inter-region pair entropies
- Mapping entanglement fluctuations to Facial Action Coding System (FACS) Action Units (AUs)
- Serializing structured JSON analysis outputs for downstream interpretability
- Self-verifying checkpoint tests
"""

import sys
import os
import json
import numpy as np

# Add implementation directory to path
sys.path.insert(0, os.path.dirname(__file__))
from stage4_quantum_engine import (
    INTER_REGION_PAIRS,
    compute_all_qiei_entropies
)

# Canonical 24-dimensional feature vector ordering
QIEI_FEATURE_NAMES = [
    # 7 Intra-region Entropies
    "intra_right_eyebrow",
    "intra_left_eyebrow",
    "intra_nose",
    "intra_right_eye",
    "intra_left_eye",
    "intra_jaw",
    "intra_mouth_averaged",
    
    # 17 Inter-region Entropies
    "inter_right_eyebrow__left_eyebrow",
    "inter_right_eyebrow__right_eye",
    "inter_left_eyebrow__left_eye",
    "inter_right_eye__left_eye",
    "inter_right_eyebrow__nose",
    "inter_left_eyebrow__nose",
    "inter_right_eye__nose",
    "inter_left_eye__nose",
    "inter_nose__mouth_outer",
    "inter_nose__mouth_inner",
    "inter_right_eyebrow__mouth_outer",
    "inter_left_eyebrow__mouth_outer",
    "inter_right_eye__mouth_outer",
    "inter_left_eye__mouth_outer",
    "inter_jaw__mouth_outer",
    "inter_jaw__nose",
    "inter_mouth_outer__mouth_inner"
]

# Anatomical FACS Action Unit mappings (Table 2 & Section 3.3)
FACS_AU_MAPPING = {
    "AU1_Inner_Brow_Raiser": ["inter_right_eyebrow__left_eyebrow"],
    "AU2_Outer_Brow_Raiser": ["intra_right_eyebrow", "intra_left_eyebrow"],
    "AU4_Brow_Lowerer": ["inter_right_eyebrow__nose", "inter_left_eyebrow__nose"],
    "AU6_Cheek_Raiser": ["inter_right_eye__mouth_outer", "inter_left_eye__mouth_outer"],
    "AU9_Nose_Wrinkler": ["intra_nose", "inter_right_eye__nose", "inter_left_eye__nose"],
    "AU12_Lip_Corner_Puller": ["intra_mouth_averaged", "inter_jaw__mouth_outer"],
    "AU20_Lip_Stretcher": ["inter_jaw__mouth_outer"],
    "AU25_Lips_Part": ["inter_mouth_outer__mouth_inner"],
    "AU26_Jaw_Drop": ["inter_jaw__mouth_outer", "intra_jaw"]
}


def build_qiei_descriptor(landmarks_68: np.ndarray) -> tuple:
    """
    Constructs the canonical 24D QIEI feature vector from 68 landmarks.
    
    Returns:
        feature_vector: 1D numpy array of shape (24,) with float64 values
        feature_dict: dictionary of named feature names mapped to scalar values
    """
    entropies = compute_all_qiei_entropies(landmarks_68)
    intra = entropies["intra"]
    inter = entropies["inter"]
    
    feature_dict = {}
    
    # 7 Intra-region features
    feature_dict["intra_right_eyebrow"] = intra["right_eyebrow"]
    feature_dict["intra_left_eyebrow"] = intra["left_eyebrow"]
    feature_dict["intra_nose"] = intra["nose"]
    feature_dict["intra_right_eye"] = intra["right_eye"]
    feature_dict["intra_left_eye"] = intra["left_eye"]
    feature_dict["intra_jaw"] = intra["jaw"]
    feature_dict["intra_mouth_averaged"] = intra["mouth_averaged"]
    
    # 17 Inter-region features
    for r1, r2 in INTER_REGION_PAIRS:
        pair_key = f"{r1}__x__{r2}"
        desc_key = f"inter_{r1}__{r2}"
        feature_dict[desc_key] = inter[pair_key]
        
    feature_vector = np.array([feature_dict[name] for name in QIEI_FEATURE_NAMES], dtype=np.float64)
    return feature_vector, feature_dict


def map_descriptor_to_facs(feature_dict: dict) -> dict:
    """
    Computes surrogate FACS Action Unit activation intensities
    by aggregating the corresponding QIEI entanglement metrics.
    """
    au_scores = {}
    for au_name, feature_keys in FACS_AU_MAPPING.items():
        vals = [feature_dict[k] for k in feature_keys if k in feature_dict]
        au_scores[au_name] = float(np.mean(vals)) if len(vals) > 0 else 0.0
    return au_scores


def run_stage5_checkpoint():
    """Stage 5 Test & Verification Checkpoint."""
    print("=" * 65)
    print("STAGE 5: 24D QIEI DESCRIPTOR BUILDER & FACS MAPPING CHECKPOINT")
    print("=" * 65)
    
    from stage2_landmark_detector import generate_anatomical_synthetic_landmarks
    lms = generate_anatomical_synthetic_landmarks(400, 400)
    
    vec, feat_dict = build_qiei_descriptor(lms)
    
    # 1. Shape check
    assert vec.shape == (24,), f"Expected shape (24,), got {vec.shape}"
    print(f"  [PASS] Test 1: Feature vector shape is strictly (24,)")
    
    # 2. Bound checks
    # Intra features (0..6) must be in [0, 1]
    assert np.all(vec[:7] >= 0.0) and np.all(vec[:7] <= 1.0), "Intra features out of bounds [0, 1]"
    print(f"  [PASS] Test 2: All 7 intra-region features within [0.0, 1.0]")
    
    # Inter features (7..23) must be in [0, 3]
    assert np.all(vec[7:] >= 0.0) and np.all(vec[7:] <= 3.0), "Inter features out of bounds [0, 3]"
    print(f"  [PASS] Test 3: All 17 inter-region features within [0.0, 3.0]")
    
    # 3. FACS mapping check
    au_scores = map_descriptor_to_facs(feat_dict)
    assert len(au_scores) == len(FACS_AU_MAPPING), "FACS AU mapping size mismatch"
    print(f"  [PASS] Test 4: Mapped all {len(au_scores)} FACS Action Units successfully")
    
    # 4. Serialize sample JSON artifact
    out_dir = os.path.join(os.path.dirname(__file__), "artifacts", "stages")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "stage5_facs_sample_output.json")
    output_payload = {
        "descriptor_length": len(vec),
        "feature_names": QIEI_FEATURE_NAMES,
        "feature_values": vec.tolist(),
        "facs_action_units": au_scores
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    assert os.path.exists(json_path), "Failed to save sample JSON artifact"
    print(f"  [PASS] Test 5: Saved sample FACS JSON artifact: {json_path}")
    
    print("-" * 65)
    print("[STAGE 5 CHECKPOINT PASSED: 24D DESCRIPTOR & FACS MAPPING VALIDATED]")
    print("=" * 65)


if __name__ == "__main__":
    run_stage5_checkpoint()
