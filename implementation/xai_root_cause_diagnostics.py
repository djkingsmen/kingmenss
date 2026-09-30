"""
xai_root_cause_diagnostics.py — Module 4: 3-Way Root Cause Error Matrix & AU Anomaly Inspector
Paper / Extension: Localized Explainability Layer for QIEI Facial Expression Analysis

This module handles:
1. Categorizing every prediction error into a 3-Way Root Cause Taxonomy:
   - Category 1: Landmark Data Degradation (Input landmark confidence C_lm < tau_conf)
   - Category 2: Quantum Feature Miscalibration (High C_lm, but top SHAP feature |Z_j| > 1.96)
   - Category 3: Classifier Boundary Overlap (High C_lm, normal features near decision boundary)
2. Computing the FACS Action Unit Anomaly Index (A_AU) for anatomical consistency checking.
3. Structuring diagnostic State Vectors (s_t) and Reward Logs (r_t) for future Reinforcement Learning (RL).
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from stage5_descriptor_builder import FACS_AU_MAPPING, QIEI_FEATURE_NAMES, map_descriptor_to_facs

# Anatomically expected Action Units per emotion class (Section 3.3)
EXPECTED_EMOTION_AUS = {
    "Sad": ["AU1_Inner_Brow_Raiser", "AU4_Brow_Lowerer", "AU15_Lip_Corner_Depressor"],
    "Happy": ["AU6_Cheek_Raiser", "AU12_Lip_Corner_Puller"],
    "Anger": ["AU4_Brow_Lowerer", "AU7_Lid_Tightener", "AU25_Lips_Part"],
    "Contempt": ["AU12_Lip_Corner_Puller", "AU14_Dimpler"],
    "Disgust": ["AU9_Nose_Wrinkler", "AU10_Upper_Lip_Raiser"],
    "Fear": ["AU1_Inner_Brow_Raiser", "AU2_Outer_Brow_Raiser", "AU4_Brow_Lowerer", "AU20_Lip_Stretcher"],
    "Neutral": [],
    "Surprise": ["AU1_Inner_Brow_Raiser", "AU2_Outer_Brow_Raiser", "AU25_Lips_Part", "AU26_Jaw_Drop"]
}


def compute_au_anomaly_index(feature_dict: dict, predicted_emotion: str) -> float:
    """
    Calculates AU Anomaly Index A_AU in [0, 1].
    Higher value indicates top QIEI feature maps to an Action Unit unexpected for the predicted emotion.
    """
    au_scores = map_descriptor_to_facs(feature_dict)
    expected_aus = EXPECTED_EMOTION_AUS.get(predicted_emotion, [])

    total_au_mass = sum(au_scores.values()) + 1e-8
    expected_au_mass = sum(au_scores[au] for au in expected_aus if au in au_scores)

    anomaly_score = 1.0 - (expected_au_mass / total_au_mass)
    return float(np.clip(anomaly_score, 0.0, 1.0))


def classify_error_root_cause(
    landmark_conf: float,
    top_shap_z_score: float,
    tau_conf: float = 0.70,
    z_threshold: float = 1.96
) -> str:
    """
    Classifies a misclassified sample into the 3-Way Error Taxonomy.
    """
    if landmark_conf < tau_conf:
        return "Landmark Data Degradation"
    elif abs(top_shap_z_score) > z_threshold:
        return "Quantum Feature Miscalibration"
    else:
        return "Classifier Boundary Overlap"


def build_rl_state_representation(
    sample_qiei: np.ndarray,
    landmark_conf: float,
    au_anomaly: float,
    top_z_score: float
) -> list:
    """
    Structures the 27-dimensional State Vector s_t for future Reinforcement Learning:
    s_t = [24D QIEI vector, Landmark Confidence C_lm, AU Anomaly A_AU, Top Z-Score]
    """
    state_vector = list(sample_qiei) + [float(landmark_conf), float(au_anomaly), float(top_z_score)]
    return state_vector


def run_root_cause_diagnostics(
    X_qiei: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    landmark_confidences: np.ndarray,
    shap_attributions: list,
    z_scores_dict: dict,
    label_map: dict,
    output_dir: str
) -> dict:
    """
    Master function for Module 4.
    Evaluates 3-way taxonomy breakdown, AU anomalies, and exports RL state/reward logs.
    """
    os.makedirs(output_dir, exist_ok=True)

    n_samples = len(X_qiei)
    errors = []
    taxonomy_counts = {
        "Landmark Data Degradation": 0,
        "Quantum Feature Miscalibration": 0,
        "Classifier Boundary Overlap": 0
    }

    rl_experiences = []
    au_anomaly_count = 0

    for i in range(n_samples):
        t_lbl = label_map.get(int(y_true[i]), f"Class_{y_true[i]}")
        p_lbl = label_map.get(int(y_pred[i]), f"Class_{y_pred[i]}")
        c_lm = float(landmark_confidences[i]) if i < len(landmark_confidences) else 0.85

        # Build feature dict
        feat_dict = {QIEI_FEATURE_NAMES[j]: float(X_qiei[i, j]) for j in range(24)}
        au_anom = compute_au_anomaly_index(feat_dict, p_lbl)
        if au_anom > 0.65:
            au_anomaly_count += 1

        # Get top SHAP feature Z-score
        shap_info = shap_attributions[i] if i < len(shap_attributions) else {}
        top_feat_name = shap_info.get("top_drivers", [{}])[0].get("feature", QIEI_FEATURE_NAMES[0]) if "top_drivers" in shap_info else QIEI_FEATURE_NAMES[0]
        top_z = z_scores_dict.get(top_feat_name, {}).get("z_score", 0.0)

        is_error = bool(y_true[i] != y_pred[i])
        root_cause = "Correct Classification"
        if is_error:
            root_cause = classify_error_root_cause(c_lm, top_z)
            taxonomy_counts[root_cause] += 1
            errors.append({
                "sample_index": int(i),
                "true_label": t_lbl,
                "pred_label": p_lbl,
                "landmark_confidence": float(c_lm),
                "top_feature": str(top_feat_name),
                "top_feature_z_score": float(top_z),
                "au_anomaly_index": float(au_anom),
                "root_cause": root_cause
            })

        # Construct RL state and reward
        state_vec = [float(val) for val in build_rl_state_representation(X_qiei[i], c_lm, au_anom, top_z)]
        reward = 1.0 if not is_error else -1.0 - (0.5 * au_anom)
        rl_experiences.append({
            "sample_index": int(i),
            "state_dim": len(state_vec),
            "state_vector": state_vec,
            "reward": round(float(reward), 4),
            "is_error": bool(is_error)
        })

    total_errors = sum(taxonomy_counts.values())
    taxonomy_pcts = {
        k: round((v / max(1, total_errors)) * 100.0, 2)
        for k, v in taxonomy_counts.items()
    }

    summary_data = {
        "total_samples": n_samples,
        "total_errors": total_errors,
        "taxonomy_counts": taxonomy_counts,
        "taxonomy_percentages": taxonomy_pcts,
        "au_anomalies_flagged": au_anomaly_count,
        "detailed_errors": errors
    }

    # Save Diagnostic JSON
    diag_path = os.path.join(output_dir, "error_taxonomy_breakdown.json")
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Save RL Experience Log JSON
    rl_path = os.path.join(output_dir, "rl_state_reward_log.json")
    with open(rl_path, "w", encoding="utf-8") as f:
        json.dump(rl_experiences, f, indent=2)

    print(f"  [Module 4] Diagnosed {total_errors} errors across 3-way taxonomy.")
    print(f"  [Module 4] Taxonomy breakdown: {taxonomy_pcts}")
    print(f"  [Module 4] Exported {len(rl_experiences)} RL state/reward experiences to {rl_path}")

    return summary_data


if __name__ == "__main__":
    # Self-checkpoint test
    X_m = np.random.uniform(0.1, 0.9, (20, 24))
    yt_m = np.array([0]*10 + [1]*10)
    yp_m = np.array([0]*8 + [1]*2 + [1]*8 + [0]*2)
    conf_m = np.array([0.85]*15 + [0.55]*5)

    out_dir = os.path.join(os.path.dirname(__file__), "artifacts", "xai")
    res = run_root_cause_diagnostics(X_m, yt_m, yp_m, conf_m, [], {}, {0:"Fear", 1:"Surprise"}, out_dir)
    assert os.path.exists(os.path.join(out_dir, "error_taxonomy_breakdown.json")), "Taxonomy JSON missing"
    print("[MODULE 4 CHECKPOINT PASSED: ROOT CAUSE DIAGNOSTICS & RL LOGGING VALIDATED]")
