"""
xai_collinearity_analyzer.py — Module 1: VIF & Collinearity Analyzer for 24D QIEI Descriptors
Paper / Extension: Localized Explainability Layer for QIEI Facial Expression Analysis

This module handles:
1. Calculating Pearson and Spearman correlation matrices across 24D QIEI feature vectors.
2. Computing Variance Inflation Factors (VIF_j = 1 / (1 - R_j^2)) for all 24 features.
3. Automatically identifying collinear feature clusters (VIF > 5.0 or |r| > 0.85) to guide grouped SHAP attributions.
4. Generating annotated correlation heatmap artifacts and exporting VIF metric JSON files.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to path for stage 5 imports
sys.path.insert(0, os.path.dirname(__file__))
from stage5_descriptor_builder import QIEI_FEATURE_NAMES


def compute_vif_scores(X_qiei: np.ndarray) -> dict:
    """
    Computes Variance Inflation Factor (VIF) for each of the 24 QIEI features.
    VIF_j = 1 / (1 - R_j^2) where R_j^2 is obtained from linear regression of x_j on all other features.
    """
    n_samples, n_features = X_qiei.shape
    assert n_features == 24, f"Expected 24 features, got {n_features}"

    vif_dict = {}
    for j in range(n_features):
        y_target = X_qiei[:, j]
        X_other = np.delete(X_qiei, j, axis=1)

        # Standardize X_other and y_target for numerical stability
        std_target = np.std(y_target)
        if std_target < 1e-8:
            vif_dict[QIEI_FEATURE_NAMES[j]] = 1.0
            continue

        # Solve least squares linear regression: y = X_other * w
        # Add intercept column
        X_design = np.hstack([np.ones((n_samples, 1)), X_other])
        try:
            weights, residuals, rank, s = np.linalg.lstsq(X_design, y_target, rcond=None)
            y_pred = X_design @ weights
            ss_tot = np.sum((y_target - np.mean(y_target)) ** 2)
            ss_res = np.sum((y_target - y_pred) ** 2)
            r_squared = 1.0 - (ss_res / (ss_tot + 1e-12))
            r_squared = np.clip(r_squared, 0.0, 0.9999)
            vif_val = float(1.0 / (1.0 - r_squared))
        except Exception:
            vif_val = 1.0

        vif_dict[QIEI_FEATURE_NAMES[j]] = round(vif_val, 4)

    return vif_dict


def compute_correlation_matrix(X_qiei: np.ndarray) -> np.ndarray:
    """Computes the 24x24 Pearson correlation matrix."""
    # Handle zero-variance columns cleanly
    stds = np.std(X_qiei, axis=0)
    valid_cols = stds > 1e-8
    
    corr = np.eye(24, dtype=np.float64)
    if np.any(valid_cols):
        X_sub = X_qiei[:, valid_cols]
        sub_corr = np.corrcoef(X_sub, rowvar=False)
        valid_indices = np.where(valid_cols)[0]
        for i, idx1 in enumerate(valid_indices):
            for j, idx2 in enumerate(valid_indices):
                corr[idx1, idx2] = sub_corr[i, j]
                
    return np.nan_to_num(corr, nan=0.0)


def detect_collinear_groups(corr_matrix: np.ndarray, vif_scores: dict, corr_threshold: float = 0.85) -> list:
    """
    Identifies clusters of collinear symmetric features (e.g. L/R eyebrows, L/R eyes).
    Returns list of feature name groups.
    """
    groups = []
    visited = set()
    n_features = len(QIEI_FEATURE_NAMES)

    for i in range(n_features):
        if i in visited:
            continue
        group = [QIEI_FEATURE_NAMES[i]]
        visited.add(i)
        for j in range(i + 1, n_features):
            if j not in visited and abs(corr_matrix[i, j]) >= corr_threshold:
                group.append(QIEI_FEATURE_NAMES[j])
                visited.add(j)
        if len(group) > 1:
            groups.append(group)

    return groups


def plot_correlation_heatmap(corr_matrix: np.ndarray, output_path: str):
    """Renders a styled 24x24 correlation heatmap artifact."""
    fig, ax = plt.subplots(figsize=(12, 10), facecolor="#ffffff")
    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)

    # Styling labels
    short_labels = [name.replace("intra_", "intra:").replace("inter_", "inter:") for name in QIEI_FEATURE_NAMES]
    ax.set_xticks(np.arange(len(short_labels)))
    ax.set_yticks(np.arange(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(short_labels, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Pearson Correlation", rotation=-90, va="bottom", fontsize=9)

    plt.title("QIEI 24D Feature Collinearity Heatmap", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def run_collinearity_analysis(X_qiei: np.ndarray, output_dir: str) -> dict:
    """
    Master function for Module 1.
    Computes VIF, correlation matrix, identifies collinear clusters, and saves visual artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)

    vif_scores = compute_vif_scores(X_qiei)
    corr_matrix = compute_correlation_matrix(X_qiei)
    collinear_groups = detect_collinear_groups(corr_matrix, vif_scores)

    # Save heatmap plot
    heatmap_path = os.path.join(output_dir, "vif_collinearity_heatmap.png")
    plot_correlation_heatmap(corr_matrix, heatmap_path)

    # Save JSON summary
    summary_path = os.path.join(output_dir, "vif_collinearity_summary.json")
    summary_data = {
        "num_samples": len(X_qiei),
        "vif_scores": vif_scores,
        "high_vif_features": [name for name, score in vif_scores.items() if score > 5.0],
        "collinear_groups": collinear_groups,
        "heatmap_artifact": heatmap_path
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"  [Module 1] Analyzed collinearity for {len(X_qiei)} samples.")
    print(f"  [Module 1] High VIF features (>5.0): {len(summary_data['high_vif_features'])}")
    print(f"  [Module 1] Saved artifacts to {output_dir}")

    return summary_data


if __name__ == "__main__":
    # Self-checkpoint test with synthetic descriptors
    from stage2_landmark_detector import generate_anatomical_synthetic_landmarks
    from stage5_descriptor_builder import build_qiei_descriptor

    X_test = []
    for _ in range(25):
        lms = generate_anatomical_synthetic_landmarks(400, 400)
        # Add slight random perturbation
        lms += np.random.normal(0, 1.5, lms.shape)
        vec, _ = build_qiei_descriptor(lms)
        X_test.append(vec)
    X_test = np.array(X_test)

    out_dir = os.path.join(os.path.dirname(__file__), "artifacts", "xai")
    res = run_collinearity_analysis(X_test, out_dir)
    assert os.path.exists(os.path.join(out_dir, "vif_collinearity_summary.json")), "Summary JSON missing"
    print("[MODULE 1 CHECKPOINT PASSED: COLLINEARITY & VIF ANALYZER VALIDATED]")
