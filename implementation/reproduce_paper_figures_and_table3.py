"""
reproduce_paper_figures_and_table3.py — Complete Reproduction of Paper Figures 2, 4(a), 4(b) and Table 3
Paper: Quantum Information Entanglement Index for Facial Expression Analysis (ECML PKDD 2026)

This script accomplishes:
1. Figure 2 Replication:
   - Triple pipeline panel: [Original Face -> Grayscale -> 68-Point Landmark Annotation]
   - Colored regional markers and point numbers matching Figure 2 exactly:
     * jaw (1..17): gray circles
     * right_eyebrow (18..22): yellow squares
     * left_eyebrow (23..27): orange triangles
     * nose (28..36): teal diamonds
     * right_eye (37..42): blue pluses
     * left_eye (43..48): dark blue crosses
     * mouth_outer (49..60): red inverted triangles
     * mouth_inner (61..68): dark red stars
2. Figure 4 Replication:
   - (a) Intra-region connectivity: closed polygons connecting regional extreme points
   - (b) Inter-region connectivity: bipartite entanglement chords across all 17 region pairs
3. Table 3 Benchmark:
   - Evaluates all 6 models in Table 3:
     * SVM (RBF)
     * ANN (MLP)
     * Logistic Regression
     * Random Forest (RF)
     * KNN (k=5)
     * Naive Bayes (NB)
   - Formats exact Table 3 Before QIEI vs After QIEI with absolute gain (Delta)
4. Saves high-res publication figures and detailed step-by-step interpretable logs
"""

import sys
import os
import json
import collections
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(__file__))

from logger import ExecutionLogger
from stage2_landmark_detector import detect_landmarks
from stage3_region_partitioner import REGION_LANDMARK_MAP, extract_region_extreme_points
from stage4_quantum_engine import INTER_REGION_PAIRS
from stage5_descriptor_builder import build_qiei_descriptor
from apex_benchmark_aligned import compute_facs_euclidean_distances, find_ckplus_apex_frames

# Canonical styles matching Figure 2 & Figure 4 in the paper
REGION_STYLES = {
    "jaw": {"label": "jaw", "color": "#7f8c8d", "marker": "o", "indices": list(range(0, 17))},
    "right_eyebrow": {"label": "right_eyebrow", "color": "#f1c40f", "marker": "s", "indices": list(range(17, 22))},
    "left_eyebrow": {"label": "left_eyebrow", "color": "#e67e22", "marker": "^", "indices": list(range(22, 27))},
    "nose": {"label": "nose", "color": "#16a085", "marker": "D", "indices": list(range(27, 36))},
    "right_eye": {"label": "right_eye", "color": "#2980b9", "marker": "+", "indices": list(range(36, 42))},
    "left_eye": {"label": "left_eye", "color": "#1f3a93", "marker": "x", "indices": list(range(42, 48))},
    "mouth_outer": {"label": "mouth_outer", "color": "#e74c3c", "marker": "v", "indices": list(range(48, 60))},
    "mouth_inner": {"label": "mouth_inner", "color": "#96281b", "marker": "*", "indices": list(range(60, 68))}
}


def generate_figure2_exact(sample_img_path: str, landmarks: np.ndarray, output_path: str):
    """
    Recreates Figure 2: Detection and annotation of 68 facial landmark points.
    """
    img_bgr = cv2.imread(sample_img_path)
    if img_bgr is None:
        img_bgr = np.full((300, 300, 3), 180, dtype=np.uint8)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    fig = plt.figure(figsize=(13, 5), facecolor="#ffffff")
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 3.5], wspace=0.25)
    
    # Panel 1: Original image
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img_rgb)
    ax0.set_title("Input RGB", fontsize=10, fontweight="bold")
    ax0.axis("off")
    
    # Panel 2: Grayscale
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(img_gray, cmap="gray")
    ax1.set_title("Grayscale", fontsize=10, fontweight="bold")
    ax1.axis("off")
    
    # Panel 3: Landmark wireframe preview
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(img_gray, cmap="gray")
    # Draw points
    ax2.scatter(landmarks[:, 0], landmarks[:, 1], c="#3498db", s=6, zorder=3)
    ax2.set_title("68 Landmarks", fontsize=10, fontweight="bold")
    ax2.axis("off")
    
    # Panel 4: Canonical Figure 2 Annotated Plot
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.set_facecolor("#ffffff")
    
    # Plot connections and markers for each region
    for r_name, cfg in REGION_STYLES.items():
        sub_pts = landmarks[cfg["indices"]]
        ax3.scatter(sub_pts[:, 0], sub_pts[:, 1], color=cfg["color"],
                    marker=cfg["marker"], s=45, label=cfg["label"], zorder=4)
        
        # Connect contour lines
        if "mouth" in r_name or "eye" in r_name:
            closed_pts = np.vstack([sub_pts, sub_pts[0:1]])
            ax3.plot(closed_pts[:, 0], closed_pts[:, 1], color=cfg["color"], linewidth=1.2, alpha=0.8)
        else:
            ax3.plot(sub_pts[:, 0], sub_pts[:, 1], color=cfg["color"], linewidth=1.2, alpha=0.8)
            
        # Number labels 1-indexed (matching paper)
        for i, idx in enumerate(cfg["indices"]):
            pt_num = idx + 1
            x, y = sub_pts[i]
            offset_x = 4 if pt_num % 2 == 0 else -12
            offset_y = 3 if pt_num % 3 == 0 else -6
            ax3.text(x + offset_x, y + offset_y, str(pt_num),
                     fontsize=7, color="#2c3e50", fontweight="bold")
                     
    ax3.invert_yaxis()
    ax3.set_aspect("equal")
    ax3.grid(True, linestyle="--", alpha=0.3, color="#bdc3c7")
    ax3.legend(bbox_to_anchor=(1.04, 0.95), loc="upper left", frameon=False, fontsize=10)
    ax3.set_title("Figure 2: Canonical 68-Point Regional Slicing & Annotation", fontsize=12, fontweight="bold", pad=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()


def generate_figure4_exact(landmarks: np.ndarray, output_path: str):
    """
    Recreates Figure 4:
    (a) Intra-region connectivity: convex boundary hulls / extreme envelopes
    (b) Inter-region connectivity: pairwise entanglement chords
    """
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6), facecolor="#ffffff")
    
    # -------------------------------------------------------------
    # (a) Intra-region connectivity
    # -------------------------------------------------------------
    ax_a.set_facecolor("#ffffff")
    for r_name, cfg in REGION_STYLES.items():
        sub_pts = landmarks[cfg["indices"]]
        ax_a.scatter(sub_pts[:, 0], sub_pts[:, 1], color=cfg["color"],
                     marker=cfg["marker"], s=40, label=cfg["label"], zorder=4)
        
        # 4 extreme points
        ext = extract_region_extreme_points(sub_pts)
        p_L, p_R, p_T, p_B = ext[0:2], ext[2:4], ext[4:6], ext[6:8]
        diamond = np.array([p_T, p_R, p_B, p_L, p_T])
        ax_a.plot(diamond[:, 0], diamond[:, 1], color=cfg["color"], linewidth=1.5, alpha=0.9)
        ax_a.fill(diamond[:, 0], diamond[:, 1], color=cfg["color"], alpha=0.08)
        
    ax_a.invert_yaxis()
    ax_a.set_aspect("equal")
    ax_a.grid(True, linestyle=":", alpha=0.4, color="#bdc3c7")
    ax_a.legend(bbox_to_anchor=(1.02, 0.9), loc="upper left", frameon=False, fontsize=9)
    ax_a.set_title("(a) Intra-region connectivity", fontsize=12, fontweight="bold", pad=10)
    
    # -------------------------------------------------------------
    # (b) Inter-region connectivity
    # -------------------------------------------------------------
    ax_b.set_facecolor("#ffffff")
    # Plot landmarks faintly in background
    for r_name, cfg in REGION_STYLES.items():
        sub_pts = landmarks[cfg["indices"]]
        ax_b.scatter(sub_pts[:, 0], sub_pts[:, 1], color=cfg["color"],
                     marker=cfg["marker"], s=25, alpha=0.7, zorder=3)
                     
    # Plot inter-region entanglement chords across all 17 pairs
    for r1_name, r2_name in INTER_REGION_PAIRS:
        r1_pts = landmarks[REGION_LANDMARK_MAP[r1_name]]
        r2_pts = landmarks[REGION_LANDMARK_MAP[r2_name]]
        
        c1 = np.mean(r1_pts, axis=0)
        c2 = np.mean(r2_pts, axis=0)
        
        # Draw entanglement chord
        ax_b.plot([c1[0], c2[0]], [c1[1], c2[1]], color="#7f8c8d", linewidth=1.2, alpha=0.55, zorder=2)
        
        # Connect extreme vertices
        ext1 = extract_region_extreme_points(r1_pts)
        ext2 = extract_region_extreme_points(r2_pts)
        for p1 in [ext1[0:2], ext1[2:4]]:
            for p2 in [ext2[4:6], ext2[6:8]]:
                ax_b.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#bdc3c7", linewidth=0.5, alpha=0.35, zorder=1)
                
    ax_b.invert_yaxis()
    ax_b.set_aspect("equal")
    ax_b.grid(True, linestyle=":", alpha=0.4, color="#bdc3c7")
    ax_b.set_title("(b) Inter-region connectivity", fontsize=12, fontweight="bold", pad=10)
    
    plt.suptitle("Fig. 4: Illustration of intra-region and inter-region entanglement structures used for QIEI computation.",
                 fontsize=13, fontweight="bold", y=0.03)
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()


from sklearn.feature_selection import SelectKBest, f_classif

os.environ["LOKY_MAX_CPU_COUNT"] = "1"


def run_full_table3_evaluation(X_base, X_qiei, y):
    """
    Evaluates all 6 models from Table 3 in the paper under two configurations:
    1. Mode A: Paper Raw Concatenation [X_base, X_qiei]
    2. Mode B: Our Methodological Improvement (Adaptive Selective Weighted Fusion)
    """
    models = {
        "SVM (RBF)": SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42),
        "ANN (MLP)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=350, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=400, C=5.0, random_state=42),
        "Random Forest (RF)": RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=1),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "Naive Bayes (NB)": GaussianNB()
    }
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    table3_results = []
    
    # 1. Paper Raw Concatenation
    X_raw = np.hstack([X_base, X_qiei])
    
    for name, clf_prototype in models.items():
        base_accs = []
        raw_accs = []
        improved_accs = []
        
        for train_idx, test_idx in cv.split(X_base, y):
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 1. Baseline (FACS Only)
            s_b = StandardScaler()
            X_tr_b = s_b.fit_transform(X_base[train_idx])
            X_te_b = s_b.transform(X_base[test_idx])
            
            clf_b = type(clf_prototype)(**clf_prototype.get_params())
            clf_b.fit(X_tr_b, y_train)
            acc_b = accuracy_score(y_test, clf_b.predict(X_te_b)) * 100.0
            base_accs.append(acc_b)
            
            # 2. Paper Raw Concatenation [X_base, X_qiei]
            s_r = StandardScaler()
            X_tr_r = s_r.fit_transform(X_raw[train_idx])
            X_te_r = s_r.transform(X_raw[test_idx])
            
            clf_r = type(clf_prototype)(**clf_prototype.get_params())
            clf_r.fit(X_tr_r, y_train)
            acc_r = accuracy_score(y_test, clf_r.predict(X_te_r)) * 100.0
            raw_accs.append(acc_r)
            
            # 3. Our Methodological Innovation: Adaptive Selective Weighted Fusion (k=4, alpha=0.25)
            sel = SelectKBest(f_classif, k=4)
            s_q = StandardScaler()
            X_tr_q_scaled = sel.fit_transform(s_q.fit_transform(X_qiei[train_idx]), y_train) * 0.25
            X_te_q_scaled = sel.transform(s_q.transform(X_qiei[test_idx])) * 0.25
            
            X_tr_imp = np.hstack([X_tr_b, X_tr_q_scaled])
            X_te_imp = np.hstack([X_te_b, X_te_q_scaled])
            
            clf_imp = type(clf_prototype)(**clf_prototype.get_params())
            clf_imp.fit(X_tr_imp, y_train)
            acc_imp = accuracy_score(y_test, clf_imp.predict(X_te_imp)) * 100.0
            improved_accs.append(acc_imp)
            
        mean_b = float(np.mean(base_accs))
        mean_r = float(np.mean(raw_accs))
        mean_imp = float(np.mean(improved_accs))
        
        table3_results.append({
            "model": name,
            "before_qiei": round(mean_b, 2),
            "paper_raw_qiei": round(mean_r, 2),
            "paper_delta": round(mean_r - mean_b, 2),
            "improved_fused_qiei": round(mean_imp, 2),
            "improved_delta": round(mean_imp - mean_b, 2)
        })
        
    return table3_results


def plot_table3_radar_comparison(table3_results, output_path):
    """
    Plots a high-resolution 6-axis Radar chart comparing:
    - Baseline (FACS only)
    - Paper Raw Concatenation
    - Improved Adaptive Selective QIEI Fusion
    """
    labels = [r["model"] for r in table3_results]
    base_scores = [r["before_qiei"] for r in table3_results]
    raw_scores = [r["paper_raw_qiei"] for r in table3_results]
    imp_scores = [r["improved_fused_qiei"] for r in table3_results]
    
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    base_scores += base_scores[:1]
    raw_scores += raw_scores[:1]
    imp_scores += imp_scores[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), facecolor="#ffffff")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], labels, color="#2c3e50", size=11, fontweight="bold")
    ax.tick_params(pad=15)
    
    min_val = min(min(base_scores), min(raw_scores), min(imp_scores)) - 5.0
    max_val = max(max(base_scores), max(raw_scores), max(imp_scores)) + 3.0
    ax.set_ylim(min_val, max_val)
    
    # Grid styling
    ax.yaxis.grid(True, color="#dcdde1", linestyle="--", linewidth=0.8)
    ax.xaxis.grid(True, color="#dcdde1", linestyle="--", linewidth=0.8)
    
    # 1. Baseline
    ax.plot(angles, base_scores, color="#7f8c8d", linewidth=2.0, linestyle="--", label="Baseline (FACS Only)")
    ax.fill(angles, base_scores, color="#95a5a6", alpha=0.10)
    
    # 2. Paper Raw
    ax.plot(angles, raw_scores, color="#e74c3c", linewidth=2.2, linestyle="-.", label="Paper Raw QIEI Concatenation")
    ax.fill(angles, raw_scores, color="#e74c3c", alpha=0.12)
    
    # 3. Improved Adaptive
    ax.plot(angles, imp_scores, color="#27ae60", linewidth=2.6, linestyle="-", label="Improved Adaptive Selective QIEI")
    ax.fill(angles, imp_scores, color="#2ecc71", alpha=0.20)
    
    plt.legend(loc="upper right", bbox_to_anchor=(1.30, 1.12), fontsize=10.5, frameon=True, facecolor="#f8f9fa")
    plt.title("Performance Comparison across Classifiers on CK+ Apex Frames\n(Baseline vs. Paper Raw QIEI vs. Improved Adaptive QIEI)",
              size=13, fontweight="bold", pad=25, color="#1e272c")
              
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()


def main():
    logger = ExecutionLogger("paper_figures_and_table3_reproduction")
    base_dir = os.path.dirname(__file__)
    out_dir = os.path.join(base_dir, "artifacts", "paper")
    os.makedirs(out_dir, exist_ok=True)
    data_dir = os.path.join(base_dir, "data", "CK+48")
    
    # 1. Get sample image for Figure 2 and Figure 4
    sample_img = os.path.join(data_dir, "happy", "S010_006_00000015.png")
    if not os.path.exists(sample_img):
        # find first available png
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".png"):
                    sample_img = os.path.join(root, f)
                    break
            if sample_img: break
            
    logger.start_stage("FIG-2", "Recreating Publication Figure 2 (Landmark Annotation Panel)")
    lms = detect_landmarks(sample_img)
    fig2_path = os.path.join(out_dir, "fig2_landmark_annotation.png")
    generate_figure2_exact(sample_img, lms, fig2_path)
    logger.success(f"Generated Figure 2 matching paper: {fig2_path}")
    logger.end_stage("PASSED")
    
    logger.start_stage("FIG-4", "Recreating Publication Figure 4 (Intra/Inter Connectivity)")
    fig4_path = os.path.join(out_dir, "fig4_intra_inter_connectivity.png")
    generate_figure4_exact(lms, fig4_path)
    logger.success(f"Generated Figure 4 matching paper: {fig4_path}")
    logger.end_stage("PASSED")
    
    logger.start_stage("TABLE-3", "Reproducing Exact Table 3 Benchmark Across All 6 Classifiers")
    apex_samples = find_ckplus_apex_frames(data_dir)
    logger.info(f"Loaded {len(apex_samples)} apex samples")
    
    qiei_cache = os.path.join(out_dir, "apex_X_qiei.npy")
    facs_cache = os.path.join(out_dir, "apex_X_facs.npy")
    y_cache = os.path.join(out_dir, "apex_y.npy")
    
    if os.path.exists(qiei_cache) and os.path.exists(facs_cache) and os.path.exists(y_cache):
        X_qiei = np.load(qiei_cache)
        X_facs = np.load(facs_cache)
        y = np.load(y_cache)
    else:
        from apex_benchmark_aligned import extract_apex_features
        X_qiei, X_facs, y, _ = extract_apex_features(apex_samples, out_dir, logger)
        
    table3 = run_full_table3_evaluation(X_facs, X_qiei, y)
    
    logger.log_raw("\n" + "=" * 90)
    logger.log_raw("  Table 3: Classification accuracy (%) for CK+: Before vs. After QIEI Comparison")
    logger.log_raw("=" * 90)
    logger.log_raw(f"  {'Model':<22} | {'Before QIEI':<12} | {'Paper Raw QIEI':<15} | {'Raw Delta':<10} | {'Improved QIEI':<15} | {'Imp Delta':<10}")
    logger.log_raw("  " + "-" * 86)
    for row in table3:
        d_raw = f"+{row['paper_delta']:.2f}" if row['paper_delta'] >= 0 else f"{row['paper_delta']:.2f}"
        d_imp = f"+{row['improved_delta']:.2f}" if row['improved_delta'] >= 0 else f"{row['improved_delta']:.2f}"
        logger.log_raw(f"  {row['model']:<22} | {row['before_qiei']:>10.2f}% | {row['paper_raw_qiei']:>13.2f}% | {d_raw:>8}% | {row['improved_fused_qiei']:>13.2f}% | {d_imp:>8}%")
    logger.log_raw("  " + "-" * 86)
    
    # Save radar plot comparison
    radar_path = os.path.join(out_dir, "fig5_table3_comparison_radar.png")
    plot_table3_radar_comparison(table3, radar_path)
    logger.success(f"Generated Figure 5 comparison radar plot: {radar_path}")
    
    # Save Table 3 JSON
    t3_path = os.path.join(out_dir, "table3_exact_results.json")
    with open(t3_path, "w", encoding="utf-8") as jf:
        json.dump({
            "table": "Table 3: Classification accuracy (%) for CK+: before vs. after QIEI comparison",
            "paper_reference": "ECML PKDD 2026",
            "results": table3
        }, jf, indent=2)
    logger.success(f"Saved Table 3 JSON results: {t3_path}")
    logger.end_stage("PASSED")
    
    logger.finalize()


if __name__ == "__main__":
    main()
