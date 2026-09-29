"""
apex_benchmark_aligned.py — Canonical CK+ Apex Benchmark Aligned to Paper Specifications
Paper: Quantum Information Entanglement Index for Facial Expression Analysis (ECML PKDD 2026)

Key Specifications:
1. Apex Frame Filtering:
   - Exactly 327 peak-expression apex frames from the CK+ dataset
2. FACS-Aligned Baseline Features:
   - Scale-invariant inter- and intra-region pairwise Euclidean landmark distances
   - Normalized by inter-ocular distance (outer eye corners p36-p45)
3. 24D QIEI Feature Augmentation:
   - 7 intra-region and 17 inter-region quantum entanglement entropy features
4. Strict 10-Fold Subject-Disjoint Cross-Validation:
   - GroupKFold over all unique subjects (0% subject leakage)
5. Tuned Classifier Hyperparameters:
   - RBF-SVM (C=10.0, gamma='scale')
   - Random Forest (n_estimators=200, max_depth=12)
   - Multi-Layer Perceptron (hidden_layer_sizes=(128, 64), max_iter=300)
6. Full persistent time logs, paired t-tests, and publication radar plots
"""

import sys
import os
import time
import json
import collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))

from logger import ExecutionLogger
from stage2_landmark_detector import detect_landmarks
from stage5_descriptor_builder import build_qiei_descriptor
from stage6_dataset_manager import (
    EMOTION_LABEL_MAP,
    LABEL_EMOTION_MAP
)


def compute_facs_euclidean_distances(landmarks_68: np.ndarray) -> np.ndarray:
    """
    Computes scale-invariant intra- and inter-region Euclidean distances between
    facial landmarks based on FACS principles (Sections 3 & 4.1).
    Normalized by inter-ocular distance (distance between eye corners 36 and 45).
    """
    pts = landmarks_68.astype(np.float64)
    
    # Scale normalization factor: inter-ocular distance (p36 to p45)
    iod = np.linalg.norm(pts[36] - pts[45])
    scale = iod if iod > 1e-6 else 1.0
    
    distances = []
    
    # 1. Eyebrows (AU1, AU2, AU4)
    # Inter-eyebrow distance (inner corners 21 and 22)
    distances.append(np.linalg.norm(pts[21] - pts[22]))
    # Eyebrow heights above eyes
    distances.append(np.linalg.norm(pts[19] - pts[37]))  # right mid-brow to eye
    distances.append(np.linalg.norm(pts[24] - pts[44]))  # left mid-brow to eye
    distances.append(np.linalg.norm(pts[17] - pts[36]))  # right outer brow to eye
    distances.append(np.linalg.norm(pts[26] - pts[45]))  # left outer brow to eye
    # Eyebrow to nose root (AU4 brow lowerer)
    distances.append(np.linalg.norm(pts[21] - pts[27]))
    distances.append(np.linalg.norm(pts[22] - pts[27]))
    
    # 2. Eyes (AU6 cheek raiser & squint)
    # Right eye vertical openings
    distances.append(np.linalg.norm(pts[37] - pts[41]))
    distances.append(np.linalg.norm(pts[38] - pts[40]))
    distances.append(np.linalg.norm(pts[36] - pts[39]))  # right eye width
    # Left eye vertical openings
    distances.append(np.linalg.norm(pts[43] - pts[47]))
    distances.append(np.linalg.norm(pts[44] - pts[46]))
    distances.append(np.linalg.norm(pts[42] - pts[45]))  # left eye width
    
    # 3. Nose
    distances.append(np.linalg.norm(pts[27] - pts[33]))  # nose length
    distances.append(np.linalg.norm(pts[31] - pts[35]))  # nose width
    
    # 4. Mouth (AU12 lip pull, AU20 stretcher, AU25 lips part, AU26 jaw drop)
    distances.append(np.linalg.norm(pts[48] - pts[54]))  # mouth width (lip corners)
    distances.append(np.linalg.norm(pts[51] - pts[57]))  # outer mouth height
    distances.append(np.linalg.norm(pts[62] - pts[66]))  # inner lip separation (parting)
    distances.append(np.linalg.norm(pts[60] - pts[64]))  # inner mouth width
    
    # 5. Cross-Region (AU6, AU9, AU12, AU26)
    # Mouth corners to nose tip
    distances.append(np.linalg.norm(pts[48] - pts[33]))  # right mouth corner to nose
    distances.append(np.linalg.norm(pts[54] - pts[33]))  # left mouth corner to nose
    # Mouth corners to outer eye corners
    distances.append(np.linalg.norm(pts[48] - pts[36]))  # right mouth to right eye
    distances.append(np.linalg.norm(pts[54] - pts[45]))  # left mouth to left eye
    # Mouth to jaw / chin (AU26 jaw drop)
    distances.append(np.linalg.norm(pts[57] - pts[8]))   # lower lip to chin
    distances.append(np.linalg.norm(pts[66] - pts[8]))   # inner lower lip to chin
    # Jaw width
    distances.append(np.linalg.norm(pts[0] - pts[16]))
    
    return np.array(distances, dtype=np.float64) / scale


def find_ckplus_apex_frames(data_dir: str) -> list:
    """
    Scans the CK+48 dataset and selects exclusively the apex (peak expression)
    frame for each recorded subject sequence.
    """
    sequences = collections.defaultdict(list)
    for emotion_name, label_idx in EMOTION_LABEL_MAP.items():
        folder = os.path.join(data_dir, emotion_name)
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            if f.endswith(".png"):
                parts = f.split("_")
                if len(parts) >= 3:
                    sub_id = parts[0]
                    sub_seq = f"{parts[0]}_{parts[1]}"
                    frame_num = int(parts[2].replace(".png", ""))
                    full_path = os.path.join(folder, f)
                    sequences[sub_seq].append({
                        "frame_num": frame_num,
                        "path": full_path,
                        "emotion": emotion_name,
                        "label": label_idx,
                        "subject": sub_id
                    })
                    
    apex_samples = []
    for sub_seq, frame_list in sequences.items():
        frame_list.sort(key=lambda x: x["frame_num"])
        apex_samples.append(frame_list[-1])  # Max frame index = Apex
        
    return apex_samples


def extract_apex_features(apex_samples: list, cache_dir: str, logger: ExecutionLogger):
    """
    Extracts FACS Euclidean distance baseline features and 24D QIEI descriptors
    for all 327 apex frames.
    """
    qiei_path = os.path.join(cache_dir, "apex_X_qiei.npy")
    facs_path = os.path.join(cache_dir, "apex_X_facs.npy")
    y_path = os.path.join(cache_dir, "apex_y.npy")
    subs_path = os.path.join(cache_dir, "apex_subjects.npy")
    
    if (os.path.exists(qiei_path) and os.path.exists(facs_path) and 
        os.path.exists(y_path) and os.path.exists(subs_path)):
        logger.info("Loading cached apex feature matrices...")
        X_qiei = np.load(qiei_path)
        X_facs = np.load(facs_path)
        y = np.load(y_path)
        subjects = np.load(subs_path)
        if len(y) == len(apex_samples):
            logger.success(f"Loaded {len(y)} apex samples from cache (X_qiei: {X_qiei.shape}, X_facs: {X_facs.shape})")
            return X_qiei, X_facs, y, subjects
            
    logger.info(f"Extracting features from {len(apex_samples)} apex frames...")
    X_qiei_list = []
    X_facs_list = []
    y_list = []
    subs_list = []
    
    t0 = time.perf_counter()
    for i, s in enumerate(apex_samples):
        lms = detect_landmarks(s["path"])
        facs_vec = compute_facs_euclidean_distances(lms)
        qiei_vec, _ = build_qiei_descriptor(lms)
        
        X_facs_list.append(facs_vec)
        X_qiei_list.append(qiei_vec)
        y_list.append(s["label"])
        subs_list.append(s["subject"])
        
        if (i + 1) % 50 == 0 or (i + 1) == len(apex_samples):
            elapsed = time.perf_counter() - t0
            logger.info(f"  Processed {i + 1}/{len(apex_samples)} apex frames ({elapsed:.1f}s)")
            
    X_qiei = np.array(X_qiei_list, dtype=np.float64)
    X_facs = np.array(X_facs_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int64)
    subjects = np.array(subs_list)
    
    np.save(qiei_path, X_qiei)
    np.save(facs_path, X_facs)
    np.save(y_path, y)
    np.save(subs_path, subjects)
    logger.success(f"Saved {len(y)} apex features to {cache_dir}")
    
    return X_qiei, X_facs, y, subjects


def run_aligned_apex_10fold_cv(X_base, X_qiei, y, subjects, logger: ExecutionLogger):
    """
    Executes 10-Fold Subject-Disjoint Cross-Validation on Apex frames.
    Evaluates RBF-SVM, Random Forest, and MLP before and after QIEI augmentation.
    """
    gkf = GroupKFold(n_splits=10)
    X_aug = np.hstack([X_base, X_qiei])
    
    models = {
        "RBF_SVM": lambda: SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42),
        "Random_Forest": lambda: RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42),
        "MLP": lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=350, random_state=42)
    }
    
    fold_results = {m: {"base": [], "aug": []} for m in models}
    confusion_matrices = {m: {"base": np.zeros((7, 7)), "aug": np.zeros((7, 7))} for m in models}
    
    logger.info("Running 10-Fold Subject-Disjoint CV on Apex Frames...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_base, y, groups=subjects)):
        t_fold = time.perf_counter()
        
        # Verify 0% subject leakage
        leak = set(subjects[train_idx]).intersection(set(subjects[test_idx]))
        assert len(leak) == 0, f"Subject leakage in fold {fold_idx + 1}"
        
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scaling
        scaler_b = StandardScaler()
        X_tr_b = scaler_b.fit_transform(X_base[train_idx])
        X_te_b = scaler_b.transform(X_base[test_idx])
        
        scaler_a = StandardScaler()
        X_tr_a = scaler_a.fit_transform(X_aug[train_idx])
        X_te_a = scaler_a.transform(X_aug[test_idx])
        
        summary_parts = []
        for m_name, m_fn in models.items():
            # Base
            clf_b = m_fn()
            clf_b.fit(X_tr_b, y_train)
            pred_b = clf_b.predict(X_te_b)
            acc_b = accuracy_score(y_test, pred_b) * 100.0
            fold_results[m_name]["base"].append(acc_b)
            confusion_matrices[m_name]["base"] += confusion_matrix(y_test, pred_b, labels=list(range(7)))
            
            # Aug
            clf_a = m_fn()
            clf_a.fit(X_tr_a, y_train)
            pred_a = clf_a.predict(X_te_a)
            acc_a = accuracy_score(y_test, pred_a) * 100.0
            fold_results[m_name]["aug"].append(acc_a)
            confusion_matrices[m_name]["aug"] += confusion_matrix(y_test, pred_a, labels=list(range(7)))
            
            summary_parts.append(f"{m_name}: {acc_b:.1f}% -> {acc_a:.1f}% ({acc_a - acc_b:+.1f}%)")
            
        dur = time.perf_counter() - t_fold
        logger.info(f"  Fold {fold_idx + 1:2d}/10 ({dur:.2f}s) | " + " | ".join(summary_parts))
        
    return fold_results, confusion_matrices


def plot_apex_visualizations(fold_results, confusion_matrices, out_dir, logger: ExecutionLogger):
    """
    Renders the publication Figure 5 radar plot and confusion matrices for the apex benchmark.
    """
    categories = list(EMOTION_LABEL_MAP.keys())
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    # 1. Radar Plot using RBF_SVM (or best model)
    cm_base = confusion_matrices["RBF_SVM"]["base"]
    cm_aug = confusion_matrices["RBF_SVM"]["aug"]
    
    rec_b = [cm_base[i, i] / max(1, np.sum(cm_base[i, :])) for i in range(7)]
    rec_a = [cm_aug[i, i] / max(1, np.sum(cm_aug[i, :])) for i in range(7)]
    
    vals_b = rec_b + rec_b[:1]
    vals_a = rec_a + rec_a[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor("#161625")
    fig.patch.set_facecolor("#0d0d16")
    
    ax.plot(angles, vals_b, color="#e74c3c", linewidth=2, linestyle="dashed", label="Baseline (FACS Euclidean)")
    ax.fill(angles, vals_b, color="#e74c3c", alpha=0.18)
    
    ax.plot(angles, vals_a, color="#2ecc71", linewidth=2.5, label="QIEI Augmented (+Quantum Entanglement)")
    ax.fill(angles, vals_a, color="#2ecc71", alpha=0.28)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.upper() for c in categories], color="#ffffff", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#bdc3c7")
    ax.grid(color="#2c3e50", linestyle="dotted")
    ax.set_ylim(0.0, 1.0)
    
    plt.title("Figure 5: Aligned CK+ Apex Benchmark (10-Fold Subject-Disjoint CV)\nBaseline (FACS Distances) vs. QIEI Augmented",
              color="#ffffff", fontsize=13, fontweight="bold", pad=25)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), facecolor="#1f2438", edgecolor="none", labelcolor="#ffffff")
    
    radar_path = os.path.join(out_dir, "apex_aligned_radar_fig5.png")
    plt.tight_layout()
    plt.savefig(radar_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.success(f"Saved aligned radar plot: {radar_path}")
    
    # 2. Confusion Matrices Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0d0d16")
    
    for ax, cm, title in zip(axes, [cm_base, cm_aug], ["Baseline (FACS Euclidean)", "QIEI Augmented (+Quantum Entanglement)"]):
        ax.set_facecolor("#161625")
        cm_norm = cm.astype('float') / np.maximum(1, cm.sum(axis=1)[:, np.newaxis])
        im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.viridis, vmin=0.0, vmax=1.0)
        ax.set_title(f"RBF-SVM (10-Fold CV on Apex)\n{title}", color="#ffffff", fontsize=12, fontweight="bold", pad=12)
        
        ticks = np.arange(len(categories))
        ax.set_xticks(ticks)
        ax.set_xticklabels([c[:4].upper() for c in categories], color="#ecf0f1", fontsize=9)
        ax.set_yticks(ticks)
        ax.set_yticklabels([c[:4].upper() for c in categories], color="#ecf0f1", fontsize=9)
        
        ax.set_ylabel("True Emotion", color="#ecf0f1", fontsize=10)
        ax.set_xlabel("Predicted Emotion", color="#ecf0f1", fontsize=10)
        
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f"{cm_norm[i, j]*100:.0f}%",
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] < 0.6 else "black",
                        fontsize=8, fontweight="bold")
                        
    cm_path = os.path.join(out_dir, "apex_aligned_confusion_matrices.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.success(f"Saved aligned confusion matrices: {cm_path}")


def main():
    logger = ExecutionLogger("ckplus_apex_aligned_benchmark")
    base_dir = os.path.dirname(__file__)
    out_dir = os.path.join(base_dir, "artifacts", "apex")
    os.makedirs(out_dir, exist_ok=True)
    data_dir = os.path.join(base_dir, "data", "CK+48")
    
    # 1. Filter to apex frames
    logger.start_stage("STAGE-APEX-FILTER", "Filtering 981 Frames to 327 Canonical Apex Expressions")
    apex_samples = find_ckplus_apex_frames(data_dir)
    logger.info(f"Identified {len(apex_samples)} apex frames across {len(set(s['subject'] for s in apex_samples))} subjects")
    emo_counts = collections.Counter([s["emotion"] for s in apex_samples])
    for emo, cnt in emo_counts.items():
        logger.info(f"  - {emo:<10}: {cnt} peak expression images")
    logger.end_stage(status="PASSED", metadata={"apex_count": len(apex_samples)})
    
    # 2. Extract features
    logger.start_stage("STAGE-APEX-FEATURES", "Extracting FACS Euclidean Baseline & 24D QIEI Features")
    X_qiei, X_facs, y, subjects = extract_apex_features(apex_samples, out_dir, logger)
    logger.end_stage(status="PASSED", metadata={"facs_dim": X_facs.shape[1], "qiei_dim": X_qiei.shape[1]})
    
    # 3. 10-Fold CV
    logger.start_stage("STAGE-10FOLD-APEX", "10-Fold Subject-Disjoint CV on Apex Frames (60 Models)")
    fold_results, confusion_matrices = run_aligned_apex_10fold_cv(X_facs, X_qiei, y, subjects, logger)
    logger.end_stage(status="PASSED")
    
    # 4. Statistical significance & reporting
    logger.start_stage("STAGE-APEX-STATS", "Paired t-test Hypothesis Testing & Statistical Significance")
    
    logger.log_raw("\n" + "=" * 85)
    logger.log_raw("       FINAL ALIGNED CK+ APEX 10-FOLD SUBJECT-DISJOINT BENCHMARK RESULTS")
    logger.log_raw("                       (327 Apex Frames, 118 Subjects)")
    logger.log_raw("=" * 85)
    logger.log_raw(f"  {'Classifier':<16} | {'Baseline Acc (FACS Distances)':<28} | {'QIEI Augmented':<20} | {'Gain':<8} | {'p-value':<9}")
    logger.log_raw("  " + "-" * 89)
    
    final_stats = {}
    for model_name in ["RBF_SVM", "Random_Forest", "MLP"]:
        b_scores = fold_results[model_name]["base"]
        a_scores = fold_results[model_name]["aug"]
        
        b_mean, b_std = float(np.mean(b_scores)), float(np.std(b_scores))
        a_mean, a_std = float(np.mean(a_scores)), float(np.std(a_scores))
        gain = a_mean - b_mean
        
        t_stat, p_val = stats.ttest_rel(a_scores, b_scores)
        
        final_stats[model_name] = {
            "baseline_mean": b_mean,
            "baseline_std": b_std,
            "qiei_mean": a_mean,
            "qiei_std": a_std,
            "gain": gain,
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "significant": bool(p_val < 0.05)
        }
        
        b_str = f"{b_mean:5.2f}% +/- {b_std:4.2f}%"
        a_str = f"{a_mean:5.2f}% +/- {a_std:4.2f}%"
        gain_str = f"{gain:+5.2f}%"
        sig_str = f"{p_val:.4f} *" if p_val < 0.05 else f"{p_val:.4f}"
        
        logger.log_raw(f"  {model_name:<16} | {b_str:<28} | {a_str:<20} | {gain_str:<8} | {sig_str:<9}")
        
    logger.log_raw("  " + "-" * 89)
    logger.log_raw("  * Statistically significant at p < 0.05\n")
    logger.end_stage(status="PASSED", metadata=final_stats)
    
    # 5. Visualizations
    logger.start_stage("STAGE-APEX-VISUALS", "Generating Publication Radar Plots & Confusion Matrices")
    plot_apex_visualizations(fold_results, confusion_matrices, out_dir, logger)
    logger.end_stage(status="PASSED")
    
    # Save structured JSON
    metrics_json_path = os.path.join(out_dir, "apex_aligned_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as jf:
        json.dump({
            "experiment": "Aligned CK+ Apex Benchmark (10-Fold Subject-Disjoint)",
            "paper": "Quantum Information Entanglement Index for Facial Expression Analysis (ECML PKDD 2026)",
            "dataset": "CK+ (327 Apex Frames)",
            "total_apex_samples": len(y),
            "unique_subjects": len(set(subjects)),
            "models_evaluated": final_stats
        }, jf, indent=2)
    logger.success(f"Saved aligned apex metrics JSON: {metrics_json_path}")
    
    logger.finalize()


if __name__ == "__main__":
    main()
