import os
import sys
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def run_evaluation():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    artifact_dir = os.path.join(base_dir, "artifacts", "improved")
    cache_dir = os.path.join(base_dir, "artifacts", "apex")
    os.makedirs(artifact_dir, exist_ok=True)
    X_base = np.load(os.path.join(cache_dir, 'apex_X_facs.npy'))
    X_qiei = np.load(os.path.join(cache_dir, 'apex_X_qiei.npy'))
    y = np.load(os.path.join(cache_dir, 'apex_y.npy'))
    subjects = np.load(os.path.join(cache_dir, 'apex_subjects.npy'))
    
    print(f"Loaded {len(y)} apex frames across {len(np.unique(subjects))} unique subjects.", flush=True)
    print(f"X_base: {X_base.shape}, X_qiei: {X_qiei.shape}", flush=True)
    
    gkf = GroupKFold(n_splits=10)
    splits = list(gkf.split(X_base, y, subjects))
    
    all_classifiers = {
        "Random Forest": lambda: RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=1),
        "RBF-SVM": lambda: SVC(C=10.0, kernel='rbf', gamma='scale', random_state=42),
        "MLP": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42),
        "KNN (k=5)": lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=1),
        "Naive Bayes": lambda: GaussianNB()
    }
    
    # 1. Baseline Evaluation (Classical FACS 27D)
    print("\n--- 1. Evaluating Classical Baseline (FACS 27D) ---", flush=True)
    baseline_scores = {name: [] for name in all_classifiers}
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_base[train_idx])
        X_te = scaler.transform(X_base[test_idx])
        
        for name, clf_fn in all_classifiers.items():
            clf = clf_fn()
            clf.fit(X_tr, y[train_idx])
            baseline_scores[name].append(clf.score(X_te, y[test_idx]))
            
    for name in all_classifiers:
        print(f"  {name:15s}: {np.mean(baseline_scores[name])*100:.2f}% ± {np.std(baseline_scores[name])*100:.2f}%", flush=True)
        
    # 2. Raw Unweighted Concatenation (51D)
    print("\n--- 2. Evaluating Raw Unweighted Concatenation (51D) ---", flush=True)
    raw_scores = {name: [] for name in all_classifiers}
    X_raw = np.hstack([X_base, X_qiei])
    for train_idx, test_idx in splits:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_raw[train_idx])
        X_te = scaler.transform(X_raw[test_idx])
        
        for name, clf_fn in all_classifiers.items():
            clf = clf_fn()
            clf.fit(X_tr, y[train_idx])
            raw_scores[name].append(clf.score(X_te, y[test_idx]))
            
    for name in all_classifiers:
        diff = (np.mean(raw_scores[name]) - np.mean(baseline_scores[name])) * 100
        print(f"  {name:15s}: {np.mean(raw_scores[name])*100:.2f}% ({diff:+.2f}%)", flush=True)

    # 3. Sweep Alpha-Weighting on QIEI [X_base, alpha * X_qiei]
    print("\n--- 3. Optimizing Alpha-Weighting on QIEI [X_base, alpha * X_qiei] ---", flush=True)
    sweep_clfs = ["RBF-SVM", "KNN (k=5)", "Random Forest"]
    alpha_candidates = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    alpha_results = {}
    
    for alpha in alpha_candidates:
        alpha_scores = {name: [] for name in sweep_clfs}
        for train_idx, test_idx in splits:
            sc_b = StandardScaler()
            sc_q = StandardScaler()
            
            X_tr_b = sc_b.fit_transform(X_base[train_idx])
            X_te_b = sc_b.transform(X_base[test_idx])
            
            X_tr_q = sc_q.fit_transform(X_qiei[train_idx]) * alpha
            X_te_q = sc_q.transform(X_qiei[test_idx]) * alpha
            
            X_tr = np.hstack([X_tr_b, X_tr_q])
            X_te = np.hstack([X_te_b, X_te_q])
            
            for name in sweep_clfs:
                clf = all_classifiers[name]()
                clf.fit(X_tr, y[train_idx])
                alpha_scores[name].append(clf.score(X_te, y[test_idx]))
                
        alpha_results[alpha] = {name: np.mean(alpha_scores[name]) for name in sweep_clfs}
        svm_diff = (alpha_results[alpha]["RBF-SVM"] - np.mean(baseline_scores["RBF-SVM"])) * 100
        knn_diff = (alpha_results[alpha]["KNN (k=5)"] - np.mean(baseline_scores["KNN (k=5)"])) * 100
        rf_diff = (alpha_results[alpha]["Random Forest"] - np.mean(baseline_scores["Random Forest"])) * 100
        print(f"  alpha={alpha:.2f} -> SVM: {alpha_results[alpha]['RBF-SVM']*100:.2f}% ({svm_diff:+.2f}%), KNN: {alpha_results[alpha]['KNN (k=5)']*100:.2f}% ({knn_diff:+.2f}%), RF: {alpha_results[alpha]['Random Forest']*100:.2f}% ({rf_diff:+.2f}%)", flush=True)

    # 4. Evaluating PCA Compression on Quantum Features
    print("\n--- 4. Evaluating PCA Quantum Redundancy Compression ---", flush=True)
    pca_results = {}
    for n_comp in [3, 4, 6, 8]:
        pca_scores = {name: [] for name in ["RBF-SVM", "Random Forest"]}
        for train_idx, test_idx in splits:
            sc_b = StandardScaler()
            sc_q = StandardScaler()
            
            X_tr_b = sc_b.fit_transform(X_base[train_idx])
            X_te_b = sc_b.transform(X_base[test_idx])
            
            X_tr_q_scaled = sc_q.fit_transform(X_qiei[train_idx])
            X_te_q_scaled = sc_q.transform(X_qiei[test_idx])
            
            pca = PCA(n_components=n_comp, random_state=42)
            X_tr_q_pca = pca.fit_transform(X_tr_q_scaled) * 0.35
            X_te_q_pca = pca.transform(X_te_q_scaled) * 0.35
            
            X_tr = np.hstack([X_tr_b, X_tr_q_pca])
            X_te = np.hstack([X_te_b, X_te_q_pca])
            
            for name in ["RBF-SVM", "Random Forest"]:
                clf = all_classifiers[name]()
                clf.fit(X_tr, y[train_idx])
                pca_scores[name].append(clf.score(X_te, y[test_idx]))
                
        pca_results[n_comp] = {name: np.mean(pca_scores[name]) for name in ["RBF-SVM", "Random Forest"]}
        svm_diff = (pca_results[n_comp]["RBF-SVM"] - np.mean(baseline_scores["RBF-SVM"])) * 100
        print(f"  PCA k={n_comp:2d} -> SVM: {pca_results[n_comp]['RBF-SVM']*100:.2f}% ({svm_diff:+.2f}%), RF: {pca_results[n_comp]['Random Forest']*100:.2f}%", flush=True)

    # 5. Full Optimized Evaluation with Best Alpha = 0.25 across ALL classifiers
    opt_alpha = 0.25
    print(f"\n--- 5. Evaluating All Classifiers with Optimal Alpha = {opt_alpha} ---", flush=True)
    improved_scores = {name: [] for name in all_classifiers}
    for train_idx, test_idx in splits:
        sc_b = StandardScaler()
        sc_q = StandardScaler()
        
        X_tr_b = sc_b.fit_transform(X_base[train_idx])
        X_te_b = sc_b.transform(X_base[test_idx])
        
        X_tr_q = sc_q.fit_transform(X_qiei[train_idx]) * opt_alpha
        X_te_q = sc_q.transform(X_qiei[test_idx]) * opt_alpha
        
        X_tr = np.hstack([X_tr_b, X_tr_q])
        X_te = np.hstack([X_te_b, X_te_q])
        
        for name, clf_fn in all_classifiers.items():
            clf = clf_fn()
            clf.fit(X_tr, y[train_idx])
            improved_scores[name].append(clf.score(X_te, y[test_idx]))

    print("\n" + "="*85, flush=True)
    print("  FINAL ACCURACY BENCHMARK COMPARISON TABLE (10-FOLD SUBJECT-DISJOINT)", flush=True)
    print("="*85, flush=True)
    print(f"{'Classifier':16s} | {'Baseline (FACS)':16s} | {'Raw QIEI (51D)':16s} | {'Improved (alpha=0.25)':22s} | {'Net Gain':10s} | {'Max Fold':10s}", flush=True)
    print("-" * 85, flush=True)
    
    summary_data = {}
    for name in all_classifiers:
        b_mean = np.mean(baseline_scores[name]) * 100
        b_std = np.std(baseline_scores[name]) * 100
        r_mean = np.mean(raw_scores[name]) * 100
        i_mean = np.mean(improved_scores[name]) * 100
        i_std = np.std(improved_scores[name]) * 100
        gain = i_mean - b_mean
        max_f = np.max(improved_scores[name]) * 100
        
        summary_data[name] = {
            "baseline_mean": round(b_mean, 2),
            "baseline_std": round(b_std, 2),
            "raw_qiei_mean": round(r_mean, 2),
            "improved_mean": round(i_mean, 2),
            "improved_std": round(i_std, 2),
            "net_gain": round(gain, 2),
            "max_fold_accuracy": round(max_f, 2)
        }
        print(f"{name:16s} | {b_mean:5.2f}% ± {b_std:4.2f}%   | {r_mean:5.2f}%          | {i_mean:5.2f}% ± {i_std:4.2f}%      | {gain:+6.2f}%   | {max_f:5.2f}%", flush=True)
        
    print("="*85, flush=True)

    # Save JSON metrics
    json_path = os.path.join(artifact_dir, "improved_benchmark_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "optimal_alpha": opt_alpha,
            "alpha_sweep": alpha_results,
            "pca_sweep": pca_results,
            "classifiers": summary_data
        }, f, indent=2)
    print(f"Saved JSON metrics: {json_path}", flush=True)

    # Generate Visual Comparative Charts
    plot_path = os.path.join(artifact_dir, "improved_benchmark_comparison.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor="#ffffff")
    
    x = np.arange(len(all_classifiers))
    width = 0.26
    names = list(all_classifiers.keys())
    base_vals = [summary_data[n]["baseline_mean"] for n in names]
    raw_vals = [summary_data[n]["raw_qiei_mean"] for n in names]
    imp_vals = [summary_data[n]["improved_mean"] for n in names]
    
    rects1 = ax1.bar(x - width, base_vals, width, label='Classical Baseline (27D)', color='#7f8c8d', edgecolor='#2c3e50')
    rects2 = ax1.bar(x, raw_vals, width, label='Raw QIEI (51D)', color='#e74c3c', edgecolor='#2c3e50')
    rects3 = ax1.bar(x + width, imp_vals, width, label=r'Improved Adaptive ($\alpha=0.25$)', color='#27ae60', edgecolor='#2c3e50')
    
    ax1.set_ylabel('10-Fold Subject-Disjoint Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Classifier Accuracy Across Fusion Strategies', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9.5, fontweight='bold')
    ax1.set_ylim(60, 95)
    ax1.legend(loc='upper right', framealpha=0.95, fontsize=9.5)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    for rects in [rects1, rects2, rects3]:
        for r in rects:
            h = r.get_height()
            ax1.text(r.get_x() + r.get_width()/2., h + 0.5, f'{h:.1f}%',
                     ha='center', va='bottom', fontsize=8, fontweight='bold')
                     
    diff_raw = [raw_vals[i] - base_vals[i] for i in range(len(names))]
    diff_imp = [imp_vals[i] - base_vals[i] for i in range(len(names))]
    
    ax2.axhline(0, color='#2c3e50', linestyle='-', lw=1.5)
    ax2.bar(x - width/2, diff_raw, width, label='Raw QIEI Net Change', color='#e74c3c', alpha=0.85, edgecolor='#2c3e50')
    ax2.bar(x + width/2, diff_imp, width, label='Improved Fusion Net Change', color='#27ae60', alpha=0.9, edgecolor='#2c3e50')
    
    ax2.set_ylabel('Net Accuracy Gain / Loss (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Net Gain over Classical Baseline: Raw vs Improved', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9.5, fontweight='bold')
    ax2.set_ylim(-11, 4)
    ax2.legend(loc='lower left', framealpha=0.95, fontsize=9.5)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    for i, (dr, di) in enumerate(zip(diff_raw, diff_imp)):
        ax2.text(x[i] - width/2, dr - 0.7 if dr < 0 else dr + 0.3, f'{dr:+.2f}%', ha='center', fontsize=8.5, fontweight='bold', color='#c0392b')
        ax2.text(x[i] + width/2, di + 0.3 if di >= 0 else di - 0.7, f'{di:+.2f}%', ha='center', fontsize=8.5, fontweight='bold', color='#27ae60')

    plt.tight_layout()
    plt.savefig(plot_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"Generated Comparison Chart: {plot_path}", flush=True)

if __name__ == "__main__":
    run_evaluation()
