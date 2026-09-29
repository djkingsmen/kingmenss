# Experiment 02: Adaptive Alpha Regularization

By calibrating $\alpha \in [0.15, 0.25]$, the quantum features provide non-linear guidance without overwhelming Euclidean distance geometry.

| Model | Baseline | Best Alpha | Optimized Accuracy | Net Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | 81.97% | `0.05` | **82.87%** | **`+0.90%`** |
| **RBF-SVM** | 82.63% | `0.15` | **82.92%** | **`+0.28%`** |
| **MLP** | 83.8% | `0.4` | **84.72%** | **`+0.92%`** |
| **KNN (k=5)** | 76.8% | `0.15` | **77.73%** | **`+0.93%`** |
| **Naive Bayes** | 81.65% | `0.05` | **79.54%** | **`-2.11%`** |
