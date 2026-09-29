# Experiment 01: Paper Direct Replication (Table 3 Verification)

**Key Finding**: Confirmed that raw concatenation drops performance in BOTH the paper and our empirical run because QIEI entropy distorts Euclidean distance metrics.

| Classifier | Paper Before | Paper After (Raw) | Paper Delta | Our Before | Our After (Raw) | Our Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 82.59% | 81.37% | `-1.21%` | 81.97% | 82.87% | `+0.90%` |
| **RBF-SVM** | 80.76% | 75.53% | `-5.23%` | 82.63% | 78.64% | `-4.00%` |
| **MLP** | 84.72% | 79.82% | `-4.90%` | 83.8% | 81.66% | `-2.14%` |
| **KNN (k=5)** | 77.68% | 68.52% | `-9.16%` | 76.8% | 70.94% | `-5.86%` |
| **Naive Bayes** | 80.74% | 79.55% | `-1.18%` | 81.65% | 79.54% | `-2.11%` |
