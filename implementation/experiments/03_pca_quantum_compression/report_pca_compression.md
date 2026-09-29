# Experiment 03: PCA Quantum Redundancy Compression

**Key Finding**: The 24 QIEI features contain collinearities between adjacent anatomical pairs. Compressing to top 6 components retains 85.4% of total quantum entropy variance while boosting Random Forest accuracy to **83.18%** (+1.21% over baseline) using 75% fewer quantum parameters.

- **Optimal Compression**: $k = 6$ principal components
- **Random Forest Accuracy at k=6**: `83.18181818181817%`
- **SVM Accuracy at k=4**: `83.21022727272727%`
