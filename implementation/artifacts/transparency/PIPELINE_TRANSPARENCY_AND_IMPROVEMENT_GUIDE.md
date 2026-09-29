# Comprehensive Pipeline Transparency, Interpretability & Improvement Guide

**Paper Reference**: *Quantum Information Entanglement Index for Facial Expression Analysis* (ECML PKDD 2026)  
**System Architecture**: Standalone Pure-NumPy Statevector Engine on Local CPU  
**Artifacts Generated**: Visual Dashboard (`pipeline_transparency_dashboard.png`), Audit JSON (`pipeline_transparency_audit.json`)  

---

## 1. Interpretability by Values & Exact Mathematical Quantities

### Stage 1: Quantum Linear Algebra Core
- **Statevector Dimension**: Intra-region = 8 ($2^3$), Inter-region = 32 ($2^5$)
- **Statevector Conservation Norm**: 1.0
- **Unitary Invariant Error**: 1.11e-16 (Zero information loss)
- **Subsystem Density Matrix Trace**: 1.0
- **Subsystem Purity**: 0.5212 (Mixed Subsystem (Entangled))
- **Eigenspectrum**: lambda_0 = 0.60305, lambda_1 = 0.39695
- **Von Neumann Entanglement Entropy**: 0.96914 bits (Max: 1.0 bit)

### Stage 2: Facial Landmark Detection & Pose Extraction
- **Detected Landmark Points**: 68 / 68 (Canonical Multi-PIE Layout)
- **Bounding Box**: Width = 36.3 px, Height = 35.7 px
- **Face Aspect Ratio**: 0.982
- **Inter-Ocular Distance (IOD)**: 25.92 px (Invariance scale factor)
- **Estimated Head Roll Tilt**: -2.0 degrees

### Stage 3: Regional Partitioning & Extreme Points
- **Partitioned Regions**: 7 canonical facial regions (jaw, right_eyebrow, left_eyebrow, nose, right_eye, left_eye, mouth)
- **Extreme Point Vectors**: 4 points per region (x_L, y_L, x_R, y_R, x_T, y_T, x_B, y_B) in R^8
- **L2-Norm Residual**: 1.000000 (Unitary mapping condition satisfied)

### Stage 4: Quantum Entanglement Simulation
- **Intra-Region Quantum Circuits**: 7 circuits simulated in < 1 ms
- **Dual-Contour Lip Aperture**: Outer Lip = 0.8258, Inner Aperture = 0.8186, Averaged = 0.8222
- **Inter-Region Coupling Pairs**: 21 bipartite quantum circuits
- **Tightest Coupled Zones**: []
- **Loosest Coupled Zones**: [('left_eye__x__nose', 0.821), ('left_eye__x__mouth_outer', 0.821), ('left_eyebrow__x__left_eye', 0.851)]

### Stage 5: FACS Action Unit Activation Intensities
- **AU1_Inner_Brow_Raiser**: `1.1656`
- **AU2_Outer_Brow_Raiser**: `0.5173`
- **AU4_Brow_Lowerer**: `1.0081`
- **AU6_Cheek_Raiser**: `1.0963`
- **AU9_Nose_Wrinkler**: `0.9368`
- **AU12_Lip_Corner_Puller**: `1.0578`
- **AU20_Lip_Stretcher**: `1.2934`
- **AU25_Lips_Part**: `1.4567`
- **AU26_Jaw_Drop**: `1.1328`

### Stage 7: Feature Importance & Decision Weights (Top 10)
- Rank 1: **FACS_Dist_17** (Classical Landmark (FACS)) | Gini Importance: `0.0782` | ANOVA F-Score: `300.33`
- Rank 2: **FACS_Dist_23** (Classical Landmark (FACS)) | Gini Importance: `0.074` | ANOVA F-Score: `216.33`
- Rank 3: **FACS_Dist_18** (Classical Landmark (FACS)) | Gini Importance: `0.0716` | ANOVA F-Score: `304.04`
- Rank 4: **FACS_Dist_19** (Classical Landmark (FACS)) | Gini Importance: `0.0659` | ANOVA F-Score: `244.17`
- Rank 5: **FACS_Dist_16** (Classical Landmark (FACS)) | Gini Importance: `0.0607` | ANOVA F-Score: `232.57`
- Rank 6: **inter_mouth_outer__mouth_inner** (Quantum Entanglement (QIEI)) | Gini Importance: `0.0569` | ANOVA F-Score: `128.13`
- Rank 7: **FACS_Dist_22** (Classical Landmark (FACS)) | Gini Importance: `0.0547` | ANOVA F-Score: `189.04`
- Rank 8: **FACS_Dist_7** (Classical Landmark (FACS)) | Gini Importance: `0.0369` | ANOVA F-Score: `114.23`
- Rank 9: **FACS_Dist_12** (Classical Landmark (FACS)) | Gini Importance: `0.0303` | ANOVA F-Score: `74.9`
- Rank 10: **FACS_Dist_15** (Classical Landmark (FACS)) | Gini Importance: `0.0285` | ANOVA F-Score: `109.74`

---

## 2. Interpretability by Data & Distributions

| Dimension | Classical Baseline | Raw QIEI Concatenation | Improved Adaptive Fusion |
| :--- | :---: | :---: | :---: |
| **Feature Space Size** | 27 dimensions (FACS) | 51 dimensions (27 + 24) | 31 dimensions (27 + 4 selective) |
| **Quantum Feature Share** | 0% | 47.1% (Causes metric distortion) | 12.9% (Preserves Euclidean geometry) |
| **SVM Accuracy** | 80.76% | 75.53% (-5.23%) | **81.68% (+0.92%)** |
| **KNN (k=5) Accuracy** | 77.68% | 68.52% (-9.16%) | **77.97% (+0.29%)** |
| **Random Forest Accuracy**| 82.59% | 81.37% (-1.21%) | **82.59% (+-0.00%)** |
| **Naive Bayes Accuracy** | 80.74% | 79.55% (-1.18%) | **80.75% (+0.01%)** |

---

## 3. Actionable Improvement Levers: How to Maximize Performance

1. **Selective Alpha-Weighting**: Scale quantum features by alpha = 0.20-0.30 before concatenating to prevent Euclidean distance distortion.
2. **Sub-Pixel Landmark Pre-processing**: Upscale small 48x48 crops to 256x256 using bicubic filtering to eliminate landmark discretization on onset frames.
3. **Dual Mouth Contour Fusion**: Always evaluate both outer vermilion and inner oral cavity to disambiguate AU25 (lips part) and AU26 (jaw drop).
4. **PCA Quantum Compression**: Use top 4-6 principal components of QIEI to remove collinear inter-region redundancies.

---

## 4. Visual Diagnostic Artifacts
- 📊 **Full Diagnostic Dashboard**: [pipeline_transparency_dashboard.png](file:///c:/Users/likhi/OneDrive/Pictures/Desktop/CollabProj/CogniWeave-Personalized-Learning-Intervention-System/pipeline/research_paper/output/ECMLPKDD2026/implementation/pipeline_transparency_dashboard.png)
- 📈 **Exact Paper Figure 5 Radar Reproduction**: [fig5_exact_paper_radar_plots.png](file:///c:/Users/likhi/OneDrive/Pictures/Desktop/CollabProj/CogniWeave-Personalized-Learning-Intervention-System/pipeline/research_paper/output/ECMLPKDD2026/implementation/fig5_exact_paper_radar_plots.png)
- 🕸️ **Figure 4 Anatomical Connectivity Graph**: [fig4_intra_inter_connectivity.png](file:///c:/Users/likhi/OneDrive/Pictures/Desktop/CollabProj/CogniWeave-Personalized-Learning-Intervention-System/pipeline/research_paper/output/ECMLPKDD2026/implementation/fig4_intra_inter_connectivity.png)
- 🖼️ **Figure 2 Facial Region Landmark Annotation**: [fig2_landmark_annotation.png](file:///c:/Users/likhi/OneDrive/Pictures/Desktop/CollabProj/CogniWeave-Personalized-Learning-Intervention-System/pipeline/research_paper/output/ECMLPKDD2026/implementation/fig2_landmark_annotation.png)