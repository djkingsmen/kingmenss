# Experiment Checklist & System Requirements
**Paper**: *Quantum Information Entanglement Index for Facial Expression Analysis* (ECML PKDD 2026)  
**Output Directory**: `pipeline/research_paper/output/ECMLPKDD2026/`

This document compiles the complete hardware/GPU specifications, baseline comparisons, benchmark dataset protocols, software dependencies, and actionable step-by-step verification checklists required to reproduce or integrate the Quantum Information Entanglement Index (QIEI) methodology.

---

## Table of Contents
1. [Hardware & GPU Specifications](#1-hardware--gpu-specifications)
2. [Benchmark Datasets & Experimental Protocols](#2-benchmark-datasets--experimental-protocols)
3. [Baseline Comparisons & Quantitative Results](#3-baseline-comparisons--quantitative-results)
4. [Software Stack & Environment Setup](#4-software-stack--environment-setup)
5. [Pre-Implementation & Verification Checklist](#5-pre-implementation--verification-checklist)
6. [Statistical Significance Testing Protocol](#6-statistical-significance-testing-protocol)
7. [Hyperparameter Analysis (Explicit vs. Omitted Defaults)](#7-hyperparameter-analysis-explicit-vs-omitted-defaults)

---

## 1. Hardware & GPU Specifications

### A. Quantum Feature Extraction (Simulation Mode)
* **GPU Requirement**: **Not Required (Optional)**
* **State Vector Space**:
  * **Intra-region**: $q = 3$ qubits $\to 2^3 = 8$ amplitudes.
  * **Inter-region**: $q = 5$ qubits $\to 2^5 = 32$ amplitudes.
* **CPU / Memory**:
  * Any standard multi-core processor (Intel Core i5/i7/i9, AMD Ryzen 5/7/9, Apple Silicon M1/M2/M3).
  * RAM: Minimum 8 GB (exact statevector simulation takes $< 50$ MB RAM).
* **Extraction Speed**: $\approx 3$–$10$ ms per face on CPU using the included exact NumPy simulator or Qiskit Aer.

### B. Quantum Physical Execution (IBM NISQ Hardware)
* **Tested Quantum Processors**:
  * **IBM Eagle** (127 superconducting qubits)
  * **IBM Heron** (133 superconducting qubits)
* **Hardware Execution Protocol**:
  * Measurement budget: **8,192 shots per circuit setting**.
  * Method: **Reduced-state quantum tomography** (measuring Pauli $X, Y, Z$ bases over the subsystem of interest).
  * Noise mitigation: **Ensemble averaging** across multiple repeated runs ($\bar{E} = \frac{1}{N}\sum_{i=1}^N E^{(i)}$).

### C. Downstream Classifiers & Deep Learning Baselines
| Stage / Model | Minimum Hardware | Recommended Hardware | Minimum VRAM / RAM |
| :--- | :--- | :--- | :--- |
| **Classical ML** (SVM, RF, MLP, KNN, Logistic Reg.) | Standard CPU (4+ cores) | Modern 8+ core CPU | 8 GB–16 GB System RAM |
| **Standard CNNs** (ResNet-18/50, VGG-16, ANN) | 1x NVIDIA GPU (GTX 1660 / RTX 3050) | RTX 3060 / 4060 / T4 | 6 GB–8 GB VRAM |
| **SOTA Models** (Poster++, ViT, QUADCANN on AffectNet) | 1x NVIDIA RTX 3070 (8 GB) | RTX 3090 / 4080 / 4090 / A100 | 12 GB–24 GB VRAM |

---

## 2. Benchmark Datasets & Experimental Protocols

The paper evaluates QIEI across **8 diverse benchmark datasets** spanning controlled laboratories, unconstrained in-the-wild captures, and micro-expression scenarios:

| Dataset | Setting / Domain | Challenges Addressed | Emotion Categories |
| :--- | :--- | :--- | :--- |
| **CK+** (Cohn-Kanade+) | Controlled laboratory | Posed apex frames, clean background | 8 classes (Anger, Contempt, Disgust, Fear, Happy, Sad, Surprise, Neutral) |
| **AffectNet** | Unconstrained in-the-wild | Severe occlusion, extreme poses, varied lighting, label noise | 7 or 8 basic emotion classes |
| **RAF-DB** | Real-world crowdsourced | Naturalistic, illumination & head pose | 7 basic emotions |
| **FER / FER-2013** | In-the-wild web crawl | Low resolution ($48 \times 48$), grayscale | 7 emotion classes |
| **SAMM** | High-speed laboratory camera | Subtle micro-movements, spontaneous actions | FACS Action Units & emotions |
| **KMU-FED** | Real driving environments | Heavy occlusions (masks, sunglasses, driving glare) | 6/7 basic emotions |
| **KDEF** | Controlled multi-angle | Posed expressions from 5 distinct viewing angles | 7 emotions |
| **JAFFE** | Controlled laboratory | Japanese female facial expressions | 6 emotions + Neutral |

### Data Partitioning & Slicing Protocol
1. **Split Ratio**: **80% Training : 10% Validation : 10% Testing**.
2. **Subject-Disjoint Rule**: In all datasets with subject identities (e.g. CK+, RAF-DB, KDEF, JAFFE, SAMM), the splits are strictly **subject-disjoint** to prevent identity leakage between train and test sets.
3. **Preprocessing Pipeline**:
   * All images converted to **single-channel Grayscale** to mitigate illumination bias.
   * 68-point landmarks detected using **HOG-based landmark detector** (dlib 68-point shape predictor).
   * **No facial alignment, cropping, or resizing applied**, ensuring natural facial distance ratios and physiological muscle stretch distances remain intact.

---

## 3. Baseline Comparisons & Quantitative Results

### A. Feature Extraction Model Capabilities (Table 2 from Paper)
| Feature Model | Texture | Geometry | Coordination | Occlusion Robustness | Interpretability | Dimensionality Reduction | Best-Suited Scenarios |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **QIEI (Proposed)** | — | **✓** | **✓** | **✓** | **✓** | **✓** | Occlusion, low-res, interpretable fusion |
| **Graph (Coral)** | — | **✓** | **✓** | **✓** | **✓** | — | Region-level coordination, explainable |
| **Graph (Complete)** | — | **✓** | — | — | — | — | Fine-grained, high dimensional |
| **PCA** | **✓** | — | — | — | **✓** | **✓** | Dimensionality reduction, linear baseline |
| **LBP** | **✓** | — | — | — | **✓** | — | Texture, illumination invariant |
| **HOG** | **✓** | **✓** | — | — | **✓** | — | Shape, edge, pose analysis |
| **Gabor** | **✓** | — | — | — | — | — | Micro-expressions, fine frequency texture |
| **Geometric (Classic)** | — | **✓** | **✓** | **✓** | **✓** | — | Occlusion, low resolution, explainable |
| **Deep CNN** | **✓** | **✓** | **✓** | **✓\*** | — | — | Large-scale, unconstrained black-box |

### B. Classical Classifiers: Before vs. After QIEI (Table 3 from Paper)
| Model | CK+ Baseline | CK+ with QIEI | CK+ Gain ($\mathbf{+\Delta}$) | AffectNet Baseline | AffectNet with QIEI | AffectNet Gain ($\mathbf{+\Delta}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 98.90% | 99.45% | **+0.55%** | 68.50% | 76.62% | **+8.12%** |
| **ANN (MLP)** | 96.12% | 99.34% | **+3.22%** | 71.23% | 78.73% | **+7.50%** |
| **Random Forest (RF)** | 90.00% | 94.56% | **+4.56%** | 69.72% | 74.09% | **+4.37%** |
| **SVM (RBF)** | 97.59% | 99.12% | **+1.53%** | 63.31% | 67.06% | **+3.75%** |
| **Naive Bayes (NB)** | 99.12% | 99.87% | **+0.75%** | 69.08% | 70.33% | **+1.25%** |
| **KNN ($k=5$)** | 94.80% | 95.08% | **+0.28%** | 64.57% | 65.19% | **+0.62%** |

### C. State-of-the-Art (SOTA) Augmented Benchmarks (Table 4 from Paper)
| Model Architecture | Target Dataset | Baseline Acc (%) | + QIEI Acc (%) | Absolute Improvement ($\mathbf{\Delta}$) |
| :--- | :--- | :---: | :---: | :---: |
| **AU-Based Micro-Expression** | SAMM | 93.10% | **98.83%** | **+5.73%** |
| **QUADCANN** | AffectNet | 74.60% | **80.12%** | **+5.52%** |
| **Attention-Optimized Ensemble** | FER | 78.60% | **83.52%** | **+4.92%** |
| **Poster++ FEA** | AffectNet | 66.42% | **69.80%** | **+3.38%** |
| **Deep Ensemble (Occluded FER)** | RAF-DB | 91.66% | **94.87%** | **+3.21%** |
| **Attention-Optimized Ensemble** | KDEF | 96.83% | **99.53%** | **+2.70%** |
| **AU-Based Micro-Expression** | KMU-FED | 94.23% | **96.83%** | **+2.60%** |
| **Enhanced Hybrid FER** | CK+ | 96.85% | **99.08%** | **+2.23%** |
| **Poster++ FEA** | CK+ | 97.25% | **99.40%** | **+2.15%** |
| **CNN (Emognition)** | JAFFE | 97.62% | **98.12%** | **+0.50%** |

---

## 4. Software Stack & Environment Setup

### Installation Commands
```bash
# 1. Base Scientific Stack & Machine Learning
pip install numpy scipy scikit-learn

# 2. Computer Vision & Landmark Extraction
pip install opencv-python dlib

# 3. Quantum Circuit Modeling & State Simulation (Optional - NumPy fallback included)
pip install qiskit qiskit-aer

# 4. Deep Learning Baselines (If reproducing CNN/Transformer baselines)
pip install torch torchvision
```

### Pretrained Models Required
* **dlib 68-Point Landmark Model**:
  Download `shape_predictor_68_face_landmarks.dat.bz2` from [dlib.net](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), extract, and place in your working directory.

---

## 5. Pre-Implementation & Verification Checklist

### Phase 1: Environment & Tooling Verification
- [ ] Python 3.9+ installed and virtual environment created.
- [ ] Install base dependencies: `pip install numpy scipy scikit-learn opencv-python`.
- [ ] Test the standalone QIEI extractor:
  ```powershell
  python tools/qiei_extractor.py
  ```
- [ ] Confirm output returns `Extraction Successful! Total Descriptor Dimensions: 24`.

### Phase 2: Data Acquisition & Preprocessing
- [ ] Obtain benchmark dataset images (e.g. CK+ or RAF-DB).
- [ ] Convert images to single-channel Grayscale (`cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`).
- [ ] Ensure **no face alignment, scaling, or cropping** is applied.
- [ ] Run 68-point landmark detection and extract $(x_i, y_i)$ coordinate array of shape `(68, 2)`.
- [ ] Slice the 68 points into the 7 anatomical regions:
  * `jaw`: `[0..16]`
  * `right_eyebrow`: `[17..21]`
  * `left_eyebrow`: `[22..26]`
  * `nose`: `[27..35]`
  * `right_eye`: `[36..41]`
  * `left_eye`: `[42..47]`
  * `mouth`: outer `[48..59]`, inner `[60..67]`

### Phase 3: Extreme Points Extraction & Normalization
- [ ] For each region $R$, extract the 4 extreme coordinates:
  $$\text{extreme}(R) = [x_{\text{left}}, y_{\text{left}}, x_{\text{right}}, y_{\text{right}}, x_{\text{top}}, y_{\text{top}}, x_{\text{bot}}, y_{\text{bot}}]$$
- [ ] For Intra-region ($q=3$ qubits): $\ell_2$-normalize the 8-coordinate vector.
- [ ] For Inter-region ($q=5$ qubits): Concatenate 8 coordinates from Region 1 and 8 coordinates from Region 2 ($16$ floats), zero-pad to length 32, and $\ell_2$-normalize.

### Phase 4: Quantum Circuit Execution & Entropy Measurement
- [ ] Initialize circuit with $H^{\otimes q}$ (superposition layer).
- [ ] Apply phase rotation $R_z(2\theta_k)$ on each qubit $k$.
- [ ] Apply adjacent $CZ(q, q+1)$ and $RZZ(q, q+1; \theta_{q+1})$, plus circular boundary entanglement.
- [ ] Intra-region: Trace out qubits $[1, 2]$ to obtain $2 \times 2$ reduced density matrix $\rho_A$ on qubit $[0]$.
- [ ] Inter-region: Trace out subsystem of Region 2 (qubits $[3, 4]$) to obtain reduced density matrix on Region 1.
- [ ] Compute Von Neumann entropy:
  $$S(\rho_A) = -\text{Tr}(\rho_A \log_2 \rho_A)$$
- [ ] Calculate mouth score as average of outer and inner mouth:
  $$\bar{E}_{\text{mouth}} = \frac{1}{2}(\bar{E}_{\text{mouth\_outer}} + \bar{E}_{\text{mouth\_inner}})$$

### Phase 5: Descriptor Assembly & Classification
- [ ] Concatenate the 7 intra-region scores + 17 inter-region scores into the 24-dimensional feature vector.
- [ ] Create subject-disjoint 80:10:10 stratified splits.
- [ ] Train baseline classifiers using standard landmark distances.
- [ ] Train augmented classifiers using Baseline + QIEI features.
- [ ] Verify accuracy improvement matches expected trends (e.g. $+4.56\%$ for Random Forest on CK+, $+8.12\%$ for Logistic Regression on AffectNet).

---

## 6. Statistical Significance Testing Protocol

To verify that improvements are statistically significant and not due to random variation:
1. Conduct a **paired $t$-test** between the baseline accuracy $x_i$ and the QIEI-augmented accuracy $y_i$ for each fold/classifier $i$.
2. Compute paired difference:
   $$d_i = y_i - x_i$$
3. Test the null hypothesis:
   $$H_0: \mu_d = 0 \quad \text{vs.} \quad H_1: \mu_d > 0 \quad (\alpha = 0.05)$$
4. A computed $p$-value $< 0.05$ confirms statistical significance across both controlled and in-the-wild benchmarks.

---

## 7. Hyperparameter Analysis (Explicit vs. Omitted Defaults)

Understanding what is explicitly specified in the paper versus what relies on standard defaults is critical for reproducibility:

### A. Explicitly Specified Hyperparameters
| Parameter / Setting | Value in Paper | Source in Paper | Significance |
| :--- | :--- | :--- | :--- |
| **Landmark Count** | **68 points** | Section 3.1, Page 4 | Standard HOG/dlib detector |
| **Regional Extreme Points** | **4 points** (8 coordinates: Left, Right, Top, Bottom) | Section 3.3, Page 6 | Geometric compression discarding point-density bias |
| **Intra-Region Qubits ($q$)** | **3 qubits** | Section 3.7, Page 8 | $2^3 = 8$ amplitudes matching 8 normalized coordinates |
| **Inter-Region Qubits ($q$)** | **5 qubits** | Section 3.7, Page 8 | $2^5 = 32$ amplitudes padding 16 coordinates from 2 regions |
| **Number of Facial Regions ($m$)** | **7 regions** | Section 3.5, Page 6 | jaw, R-eyebrow, L-eyebrow, nose, R-eye, L-eye, mouth |
| **Quantum Measurement Shots ($S$)** | **8,192 shots** | Section 3.4, 3.7, 3.8, Page 6, 8 | Standard NISQ sample budget per circuit setting |
| **Noise Mitigation** | **Ensemble averaging** ($\bar{E} = \frac{1}{N} \sum_{i=1}^N E^{(i)}$) | Section 3.4, Page 6 | Repeated executions to suppress device decoherence |
| **Data Split Ratio** | **80% Train : 10% Val : 10% Test** | Section 3.6, Page 7 | Preserves class distributions across sets |
| **KNN Neighbor Count ($k$)** | **$k = 5$** | Table 3, Page 9 | Explicitly specified as `KNN (k=5)` |
| **Statistical Significance ($\alpha$)** | **$\alpha = 0.05$** | Section 4.3, Page 10 | One-tailed paired $t$-test threshold ($p < 0.05$) |
| **Input Image Mode** | **Single-channel Grayscale** | Section 3.6, Page 8 | Raw geometry without face alignment or resizing |

### B. Omitted Hyperparameters & Standard Defaults
The paper does not report exhaustive training hyperparameters for the downstream classifiers because standard library defaults were used:

1. **Support Vector Machine (`RBF-SVM`)**:
   * *Kernel*: Radial Basis Function (explicitly stated).
   * *Regularization parameter $C$ & kernel coefficient $\gamma$*: Omitted.
   * *Recommended default*: $C = 10.0$, $\gamma = \text{"scale"}$ with `StandardScaler()`.
2. **Random Forest (`RF`)**:
   * *Number of trees (`n_estimators`) & tree depth (`max_depth`)*: Omitted.
   * *Recommended default*: `n_estimators=200`, `max_depth=15`, `random_state=42`.
3. **Artificial Neural Network (`ANN / MLP`)**:
   * *Hidden layer architecture, learning rate, optimizer, batch size, and epochs*: Omitted.
   * *Recommended default*: 2 hidden layers `(128, 64)`, Adam optimizer, learning rate $\eta = 10^{-3}$, batch size $32$, $200$ epochs with early stopping.
4. **Logistic Regression**:
   * *Solver & penalty*: Omitted.
   * *Recommended default*: `penalty='l2'`, solver `'lbfgs'`, `max_iter=1000`.
5. **State-of-the-Art Deep Models (Poster++, QUADCANN, Emognition, Attention Ensembles)**:
   * *Note from authors (Page 10)*: *"For these comparisons, baseline results correspond to the cited methods, while '+QIEI' scores are obtained by augmenting reproduced pipelines under our evaluation protocol and data splits."*
   * *Rule*: Replicate the original authors' published hyperparameters (learning rate, scheduler, epochs, batch size) from their respective papers, and concatenate the 24-dimensional QIEI descriptor directly into the penultimate feature layer prior to the final classification head.

### C. Ready-to-Use Classifier Instantiation Code
```python
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def get_recommended_classifiers():
    """
    Returns pre-configured classifiers matching the experimental setup
    described in Table 3 of the ECML PKDD 2026 paper.
    """
    return {
        # RBF-SVM with standard feature normalization
        "SVM (RBF)": make_pipeline(
            StandardScaler(), 
            SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42)
        ),
        
        # Random Forest with 200 estimators
        "Random Forest": RandomForestClassifier(
            n_estimators=200, 
            max_depth=15, 
            random_state=42
        ),
        
        # Two-layer Multilayer Perceptron
        "ANN (MLP)": make_pipeline(
            StandardScaler(), 
            MLPClassifier(
                hidden_layer_sizes=(128, 64), 
                learning_rate_init=1e-3, 
                max_iter=500, 
                random_state=42
            )
        ),
        
        # L2-regularized Logistic Regression
        "Logistic Regression": make_pipeline(
            StandardScaler(), 
            LogisticRegression(max_iter=1000, random_state=42)
        ),
        
        # K-Nearest Neighbors (k=5 explicitly defined in paper)
        "KNN (k=5)": make_pipeline(
            StandardScaler(), 
            KNeighborsClassifier(n_neighbors=5)
        )
    }
```

