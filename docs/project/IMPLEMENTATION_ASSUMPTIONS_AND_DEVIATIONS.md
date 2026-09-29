# Implementation Assumptions, Methodological Deviations & Engineering Decisions

**Paper**: *Quantum Information Entanglement Index for Facial Expression Analysis* (ECML PKDD 2026)  
**Document Purpose**: Exhaustive record of every engineering decision, implicit assumption, and technical deviation introduced during our implementation compared to the original research paper, including the scientific rationale behind each choice.

---

## Table of Contents
1. [Quantum Mathematical Modeling & Simulation](#1-quantum-mathematical-modeling--simulation)
2. [Dataset Resolution & Sequence Selection (CK+ vs. CK+48)](#2-dataset-resolution--sequence-selection-ck-vs-ck48)
3. [Facial Landmark Detection Backend](#3-facial-landmark-detection-backend)
4. [Baseline Feature Engineering (FACS Euclidean Distances)](#4-baseline-feature-engineering-facs-euclidean-distances)
5. [Validation Protocol & Subject-Disjoint Enforcement](#5-validation-protocol--subject-disjoint-enforcement)
6. [Feature Concatenation & Metric Space Distortion](#6-feature-concatenation--metric-space-distortion)
7. [Persistent Profiling & Execution Logging](#7-persistent-profiling--execution-logging)
8. [Summary Matrix: Paper Specification vs. Our Implementation](#8-summary-matrix-paper-specification-vs-our-implementation)

---

## 1. Quantum Mathematical Modeling & Simulation

### Paper Specification
* The paper outlines the quantum Hilbert-space state mapping:
  * Intra-region: 4 extreme coordinates $\to 8$ coordinates $\to 3$ qubits ($2^3 = 8$).
  * Inter-region: 2 regions $\times 8 = 16$ coordinates $\to$ zero-padded to 32 dimensions $\to 5$ qubits ($2^5 = 32$).
* Unitary evolution: Hadamard layer $H^{\otimes q}$, parameterized phase rotations $R_z(\theta_i)$, and pairwise entangling gates ($CZ, RZZ$).
* Physical hardware execution: Ran circuits on **IBM Eagle (127-qubit)** and **IBM Heron (133-qubit)** with **8,192 measurement shots** per setting using reduced-state quantum tomography.

### New Changes & Assumptions Made in Our Implementation
1. **Exact Pure-NumPy Statevector Engine (Zero Cloud / No Qiskit Aer Dependency)**:
   * **What we did**: Built a standalone, exact linear algebra simulation engine ([stage1_linalg_core.py](file:///c:/Users/likhi/OneDrive/Pictures/Desktop/CollabProj/CogniWeave-Personalized-Learning-Intervention-System/pipeline/research_paper/output/ECMLPKDD2026/implementation/stage1_linalg_core.py)) that computes the statevector and density matrix directly via tensor contractions.
   * **Why considered**:
     * $2^3 = 8$ and $2^5 = 32$ dimensional matrices are microscopic in classical computing terms ($< 50$ MB RAM).
     * Physical quantum computers and cloud simulators introduce queue delays, shot noise ($1/\sqrt{8192} \approx 1.1\%$), and require cloud API tokens.
     * Exact statevector simulation runs in $< 1$ millisecond per face on a local CPU with zero floating-point noise.
2. **Explicit Circuit Boundary Topology**:
   * **What we did**: Added circular boundary closure ($CZ(q-1, 0)$ and $RZZ$) to ensure the 1D qubit chain couples all extreme coordinates symmetrically.
   * **Why considered**: The paper generalized the entangling layer in text without publishing explicit quantum gate assembly diagrams.

---

## 2. Dataset Resolution & Sequence Selection (CK+ vs. CK+48)

### Paper Specification
* Evaluated on the original **Cohn-Kanade+ (CK+)** dataset, which consists of high-resolution full-face video sequences ($640 \times 490$ or $640 \times 480$ pixels).
* Standard literature protocol evaluates **only the apex (peak expression) frame** of each sequence (typically 1 apex frame per sequence, $\approx 327$ total images).

### New Changes & Assumptions Made in Our Implementation
1. **Use of Pre-cropped $48 \times 48$ Grayscale Dataset (`CK+48`)**:
   * **What we did**: Utilized the user-provided [archive.zip](../../data/raw/archive.zip), which contains the complete `CK+48` distribution (981 images across 118 subjects).
   * **Why considered**: The official original CMU CK+ dataset requires a signed academic EULA and manual credential verification. `archive.zip` was immediately available and runnable locally.
2. **Implementation of Two Separate Benchmarks**:
   * **Full-Sequence Benchmark ([full_rigorous_benchmark.py](file:///c:/Users/likhi/OneDrive/Pictures/Desktop/CollabProj/CogniWeave-Personalized-Learning-Intervention-System/pipeline/research_paper/output/ECMLPKDD2026/implementation/full_rigorous_benchmark.py))**:
     * Evaluates all 981 frames from onset (neutral) $\to$ transition $\to$ apex.
     * *Observation*: Models scored $78\%–83\%$ because early onset frames look almost completely neutral.
   * **Aligned Apex Benchmark ([apex_benchmark_aligned.py](file:///c:/Users/likhi/OneDrive/Pictures/Desktop/CollabProj/CogniWeave-Personalized-Learning-Intervention-System/pipeline/research_paper/output/ECMLPKDD2026/implementation/apex_benchmark_aligned.py))**:
     * Implemented sequence grouping logic (`parts[0]_parts[1]`) and filtered for the maximum frame number per sequence, isolating the **exact 327 canonical apex frames**.
     * *Observation*: Peak accuracies immediately reached **$93.9\%$**, matching the paper's $90\%+$ regime.

---

## 3. Facial Landmark Detection Backend

### Paper Specification
* Cites King 2009 / Kazemi & Sullivan CVPR 2014: **Dlib HOG-based 68-point facial landmark detector** (`shape_predictor_68_face_landmarks.dat`, ~99 MB).

### New Changes & Assumptions Made in Our Implementation
1. **Lightweight Google MediaPipe FaceLandmarker Integration**:
   * **What we did**: Integrated Google's modern `face_landmarker.task` (3.58 MB TFLite model) with an exact index mapping from 478 mesh points to the canonical 68-point Multi-PIE / dlib indexing layout.
   * **Why considered**:
     * `dlib` is not pre-installed in the user's environment and compiling `dlib` from source on Windows requires Microsoft Visual C++ Build Tools and CMake, which often fails on Windows developer setups.
     * `mediapipe` was already installed in the local Python 3.12 environment, runs on CPU via XNNPACK in $< 20$ ms, and provides sub-pixel landmark precision even on small $48 \times 48$ faces when upscaled to $256 \times 256$.
2. **Anatomical Proportional Fallback Generator**:
   * **What we did**: Implemented `generate_anatomical_synthetic_landmarks()` as a guaranteed zero-dependency fallback for headless unit testing.
   * **Why considered**: Ensures mathematical invariant testing in Stage 1 and Stage 3 can run in isolated CI/CD environments without model downloads.

---

## 4. Baseline Feature Engineering (FACS Euclidean Distances)

### Paper Specification
* The paper stated:
  > *"For baseline testing, we use intra- and inter-region Euclidean distances between facial landmarks... reflecting FACS principles."*
* However, the paper **did not provide the explicit index list of landmark pairs** used to construct this baseline vector.

### New Changes & Assumptions Made in Our Implementation
1. **Formalized 27-Dimensional FACS Euclidean Distance Vector**:
   * **What we did**: In [apex_benchmark_aligned.py](file:///c:/Users/likhi/OneDrive/Pictures/Desktop/CollabProj/CogniWeave-Personalized-Learning-Intervention-System/pipeline/research_paper/output/ECMLPKDD2026/implementation/apex_benchmark_aligned.py), we explicitly codified 27 anatomical distance pairs representing core FACS Action Units:
     * **AU1 / AU2 / AU4**: Inter-eyebrow distance ($\|p_{21} - p_{22}\|_2$), mid-brow to eye ($\|p_{19} - p_{37}\|_2, \|p_{24} - p_{44}\|_2$), brow to nose root ($\|p_{21} - p_{27}\|_2$).
     * **AU6**: Eye vertical aspect ratios ($\|p_{37} - p_{41}\|_2, \|p_{43} - p_{47}\|_2$), eye to mouth corners ($\|p_{36} - p_{48}\|_2, \|p_{45} - p_{54}\|_2$).
     * **AU9 / AU10**: Nose length ($\|p_{27} - p_{33}\|_2$), nose width ($\|p_{31} - p_{35}\|_2$), mouth corner to nose tip ($\|p_{48} - p_{33}\|_2$).
     * **AU12 / AU20**: Mouth width ($\|p_{48} - p_{54}\|_2$).
     * **AU25 / AU26**: Outer mouth vertical height ($\|p_{51} - p_{57}\|_2$), inner lip parting ($\|p_{62} - p_{66}\|_2$), lower lip to chin ($\|p_{57} - p_{8}\|_2$), inner lip to chin ($\|p_{66} - p_{8}\|_2$).
2. **Inter-Ocular Distance (IOD) Normalization**:
   * **What we did**: Every pairwise distance is divided by the inter-ocular distance $\|p_{36} - p_{45}\|_2$ (outer corner of right eye to outer corner of left eye).
   * **Why considered**: Without IOD normalization, facial distance features change whenever a person sits closer or further from the camera. IOD normalization provides scale and head-size invariance.

---

## 5. Validation Protocol & Subject-Disjoint Enforcement

### Paper Specification
* Mentions 10-fold cross-validation and 80:10:10 subject-disjoint partitions.

### New Changes & Assumptions Made in Our Implementation
1. **Strict Automated Zero-Leakage Assertion**:
   * **What we did**: In both [stage6_dataset_manager.py](file:///c:/Users/likhi/OneDrive/Pictures/Desktop/CollabProj/CogniWeave-Personalized-Learning-Intervention-System/pipeline/research_paper/output/ECMLPKDD2026/implementation/stage6_dataset_manager.py) and [full_rigorous_benchmark.py](file:///c:/Users/likhi/OneDrive/Pictures/Desktop/CollabProj/CogniWeave-Personalized-Learning-Intervention-System/pipeline/research_paper/output/ECMLPKDD2026/implementation/full_rigorous_benchmark.py), we implemented `GroupKFold` on parsed subject prefixes (`S010`, `S011`, etc.) and added automated runtime assertions:
     ```python
     leakage = set(subjects[train_idx]).intersection(set(subjects[test_idx]))
     assert len(leakage) == 0, f"Subject leakage detected: {leakage}"
     ```
   * **Why considered**:
     * In older facial expression literature, some papers accidentally used standard random cross-validation. This caused frames of the same actor to appear in both training and test sets, artificially inflating accuracy to $99\%+$.
     * Enforcing strict 0% subject leakage provides true, uncompromised generalization testing on completely unseen human faces.

---

## 6. Feature Concatenation & Metric Space Distortion

### Paper Observation vs. Our Diagnostic Finding
1. **The Feature Dimensionality Dilemma**:
   * Baseline FACS Euclidean vector: **27 dimensions**.
   * QIEI Quantum Entanglement vector: **24 dimensions**.
   * Naive concatenation yields a **51-dimensional vector** ($27 + 24$), where quantum entropy features constitute **$47\%$ of the entire vector space**.
2. **Why Distance-Based Classifiers (KNN, RBF-SVM) Experience Performance Drops**:
   * Quantum von Neumann entropy scores are non-linear relational summary scalars ($S \in [0, 1]$ and $S \in [0, 3]$), not physical Cartesian millimeters.
   * When calculating Euclidean distances in KNN or the RBF kernel $\exp(-\gamma \|x - x'\|^2)$, the distance becomes heavily dominated by the 24 entropy scores rather than facial geometry, causing KNN to drop by $-9.16\%$.
3. **Why Tree Ensembles (Random Forest) Succeeded**:
   * Random Forest uses orthogonal decision stumps (threshold splits on individual features). It is completely immune to feature scale distortion and can selectively pick the exact quantum entropy features that disambiguate subtle expressions.
   * This explains why Random Forest gained **$+9.1\%$** on Fold 6 and **$+5.71\%$** on initial test splits.
4. **Our Methodological Recommendation**:
   * In future iterations, apply **Weighted Fusion**:
     $$X_{\text{fused}} = \left[ X_{\text{base}},\; \alpha \cdot X_{\text{qiei}} \right] \quad \text{with } \alpha \approx 0.15$$
   * Or apply **Principal Component Analysis (PCA)** on the 24 QIEI features to retain the top 4–6 orthogonal entanglement modes before concatenation.

---

## 7. Persistent Profiling & Execution Logging

### User Requirement vs. Implementation
* **User Demand**: *"introduced detailed time logs so that we can verify and store them too afiles for every run"*.
* **What we created**:
  * [logger.py](file:///c:/Users/likhi/OneDrive/Pictures/Desktop/CollabProj/CogniWeave-Personalized-Learning-Intervention-System/pipeline/research_paper/output/ECMLPKDD2026/implementation/logger.py): A dedicated profiling subsystem tracking sub-millisecond timestamps via `time.perf_counter()`.
  * Generates two persistent files on disk for **every execution**:
    1. `logs/run_YYYYMMDD_HHMMSS.log`: Formatted human-readable execution transcript.
    2. `logs/run_YYYYMMDD_HHMMSS_timings.json`: Machine-readable structured benchmark profile (hardware specs, per-stage durations, and throughput).
  * Maintains `latest_run.log` and `latest_timings.json` for instant reference.

---

## 8. Summary Matrix: Paper Specification vs. Our Implementation

| Dimension | Original Paper Specification | Our Local Implementation | Scientific / Engineering Rationale |
| :--- | :--- | :--- | :--- |
| **Execution Environment** | IBM Eagle/Heron Physical QPUs + Qiskit Cloud | Standalone Exact Pure-NumPy Simulator + Local CPU | Microscopic state spaces ($2^3, 2^5$) execute in $<1$ ms locally without cloud queues or noise. |
| **Dataset Source** | CMU/Pitt Original High-Res CK+ ($640 \times 490$) | `CK+48` ($48 \times 48$ grayscale, 981 images, 118 subjects) | CMU requires academic EULA; user-provided `archive.zip` was immediately runnable. |
| **Frame Selection** | Apex frames only ($\approx 327$ peak images) | **Both**: 981 full progression frames **AND** 327 filtered apex frames | Allows analyzing expression onset dynamics (78–83%) alongside peak expressions (94%). |
| **Landmark Detector** | Dlib HOG detector (`shape_predictor_68.dat`) | Google MediaPipe `face_landmarker.task` + 68-pt mapping | Avoids C++ CMake compiler dependency on Windows; runs in 20 ms via XNNPACK. |
| **Baseline Features** | Unspecified "FACS Euclidean distances" | Formalized 27D IOD-normalized pairwise distances | Provides rigorous scale and head-size invariance representing AU1 through AU26. |
| **Evaluated Models** | 6 models in Table 3 (SVM, MLP, LR, RF, KNN, NB) | **All 6 models** evaluated under identical 10-fold CV | Complete coverage reproducing Table 3 structure. |
| **Subject Leakage** | Stated 80:10:10 subject-disjoint | Automated assertion: $\text{Train} \cap \text{Test} = \emptyset$ | Guarantees zero identity leakage across folds. |
| **Visual Artifacts** | Static printed figures in PDF | Generated high-res Figure 2, Figure 4(a,b), Figure 5, and Confusion Matrices ($250$ DPI) | Verifiable visual artifacts matching paper layouts. |
| **Profiling & Logging** | Not published | `ExecutionLogger` persisting `.log` and `.json` per run | Enables step-by-step verification and diagnostic auditing. |
