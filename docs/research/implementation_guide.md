# Implementation Guide: Quantum Information Entanglement Index (QIEI)
**Paper**: *Quantum Information Entanglement Index for Facial Expression Analysis* (ECML PKDD 2026)

This implementation guide provides complete, production-ready specifications, mathematical formulations, architectural diagrams, and Python/Qiskit code templates to build and deploy the **Quantum Information Entanglement Index (QIEI)** pipeline.

---

## Table of Contents
1. [Core Concept & Architectural Overview](#1-core-concept--architectural-overview)
2. [End-to-End Processing Pipeline](#2-end-to-end-processing-pipeline)
3. [Mathematical Foundations](#3-mathematical-foundations)
4. [Facial Landmarks & Regional Decomposition](#4-facial-landmarks--regional-decomposition)
5. [Quantum Circuit Architecture (QIEImap)](#5-quantum-circuit-architecture-qieimap)
6. [FACS Action Units (AU) Interpretability Mapping](#6-facs-action-units-au-interpretability-mapping)
7. [Complete Standalone Python/Qiskit Implementation](#7-complete-standalone-pythonqiskit-implementation)
8. [Downstream Classifier Integration & Benchmarking](#8-downstream-classifier-integration--benchmarking)
9. [Hardware Execution (IBM NISQ / Aer Simulator)](#9-hardware-execution-ibm-nisq--aer-simulator)

---

## 1. Core Concept & Architectural Overview

Traditional geometric approaches treat facial landmarks as independent points or calculate raw Euclidean distances, failing to capture subtle, non-linear coordinate movements across facial muscle groups. 

**QIEI** bridges quantum information theory and affective computing:
1. **Geometric Coordinate Compression**: Extracts extreme landmark coordinates from facial regions.
2. **Quantum Feature Map ($U_{\text{QIEImap}}$)**: Encodes normalized features into quantum states using Hadamard superposition, phase rotations ($R_z$), and pairwise entanglement ($CZ$, $RZZ$).
3. **Subsystem Entanglement Quantification**: Takes the **partial trace** over a subsystem to form a **reduced density matrix** $\rho_A$, and computes the **von Neumann entropy** $S(\rho_A) = -\text{Tr}(\rho_A \log_2 \rho_A)$.
4. **Relational Descriptor**: Concatenates intra-region coherence scores and cross-region coordination scores into a compact vector $[QIEI_{\text{intra}}, QIEI_{\text{inter}}]^\top$.
5. **Physiological Grounding**: Maps directly to the Facial Action Coding System (FACS) Action Units (AUs).

```mermaid
flowchart TD
    A[Input Facial Image] --> B[Grayscale & 68-Point HOG Landmark Extraction]
    B --> C[Partition into 7 Facial Regions]
    C --> D[Extract 4 Extreme Points per Region: Left, Right, Top, Bottom]
    D --> E[L2 Normalization & Zero Padding]
    
    subgraph Quantum Feature Map U_QIEImap
        E --> F[Superposition Layer: H gates]
        F --> G[Phase Encoding Layer: Rz rotations]
        G --> H[Entanglement Layer: CZ and RZZ gates]
    end
    
    H --> I[Form Density Matrix rho = |psi><psi|]
    I --> J[Partial Trace over Subsystem -> Reduced rho_A]
    J --> K[Von Neumann Entropy: S = -Tr rho_A log2 rho_A]
    
    K --> L1[Intra-Region QIEI: 7 Regions]
    K --> L2[Inter-Region QIEI: Region Pairs]
    
    L1 --> M[Concatenate QIEI Descriptor Vector]
    L2 --> M
    M --> N[FACS Action Unit Mapping & Classification SVM / MLP / CNN / ViT]
```

---

## 2. End-to-End Processing Pipeline

The execution flow consists of six deterministic steps:

| Step | Operation | Description | Target Dimension / Output |
| :--- | :--- | :--- | :--- |
| **1** | **Face Preprocessing** | Convert RGB to Grayscale, run HOG/dlib 68-point detector | $68 \times 2$ coordinates $(x_i, y_i)$ |
| **2** | **Extreme Points Extraction** | For each region $R$, find leftmost, rightmost, topmost, bottommost points | 4 points ($8$ coordinates) per region |
| **3** | **Feature Normalization** | $\ell_2$ normalization: $v = \frac{v}{\|v\|_2}$, pad to nearest power-of-2 ($2^q$) | Intra: 8 floats ($q=3$ qubits)<br>Inter: 16/32 floats ($q=5$ qubits) |
| **4** | **Quantum Unitary Map ($U_{\text{QIEImap}}$)** | Apply $H^{\otimes q} \to R_z(\theta_i) \to CZ / RZZ(i, j)$ | Quantum state $|\psi\rangle$ |
| **5** | **Entropy Calculation** | Compute reduced density matrix $\rho_A = \text{Tr}_B(|\psi\rangle\langle\psi|)$ and $S(\rho_A)$ | Scalar entanglement index $E \in [0, 1]$ |
| **6** | **Descriptor Aggregation** | Concatenate intra- and inter-region scores; average mouth inner/outer | Final compact feature vector $\text{QIEI} \in \mathbb{R}^{D}$ |

---

## 3. Mathematical Foundations

### 3.1 Quantum State Preparation
For a normalized coordinate parameter vector $\boldsymbol{\theta} = (\theta_0, \theta_1, \dots, \theta_{2^q-1})^\top$, the quantum circuit executes:
$$|\psi\rangle = U_{\text{QIEImap}} |0\rangle^{\otimes q}$$
where the unitary operator factorizes into three distinct layers:
$$U_{\text{QIEImap}} = U_{\text{ent}} \cdot U_{\text{phase}} \cdot U_{\text{super}}$$

1. **Superposition Layer**:
   $$U_{\text{super}} = H^{\otimes q} = \bigotimes_{k=0}^{q-1} \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

2. **Phase Encoding Layer**:
   $$U_{\text{phase}} = \bigotimes_{k=0}^{q-1} R_z(2\theta_k), \quad R_z(\theta) = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$

3. **Pairwise Entanglement Layer**:
   $$U_{\text{ent}} = \prod_{(j, k) \in \text{Pairs}} CZ(j, k) \cdot RZZ(j, k; \theta_{j, k})$$
   where $CZ = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes Z$ and $RZZ(\phi) = \exp\left(-i \frac{\phi}{2} Z \otimes Z\right)$.

### 3.2 Reduced Density Matrix & Von Neumann Entropy
For a bipartite quantum system $\mathcal{H} = \mathcal{H}_A \otimes \mathcal{H}_B$, the full density matrix is $\rho = |\psi\rangle\langle\psi|$. The reduced density matrix of subsystem $A$ is obtained via partial trace:
$$\rho_A = \text{Tr}_B(\rho) = \sum_k (I_A \otimes \langle k|_B) \rho (I_A \otimes |k\rangle_B)$$

The degree of expressive dependency (entanglement) is given by the von Neumann entropy:
$$S(\rho_A) = -\text{Tr}(\rho_A \log_2 \rho_A) = -\sum_{i} \lambda_i \log_2 \lambda_i$$
where $\lambda_i$ are the eigenvalues of $\rho_A$.

### 3.3 Descriptor Aggregation
- **Intra-region vector** ($m = 7$ regions):
  $$\text{QIEI}_{\text{intra}} = [\bar{E}_{\text{jaw}}, \bar{E}_{\text{right\_eyebrow}}, \bar{E}_{\text{left\_eyebrow}}, \bar{E}_{\text{nose}}, \bar{E}_{\text{right\_eye}}, \bar{E}_{\text{left\_eye}}, \bar{E}_{\text{mouth}}]^\top$$
  where $\bar{E}_{\text{mouth}} = \frac{1}{2}\left(\bar{E}_{\text{mouth\_outer}} + \bar{E}_{\text{mouth\_inner}}\right)$.

- **Inter-region vector**:
  $$\text{QIEI}_{\text{inter}} = [\bar{E}_{R_1, R_2}, \dots, \bar{E}_{R_{m-1}, R_m}]^\top$$

- **Final Concatenated Descriptor**:
  $$\text{QIEI} = \begin{bmatrix} \text{QIEI}_{\text{intra}} \\ \text{QIEI}_{\text{inter}} \end{bmatrix}$$

---

## 4. Facial Landmarks & Regional Decomposition

The standard 68-landmark annotations (dlib / Multi-PIE convention) are partitioned into **7 anatomical facial regions**:

```
Landmark Indices Map (68 Points):
  Jawline:           [0 .. 16]   (17 points)
  Right Eyebrow:     [17 .. 21]  (5 points)
  Left Eyebrow:      [22 .. 26]  (5 points)
  Nose Bridge/Tip:   [27 .. 35]  (9 points)
  Right Eye:         [36 .. 41]  (6 points)
  Left Eye:          [42 .. 47]  (6 points)
  Mouth Outer:       [48 .. 59]  (12 points)
  Mouth Inner:       [60 .. 67]  (8 points)
```

### Extreme Points Selection Rule
For any subset of landmark coordinates $\{(x_i, y_i)\}_{i=1}^P$:
- **Leftmost point**: $\arg\min_i x_i$
- **Rightmost point**: $\arg\max_i x_i$
- **Topmost point**: $\arg\min_i y_i$ (image coordinate origin top-left)
- **Bottommost point**: $\arg\max_i y_i$

This compresses each region into 4 points $\times$ 2 coordinates = **8 values**, preserving boundary deformation while eliminating point-density bias.

---

## 5. Quantum Circuit Architecture (QIEImap)

### Intra-Region Circuit ($q = 3$ qubits)
- Input: 8 normalized coordinates $\to$ 8 amplitudes $\to 3$ qubits ($2^3 = 8$).
- Subsystem $A$: Qubit 0 (traced over qubits 1 and 2) or Qubits 0, 1 (traced over qubit 2).

```
q0: ──[ H ]──[ Rz(θ0) ]──■────────────■────── ... 
                         │            │
q1: ──[ H ]──[ Rz(θ1) ]──■──[ RZZ ]───┼────── ... 
                            │         │
q2: ──[ H ]──[ Rz(θ2) ]─────[ RZZ ]───■────── ... 
```

### Inter-Region Circuit ($q = 5$ qubits)
- Input: Combined extreme points from Region 1 and Region 2 (16 coordinates).
- Padded with zeros to length 32 $\to 5$ qubits ($2^5 = 32$).
- Subsystem $A$: Subsystem representing Region 1 qubits (qubits 0, 1, 2), tracing out Region 2 qubits (3, 4).

---

## 6. FACS Action Units (AU) Interpretability Mapping

The paper maps computed QIEI features directly to standard Facial Action Coding System (FACS) codes (Table 1):

| QIEI Feature | Associated Action Units (AUs) | Emotional Labels | Physiological Interpretation |
| :--- | :--- | :--- | :--- |
| **`jaw`** | AU25, AU26, AU27 | Surprise (O) | Jaw drop, lips part, mouth stretch |
| **`right_eyebrow`** | AU2, AU4 | Surprise (O), Anger (A) | Outer brow raiser, brow lowerer |
| **`left_eyebrow`** | AU1, AU4 | Surprise (O), Anger (A), Sadness (S) | Inner brow raiser, brow lowerer |
| **`nose`** | AU9, AU10 | Disgust (D), Contempt (C) | Nose wrinkler, upper lip raiser |
| **`right_eye`** | AU5, AU7 | Surprise (O), Fear (F) | Upper lid raiser, lid tightener |
| **`left_eye`** | AU5, AU7 | Surprise (O), Fear (F) | Upper lid raiser, lid tightener |
| **`mouth`** | AU12, AU15, AU20, AU23+24 | Happiness (H), Sadness (S), Fear (F), Neutral (N) | Lip corner puller, lip corner depressor, lip stretcher |
| **`jaw-eyebrow`** | AU4, AU25–27 | Anger (A), Surprise (O) | Coordinated open mouth + furrowed brows |
| **`jaw-nose`** | AU9, AU25–27 | Disgust (D), Surprise (O) | Snarl with dropped jaw |
| **`jaw-eye`** | AU5, AU25–27 | Surprise (O) | Wide open eyes and open mouth |
| **`jaw-mouth`** | AU14, AU25–27 | Surprise (O), Contempt (C) | Dimpler / asymmetrical mouth drop |
| **`eyebrow-eyebrow`** | AU1, AU2, AU4 | Surprise (O), Anger (A), Sadness (S) | Symmetric / asymmetric brow knitting |
| **`eyebrow-nose`** | AU4, AU9 | Anger (A), Disgust (D) | Wrinkled nose + brow furrow |
| **`eyebrow-eye`** | AU2, AU5 | Surprise (O) | Wide open eye aperture and raised brow |
| **`eyebrow-mouth`** | AU4, AU12, AU15 | Anger (A), Happiness (H), Sadness (S) | Smiling/frowning with brow activation |
| **`nose-eye`** | AU5, AU9 | Disgust (D), Surprise (O) | Eye squinting during nose wrinkling |
| **`nose-mouth`** | AU9, AU12, AU15 | Disgust (D), Happiness (H), Sadness (S) | Upper lip retraction into nasolabial fold |
| **`eye-eye`** | AU5, AU7 | Surprise (O), Fear (F) | Bilateral eye widening / tensing |
| **`eye-mouth`** | AU5, AU12, AU15 | Surprise (O), Happiness (H), Sadness (S) | Eye crinkle with smile / eye widening with grimace |

*Emotion Label Keys: H = Happiness, S = Sadness, A = Anger, O = Surprise, F = Fear, D = Disgust, C = Contempt, N = Neutral.*

---

## 7. Complete Standalone Python/Qiskit Implementation

Save this script as `qiei_pipeline.py` or import it directly into your projects:

```python
"""
qiei_pipeline.py — Quantum Information Entanglement Index (QIEI) Feature Extractor
Reference Implementation from ECML PKDD 2026.
Requirements: qiskit, numpy, scipy
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace

# ---------------------------------------------------------------------------
# 1. LANDMARK REGION DEFINITIONS (68-Point Scheme)
# ---------------------------------------------------------------------------
FACIAL_REGIONS = {
    "jaw": list(range(0, 17)),
    "right_eyebrow": list(range(17, 22)),
    "left_eyebrow": list(range(22, 27)),
    "nose": list(range(27, 36)),
    "right_eye": list(range(36, 42)),
    "left_eye": list(range(42, 48)),
    "mouth_outer": list(range(48, 60)),
    "mouth_inner": list(range(60, 68))
}

INTER_REGION_PAIRS = [
    ("jaw", "right_eyebrow"),
    ("jaw", "left_eyebrow"),
    ("jaw", "nose"),
    ("jaw", "right_eye"),
    ("jaw", "left_eye"),
    ("jaw", "mouth_outer"),
    ("right_eyebrow", "left_eyebrow"),
    ("right_eyebrow", "nose"),
    ("right_eyebrow", "right_eye"),
    ("right_eyebrow", "mouth_outer"),
    ("left_eyebrow", "nose"),
    ("left_eyebrow", "left_eye"),
    ("left_eyebrow", "mouth_outer"),
    ("nose", "right_eye"),
    ("nose", "left_eye"),
    ("nose", "mouth_outer"),
    ("right_eye", "left_eye"),
    ("right_eye", "mouth_outer"),
    ("left_eye", "mouth_outer"),
]


# ---------------------------------------------------------------------------
# 2. GEOMETRIC PREPROCESSING: 4 EXTREME POINTS & L2 NORMALIZATION
# ---------------------------------------------------------------------------
def extract_extreme_points(landmarks_xy):
    """
    Extracts leftmost, rightmost, topmost, and bottommost points.
    Input: landmarks_xy - array of shape (N, 2)
    Output: 1D numpy array of 8 coordinates [x_left, y_left, x_right, ...]
    """
    if len(landmarks_xy) == 0:
        return np.zeros(8, dtype=np.float32)

    left = landmarks_xy[np.argmin(landmarks_xy[:, 0])]
    right = landmarks_xy[np.argmax(landmarks_xy[:, 0])]
    top = landmarks_xy[np.argmin(landmarks_xy[:, 1])]
    bottom = landmarks_xy[np.argmax(landmarks_xy[:, 1])]

    extreme_coords = np.concatenate([left, right, top, bottom])
    return extreme_coords


def prepare_feature_vector(feature_array, num_qubits):
    """
    Normalizes coordinates via L2 norm and pads with zeros to 2^num_qubits.
    """
    target_dim = 2 ** num_qubits
    vec = np.zeros(target_dim, dtype=np.float32)
    
    n = min(len(feature_array), target_dim)
    vec[:n] = feature_array[:n]
    
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec = vec / norm
    else:
        vec[0] = 1.0  # Safe ground state
        
    return vec


# ---------------------------------------------------------------------------
# 3. QUANTUM CIRCUIT: QIEImap (Superposition, Phase, Entanglement)
# ---------------------------------------------------------------------------
def build_qiei_circuit(theta_params, num_qubits):
    """
    Builds the QIEImap Unitary circuit in Qiskit:
      1. Superposition Layer: H on all qubits
      2. Phase Encoding Layer: Rz(theta_i) on each qubit
      3. Entanglement Layer: CZ and RZZ between adjacent and closing pairs
    """
    qc = QuantumCircuit(num_qubits)
    
    # 1. Superposition Layer
    for q in range(num_qubits):
        qc.h(q)
        
    # 2. Phase Encoding Layer
    for q in range(num_qubits):
        angle = 2.0 * float(theta_params[q % len(theta_params)])
        qc.rz(angle, q)
        
    # 3. Pairwise Entanglement Layer
    for q in range(num_qubits - 1):
        qc.cz(q, q + 1)
        angle_zz = float(theta_params[(q + 1) % len(theta_params)])
        qc.rzz(angle_zz, q, q + 1)
        
    # Circular boundary entanglement if > 2 qubits
    if num_qubits > 2:
        qc.cz(num_qubits - 1, 0)
        angle_zz = float(theta_params[0])
        qc.rzz(angle_zz, num_qubits - 1, 0)
        
    return qc


# ---------------------------------------------------------------------------
# 4. VON NEUMANN ENTROPY OF REDUCED SUBSYSTEM
# ---------------------------------------------------------------------------
def compute_entanglement_entropy(qc, trace_qubits):
    """
    Simulates the statevector, computes reduced density matrix by tracing out
    `trace_qubits`, and calculates von Neumann entropy: S(rho_A) = -Tr(rho_A log2 rho_A).
    """
    # 1. Get exact statevector
    state = Statevector.from_instruction(qc)
    
    # 2. Trace out specified qubits to get reduced density matrix rho_A
    rho_a = partial_trace(state, trace_qubits)
    
    # 3. Compute eigenvalues of rho_A
    evals = np.linalg.eigvalsh(rho_a.data)
    
    # 4. Filter strictly positive eigenvalues to avoid log(0)
    evals = evals[evals > 1e-12]
    evals = evals / np.sum(evals)  # Re-normalize for numerical precision
    
    # 5. Von Neumann Entropy (base 2)
    s = -np.sum(evals * np.log2(evals))
    return float(s)


# ---------------------------------------------------------------------------
# 5. HIGH-LEVEL QIEI EXTRACTOR
# ---------------------------------------------------------------------------
def extract_qiei_descriptor(landmarks_68):
    """
    Full QIEI Feature Extraction from 68 facial landmarks.
    Input:
      landmarks_68: array-like of shape (68, 2)
    Output:
      qiei_full: 1D numpy array containing concatenated intra- and inter-region scores
      metadata: dict of breakdown scores for Action Unit interpretability
    """
    landmarks_68 = np.asarray(landmarks_68, dtype=np.float32)
    
    # Extract extreme coordinates for each region
    region_extremes = {}
    for r_name, indices in FACIAL_REGIONS.items():
        sub_pts = landmarks_68[indices]
        region_extremes[r_name] = extract_extreme_points(sub_pts)
        
    # -----------------------------------------------------------------------
    # A. INTRA-REGION SCORES (3 Qubits each)
    # -----------------------------------------------------------------------
    intra_scores = {}
    for r_name, ext_pts in region_extremes.items():
        params = prepare_feature_vector(ext_pts, num_qubits=3)
        qc = build_qiei_circuit(params, num_qubits=3)
        # Trace out qubits [1, 2] to measure entanglement on subsystem qubit [0]
        s_intra = compute_entanglement_entropy(qc, trace_qubits=[1, 2])
        intra_scores[r_name] = s_intra
        
    # Aggregate Mouth score: 0.5 * (mouth_outer + mouth_inner)
    e_mouth = 0.5 * (intra_scores["mouth_outer"] + intra_scores["mouth_inner"])
    
    # Order of 7 primary intra features
    primary_intra_keys = [
        "jaw", "right_eyebrow", "left_eyebrow", "nose", 
        "right_eye", "left_eye"
    ]
    qiei_intra = [intra_scores[k] for k in primary_intra_keys] + [e_mouth]
    
    # -----------------------------------------------------------------------
    # B. INTER-REGION SCORES (5 Qubits each)
    # -----------------------------------------------------------------------
    inter_scores = {}
    qiei_inter = []
    for r1, r2 in INTER_REGION_PAIRS:
        # Combine extreme points: 8 + 8 = 16 coordinates
        pair_feature = np.concatenate([region_extremes[r1], region_extremes[r2]])
        params = prepare_feature_vector(pair_feature, num_qubits=5)
        qc = build_qiei_circuit(params, num_qubits=5)
        # Trace out second subsystem (qubits [3, 4])
        s_inter = compute_entanglement_entropy(qc, trace_qubits=[3, 4])
        pair_key = f"{r1}__{r2}"
        inter_scores[pair_key] = s_inter
        qiei_inter.append(s_inter)
        
    # -----------------------------------------------------------------------
    # C. FINAL CONCATENATED DESCRIPTOR
    # -----------------------------------------------------------------------
    qiei_full = np.array(qiei_intra + qiei_inter, dtype=np.float32)
    
    metadata = {
        "intra_scores": intra_scores,
        "inter_scores": inter_scores,
        "mouth_averaged": e_mouth,
        "feature_dim": len(qiei_full)
    }
    
    return qiei_full, metadata
```

---

## 8. Downstream Classifier Integration & Benchmarking

The paper demonstrates that augmenting classical features with QIEI yields consistent accuracy improvements ($\Delta$ up to $+8.12$ points in AffectNet and $+4.56$ points in CK+).

```python
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def create_qiei_augmented_classifiers():
    return {
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale")),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
        "ANN (MLP)": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500))
    }
```

---

## 9. Hardware Execution (IBM NISQ / Aer Simulator)

### Simulator vs Hardware Trade-Offs (Section 3.8 of Paper)
1. **Qiskit Aer Simulator (Default)**:
   - Evaluates full statevector directly via exact linear algebra.
   - Computes $\rho_A$ and $S(\rho_A)$ in milliseconds without shot noise.
2. **IBM NISQ Hardware (Eagle 127q, Heron 133q)**:
   - Statevectors cannot be read out directly.
   - Requires **Reduced-State Quantum Tomography** over subsystem $A$ (measuring Pauli $X, Y, Z$ bases).
   - Use **8,192 shots** and ensemble averaging ($\bar{E} = \frac{1}{N}\sum_{i=1}^N E^{(i)}$) to mitigate device decoherence.
