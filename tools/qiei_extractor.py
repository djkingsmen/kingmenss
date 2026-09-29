"""
qiei_extractor.py — Quantum Information Entanglement Index (QIEI) Feature Extractor
Reference Implementation based on ECML PKDD 2026:
"Quantum Information Entanglement Index for Facial Expression Analysis"

Requirements:
    pip install numpy scipy qiskit
"""

import numpy as np

try:
    import qiskit
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, partial_trace
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

# ---------------------------------------------------------------------------
# 1. 68-LANDMARK REGION PARTITIONING
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

# 12 Primary FACS Action Unit relational pairs defined in paper (Table 1)
INTER_REGION_PAIRS = [
    ("jaw", "right_eyebrow"),
    ("jaw", "left_eyebrow"),
    ("jaw", "nose"),
    ("jaw", "right_eye"),
    ("jaw", "mouth_outer"),
    ("right_eyebrow", "left_eyebrow"),
    ("right_eyebrow", "nose"),
    ("right_eyebrow", "right_eye"),
    ("right_eyebrow", "mouth_outer"),
    ("left_eyebrow", "nose"),
    ("left_eyebrow", "left_eye"),
    ("left_eyebrow", "mouth_outer"),
    ("nose", "right_eye"),
    ("nose", "mouth_outer"),
    ("right_eye", "left_eye"),
    ("right_eye", "mouth_outer"),
    ("left_eye", "mouth_outer"),
]

# Action Units (FACS) mapping dictionary (Table 1 from paper)
FACS_AU_MAPPING = {
    "jaw": {"aus": ["AU25", "AU26", "AU27"], "labels": ["Surprise"]},
    "right_eyebrow": {"aus": ["AU2", "AU4"], "labels": ["Surprise", "Anger"]},
    "left_eyebrow": {"aus": ["AU1", "AU4"], "labels": ["Surprise", "Anger", "Sadness"]},
    "nose": {"aus": ["AU9", "AU10"], "labels": ["Disgust", "Contempt"]},
    "right_eye": {"aus": ["AU5", "AU7"], "labels": ["Surprise", "Fear"]},
    "left_eye": {"aus": ["AU5", "AU7"], "labels": ["Surprise", "Fear"]},
    "mouth": {"aus": ["AU12", "AU15", "AU20", "AU23+24"], "labels": ["Happiness", "Sadness", "Fear", "Neutral"]},
    "jaw-eyebrow": {"aus": ["AU4", "AU25-27"], "labels": ["Anger", "Surprise"]},
    "jaw-nose": {"aus": ["AU9", "AU25-27"], "labels": ["Disgust", "Surprise"]},
    "jaw-eye": {"aus": ["AU5", "AU25-27"], "labels": ["Surprise"]},
    "jaw-mouth": {"aus": ["AU14", "AU25-27"], "labels": ["Surprise", "Contempt"]},
    "eyebrow-eyebrow": {"aus": ["AU1", "AU2", "AU4"], "labels": ["Surprise", "Anger", "Sadness"]},
    "eyebrow-nose": {"aus": ["AU4", "AU9"], "labels": ["Anger", "Disgust"]},
    "eyebrow-eye": {"aus": ["AU2", "AU5"], "labels": ["Surprise"]},
    "eyebrow-mouth": {"aus": ["AU4", "AU12", "AU15"], "labels": ["Anger", "Happiness", "Sadness"]},
    "nose-eye": {"aus": ["AU5", "AU9"], "labels": ["Disgust", "Surprise"]},
    "nose-mouth": {"aus": ["AU9", "AU12", "AU15"], "labels": ["Disgust", "Happiness", "Sadness"]},
    "eye-eye": {"aus": ["AU5", "AU7"], "labels": ["Surprise", "Fear"]},
    "eye-mouth": {"aus": ["AU5", "AU12", "AU15"], "labels": ["Surprise", "Happiness", "Sadness"]},
}


# ---------------------------------------------------------------------------
# 2. EXTREME LANDMARK COORDINATE EXTRACTION & NORMALIZATION
# ---------------------------------------------------------------------------
def get_extreme_coordinates(points):
    """
    Finds 4 extreme landmarks: leftmost, rightmost, topmost, and bottommost.
    Returns array of 8 coordinates: [x_left, y_left, x_right, y_right, x_top, y_top, x_bot, y_bot]
    """
    if len(points) == 0:
        return np.zeros(8, dtype=np.float32)

    p_left = points[np.argmin(points[:, 0])]
    p_right = points[np.argmax(points[:, 0])]
    p_top = points[np.argmin(points[:, 1])]
    p_bottom = points[np.argmax(points[:, 1])]

    return np.concatenate([p_left, p_right, p_top, p_bottom]).astype(np.float32)


def normalize_and_pad(vector, num_qubits):
    """
    L2 normalization followed by zero-padding to 2^num_qubits.
    """
    dim = 2 ** num_qubits
    padded = np.zeros(dim, dtype=np.float32)
    n = min(len(vector), dim)
    padded[:n] = vector[:n]
    
    norm = np.linalg.norm(padded)
    if norm > 1e-8:
        padded = padded / norm
    else:
        padded[0] = 1.0
    return padded


# ---------------------------------------------------------------------------
# 3. QUANTUM UNITARY MAP CIRCUIT (QIEImap)
# ---------------------------------------------------------------------------
def construct_qieimap_circuit(normalized_params, num_qubits):
    """
    Constructs the 3-stage quantum unitary feature map:
      Stage 1: Superposition via Hadamard gates on all qubits
      Stage 2: Phase Encoding via Rz rotations with parameter angles
      Stage 3: Pairwise Entanglement via CZ and RZZ gates
    """
    qc = QuantumCircuit(num_qubits)
    
    # 1. Superposition Layer
    for q in range(num_qubits):
        qc.h(q)
        
    # 2. Phase Encoding Layer
    for q in range(num_qubits):
        angle = 2.0 * float(normalized_params[q % len(normalized_params)])
        qc.rz(angle, q)
        
    # 3. Entanglement Layer
    for q in range(num_qubits - 1):
        qc.cz(q, q + 1)
        rzz_angle = float(normalized_params[(q + 1) % len(normalized_params)])
        qc.rzz(rzz_angle, q, q + 1)
        
    if num_qubits > 2:
        qc.cz(num_qubits - 1, 0)
        qc.rzz(float(normalized_params[0]), num_qubits - 1, 0)
        
    return qc


def simulate_entropy_numpy(normalized_params, num_qubits, trace_out_qubits):
    """
    Exact pure-NumPy simulation of the QIEImap quantum circuit and Von Neumann entropy.
    Requires no external quantum packages.
    """
    dim = 2 ** num_qubits
    # 1. Statevector initialization with Hadamard superposition: H^q |0^q> = 1/sqrt(dim)
    state = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)
    
    # 2. Phase Encoding Layer: Rz(2 * theta_k) on each qubit
    for b in range(dim):
        phase = 0.0
        for k in range(num_qubits):
            bit = (b >> k) & 1
            theta = float(normalized_params[k % len(normalized_params)])
            phase += (1.0 if bit == 1 else -1.0) * theta
        state[b] *= np.exp(1j * phase)
        
    # 3. Entanglement Layer: CZ and RZZ gates
    # Adjacent pairs (q, q + 1)
    for q in range(num_qubits - 1):
        angle_zz = float(normalized_params[(q + 1) % len(normalized_params)])
        for b in range(dim):
            bit_q = (b >> q) & 1
            bit_next = (b >> (q + 1)) & 1
            # CZ gate
            if bit_q == 1 and bit_next == 1:
                state[b] *= -1.0
            # RZZ gate
            zz_parity = -1.0 if (bit_q == bit_next) else 1.0
            state[b] *= np.exp(1j * 0.5 * zz_parity * angle_zz)
            
    # Circular boundary entanglement
    if num_qubits > 2:
        angle_0 = float(normalized_params[0])
        q_last = num_qubits - 1
        for b in range(dim):
            bit_last = (b >> q_last) & 1
            bit_0 = b & 1
            if bit_last == 1 and bit_0 == 1:
                state[b] *= -1.0
            zz_parity = -1.0 if (bit_last == bit_0) else 1.0
            state[b] *= np.exp(1j * 0.5 * zz_parity * angle_0)

    # 4. Partial trace over trace_out_qubits
    tensor_shape = [2] * num_qubits
    psi_tensor = state.reshape(tensor_shape)
    
    # Kept qubits (Subsystem A) vs Traced qubits (Subsystem B)
    kept_qubits = [q for q in range(num_qubits) if q not in trace_out_qubits]
    num_kept = len(kept_qubits)
    num_traced = len(trace_out_qubits)
    
    # Permute axes so kept are first, followed by traced
    perm = kept_qubits + trace_out_qubits
    psi_permuted = np.transpose(psi_tensor, perm)
    
    # Reshape into matrix: shape (2^num_kept, 2^num_traced)
    mat = psi_permuted.reshape((2 ** num_kept, 2 ** num_traced))
    
    # Reduced density matrix rho_A = mat * mat^dagger
    rho_a = np.dot(mat, mat.conj().T)
    
    # 5. Von Neumann Entropy
    evals = np.linalg.eigvalsh(rho_a)
    positive_evals = evals[evals > 1e-12]
    positive_evals = positive_evals / np.sum(positive_evals)
    
    s = -np.sum(positive_evals * np.log2(positive_evals))
    return float(s)


def calculate_subsystem_entropy(params, num_qubits, trace_out_qubits):
    """
    Computes reduced density matrix rho_A by partial trace over `trace_out_qubits`
    and calculates Von Neumann entropy: S(rho_A) = -Tr(rho_A * log2(rho_A)).
    Uses Qiskit if available, or falls back to exact NumPy simulation.
    """
    if HAS_QISKIT:
        qc = construct_qieimap_circuit(params, num_qubits)
        state = Statevector.from_instruction(qc)
        rho_reduced = partial_trace(state, trace_out_qubits)
        eigenvalues = np.linalg.eigvalsh(rho_reduced.data)
        positive_evals = eigenvalues[eigenvalues > 1e-12]
        positive_evals = positive_evals / np.sum(positive_evals)
        return float(-np.sum(positive_evals * np.log2(positive_evals)))
    else:
        return simulate_entropy_numpy(params, num_qubits, trace_out_qubits)


# ---------------------------------------------------------------------------
# 5. FULL QIEI PIPELINE EXTRACTOR
# ---------------------------------------------------------------------------
def extract_qiei_features(landmarks_68):
    """
    Extracts complete QIEI descriptor from 68 2D facial landmarks.
    
    Parameters:
        landmarks_68 (np.ndarray): Shape (68, 2) array of landmark coordinates.
        
    Returns:
        feature_vector (np.ndarray): 1D array of intra- and inter-region scores.
        detailed_info (dict): Subsystem scores, mouth averaging, and metadata.
    """
    landmarks = np.asarray(landmarks_68, dtype=np.float32)
    
    # Extract 4 extreme points for all regions
    extremes = {}
    for r_name, indices in FACIAL_REGIONS.items():
        sub_pts = landmarks[indices]
        extremes[r_name] = get_extreme_coordinates(sub_pts)
        
    # --- A. Intra-Region Entanglement (3 Qubits) ---
    intra_scores = {}
    for r_name, coords in extremes.items():
        params = normalize_and_pad(coords, num_qubits=3)
        # Trace out qubits 1, 2 to observe subsystem qubit 0
        s = calculate_subsystem_entropy(params, num_qubits=3, trace_out_qubits=[1, 2])
        intra_scores[r_name] = s
        
    # Mouth score averaging (Section 3.5): 0.5 * (outer + inner)
    e_mouth = 0.5 * (intra_scores["mouth_outer"] + intra_scores["mouth_inner"])
    
    # Order of 7 primary intra features
    primary_regions = ["jaw", "right_eyebrow", "left_eyebrow", "nose", "right_eye", "left_eye"]
    qiei_intra = [intra_scores[r] for r in primary_regions] + [e_mouth]
    
    # --- B. Inter-Region Entanglement (5 Qubits) ---
    inter_scores = {}
    qiei_inter = []
    for r1, r2 in INTER_REGION_PAIRS:
        combined = np.concatenate([extremes[r1], extremes[r2]])
        params = normalize_and_pad(combined, num_qubits=5)
        # Trace out subsystem corresponding to region 2 (qubits 3, 4)
        s = calculate_subsystem_entropy(params, num_qubits=5, trace_out_qubits=[3, 4])
        pair_name = f"{r1}__{r2}"
        inter_scores[pair_name] = s
        qiei_inter.append(s)
        
    # Full concatenated feature vector
    qiei_descriptor = np.array(qiei_intra + qiei_inter, dtype=np.float32)
    
    details = {
        "intra_scores": intra_scores,
        "inter_scores": inter_scores,
        "mouth_averaged_score": e_mouth,
        "total_dimensions": len(qiei_descriptor),
        "intra_dimensions": len(qiei_intra),
        "inter_dimensions": len(qiei_inter)
    }
    
    return qiei_descriptor, details


# ---------------------------------------------------------------------------
# 6. QUICK SELF-TEST / DEMONSTRATION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing QIEI Extractor with synthetic 68-landmark facial coordinates...")
    
    # Create realistic synthetic face geometry
    t = np.linspace(0, 2 * np.pi, 68, endpoint=False)
    synthetic_landmarks = np.stack([
        150.0 + 60.0 * np.cos(t),
        150.0 + 80.0 * np.sin(t)
    ], axis=1).astype(np.float32)
    
    features, meta = extract_qiei_features(synthetic_landmarks)
    
    print("\nExtraction Successful!")
    print(f"Total Descriptor Dimensions: {meta['total_dimensions']}")
    print(f"  - Intra-region count     : {meta['intra_dimensions']}")
    print(f"  - Inter-region count     : {meta['inter_dimensions']}")
    print(f"  - First 7 Intra Scores   : {np.round(features[:7], 4)}")
    print(f"  - First 5 Inter Scores   : {np.round(features[7:12], 4)}")
