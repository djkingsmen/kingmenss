"""
stage1_linalg_core.py — Stage 1: Linear Algebra & Quantum Mathematical Engine Core
Paper: Quantum Information Entanglement Index for Facial Expression Analysis (ECML PKDD 2026)

This module implements the foundational quantum mathematical primitives:
- Statevector normalization and pure state representation
- Unitary gates (Hadamard, Rz phase rotation, CZ, RZZ)
- Tensor-network partial trace over bipartite quantum subsystems
- Von Neumann entropy computation with rigorous eigenvalue stability
- Self-verifying checkpoint test suite validating mathematical invariants
"""

import sys
import numpy as np


# ---------------------------------------------------------------------------
# 1. QUANTUM STATE & GATE OPERATORS
# ---------------------------------------------------------------------------
H_GATE = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)

def rz_gate(theta: float) -> np.ndarray:
    """Single-qubit Z-rotation gate: Rz(theta) = diag(exp(-i*theta/2), exp(i*theta/2))."""
    return np.array([
        [np.exp(-1j * 0.5 * theta), 0.0],
        [0.0, np.exp(1j * 0.5 * theta)]
    ], dtype=np.complex128)


def initialize_hadamard_state(num_qubits: int) -> np.ndarray:
    r"""
    Initializes a system of `num_qubits` into equal superposition:
    |psi> = H^{\otimes num_qubits} |0>^{\otimes num_qubits} = (1/sqrt(2^q)) * [1, 1, ..., 1]^T
    """
    dim = 2 ** num_qubits
    state = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)
    return state


def apply_phase_rotations(state: np.ndarray, num_qubits: int, theta_params: np.ndarray) -> np.ndarray:
    """
    Applies single-qubit phase rotations Rz(2 * theta_k) on each qubit k:
    |b> -> exp(i * sum_k (-1)^{b_k} * theta_k) |b>
    """
    dim = 2 ** num_qubits
    state_out = state.copy()
    for b in range(dim):
        total_phase = 0.0
        for k in range(num_qubits):
            bit = (b >> k) & 1
            theta = float(theta_params[k % len(theta_params)])
            # bit == 1 gets +theta, bit == 0 gets -theta
            total_phase += (1.0 if bit == 1 else -1.0) * theta
        state_out[b] *= np.exp(1j * total_phase)
    return state_out


def apply_pairwise_entanglement(state: np.ndarray, num_qubits: int, theta_params: np.ndarray) -> np.ndarray:
    """
    Applies pairwise entanglement gates:
    - CZ(j, j+1): flips sign if both qubits are 1
    - RZZ(j, j+1; theta): introduces phase based on parity b_j ^ b_{j+1}
    Includes circular boundary entanglement gate between qubit (q-1) and qubit 0.
    """
    dim = 2 ** num_qubits
    state_out = state.copy()
    
    # Adjacent pairs (q, q + 1)
    for q in range(num_qubits - 1):
        angle_zz = float(theta_params[(q + 1) % len(theta_params)])
        for b in range(dim):
            bit_q = (b >> q) & 1
            bit_next = (b >> (q + 1)) & 1
            # CZ gate: phase -1 if both 1
            if bit_q == 1 and bit_next == 1:
                state_out[b] *= -1.0
            # RZZ gate: exp(-i*theta/2) if same, exp(+i*theta/2) if different
            zz_parity = -1.0 if (bit_q == bit_next) else 1.0
            state_out[b] *= np.exp(1j * 0.5 * zz_parity * angle_zz)
            
    # Circular boundary closure for circuits with > 2 qubits
    if num_qubits > 2:
        angle_0 = float(theta_params[0])
        q_last = num_qubits - 1
        for b in range(dim):
            bit_last = (b >> q_last) & 1
            bit_0 = b & 1
            if bit_last == 1 and bit_0 == 1:
                state_out[b] *= -1.0
            zz_parity = -1.0 if (bit_last == bit_0) else 1.0
            state_out[b] *= np.exp(1j * 0.5 * zz_parity * angle_0)
            
    return state_out


# ---------------------------------------------------------------------------
# 2. PARTIAL TRACE & DENSITY MATRIX COMPUTATION
# ---------------------------------------------------------------------------
def compute_reduced_density_matrix(state: np.ndarray, num_qubits: int, trace_out_qubits: list) -> np.ndarray:
    """
    Computes the reduced density matrix rho_A = Tr_B(|psi><psi|)
    by reshaping the statevector into a multi-index tensor and contracting
    over the traced-out qubit indices.
    
    Parameters:
        state: 1D complex statevector of length 2^num_qubits
        num_qubits: Total number of qubits in the bipartite system
        trace_out_qubits: List of 0-indexed qubits to trace out (Subsystem B)
        
    Returns:
        rho_a: 2D square complex matrix of dimension (2^num_kept, 2^num_kept)
    """
    tensor_shape = [2] * num_qubits
    psi_tensor = state.reshape(tensor_shape)
    
    kept_qubits = [q for q in range(num_qubits) if q not in trace_out_qubits]
    num_kept = len(kept_qubits)
    num_traced = len(trace_out_qubits)
    
    # Permute tensor axes: kept qubits first, followed by traced qubits
    perm = kept_qubits + trace_out_qubits
    psi_permuted = np.transpose(psi_tensor, perm)
    
    # Reshape into matrix: shape (2^num_kept, 2^num_traced)
    mat = psi_permuted.reshape((2 ** num_kept, 2 ** num_traced))
    
    # rho_A = mat @ mat^dagger
    rho_a = np.dot(mat, mat.conj().T)
    return rho_a


# ---------------------------------------------------------------------------
# 3. VON NEUMANN ENTROPY
# ---------------------------------------------------------------------------
def compute_von_neumann_entropy(rho: np.ndarray, base: float = 2.0) -> float:
    """
    Calculates Von Neumann Entropy: S(rho) = -Tr(rho * log2(rho)) = -sum(lambda_i * log2(lambda_i))
    where lambda_i are the eigenvalues of Hermitian matrix rho.
    """
    # Compute real eigenvalues of Hermitian matrix
    eigenvalues = np.linalg.eigvalsh(rho)
    
    # Numerical stability filter: discard zero and negative numerical precision remnants
    positive_evals = eigenvalues[eigenvalues > 1e-12]
    
    if len(positive_evals) == 0:
        return 0.0
        
    # Re-normalize eigenvalues to ensure unit trace
    positive_evals = positive_evals / np.sum(positive_evals)
    
    if base == 2.0:
        entropy_val = -np.sum(positive_evals * np.log2(positive_evals))
    else:
        entropy_val = -np.sum(positive_evals * np.log(positive_evals))
        
    return float(entropy_val)


# ---------------------------------------------------------------------------
# 4. STAGE 1 VERIFIABLE CHECKPOINT SUITE
# ---------------------------------------------------------------------------
def run_stage1_checkpoint() -> bool:
    """
    Executes automated mathematical invariant tests.
    Returns True if all tests pass, False otherwise.
    """
    print("=" * 65)
    print("STAGE 1: LINEAR ALGEBRA & QUANTUM MATHEMATICAL ENGINE CHECKPOINT")
    print("=" * 65)
    
    np.random.seed(42)
    passed_tests = 0
    total_tests = 5
    
    # --- TEST 1: Statevector Unitarity ---
    num_qubits = 3
    dim = 2 ** num_qubits
    state = initialize_hadamard_state(num_qubits)
    norm = np.linalg.norm(state)
    test1_pass = np.isclose(norm, 1.0, atol=1e-7)
    if test1_pass:
        print("  [PASS] Test 1: Statevector Norm Unitarity ||psi|| = 1.0")
        passed_tests += 1
    else:
        print(f"  [FAIL] Test 1: Statevector norm deviates: {norm}")

    # --- TEST 2: Density Matrix Hermiticity ---
    params = np.random.uniform(0, np.pi, 8)
    state = apply_phase_rotations(state, num_qubits, params)
    state = apply_pairwise_entanglement(state, num_qubits, params)
    rho_a = compute_reduced_density_matrix(state, num_qubits, trace_out_qubits=[1, 2])
    
    hermitian_diff = np.max(np.abs(rho_a - rho_a.conj().T))
    test2_pass = hermitian_diff < 1e-10
    if test2_pass:
        print(f"  [PASS] Test 2: Reduced Density Matrix Hermiticity (max diff = {hermitian_diff:.2e} < 1e-10)")
        passed_tests += 1
    else:
        print(f"  [FAIL] Test 2: Non-Hermitian density matrix: diff = {hermitian_diff}")

    # --- TEST 3: Unit Trace Condition ---
    trace_val = np.real(np.trace(rho_a))
    test3_pass = np.isclose(trace_val, 1.0, atol=1e-7)
    if test3_pass:
        print(f"  [PASS] Test 3: Density Matrix Unit Trace Tr(rho_A) = {trace_val:.6f} == 1.0")
        passed_tests += 1
    else:
        print(f"  [FAIL] Test 3: Trace violates unity: {trace_val}")

    # --- TEST 4: Positive Semi-Definiteness ---
    evals = np.linalg.eigvalsh(rho_a)
    min_eval = np.min(evals)
    test4_pass = min_eval >= -1e-12
    if test4_pass:
        print(f"  [PASS] Test 4: Eigenvalues Positive Semi-Definite (min eigenvalue = {min_eval:.2e} >= 0)")
        passed_tests += 1
    else:
        print(f"  [FAIL] Test 4: Negative eigenvalue detected: {min_eval}")

    # --- TEST 5: Theoretical Entropy Bounds ---
    entropy_1q = compute_von_neumann_entropy(rho_a, base=2.0)
    # Maximum entropy for 1-qubit subsystem (dim=2) is log2(2) = 1.0
    test5_pass = (0.0 <= entropy_1q <= 1.0000001)
    
    # 5-qubit system test (tracing out 2 qubits -> 3-qubit subsystem, max entropy = log2(8) = 3.0)
    state_5q = initialize_hadamard_state(5)
    params_5q = np.random.uniform(0, np.pi, 16)
    state_5q = apply_phase_rotations(state_5q, 5, params_5q)
    state_5q = apply_pairwise_entanglement(state_5q, 5, params_5q)
    rho_3q = compute_reduced_density_matrix(state_5q, 5, trace_out_qubits=[3, 4])
    entropy_3q = compute_von_neumann_entropy(rho_3q, base=2.0)
    test5_pass = test5_pass and (0.0 <= entropy_3q <= 3.0000001)
    
    if test5_pass:
        print(f"  [PASS] Test 5: Von Neumann Entropy Bounds (1-qubit: {entropy_1q:.4f} in [0, 1], 3-qubit: {entropy_3q:.4f} in [0, 3])")
        passed_tests += 1
    else:
        print(f"  [FAIL] Test 5: Entropy out of theoretical bounds: 1q={entropy_1q}, 3q={entropy_3q}")

    print("-" * 65)
    all_passed = (passed_tests == total_tests)
    if all_passed:
        print("[STAGE 1 CHECKPOINT PASSED: ALL 5 MATHEMATICAL INVARIANTS VERIFIED]")
    else:
        print(f"[STAGE 1 CHECKPOINT FAILED: {passed_tests}/{total_tests} TESTS PASSED]")
    print("=" * 65)
    return all_passed


if __name__ == "__main__":
    success = run_stage1_checkpoint()
    sys.exit(0 if success else 1)
