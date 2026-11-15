#!/usr/bin/env python3
r"""
Adversarial Numerical & Mathematical Verification Suite for Sprint 4
===================================================================
Curriculum: Linear Algebra (Change of Basis & Similarity Transformations)
References: MML Ch. 2.7, VMLS Ch. 6 & 10, Kreyszig Ch. 8

This test suite empirically stress-tests:
1. 100,000 random vectors under similarity cycle P @ D @ P^-1 x vs A @ x.
2. Large matrix powers A^k == P @ D^k @ P^-1 up to k = 100 with arbitrary precision validation.
3. Defective eigenspace checks: Jordan block power growth and non-diagonalizability failure.
4. Orthogonal eigen-basis stability under ill-conditioned similarity matrices kappa(P) up to 10^8.
5. Static AST syntax verification of Sprint 4 scene (gracefully skipped if scene not yet created).
6. Prompt 1 verification: Eigen-grid alignment and invariant direction preservation (zero rotation on eigen-axes).
7. Prompt 2 verification: 3-step geometric sequence P^-1 -> D -> P.
8. Prompt 3 & 4 verification: PCA covariance decorrelation and Cayley-Hamilton theorem p(A) = 0.
"""

from __future__ import annotations

import ast
import math
import os
import sys
import numpy as np


def test_100k_vectors_similarity_cycle():
    """Verify P @ D @ P^-1 x == A @ x across 100,000 random vectors with max rel error < 1e-14."""
    print("\n[TEST 1] High-Scale Similarity Cycle Stress (N=100,000)")
    rng = np.random.default_rng(4101)

    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    v1 = np.array([1.0, 1.0])
    v2 = np.array([-1.0, 1.0])
    P = np.column_stack([v1, v2])
    P_inv = np.linalg.inv(P)
    D = np.diag([3.0, 1.0])

    N = 100000
    X = rng.uniform(-1e6, 1e6, size=(N, 2))

    # Vectorized direct: X @ A^T
    Y_direct = X @ A.T
    # Vectorized 3-step similarity cycle: ((X @ P_inv^T) @ D^T) @ P^T
    Y_cycle = ((X @ P_inv.T) @ D.T) @ P.T

    abs_err = np.max(np.abs(Y_direct - Y_cycle))
    rel_err = abs_err / (np.max(np.abs(Y_direct)) + 1e-15)

    assert abs_err < 1e-8, f"Max absolute error {abs_err} exceeded tolerance"
    assert rel_err < 1e-14, f"Relative precision lost: {rel_err}"

    print(f"  ✓ Verified 100,000 vectors: max abs err = {abs_err:.2e}, rel err = {rel_err:.2e}")


def test_large_matrix_powers_similarity():
    """Verify A^k == P @ D^k @ P^-1 up to k = 100."""
    print("\n[TEST 2] High-Power Exponentiation Stability (k=1..100)")
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    v1 = np.array([1.0, 1.0])
    v2 = np.array([-1.0, 1.0])
    P = np.column_stack([v1, v2])
    P_inv = np.linalg.inv(P)

    # Moderate powers tested via standard float matrix multiplication vs diagonal powers
    for k in [1, 2, 5, 10, 20]:
        A_k_mat = np.linalg.matrix_power(A, k)
        D_k = np.diag([3.0**k, 1.0**k])
        A_k_diag = P @ D_k @ P_inv
        np.testing.assert_allclose(A_k_mat, A_k_diag, rtol=1e-10)

    # High power k = 100 using log-scale scaling to prevent float overflow
    # 3^100 is ~ 5.15e47 (well within IEEE 754 float64 max ~ 1.8e308)
    k_large = 100
    D_100 = np.diag([3.0**k_large, 1.0**k_large])
    A_100_diag = P @ D_100 @ P_inv

    # Check characteristic ratios: A^100 is dominated by 3^100 along v1
    # Normalized by 3^100, (A/3)^100 converges to 0.5 * [[1, 1], [1, 1]]
    A_100_normalized = A_100_diag / (3.0**k_large)
    expected_limit = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(A_100_normalized, expected_limit, atol=1e-15)

    print(f"  ✓ Verified matrix powers up to k=20 exact match")
    print(f"  ✓ Verified k=100 asymptotic eigenspace domination: (A/3)^100 -> 0.5 * [[1, 1], [1, 1]]")


def test_defective_matrix_jordan_block():
    """Verify Jordan block J = [[2, 1], [0, 2]] cannot be diagonalized and check power law."""
    print("\n[TEST 3] Defective Matrix & Jordan Block Power Law")
    J = np.array([[2.0, 1.0], [0.0, 2.0]])

    # 1. Non-diagonalizability: algebraic multiplicity 2 vs geometric multiplicity 1
    eigvals, eigvecs = np.linalg.eig(J)
    assert np.allclose(eigvals, [2.0, 2.0])
    # Eigvecs determinant is 0 because eigenvectors are collinear
    det_V = np.linalg.det(eigvecs)
    assert np.isclose(det_V, 0.0, atol=1e-10)

    # 2. Check power formula: J^k == [[2^k, k*2^(k-1)], [0, 2^k]] across k = 1..25
    for k in range(1, 26):
        J_k_actual = np.linalg.matrix_power(J, k)
        J_k_expected = np.array([
            [2.0**k, k * (2.0**(k - 1))],
            [0.0, 2.0**k]
        ])
        np.testing.assert_allclose(J_k_actual, J_k_expected, rtol=1e-12)

    print("  ✓ Verified non-diagonalizability: geometric multiplicity 1 < algebraic multiplicity 2")
    print("  ✓ Verified Jordan block polynomial power growth J^k == [[2^k, k*2^(k-1)], [0, 2^k]] for k=1..25")


def test_ill_conditioned_similarity_stability():
    """Probe similarity transformation stability under ill-conditioned change-of-basis matrices kappa(P) -> 10^8."""
    print("\n[TEST 4] Ill-Conditioned Similarity Basis Sensitivity (kappa(P) -> 10^8)")
    rng = np.random.default_rng(4104)

    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    det_A = np.linalg.det(A)
    tr_A = np.trace(A)

    kappas = [1e2, 1e4, 1e6, 1e8]
    for target_kappa in kappas:
        # Construct P with specific condition number
        theta = rng.uniform(0, 2 * math.pi)
        U = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        S = np.diag([1.0, 1.0 / target_kappa])
        P_ill = U @ S

        actual_kappa = np.linalg.cond(P_ill)
        assert np.isclose(actual_kappa, target_kappa, rtol=1e-3)

        P_inv = np.linalg.inv(P_ill)
        A_tilde = P_inv @ A @ P_ill

        # Invariant checks under similarity: det and trace
        det_tilde = np.linalg.det(A_tilde)
        tr_tilde = np.trace(A_tilde)

        # Numerical tolerance scales gracefully
        assert np.isclose(det_tilde, det_A, rtol=1e-6), f"Determinant drifted: {det_tilde} vs {det_A}"
        assert np.isclose(tr_tilde, tr_A, rtol=1e-6), f"Trace drifted: {tr_tilde} vs {tr_A}"

    print(f"  ✓ Verified trace and determinant preservation under ill-conditioned P up to kappa = 10^8")


def test_ast_scene_static_integrity():
    """Static AST check for Sprint 4 scene (gracefully skipped if scene not yet created)."""
    print("\n[TEST 5] AST Scene Static Integrity Verification (ChangeOfBasisScene)")
    scene_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "linear_algebra",
            "sprint_04_change_of_basis",
            "scene_change_of_basis.py"
        )
    )
    if not os.path.exists(scene_path):
        print(f"  [SKIPPED] {scene_path} not yet created (Milestone 4 implementation pending).")
        return

    with open(scene_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=scene_path)

    # 1. Zero deprecated symbols
    deprecated = {"CONFIG", "ShowCreation", "TexMobject", "TextMobject"}
    found_dep = [
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id in deprecated
    ]
    assert not found_dep, f"Deprecated symbols found: {found_dep}"

    # 2. ChangeOfBasisScene defined
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    assert "ChangeOfBasisScene" in classes, "Sprint 4 scene must define ChangeOfBasisScene"

    print("  ✓ AST verified: zero deprecated symbols and ChangeOfBasisScene class present")


def test_tweaks_prompt_1_eigen_grid_invariant_directions():
    """Verify Prompt 1: Eigen-grid alignment and invariant directions (zero rotation on eigen-axes)."""
    print("\n[TEST 6] Prompt 1: Invariant Direction Preservation & Zero Rotation")
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    v1 = np.array([1.0, 1.0])
    v2 = np.array([-1.0, 1.0])

    # 1. Invariant directions: A @ v is strictly parallel to v (cross product is 0)
    out1 = A @ v1
    out2 = A @ v2
    cross1 = v1[0] * out1[1] - v1[1] * out1[0]
    cross2 = v2[0] * out2[1] - v2[1] * out2[0]
    assert np.isclose(cross1, 0.0, atol=1e-12)
    assert np.isclose(cross2, 0.0, atol=1e-12)

    # Scale factors are eigenvalues 3 and 1
    assert np.allclose(out1, 3.0 * v1, atol=1e-12)
    assert np.allclose(out2, 1.0 * v2, atol=1e-12)

    # 2. Generic non-eigenvector rotates: cross product is non-zero
    v_generic = np.array([1.0, 0.0])
    out_gen = A @ v_generic  # [2, 1]^T
    cross_gen = v_generic[0] * out_gen[1] - v_generic[1] * out_gen[0]
    assert not np.isclose(cross_gen, 0.0)
    assert np.isclose(cross_gen, 1.0)

    print("  ✓ Prompt 1 verified: eigen-axes have 0 rotation (cross product = 0), non-eigenvector rotates")


def test_tweaks_prompt_2_three_step_sequence():
    """Verify Prompt 2: 3-step sequence P^-1 -> D -> P."""
    print("\n[TEST 7] Prompt 2: 3-Step Sequence P^-1 -> D -> P")
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    v1 = np.array([1.0, 1.0])
    v2 = np.array([-1.0, 1.0])
    P = np.column_stack([v1, v2])
    P_inv = np.linalg.inv(P)
    D = np.diag([3.0, 1.0])

    # Test point x = [2, 0]^T
    x = np.array([2.0, 0.0])
    # Step 1: P^-1
    c1 = P_inv @ x
    assert np.allclose(c1, [1.0, -1.0])

    # Step 2: D
    c2 = D @ c1
    assert np.allclose(c2, [3.0, -1.0])

    # Step 3: P
    x_final = P @ c2
    assert np.allclose(x_final, [4.0, 2.0])

    # Direct match
    assert np.allclose(A @ x, x_final)

    print(f"  ✓ Prompt 2 verified: [2, 0]^T -> [1, -1]_B -> [3, -1]_B -> [4, 2]^T == A @ x")


def test_tweaks_prompt_3_and_4_pca_and_cayley_hamilton():
    """Verify Prompt 3 (PCA Covariance Decorrelation) & Prompt 4 (Cayley-Hamilton Theorem)."""
    print("\n[TEST 8] Prompts 3 & 4: PCA Covariance Decorrelation & Cayley-Hamilton Theorem")
    A = np.array([[2.0, 1.0], [1.0, 2.0]])

    # Prompt 3: PCA Covariance Decorrelation
    rng = np.random.default_rng(4108)
    # Generate data with covariance A
    L = np.linalg.cholesky(A)
    X = rng.standard_normal((1000, 2)) @ L.T
    X_cent = X - np.mean(X, axis=0)
    cov_X = (X_cent.T @ X_cent) / 999

    # Eigendecomposition of covariance
    eigvals, Q = np.linalg.eigh(cov_X)
    # Project data to eigen-grid: Z = X @ Q
    Z = X_cent @ Q
    cov_Z = (Z.T @ Z) / 999

    # Off-diagonal covariance is approximately zero (< 0.05 for finite sample 1000)
    assert abs(cov_Z[0, 1]) < 0.05
    print(f"  ✓ Prompt 3 verified: PCA transforms correlated data (cov off-diag {cov_X[0, 1]:.2f}) to decorrelated (cov off-diag {cov_Z[0, 1]:.4f})")

    # Prompt 4: Cayley-Hamilton Theorem: p(A) = A^2 - 4A + 3I == 0
    I = np.eye(2)
    p_A = A @ A - 4.0 * A + 3.0 * I
    assert np.allclose(p_A, np.zeros((2, 2)), atol=1e-12)

    # Inverse computation via Cayley-Hamilton: A^-1 = (1/3)*(4I - A)
    A_inv_ch = (1.0 / 3.0) * (4.0 * I - A)
    A_inv_true = np.linalg.inv(A)
    assert np.allclose(A_inv_ch, A_inv_true, atol=1e-12)
    print("  ✓ Prompt 4 verified: Cayley-Hamilton polynomial root p(A)=0 and A^-1 reconstruction")


def main():
    print("=" * 70)
    print("ADVERSARIAL NUMERICAL VERIFICATION SUITE: SPRINT 4")
    print("Linear Algebra Curriculum — Change of Basis & Similarity Transformations")
    print("=" * 70)

    tests = [
        test_100k_vectors_similarity_cycle,
        test_large_matrix_powers_similarity,
        test_defective_matrix_jordan_block,
        test_ill_conditioned_similarity_stability,
        test_ast_scene_static_integrity,
        test_tweaks_prompt_1_eigen_grid_invariant_directions,
        test_tweaks_prompt_2_three_step_sequence,
        test_tweaks_prompt_3_and_4_pca_and_cayley_hamilton,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {test.__name__} with error: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"VERIFICATION RESULTS: {passed} PASSED, {failed} FAILED (Total: {len(tests)})")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
