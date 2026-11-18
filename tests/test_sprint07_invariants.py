#!/usr/bin/env python3
"""
5-Tier Mathematical Invariant & AST Verification Suite for Sprint 7
===================================================================
Topic: Gram-Schmidt Orthonormalization & QR Decomposition
References: MML §§3.5, 3.8.2-3.8.3; VMLS §§5.3-5.4, 10.4, 11.3, 11.5, 12.1-12.3; Kreyszig §20.9
"""

from __future__ import annotations

import ast
import os
import unittest

import numpy as np


# -----------------------------------------------------------------------------
# Canonical Constants & Standalone Deterministic Algorithms Under Test
# -----------------------------------------------------------------------------
A_CANON = np.array([
    [2.0, 0.0, -1.0],
    [1.0, 2.0, -1.0],
    [2.0, 2.0, 3.0],
])
Q_CANON = np.array([
    [2.0, -2.0, -1.0],
    [1.0, 2.0, -2.0],
    [2.0, 1.0, 2.0],
]) / 3.0
R_CANON = np.array([
    [3.0, 2.0, 1.0],
    [0.0, 2.0, 1.0],
    [0.0, 0.0, 3.0],
])
B_CANON = np.array([1.0, -2.0, 3.0])
X_CANON = np.array([1.0, -1.0, 1.0])


def cgs_qr(A: np.ndarray, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray, int]:
    """Classical Gram-Schmidt QR with early termination on linear dependence."""
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n), dtype=float)
    R = np.zeros((n, n), dtype=float)
    rank = 0
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            r_ij = float(np.dot(Q[:, i], A[:, j]))
            R[i, j] = r_ij
            v = v - r_ij * Q[:, i]
        norm_v = float(np.linalg.norm(v))
        if norm_v <= tol:
            R[j, j] = 0.0
            break
        R[j, j] = norm_v
        Q[:, j] = v / norm_v
        rank += 1
    return Q, R, rank


def mgs_qr(A: np.ndarray, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray, int]:
    """Modified Gram-Schmidt QR with early termination on linear dependence."""
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    V = A.copy()
    Q = np.zeros((m, n), dtype=float)
    R = np.zeros((n, n), dtype=float)
    rank = 0
    for i in range(n):
        norm_v = float(np.linalg.norm(V[:, i]))
        if norm_v <= tol:
            R[i, i] = 0.0
            break
        R[i, i] = norm_v
        Q[:, i] = V[:, i] / norm_v
        rank += 1
        for j in range(i + 1, n):
            r_ij = float(np.dot(Q[:, i], V[:, j]))
            R[i, j] = r_ij
            V[:, j] = V[:, j] - r_ij * Q[:, i]
    return Q, R, rank


def mgs_skip_dependent(A: np.ndarray, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Modified Gram-Schmidt skipping dependent columns to extract full column-space ONB."""
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    q_cols: list[np.ndarray] = []
    pivots: list[int] = []
    for j in range(n):
        v = A[:, j].copy()
        for q in q_cols:
            v = v - float(np.dot(q, v)) * q
        norm_v = float(np.linalg.norm(v))
        if norm_v > tol:
            q_cols.append(v / norm_v)
            pivots.append(j)
    Q_basis = np.column_stack(q_cols) if q_cols else np.zeros((m, 0), dtype=float)
    R_compact = Q_basis.T @ A if q_cols else np.zeros((0, n), dtype=float)
    return Q_basis, R_compact, pivots


def cgs2_qr(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Twice-is-enough reorthogonalized Classical Gram-Schmidt."""
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n), dtype=float)
    R = np.zeros((n, n), dtype=float)
    for j in range(n):
        v = A[:, j].copy()
        for _ in range(2):
            for i in range(j):
                coeff = float(np.dot(Q[:, i], v))
                R[i, j] += coeff
                v = v - coeff * Q[:, i]
        R[j, j] = float(np.linalg.norm(v))
        Q[:, j] = v / R[j, j]
    return Q, R


def canonical_householder_qr(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Thin Householder QR via NumPy canonicalized so diag(R) >= 0."""
    Q, R = np.linalg.qr(A, mode="reduced")
    signs = np.where(np.diag(R) >= 0.0, 1.0, -1.0)
    D = np.diag(signs)
    return Q @ D, D @ R


def back_substitute(R: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve upper-triangular system R x = y via back substitution."""
    n = R.shape[0]
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        rhs = y[i] - float(np.dot(R[i, i + 1 :], x[i + 1 :]))
        x[i] = rhs / R[i, i]
    return x


def lauchli_matrix(eps: float) -> np.ndarray:
    """4x3 Lauchli matrix with condition number ~ sqrt(3)/eps."""
    return np.array([
        [1.0, 1.0, 1.0],
        [eps, 0.0, 0.0],
        [0.0, eps, 0.0],
        [0.0, 0.0, eps],
    ], dtype=float)


def hilbert_matrix(n: int) -> np.ndarray:
    """n x n Hilbert matrix H_ij = 1 / (i + j + 1)."""
    idx = np.arange(n, dtype=float)
    return 1.0 / (idx[:, None] + idx[None, :] + 1.0)


# =============================================================================
# TIER 1: FEATURE COVERAGE (5 Classes x 5 Tests = 25 Tests)
# =============================================================================

class TestTier1OrthonormalBasisExpansion(unittest.TestCase):
    """Tier 1.1: Feature #11 — Orthonormal Basis & Inversion-Free Expansion."""

    def test_columns_unit_norm_and_mutual_orthogonality(self):
        self.assertTrue(np.allclose(Q_CANON.T @ Q_CANON, np.eye(3), atol=1e-12))
        for i in range(3):
            self.assertAlmostEqual(float(np.linalg.norm(Q_CANON[:, i])), 1.0, places=12)

    def test_orthonormal_expansion_without_inversion(self):
        Q_tall = Q_CANON[:, :2]
        x_in_span = 2.5 * Q_CANON[:, 0] - 1.75 * Q_CANON[:, 1]
        recon = Q_tall @ (Q_tall.T @ x_in_span)
        self.assertTrue(np.allclose(recon, x_in_span, atol=1e-12))

    def test_norm_preservation_parseval(self):
        c = np.array([3.0, -4.0, 12.0])
        self.assertAlmostEqual(float(np.linalg.norm(Q_CANON @ c)), float(np.linalg.norm(c)), places=12)
        Q_tall = Q_CANON[:, :2]
        x_in_span = 1.5 * Q_CANON[:, 0] - 2.0 * Q_CANON[:, 1]
        self.assertAlmostEqual(float(np.linalg.norm(Q_tall.T @ x_in_span)), float(np.linalg.norm(x_in_span)), places=12)

    def test_orthonormal_projector_axioms(self):
        Q_tall = Q_CANON[:, :2]
        P = Q_tall @ Q_tall.T
        self.assertTrue(np.allclose(P @ P, P, atol=1e-12))
        self.assertTrue(np.allclose(P.T, P, atol=1e-12))
        self.assertAlmostEqual(float(np.trace(P)), 2.0, places=12)

    def test_square_orthogonal_inverse_equals_transpose(self):
        self.assertTrue(np.allclose(np.linalg.inv(Q_CANON), Q_CANON.T, atol=1e-12))
        self.assertTrue(np.allclose(Q_CANON @ Q_CANON.T, np.eye(3), atol=1e-12))


class TestTier1GramSchmidtStripping(unittest.TestCase):
    """Tier 1.2: Feature #12 — Sequential Gram-Schmidt Stripping in 3D."""

    def test_step1_first_vector_normalization(self):
        u1 = A_CANON[:, 0]
        q1 = u1 / np.linalg.norm(u1)
        self.assertAlmostEqual(float(np.linalg.norm(q1)), 1.0, places=12)
        self.assertTrue(np.allclose(q1, Q_CANON[:, 0], atol=1e-12))
        self.assertTrue(np.allclose(np.cross(u1, q1), np.zeros(3), atol=1e-12))

    def test_step2_parallel_component_stripping(self):
        q1 = Q_CANON[:, 0]
        a2 = A_CANON[:, 1]
        p21 = float(np.dot(q1, a2)) * q1
        u2 = a2 - p21
        self.assertAlmostEqual(float(np.dot(u2, q1)), 0.0, places=12)
        self.assertTrue(np.allclose(p21 + u2, a2, atol=1e-12))
        self.assertTrue(np.allclose(u2 / np.linalg.norm(u2), Q_CANON[:, 1], atol=1e-12))

    def test_step3_sequential_stripping_3d(self):
        q1, q2 = Q_CANON[:, 0], Q_CANON[:, 1]
        a3 = A_CANON[:, 2]
        u3 = a3 - float(np.dot(q1, a3)) * q1 - float(np.dot(q2, a3)) * q2
        self.assertAlmostEqual(float(np.dot(u3, q1)), 0.0, places=12)
        self.assertAlmostEqual(float(np.dot(u3, q2)), 0.0, places=12)
        q3 = u3 / np.linalg.norm(u3)
        self.assertTrue(np.allclose(q3, Q_CANON[:, 2], atol=1e-12))
        self.assertAlmostEqual(abs(float(np.linalg.det(Q_CANON))), 1.0, places=12)

    def test_prefix_subspace_span_invariance(self):
        Q, _, rank = mgs_qr(A_CANON)
        self.assertEqual(rank, 3)
        for k in (1, 2, 3):
            Ak = A_CANON[:, :k]
            Qk = Q[:, :k]
            PA = Ak @ np.linalg.inv(Ak.T @ Ak) @ Ak.T
            PQ = Qk @ Qk.T
            self.assertTrue(np.allclose(PA, PQ, atol=1e-12))

    def test_cgs_and_mgs_agree_on_well_conditioned_3x3(self):
        Q_c, R_c, rank_c = cgs_qr(A_CANON)
        Q_m, R_m, rank_m = mgs_qr(A_CANON)
        self.assertEqual(rank_c, 3)
        self.assertEqual(rank_m, 3)
        self.assertLess(float(np.linalg.norm(Q_c - Q_m, "fro")), 1e-12)
        self.assertLess(float(np.linalg.norm(R_c - R_m, "fro")), 1e-12)


class TestTier1QRFactorizationAndUniqueness(unittest.TestCase):
    """Tier 1.3: Feature #13 — Thin/Full QR Factorization & Sign Uniqueness."""

    def test_thin_qr_reconstruction_and_upper_triangular_r(self):
        rng = np.random.default_rng(7131)
        A = rng.standard_normal((6, 4))
        Q, R, rank = mgs_qr(A)
        self.assertEqual(rank, 4)
        self.assertTrue(np.allclose(Q @ R, A, atol=1e-12))
        self.assertTrue(np.allclose(np.tril(R, -1), 0.0, atol=1e-12))
        self.assertTrue(np.allclose(Q.T @ A, R, atol=1e-12))

    def test_full_qr_orthogonal_completion(self):
        rng = np.random.default_rng(7132)
        A = rng.standard_normal((6, 3))
        Q_full, R_full = np.linalg.qr(A, mode="complete")
        self.assertEqual(Q_full.shape, (6, 6))
        self.assertEqual(R_full.shape, (6, 3))
        self.assertTrue(np.allclose(Q_full.T @ Q_full, np.eye(6), atol=1e-12))
        self.assertTrue(np.allclose(Q_full @ Q_full.T, np.eye(6), atol=1e-12))
        self.assertTrue(np.allclose(R_full[3:, :], 0.0, atol=1e-12))
        self.assertTrue(np.allclose(Q_full @ R_full, A, atol=1e-12))

    def test_positive_diagonal_convention(self):
        Q, R, _ = mgs_qr(A_CANON)
        self.assertTrue(np.all(np.diag(R) > 0.0))
        self.assertTrue(np.allclose(Q, Q_CANON, atol=1e-12))
        self.assertTrue(np.allclose(R, R_CANON, atol=1e-12))

    def test_qr_uniqueness_with_positive_diagonal_and_sign_diagonal(self):
        Q_m, R_m, _ = mgs_qr(A_CANON)
        Q_hh, R_hh = canonical_householder_qr(A_CANON)
        self.assertTrue(np.allclose(Q_m, Q_hh, atol=1e-12))
        self.assertTrue(np.allclose(R_m, R_hh, atol=1e-12))
        D = np.diag([1.0, -1.0, 1.0])
        Q_flip, R_flip = Q_m @ D, D @ R_m
        self.assertTrue(np.allclose(Q_flip @ R_flip, A_CANON, atol=1e-12))
        self.assertTrue(np.allclose(Q_m.T @ Q_flip, D, atol=1e-12))

    def test_parallelotope_volume_equals_product_of_r_diagonals(self):
        _, R, _ = mgs_qr(A_CANON)
        self.assertAlmostEqual(abs(float(np.linalg.det(A_CANON))), float(np.prod(np.diag(R))), places=12)
        self.assertAlmostEqual(float(np.prod(np.diag(R))), 18.0, places=12)


class TestTier1QRLeastSquaresBackSub(unittest.TestCase):
    """Tier 1.4: Feature #14a — QR Least Squares & Back-Substitution."""

    def test_overdetermined_qr_backsub_matches_lstsq(self):
        rng = np.random.default_rng(7141)
        A = rng.standard_normal((12, 4))
        b = rng.standard_normal(12)
        Q, R, rank = mgs_qr(A)
        self.assertEqual(rank, 4)
        x_qr = back_substitute(R, Q.T @ b)
        x_ref, *_ = np.linalg.lstsq(A, b, rcond=None)
        self.assertTrue(np.allclose(x_qr, x_ref, atol=1e-12))

    def test_least_squares_residual_orthogonality(self):
        rng = np.random.default_rng(7142)
        A = rng.standard_normal((10, 3))
        b = rng.standard_normal(10)
        Q, R, _ = mgs_qr(A)
        x_hat = back_substitute(R, Q.T @ b)
        r = b - A @ x_hat
        self.assertTrue(np.allclose(A.T @ r, np.zeros(3), atol=1e-12))
        self.assertTrue(np.allclose(Q.T @ r, np.zeros(3), atol=1e-12))

    def test_normal_equations_equivalence(self):
        rng = np.random.default_rng(7143)
        A = rng.standard_normal((8, 3))
        b = rng.standard_normal(8)
        Q, R, _ = mgs_qr(A)
        x_hat = back_substitute(R, Q.T @ b)
        self.assertTrue(np.allclose(A.T @ A @ x_hat, A.T @ b, atol=1e-11))

    def test_pythagorean_norm_decomposition(self):
        rng = np.random.default_rng(7144)
        A = rng.standard_normal((9, 3))
        b = rng.standard_normal(9)
        Q, _, _ = mgs_qr(A)
        b_proj = Q @ (Q.T @ b)
        r = b - b_proj
        lhs = float(np.dot(b, b))
        rhs = float(np.dot(Q.T @ b, Q.T @ b)) + float(np.dot(r, r))
        self.assertAlmostEqual(lhs, rhs, places=11)

    def test_consistent_system_zero_residual(self):
        Q, R, _ = mgs_qr(A_CANON)
        x_hat = back_substitute(R, Q.T @ B_CANON)
        self.assertTrue(np.allclose(x_hat, X_CANON, atol=1e-12))
        self.assertLess(float(np.linalg.norm(B_CANON - A_CANON @ x_hat)), 1e-12)


class TestTier1RankDeficientEarlyTermination(unittest.TestCase):
    """Tier 1.5: Feature #14b — Linear Dependence & Early Termination."""

    def test_dependent_third_column_zero_residual_and_r33(self):
        a3_deg = A_CANON[:, 1] - A_CANON[:, 0]
        A_deg = np.column_stack([A_CANON[:, 0], A_CANON[:, 1], a3_deg])
        _, R_deg, rank = mgs_qr(A_deg)
        self.assertEqual(rank, 2)
        self.assertAlmostEqual(R_deg[2, 2], 0.0, places=12)

    def test_dependency_coefficients_encoded_in_r(self):
        a3_deg = 2.0 * A_CANON[:, 0] - 1.5 * A_CANON[:, 1]
        A_deg = np.column_stack([A_CANON[:, 0], A_CANON[:, 1], a3_deg])
        Q_deg, R_deg, rank = mgs_qr(A_deg)
        self.assertEqual(rank, 2)
        recon_a3 = R_deg[0, 2] * Q_deg[:, 0] + R_deg[1, 2] * Q_deg[:, 1]
        self.assertTrue(np.allclose(recon_a3, a3_deg, atol=1e-12))

    def test_rank_r_compact_factorization(self):
        a3_deg = A_CANON[:, 1] - A_CANON[:, 0]
        A_deg = np.column_stack([A_CANON[:, 0], A_CANON[:, 1], a3_deg])
        Q2, R23, pivots = mgs_skip_dependent(A_deg)
        self.assertEqual(pivots, [0, 1])
        self.assertEqual(Q2.shape, (3, 2))
        self.assertEqual(R23.shape, (2, 3))
        self.assertTrue(np.allclose(Q2.T @ Q2, np.eye(2), atol=1e-12))
        self.assertTrue(np.allclose(Q2 @ R23, A_deg, atol=1e-12))

    def test_rank_1_early_termination(self):
        a1 = np.array([1.0, 2.0, -2.0])
        A_r1 = np.column_stack([a1, -3.0 * a1, 0.5 * a1])
        _, R_r1, rank = mgs_qr(A_r1)
        self.assertEqual(rank, 1)
        self.assertAlmostEqual(R_r1[1, 1], 0.0, places=12)
        self.assertAlmostEqual(R_r1[2, 2], 0.0, places=12)

    def test_skip_dependent_middle_column_recovers_full_rank_basis(self):
        a1 = A_CANON[:, 0]
        a2_dep = 2.5 * a1
        a3 = A_CANON[:, 1]
        A_mid = np.column_stack([a1, a2_dep, a3])
        Q_basis, R_comp, pivots = mgs_skip_dependent(A_mid)
        self.assertEqual(pivots, [0, 2])
        self.assertEqual(Q_basis.shape, (3, 2))
        self.assertTrue(np.allclose(Q_basis.T @ Q_basis, np.eye(2), atol=1e-12))
        self.assertTrue(np.allclose(Q_basis @ R_comp, A_mid, atol=1e-12))


# =============================================================================
# TIER 2: BOUNDARY & CORNER CASES (5 Classes x 5 Tests = 25 Tests)
# =============================================================================

class TestTier2ZeroAndCollinearInputs(unittest.TestCase):
    """Tier 2.1: Zero, collinear, single-column, and wide matrix boundaries."""

    def test_zero_matrix_3x3_rank_zero(self):
        Z = np.zeros((3, 3))
        Q, R, rank = mgs_qr(Z)
        self.assertEqual(rank, 0)
        self.assertTrue(np.allclose(Q, 0.0))
        self.assertTrue(np.allclose(R, 0.0))

    def test_zero_first_column_early_termination_vs_skip(self):
        A_z0 = np.column_stack([np.zeros(3), A_CANON[:, 0], A_CANON[:, 1]])
        _, R_stop, rank_stop = mgs_qr(A_z0)
        self.assertEqual(rank_stop, 0)
        self.assertEqual(R_stop[0, 0], 0.0)
        Q_skip, R_skip, pivots = mgs_skip_dependent(A_z0)
        self.assertEqual(pivots, [1, 2])
        self.assertTrue(np.allclose(Q_skip @ R_skip, A_z0, atol=1e-12))

    def test_duplicate_columns_detected(self):
        A_dup = np.column_stack([A_CANON[:, 0], A_CANON[:, 0], A_CANON[:, 1]])
        _, _, rank = mgs_qr(A_dup)
        self.assertEqual(rank, 1)
        Q_skip, _, pivots = mgs_skip_dependent(A_dup)
        self.assertEqual(pivots, [0, 2])
        self.assertEqual(Q_skip.shape, (3, 2))

    def test_single_column_mx1_matrix(self):
        v = np.array([[3.0], [0.0], [4.0], [0.0]])
        Q, R, rank = mgs_qr(v)
        self.assertEqual(rank, 1)
        self.assertTrue(np.allclose(Q, np.array([[0.6], [0.0], [0.8], [0.0]])))
        self.assertAlmostEqual(R[0, 0], 5.0, places=12)

    def test_wide_m_less_than_n_cannot_produce_n_orthonormal_vectors(self):
        rng = np.random.default_rng(7215)
        A_wide = rng.standard_normal((3, 5))
        Q_skip, R_skip, pivots = mgs_skip_dependent(A_wide)
        self.assertEqual(len(pivots), 3)
        self.assertEqual(Q_skip.shape, (3, 3))
        self.assertTrue(np.allclose(Q_skip @ R_skip, A_wide, atol=1e-12))


class TestTier2AlreadyOrthonormalAndDiagonal(unittest.TestCase):
    """Tier 2.2: Identity, signed diagonal, reflection, permutation, and scaled orthogonal inputs."""

    def test_identity_matrix_qr_is_identity(self):
        Q, R, rank = mgs_qr(np.eye(3))
        self.assertEqual(rank, 3)
        self.assertTrue(np.allclose(Q, np.eye(3), atol=1e-12))
        self.assertTrue(np.allclose(R, np.eye(3), atol=1e-12))

    def test_negative_diagonal_enforces_positive_r_diagonal(self):
        D_in = np.diag([-2.0, 3.0, -5.0])
        Q, R, rank = mgs_qr(D_in)
        self.assertEqual(rank, 3)
        self.assertTrue(np.allclose(Q, np.diag([-1.0, 1.0, -1.0]), atol=1e-12))
        self.assertTrue(np.allclose(R, np.diag([2.0, 3.0, 5.0]), atol=1e-12))

    def test_householder_reflection_orthogonal_det_minus_one(self):
        u = np.array([1.0, -2.0, 2.0]) / 3.0
        H = np.eye(3) - 2.0 * np.outer(u, u)
        Q, R, rank = mgs_qr(H)
        self.assertEqual(rank, 3)
        self.assertAlmostEqual(float(np.linalg.det(Q)), -1.0, places=12)
        self.assertTrue(np.allclose(Q, H, atol=1e-12))
        self.assertTrue(np.allclose(R, np.eye(3), atol=1e-12))

    def test_permutation_matrix_qr(self):
        P = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        Q, R, rank = mgs_qr(P)
        self.assertEqual(rank, 3)
        self.assertTrue(np.allclose(Q, P, atol=1e-12))
        self.assertTrue(np.allclose(R, np.eye(3), atol=1e-12))

    def test_uniformly_scaled_orthogonal_matrix(self):
        c = 7.5
        A = c * Q_CANON
        Q, R, rank = mgs_qr(A)
        self.assertEqual(rank, 3)
        self.assertTrue(np.allclose(Q, Q_CANON, atol=1e-12))
        self.assertTrue(np.allclose(R, c * np.eye(3), atol=1e-12))


class TestTier2ExtremeScalesAndPrecision(unittest.TestCase):
    """Tier 2.3: Large/small scales, mixed column norms, near-dependence, and NaN/Inf guards."""

    def test_large_scale_1e10_q_invariant_r_scales(self):
        Q, R, rank = mgs_qr(1e10 * A_CANON)
        self.assertEqual(rank, 3)
        self.assertTrue(np.allclose(Q, Q_CANON, atol=1e-12))
        self.assertTrue(np.allclose(R, 1e10 * R_CANON, rtol=1e-12))

    def test_small_scale_1e_minus10_q_invariant_r_scales(self):
        Q, R, rank = mgs_qr(1e-10 * A_CANON, tol=1e-15)
        self.assertEqual(rank, 3)
        self.assertTrue(np.allclose(Q, Q_CANON, atol=1e-11))
        self.assertTrue(np.allclose(R, 1e-10 * R_CANON, rtol=1e-11))

    def test_mixed_column_norms_1e8_and_1e_minus8(self):
        A_mix = np.column_stack([
            1e8 * np.array([1.0, 1.0, 0.0]),
            1e-8 * np.array([1.0, -1.0, 1.0]),
            np.array([0.0, 0.0, 1.0]),
        ])
        Q, R, rank = mgs_qr(A_mix, tol=1e-15)
        self.assertEqual(rank, 3)
        self.assertTrue(np.allclose(Q.T @ Q, np.eye(3), atol=1e-12))
        self.assertTrue(np.allclose(Q @ R, A_mix, rtol=1e-12, atol=1e-12))

    def test_subthreshold_noise_vs_true_independence(self):
        a1 = np.array([1.0, 0.0, 0.0])
        a2_noise = np.array([1.0, 1e-15, 0.0])
        a2_real = np.array([1.0, 1e-6, 0.0])
        _, _, r_noise = mgs_qr(np.column_stack([a1, a2_noise]), tol=1e-12)
        Q_real, _, r_real = mgs_qr(np.column_stack([a1, a2_real]), tol=1e-12)
        self.assertEqual(r_noise, 1)
        self.assertEqual(r_real, 2)
        self.assertTrue(np.allclose(Q_real.T @ Q_real, np.eye(2), atol=1e-12))

    def test_zero_norm_guard_produces_no_nan_or_inf(self):
        A_bad = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
        Q_c, R_c, rc = cgs_qr(A_bad)
        Q_m, R_m, rm = mgs_qr(A_bad)
        self.assertEqual(rc, 0)
        self.assertEqual(rm, 0)
        self.assertFalse(np.any(np.isnan(Q_c)) or np.any(np.isinf(Q_c)))
        self.assertFalse(np.any(np.isnan(R_c)) or np.any(np.isinf(R_c)))
        self.assertFalse(np.any(np.isnan(Q_m)) or np.any(np.isinf(Q_m)))
        self.assertFalse(np.any(np.isnan(R_m)) or np.any(np.isinf(R_m)))


class TestTier2IllConditioningCGSvsMGS(unittest.TestCase):
    """Tier 2.4: Lauchli & Hilbert ill-conditioning (CGS vs MGS vs CGS2 vs Householder)."""

    def test_lauchli_eps_1e_minus8_cgs_fails_mgs_and_hh_succeed(self):
        A = lauchli_matrix(1e-8)
        Q_c, _, _ = cgs_qr(A, tol=1e-15)
        Q_m, _, _ = mgs_qr(A, tol=1e-15)
        Q_h, _ = canonical_householder_qr(A)
        err_c = float(np.linalg.norm(Q_c.T @ Q_c - np.eye(3), "fro"))
        err_m = float(np.linalg.norm(Q_m.T @ Q_m - np.eye(3), "fro"))
        err_h = float(np.linalg.norm(Q_h.T @ Q_h - np.eye(3), "fro"))
        self.assertGreater(err_c, 1e-1)
        self.assertLess(err_m, 1e-7)
        self.assertLess(err_h, 1e-14)

    def test_hilbert_h6_and_h8_orthogonality_hierarchy(self):
        for n in (6, 8):
            H = hilbert_matrix(n)
            Q_c, _, _ = cgs_qr(H, tol=1e-15)
            Q_m, _, _ = mgs_qr(H, tol=1e-15)
            Q_h, _ = canonical_householder_qr(H)
            err_c = float(np.linalg.norm(Q_c.T @ Q_c - np.eye(n), "fro"))
            err_m = float(np.linalg.norm(Q_m.T @ Q_m - np.eye(n), "fro"))
            err_h = float(np.linalg.norm(Q_h.T @ Q_h - np.eye(n), "fro"))
            self.assertGreater(err_c, err_m * 100.0)
            self.assertLess(err_h, 1e-14)

    def test_nearly_collinear_pair_orthogonality(self):
        A_pair = np.array([[1.0, 1.0], [0.0, 1e-8], [0.0, 0.0]])
        Q_m, R_m, rank = mgs_qr(A_pair, tol=1e-15)
        self.assertEqual(rank, 2)
        self.assertLess(float(np.linalg.norm(Q_m.T @ Q_m - np.eye(2), "fro")), 1e-14)
        self.assertTrue(np.allclose(Q_m @ R_m, A_pair, atol=1e-15))

    def test_cgs2_reorthogonalization_restores_machine_orthogonality(self):
        A = lauchli_matrix(1e-8)
        Q_c2, R_c2 = cgs2_qr(A)
        err_c2 = float(np.linalg.norm(Q_c2.T @ Q_c2 - np.eye(3), "fro"))
        self.assertLess(err_c2, 1e-14)
        self.assertTrue(np.allclose(Q_c2 @ R_c2, A, atol=1e-14))

    def test_cgs_still_reconstructs_a_despite_losing_orthogonality(self):
        A = lauchli_matrix(1e-8)
        Q_c, R_c, _ = cgs_qr(A, tol=1e-15)
        self.assertGreater(float(np.linalg.norm(Q_c.T @ Q_c - np.eye(3), "fro")), 1e-1)
        self.assertLess(float(np.linalg.norm(Q_c @ R_c - A, "fro")), 1e-14)


class TestTier2ConditionNumberQRvsNormalEq(unittest.TestCase):
    """Tier 2.5: Condition-number squaring in Normal Equations vs QR stability."""

    def test_normal_equations_squares_condition_number(self):
        A = lauchli_matrix(1e-4)
        k_A = float(np.linalg.cond(A))
        k_AtA = float(np.linalg.cond(A.T @ A))
        self.assertAlmostEqual(k_AtA / (k_A ** 2), 1.0, delta=0.05)

    def test_lauchli_1e_minus8_normal_eq_collapses_while_qr_recovers(self):
        A = lauchli_matrix(1e-8)
        x_true = np.array([1.0, -2.0, 1.0])
        b = A @ x_true
        ne_failed = False
        try:
            x_ne = np.linalg.solve(A.T @ A, A.T @ b)
            if np.linalg.norm(x_ne - x_true) > 1e-2:
                ne_failed = True
        except np.linalg.LinAlgError:
            ne_failed = True
        self.assertTrue(ne_failed)
        Q_h, R_h = canonical_householder_qr(A)
        x_qr = back_substitute(R_h, Q_h.T @ b)
        rel_err = float(np.linalg.norm(x_qr - x_true) / np.linalg.norm(x_true))
        self.assertLess(rel_err, 1e-7)

    def test_square_m_equals_n_exact_boundary(self):
        rng = np.random.default_rng(7253)
        A = rng.standard_normal((5, 5)) + 2.0 * np.eye(5)
        b = rng.standard_normal(5)
        Q, R, _ = mgs_qr(A)
        x_qr = back_substitute(R, Q.T @ b)
        self.assertTrue(np.allclose(A @ x_qr, b, atol=1e-12))

    def test_tall_skinny_500x4_qr_accuracy(self):
        rng = np.random.default_rng(7254)
        A = rng.standard_normal((500, 4))
        x_true = np.array([2.0, -1.0, 0.5, 3.0])
        b = A @ x_true
        Q, R, rank = mgs_qr(A)
        self.assertEqual(rank, 4)
        x_qr = back_substitute(R, Q.T @ b)
        self.assertTrue(np.allclose(x_qr, x_true, atol=1e-12))

    def test_float32_vs_float64_cancellation_sensitivity(self):
        A64 = lauchli_matrix(1e-4)
        A32 = A64.astype(np.float32)
        # In float32 eps^2 = 1e-8 < float32 eps_mach (~1.19e-7) so normal eq collapses earlier
        G32 = A32.T @ A32
        self.assertTrue(np.allclose(G32, np.ones((3, 3), dtype=np.float32), atol=1e-6))
        Q64, _ = canonical_householder_qr(A64)
        self.assertLess(float(np.linalg.norm(Q64.T @ Q64 - np.eye(3), "fro")), 1e-14)


# =============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS (5 Tests)
# =============================================================================

class TestTier3CrossFeatureCombinations(unittest.TestCase):
    """Tier 3: Projectors, Cholesky-Gram link, Hadamard, Pseudoinverse, and QR eigenvalues."""

    def test_qr_projector_equals_normal_projector(self):
        rng = np.random.default_rng(7301)
        A = rng.standard_normal((7, 3))
        Q, _, _ = mgs_qr(A)
        P_qr = Q @ Q.T
        P_norm = A @ np.linalg.inv(A.T @ A) @ A.T
        self.assertTrue(np.allclose(P_qr, P_norm, atol=1e-12))

    def test_r_is_cholesky_factor_of_gram_matrix(self):
        Q, R, _ = mgs_qr(A_CANON)
        self.assertTrue(np.allclose(A_CANON.T @ A_CANON, R.T @ R, atol=1e-12))
        L_chol = np.linalg.cholesky(A_CANON.T @ A_CANON)
        self.assertTrue(np.allclose(R, L_chol.T, atol=1e-12))
        self.assertTrue(np.allclose(Q.T @ Q, np.eye(3), atol=1e-12))

    def test_hadamard_volume_inequality(self):
        _, R, _ = mgs_qr(A_CANON)
        vol = abs(float(np.linalg.det(A_CANON)))
        col_norm_prod = float(np.prod(np.linalg.norm(A_CANON, axis=0)))
        self.assertAlmostEqual(vol, float(np.prod(np.diag(R))), places=12)
        self.assertLess(vol, col_norm_prod)
        # Equality iff orthogonal columns
        self.assertAlmostEqual(abs(float(np.linalg.det(Q_CANON))), float(np.prod(np.linalg.norm(Q_CANON, axis=0))), places=12)

    def test_qr_least_squares_matches_pseudoinverse(self):
        rng = np.random.default_rng(7304)
        A = rng.standard_normal((10, 4))
        b = rng.standard_normal(10)
        Q, R, _ = mgs_qr(A)
        x_qr = back_substitute(R, Q.T @ b)
        x_pinv = np.linalg.pinv(A) @ b
        self.assertTrue(np.allclose(x_qr, x_pinv, atol=1e-12))

    def test_qr_algorithm_step_similarity_invariance(self):
        A0 = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]])
        Q0, R0, _ = mgs_qr(A0)
        A1 = R0 @ Q0
        self.assertTrue(np.allclose(A1, Q0.T @ A0 @ Q0, atol=1e-12))
        e0 = np.sort(np.linalg.eigvalsh(A0))
        e1 = np.sort(np.linalg.eigvalsh(A1))
        self.assertTrue(np.allclose(e0, e1, atol=1e-11))
        self.assertAlmostEqual(float(np.trace(A0)), float(np.trace(A1)), places=12)
        self.assertAlmostEqual(float(np.linalg.det(A0)), float(np.linalg.det(A1)), places=11)
        self.assertAlmostEqual(float(np.linalg.norm(A0, "fro")), float(np.linalg.norm(A1, "fro")), places=12)


# =============================================================================
# TIER 4: REAL-WORLD SCENARIOS (3 ML + 2 GATE = 5 Tests)
# =============================================================================

class TestTier4RealWorldScenarios(unittest.TestCase):
    """Tier 4: ML (Ridge QR, Haar Orthogonal Init, Subspace/LoRA) & GATE CS/DA."""

    def test_ml_ridge_regression_via_augmented_qr(self):
        rng = np.random.default_rng(7401)
        A = rng.standard_normal((20, 5))
        b = rng.standard_normal(20)
        lam = 0.25
        A_aug = np.vstack([A, np.sqrt(lam) * np.eye(5)])
        b_aug = np.concatenate([b, np.zeros(5)])
        Q_aug, R_aug = canonical_householder_qr(A_aug)
        x_ridge_qr = back_substitute(R_aug, Q_aug.T @ b_aug)
        x_ridge_ref = np.linalg.solve(A.T @ A + lam * np.eye(5), A.T @ b)
        self.assertTrue(np.allclose(x_ridge_qr, x_ridge_ref, atol=1e-12))

    def test_ml_orthogonal_weight_initialization_norm_preservation(self):
        rng = np.random.default_rng(7402)
        W = rng.standard_normal((16, 16))
        Q_haar, _ = canonical_householder_qr(W)
        x = rng.standard_normal(16)
        norm0 = float(np.linalg.norm(x))
        for _ in range(10):
            x = Q_haar @ x
        self.assertAlmostEqual(float(np.linalg.norm(x)), norm0, places=11)

    def test_ml_subspace_orthonormalization_sketching(self):
        rng = np.random.default_rng(7403)
        B = rng.standard_normal((32, 4))
        Q_b, R_b, rank = mgs_qr(B)
        self.assertEqual(rank, 4)
        self.assertAlmostEqual(float(np.linalg.cond(Q_b)), 1.0, places=12)
        P_B = B @ np.linalg.inv(B.T @ B) @ B.T
        P_Q = Q_b @ Q_b.T
        self.assertTrue(np.allclose(P_B, P_Q, atol=1e-12))
        self.assertTrue(np.allclose(Q_b @ R_b, B, atol=1e-12))

    def test_gate_gram_schmidt_hand_worked_3d(self):
        Q, R, rank = mgs_qr(A_CANON)
        self.assertEqual(rank, 3)
        self.assertTrue(np.allclose(Q, Q_CANON, atol=1e-12))
        self.assertTrue(np.allclose(R, R_CANON, atol=1e-12))
        x = back_substitute(R, Q.T @ B_CANON)
        self.assertTrue(np.allclose(x, X_CANON, atol=1e-12))

    def test_gate_qr_least_squares_linear_fit(self):
        t = np.array([-1.0, 0.0, 1.0, 2.0])
        y = np.array([1.0, 3.0, 5.0, 7.0])  # exact line y = 3 + 2t
        A_fit = np.column_stack([np.ones_like(t), t])
        Q, R, rank = mgs_qr(A_fit)
        self.assertEqual(rank, 2)
        c = back_substitute(R, Q.T @ y)
        self.assertTrue(np.allclose(c, [3.0, 2.0], atol=1e-12))


# =============================================================================
# TIER 5: STATIC AST & SYNTAX INTEGRITY (7 Tests)
# =============================================================================

class TestSprint07ASTAndSyntax(unittest.TestCase):
    """Tier 5: Static AST verification of Sprint 7 ManimCE 3D scene."""

    SCENE_PATH = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "linear_algebra",
            "sprint_07_gram_schmidt_qr",
            "scene_gram_schmidt_qr.py",
        )
    )

    def _get_source_and_ast(self) -> tuple[str, ast.Module]:
        if not os.path.exists(self.SCENE_PATH):
            self.skipTest(f"Scene script '{self.SCENE_PATH}' not yet created.")
        with open(self.SCENE_PATH, "r", encoding="utf-8") as f:
            src = f.read()
        return src, ast.parse(src, filename=self.SCENE_PATH)

    def test_no_deprecated_manimlib_symbols(self):
        _, tree = self._get_source_and_ast()
        deprecated = {"CONFIG", "ShowCreation", "TexMobject", "TextMobject"}
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in deprecated:
                found.append((node.id, getattr(node, "lineno", "?")))
            elif isinstance(node, ast.Attribute) and node.attr in deprecated:
                found.append((node.attr, getattr(node, "lineno", "?")))
        self.assertEqual(found, [], f"Deprecated symbols found: {found}")

    def test_scene_subclasses_threedscene(self):
        _, tree = self._get_source_and_ast()
        threed_subclasses = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if (isinstance(base, ast.Name) and base.id == "ThreeDScene") or (
                        isinstance(base, ast.Attribute) and base.attr == "ThreeDScene"
                    ):
                        threed_subclasses.append(node.name)
        self.assertGreater(len(threed_subclasses), 0, "Must subclass ThreeDScene")

    def test_hud_pinning_add_fixed_in_frame_mobjects_invoked(self):
        _, tree = self._get_source_and_ast()
        has_call = any(
            isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "add_fixed_in_frame_mobjects")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "add_fixed_in_frame_mobjects")
            )
            for node in ast.walk(tree)
        )
        self.assertTrue(has_call, "Must invoke add_fixed_in_frame_mobjects")

    def test_scene_hierarchy_classes_defined(self):
        _, tree = self._get_source_and_ast()
        class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        required = {
            "BaseGramSchmidtQRScene",
            "Act1SpatialPuzzle",
            "Act2DynamicDeformation",
            "Act3AlgebraicMorph",
            "Act4DegenerateStressTest",
            "GramSchmidtQRScene",
        }
        missing = required - class_names
        self.assertEqual(missing, set(), f"Missing required scene classes: {missing}")

    def test_make_hud_panel_usage(self):
        _, tree = self._get_source_and_ast()
        has_def = any(isinstance(n, ast.FunctionDef) and n.name == "make_hud_panel" for n in ast.walk(tree))
        has_call = any(
            isinstance(n, ast.Call)
            and (
                (isinstance(n.func, ast.Name) and n.func.id == "make_hud_panel")
                or (isinstance(n.func, ast.Attribute) and n.func.attr == "make_hud_panel")
            )
            for n in ast.walk(tree)
        )
        self.assertTrue(has_def and has_call, "Must define and call make_hud_panel")

    def test_no_forbidden_execution_calls(self):
        _, tree = self._get_source_and_ast()
        forbidden = {"eval", "exec"}
        found = [
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden
        ]
        self.assertEqual(found, [], f"Forbidden execution calls found: {found}")

    def test_3d_right_angle_elbow_and_no_3d_right_angle_crash(self):
        _, tree = self._get_source_and_ast()
        has_polygon = any(
            isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "Polygon")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "Polygon")
            )
            for node in ast.walk(tree)
        )
        has_right_angle = any(
            isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "RightAngle")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "RightAngle")
            )
            for node in ast.walk(tree)
        )
        self.assertTrue(has_polygon, "Must construct 3D right-angle elbow via Polygon")
        self.assertFalse(has_right_angle, "Must NOT call 2D RightAngle in 3D scene")


if __name__ == "__main__":
    unittest.main()
