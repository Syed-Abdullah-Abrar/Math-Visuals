#!/usr/bin/env python3
r"""
Sprint 8 E2E Invariant Test Suite: Orthogonal Matrices, Rotations SO(n) & Givens/Householder
=============================================================================================
Curriculum: Linear Algebra (Geometric Foundations & AI/Systems Invariants)
References:
    - Mathematics for Machine Learning (MML Ch. 3.4, 3.9 - Orthogonal Matrices, Rotations, Givens)
    - Introduction to Applied Linear Algebra (VMLS Ch. 7.1 & 10.4 - Rotations, Reflections, QR)
    - Erwin Kreyszig — Advanced Engineering Mathematics 10th Ed. (Ch. 8.3 & 20.9)

Architecture:
    Comprehensive 5-Tier Test Suite implemented with unittest and NumPy:
    - Tier 1: Feature Coverage (5 classes x 5 = 25 tests):
      Orthogonal group axioms & isometry, 2D rotations & reflections,
      Cartan-Dieudonne & SO(3) rotations, Givens rotations, Householder reflectors.
    - Tier 2: Boundary & Corner Cases (5 classes x 5 = 25 tests):
      Identity/negation/permutation parity, non-orthogonal unit-det shear failure,
      Givens zero-pair & extreme-scale hypot stability, Householder cancellation at
      [1, 1e-9]^T & zero-vector guard, repeated Q^1000 multiplication drift.
    - Tier 3: Cross-Feature Combinations (5 tests):
      Group closure, Householder vs Givens QR equivalence, orthogonal similarity
      conjugation, Orthogonal Procrustes SVD solution, condition number invariance.
    - Tier 4: Real-World ML & GATE CS/DA Scenarios (5 tests):
      100-layer orthogonal init norm preservation, RoPE relative position invariance,
      Householder QR least squares, GATE SO(3) Euler axis & trace 1+2cos(theta),
      GATE symmetric orthogonal involution eigenvalues & Q^-1 = Q^T shortcut.
    - Tier 5: Static AST & Syntax Integrity (5 tests):
      Parses linear_algebra/sprint_08_orthogonal_matrices_rotations/scene_orthogonal_matrices_rotations.py.

Execution:
    .venv/bin/python -m unittest tests/test_sprint08_invariants.py
    .venv/bin/pytest tests/test_sprint08_invariants.py -v
"""

from __future__ import annotations

import ast
import math
import os
import unittest
import numpy as np


# -----------------------------------------------------------------------------
# Reference Numerical Implementations (Genuine Algorithms, No Shortcuts)
# -----------------------------------------------------------------------------
def rot2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def reflect2d(alpha: float) -> np.ndarray:
    c2, s2 = math.cos(2.0 * alpha), math.sin(2.0 * alpha)
    return np.array([[c2, s2], [s2, -c2]], dtype=float)


def rot_x_3d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def rot_y_3d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def rot_z_3d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def givens_matrix(n: int, i: int, j: int, xi: float, xj: float) -> tuple[np.ndarray, float, float, float]:
    """Construct n x n Givens rotation G(i, j, theta) zeroing entry j via math.hypot guard."""
    assert 0 <= i < j < n
    r = math.hypot(xi, xj)
    G = np.eye(n, dtype=float)
    if r == 0.0:
        return G, 1.0, 0.0, 0.0
    c = xi / r
    s = -xj / r
    # Standard plane rotation: [c, -s; s, c] * [xi; xj] = [r; 0]
    G[i, i] = c
    G[i, j] = -s
    G[j, i] = s
    G[j, j] = c
    return G, c, s, r


def householder_reflector(x: np.ndarray, sign_aware: bool = True) -> np.ndarray:
    """Construct Householder reflector H = I - 2 v v^T / ||v||^2 mapping x onto e1."""
    n = len(x)
    nx = float(np.linalg.norm(x))
    if nx == 0.0:
        return np.eye(n, dtype=float)
    e1 = np.zeros(n, dtype=float)
    e1[0] = 1.0
    if sign_aware:
        sigma = 1.0 if x[0] >= 0.0 else -1.0
        v = x + sigma * nx * e1
    else:
        v = x - nx * e1
    vv = float(np.dot(v, v))
    if vv == 0.0:
        return np.eye(n, dtype=float)
    return np.eye(n, dtype=float) - 2.0 * np.outer(v, v) / vv


def householder_qr(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full A = Q R via sign-aware Householder reflectors with positive R diagonal."""
    m, n = A.shape
    R = A.astype(float).copy()
    Q = np.eye(m, dtype=float)
    for k in range(min(m, n)):
        x = R[k:, k]
        H_sub = householder_reflector(x, sign_aware=True)
        H = np.eye(m, dtype=float)
        H[k:, k:] = H_sub
        R = H @ R
        Q = Q @ H
    # Canonicalize R diagonal to > 0
    for k in range(min(m, n)):
        if R[k, k] < 0.0:
            R[k, :] *= -1.0
            Q[:, k] *= -1.0
    return Q, R


def givens_qr(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full A = Q R via Givens rotations with positive R diagonal."""
    m, n = A.shape
    R = A.astype(float).copy()
    Q = np.eye(m, dtype=float)
    for j in range(n):
        for i in range(m - 1, j, -1):
            if R[i, j] != 0.0:
                G, _, _, _ = givens_matrix(m, i - 1, i, float(R[i - 1, j]), float(R[i, j]))
                R = G @ R
                Q = Q @ G.T
    for k in range(min(m, n)):
        if R[k, k] < 0.0:
            R[k, :] *= -1.0
            Q[:, k] *= -1.0
    return Q, R


# =============================================================================
# TIER 1: FEATURE COVERAGE (25 TESTS)
# =============================================================================

class TestTier1OrthogonalGroupAndIsometry(unittest.TestCase):
    """Tier 1 Feature 1: Orthogonal Group O(n) & Isometry Invariants (MML §3.4, Kreyszig §8.3)."""

    def setUp(self):
        self.Q_pyth = np.array([[0.6, -0.8], [0.8, 0.6]], dtype=float)
        self.Q_kreyszig = (1.0 / 3.0) * np.array(
            [[2.0, 1.0, 2.0], [-2.0, 2.0, 1.0], [1.0, 2.0, -2.0]], dtype=float
        )

    def test_transpose_equals_inverse(self):
        for Q in (self.Q_pyth, self.Q_kreyszig):
            n = Q.shape[0]
            np.testing.assert_allclose(Q.T @ Q, np.eye(n), atol=1e-14)
            np.testing.assert_allclose(Q @ Q.T, np.eye(n), atol=1e-14)
            np.testing.assert_allclose(np.linalg.inv(Q), Q.T, atol=1e-14)

    def test_row_and_column_orthonormality(self):
        for Q in (self.Q_pyth, self.Q_kreyszig):
            n = Q.shape[0]
            for i in range(n):
                self.assertAlmostEqual(float(np.linalg.norm(Q[:, i])), 1.0, places=13)
                self.assertAlmostEqual(float(np.linalg.norm(Q[i, :])), 1.0, places=13)
                for j in range(i + 1, n):
                    self.assertAlmostEqual(float(np.dot(Q[:, i], Q[:, j])), 0.0, places=13)
                    self.assertAlmostEqual(float(np.dot(Q[i, :], Q[j, :])), 0.0, places=13)

    def test_norm_and_distance_preservation(self):
        rng = np.random.default_rng(801)
        X = rng.normal(size=(100, 3))
        Y = rng.normal(size=(100, 3))
        QX = X @ self.Q_kreyszig.T
        QY = Y @ self.Q_kreyszig.T
        np.testing.assert_allclose(np.linalg.norm(QX, axis=1), np.linalg.norm(X, axis=1), atol=1e-13)
        np.testing.assert_allclose(np.linalg.norm(QX - QY, axis=1), np.linalg.norm(X - Y, axis=1), atol=1e-13)

    def test_inner_product_and_angle_preservation(self):
        rng = np.random.default_rng(802)
        for _ in range(30):
            x = rng.normal(size=3)
            y = rng.normal(size=3)
            qx = self.Q_kreyszig @ x
            qy = self.Q_kreyszig @ y
            self.assertAlmostEqual(float(np.dot(qx, qy)), float(np.dot(x, y)), places=13)
            cos_orig = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
            cos_rot = np.dot(qx, qy) / (np.linalg.norm(qx) * np.linalg.norm(qy))
            self.assertAlmostEqual(float(cos_rot), float(cos_orig), places=13)

    def test_singular_values_and_condition_number(self):
        for Q in (self.Q_pyth, self.Q_kreyszig):
            s = np.linalg.svd(Q, compute_uv=False)
            np.testing.assert_allclose(s, np.ones(Q.shape[0]), atol=1e-14)
            self.assertAlmostEqual(float(np.linalg.cond(Q)), 1.0, places=13)


class TestTier1RotationsAndReflections(unittest.TestCase):
    """Tier 1 Feature 2: 2D Rotations SO(2) & Reflections O(2)\\SO(2) (MML §3.9.1, VMLS §7.1)."""

    def test_so2_determinant_and_group_addition(self):
        for t1, t2 in [(0.2, 0.7), (1.1, -0.4), (math.pi / 3, math.pi / 6)]:
            R1, R2 = rot2d(t1), rot2d(t2)
            self.assertAlmostEqual(float(np.linalg.det(R1)), 1.0, places=14)
            np.testing.assert_allclose(R1 @ R2, rot2d(t1 + t2), atol=1e-14)

    def test_so2_cyclicity_pi_over_3(self):
        R60 = rot2d(math.pi / 3.0)
        np.testing.assert_allclose(np.linalg.matrix_power(R60, 6), np.eye(2), atol=1e-14)
        np.testing.assert_allclose(np.linalg.matrix_power(R60, 3), -np.eye(2), atol=1e-14)

    def test_reflection_involution_symmetry_and_det(self):
        for alpha in [0.0, math.pi / 12.0, math.pi / 4.0, 1.23]:
            H = reflect2d(alpha)
            np.testing.assert_allclose(H, H.T, atol=1e-14)
            np.testing.assert_allclose(H @ H, np.eye(2), atol=1e-14)
            self.assertAlmostEqual(float(np.linalg.det(H)), -1.0, places=14)

    def test_reflection_eigenvalues_and_mirror_axis(self):
        alpha = math.pi / 12.0
        H = reflect2d(alpha)
        eigs = np.sort(np.linalg.eigvalsh(H))
        np.testing.assert_allclose(eigs, [-1.0, 1.0], atol=1e-14)
        u_axis = np.array([math.cos(alpha), math.sin(alpha)])
        n_perp = np.array([-math.sin(alpha), math.cos(alpha)])
        np.testing.assert_allclose(H @ u_axis, u_axis, atol=1e-14)
        np.testing.assert_allclose(H @ n_perp, -n_perp, atol=1e-14)

    def test_so2_commutativity_abelian(self):
        R1 = rot2d(0.41)
        R2 = rot2d(1.73)
        np.testing.assert_allclose(R1 @ R2, R2 @ R1, atol=1e-14)


class TestTier1CartanDieudonneAndSO3(unittest.TestCase):
    """Tier 1 Feature 3: Cartan-Dieudonne 2D Composition & 3D SO(3) Rotations (MML §3.9.2)."""

    def test_cartan_dieudonne_double_reflection_equals_rotation(self):
        a1, a2 = math.pi / 12.0, math.pi / 4.0
        H1, H2 = reflect2d(a1), reflect2d(a2)
        np.testing.assert_allclose(H2 @ H1, rot2d(2.0 * (a2 - a1)), atol=1e-14)

    def test_cartan_dieudonne_order_reversal_inverts_angle(self):
        a1, a2 = math.pi / 12.0, math.pi / 4.0
        H1, H2 = reflect2d(a1), reflect2d(a2)
        R_fwd = H2 @ H1
        R_rev = H1 @ H2
        np.testing.assert_allclose(R_rev, R_fwd.T, atol=1e-14)
        np.testing.assert_allclose(R_rev @ R_fwd, np.eye(2), atol=1e-14)

    def test_so3_elementary_rotations_orthonormality(self):
        for fn in (rot_x_3d, rot_y_3d, rot_z_3d):
            for ang in [0.0, 0.5, math.pi / 2.0, -1.3]:
                R = fn(ang)
                np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-14)
                self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=14)

    def test_so3_non_commutativity_90_degrees(self):
        R1 = rot_x_3d(math.pi / 2.0)
        R2 = rot_y_3d(math.pi / 2.0)
        e1 = np.array([1.0, 0.0, 0.0])
        out_12 = (R1 @ R2) @ e1
        out_21 = (R2 @ R1) @ e1
        np.testing.assert_allclose(out_12, [0.0, 1.0, 0.0], atol=1e-14)
        np.testing.assert_allclose(out_21, [0.0, 0.0, -1.0], atol=1e-14)
        self.assertAlmostEqual(float(np.linalg.norm(out_12 - out_21)), math.sqrt(2.0), places=13)

    def test_so3_axis_fixed_points(self):
        R1 = rot_x_3d(0.93)
        R2 = rot_y_3d(-1.17)
        R3 = rot_z_3d(2.05)
        np.testing.assert_allclose(R1 @ np.array([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-14)
        np.testing.assert_allclose(R2 @ np.array([0.0, 1.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-14)
        np.testing.assert_allclose(R3 @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0], atol=1e-14)


class TestTier1GivensRotations(unittest.TestCase):
    """Tier 1 Feature 4: Givens Planar Rotations G(i, j, theta) (MML §3.9.3, Kreyszig §20.9)."""

    def test_givens_zeros_target_entry_pythagorean(self):
        x = np.array([3.0, 2.0, 4.0], dtype=float)
        G, c, s, r = givens_matrix(3, 0, 2, float(x[0]), float(x[2]))
        self.assertAlmostEqual(c, 0.6, places=14)
        self.assertAlmostEqual(s, -0.8, places=14)
        self.assertAlmostEqual(r, 5.0, places=14)
        np.testing.assert_allclose(G @ x, [5.0, 2.0, 0.0], atol=1e-14)

    def test_givens_is_in_so_n(self):
        for n, i, j in [(2, 0, 1), (3, 0, 2), (5, 1, 4), (8, 2, 6)]:
            G, _, _, _ = givens_matrix(n, i, j, 2.5, -6.0)
            np.testing.assert_allclose(G.T @ G, np.eye(n), atol=1e-14)
            self.assertAlmostEqual(float(np.linalg.det(G)), 1.0, places=14)

    def test_givens_preserves_untouched_coordinates(self):
        x = np.array([1.1, 3.0, -7.2, 4.0, 9.5], dtype=float)
        G, _, _, _ = givens_matrix(5, 1, 3, float(x[1]), float(x[3]))
        y = G @ x
        self.assertAlmostEqual(float(y[0]), float(x[0]), places=15)
        self.assertAlmostEqual(float(y[2]), float(x[2]), places=15)
        self.assertAlmostEqual(float(y[4]), float(x[4]), places=15)
        self.assertAlmostEqual(float(y[1]), 5.0, places=14)
        self.assertAlmostEqual(float(y[3]), 0.0, places=14)

    def test_givens_sparsity_only_four_entries_differ(self):
        G, _, _, _ = givens_matrix(6, 1, 4, 5.0, 12.0)
        diff = np.abs(G - np.eye(6)) > 1e-15
        self.assertEqual(int(np.sum(diff)), 4)

    def test_givens_qr_tridiagonal_reduction(self):
        B = np.array([[4.0, 1.0, 0.0], [3.0, 2.0, 1.0], [0.0, 4.0, 3.0]], dtype=float)
        Q, R = givens_qr(B)
        np.testing.assert_allclose(Q.T @ Q, np.eye(3), atol=1e-13)
        np.testing.assert_allclose(Q @ R, B, atol=1e-13)
        np.testing.assert_allclose(np.tril(R, k=-1), np.zeros((3, 3)), atol=1e-13)


class TestTier1HouseholderReflectors(unittest.TestCase):
    """Tier 1 Feature 5: Householder Reflectors H = I - 2 v v^T / ||v||^2 (Kreyszig §20.9)."""

    def test_householder_zeros_subcolumn_3d(self):
        x = np.array([1.0, 2.0, 2.0], dtype=float)
        H = householder_reflector(x, sign_aware=True)
        np.testing.assert_allclose(H @ x, [-3.0, 0.0, 0.0], atol=1e-14)

    def test_householder_symmetry_involution_and_det(self):
        for x in [np.array([1.0, 2.0, 2.0]), np.array([-3.0, 1.0, 4.0, -2.0])]:
            H = householder_reflector(x, sign_aware=True)
            n = len(x)
            np.testing.assert_allclose(H, H.T, atol=1e-14)
            np.testing.assert_allclose(H @ H, np.eye(n), atol=1e-14)
            self.assertAlmostEqual(float(np.linalg.det(H)), -1.0, places=13)

    def test_householder_hyperplane_reflection_geometry(self):
        x = np.array([1.2, 1.6], dtype=float)
        v = x + np.linalg.norm(x) * np.array([1.0, 0.0])  # [3.2, 1.6]
        H = householder_reflector(x, sign_aware=True)
        np.testing.assert_allclose(H @ v, -v, atol=1e-14)
        w_perp = np.array([-1.6, 3.2])  # orthogonal to v
        np.testing.assert_allclose(H @ w_perp, w_perp, atol=1e-14)

    def test_householder_midpoint_lies_on_mirror_hyperplane(self):
        x = np.array([1.2, 1.6], dtype=float)
        v = np.array([3.2, 1.6], dtype=float)
        H = householder_reflector(x, sign_aware=True)
        midpoint = 0.5 * (x + H @ x)
        self.assertAlmostEqual(float(np.dot(v, midpoint)), 0.0, places=14)

    def test_householder_qr_reconstruction_rectangular(self):
        rng = np.random.default_rng(805)
        A = rng.normal(size=(5, 3))
        Q, R = householder_qr(A)
        np.testing.assert_allclose(Q.T @ Q, np.eye(5), atol=1e-13)
        np.testing.assert_allclose(Q @ R, A, atol=1e-13)
        np.testing.assert_allclose(np.tril(R, k=-1), np.zeros((5, 3)), atol=1e-13)


# =============================================================================
# TIER 2: BOUNDARY & CORNER CASES (25 TESTS)
# =============================================================================

class TestTier2IdentityNegationAndPermutationBoundaries(unittest.TestCase):
    """Tier 2 Boundary 1: Identity, Negation Parity & Permutation Matrices."""

    def test_identity_always_in_so_n(self):
        for n in [1, 2, 3, 5, 10]:
            I_n = np.eye(n)
            np.testing.assert_allclose(I_n.T @ I_n, I_n)
            self.assertAlmostEqual(float(np.linalg.det(I_n)), 1.0)

    def test_negated_identity_parity_by_dimension(self):
        for n in [1, 2, 3, 4, 5, 6]:
            neg_I = -np.eye(n)
            np.testing.assert_allclose(neg_I.T @ neg_I, np.eye(n))
            expected_det = 1.0 if n % 2 == 0 else -1.0
            self.assertAlmostEqual(float(np.linalg.det(neg_I)), expected_det)

    def test_permutation_matrices_orthogonal_and_det_parity(self):
        P_swap = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        P_cycle = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        np.testing.assert_allclose(P_swap.T @ P_swap, np.eye(3))
        np.testing.assert_allclose(P_cycle.T @ P_cycle, np.eye(3))
        self.assertAlmostEqual(float(np.linalg.det(P_swap)), -1.0)
        self.assertAlmostEqual(float(np.linalg.det(P_cycle)), 1.0)

    def test_diagonal_orthogonal_sign_matrices(self):
        for signs in [(1, 1, 1), (1, -1, 1), (-1, -1, 1), (-1, -1, -1)]:
            D = np.diag(signs).astype(float)
            np.testing.assert_allclose(D.T @ D, np.eye(3))
            expected_det = float(np.prod(signs))
            self.assertAlmostEqual(float(np.linalg.det(D)), expected_det)

    def test_1d_orthogonal_group_exact_two_elements(self):
        O1_pos = np.array([[1.0]])
        O1_neg = np.array([[-1.0]])
        self.assertAlmostEqual(float(np.linalg.det(O1_pos)), 1.0)
        self.assertAlmostEqual(float(np.linalg.det(O1_neg)), -1.0)
        np.testing.assert_allclose(O1_neg @ O1_neg, O1_pos)


class TestTier2NonOrthogonalShearAndSpectrumBoundaries(unittest.TestCase):
    """Tier 2 Boundary 2: Unit-Determinant Non-Orthogonal Shear & Spectral Failure."""

    def setUp(self):
        self.M_shear = np.array([[1.0, 1.2], [0.0, 1.0]], dtype=float)

    def test_shear_preserves_det_but_fails_orthogonality(self):
        self.assertAlmostEqual(float(np.linalg.det(self.M_shear)), 1.0, places=14)
        self.assertFalse(np.allclose(self.M_shear.T @ self.M_shear, np.eye(2), atol=1e-3))

    def test_shear_distorts_basis_norms_and_angle_to_39_81_deg(self):
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])
        me1 = self.M_shear @ e1
        me2 = self.M_shear @ e2
        self.assertAlmostEqual(float(np.linalg.norm(me1)), 1.0, places=14)
        self.assertAlmostEqual(float(np.linalg.norm(me2)), math.sqrt(2.44), places=14)
        angle_deg = math.degrees(math.acos(float(np.dot(me1, me2) / (np.linalg.norm(me1) * np.linalg.norm(me2)))))
        self.assertAlmostEqual(angle_deg, 39.8055710922652, places=10)

    def test_shear_splits_singular_values_away_from_one(self):
        s = np.linalg.svd(self.M_shear, compute_uv=False)
        self.assertGreater(s[0], 1.75)
        self.assertLess(s[1], 0.58)
        self.assertAlmostEqual(float(s[0] * s[1]), 1.0, places=13)

    def test_non_unit_eigenvalue_matrix_cannot_be_orthogonal(self):
        A_scale = np.diag([1.001, 1.0 / 1.001])
        self.assertAlmostEqual(float(np.linalg.det(A_scale)), 1.0, places=14)
        self.assertGreater(float(np.linalg.norm(A_scale.T @ A_scale - np.eye(2))), 1e-3)

    def test_complex_eigenvalues_of_so2_lie_strictly_on_unit_circle(self):
        for theta in [0.1, math.pi / 4.0, math.pi / 3.0, 2.7]:
            R = rot2d(theta)
            eigs = np.linalg.eigvals(R)
            np.testing.assert_allclose(np.abs(eigs), [1.0, 1.0], atol=1e-14)
            self.assertAlmostEqual(float(np.trace(R)), 2.0 * math.cos(theta), places=14)


class TestTier2GivensZeroPairAndExtremeScale(unittest.TestCase):
    """Tier 2 Boundary 3: Givens Zero-Pair Guard & Extreme Float64 Scale."""

    def test_givens_zero_pair_returns_identity_no_nan(self):
        G, c, s, r = givens_matrix(4, 1, 3, 0.0, 0.0)
        self.assertFalse(np.isnan(G).any())
        np.testing.assert_allclose(G, np.eye(4))
        self.assertEqual((c, s, r), (1.0, 0.0, 0.0))

    def test_givens_already_zero_target_returns_identity(self):
        G, c, s, r = givens_matrix(3, 0, 2, 5.0, 0.0)
        np.testing.assert_allclose(G, np.eye(3), atol=1e-15)
        self.assertAlmostEqual(r, 5.0, places=14)
        self.assertAlmostEqual(c, 1.0, places=14)
        self.assertAlmostEqual(s, 0.0, places=14)

    def test_givens_negative_pivot_zeroes_second_entry(self):
        x = np.array([-3.0, 4.0], dtype=float)
        G, _, _, r = givens_matrix(2, 0, 1, float(x[0]), float(x[1]))
        self.assertAlmostEqual(r, 5.0, places=14)
        np.testing.assert_allclose(G @ x, [5.0, 0.0], atol=1e-14)

    def test_givens_extreme_large_scale_no_overflow(self):
        G, c, s, r = givens_matrix(2, 0, 1, 1e200, 1e200)
        self.assertTrue(np.isfinite(G).all())
        self.assertTrue(math.isfinite(r))
        np.testing.assert_allclose(G.T @ G, np.eye(2), atol=1e-14)
        self.assertAlmostEqual(c, math.sqrt(0.5), places=14)
        self.assertAlmostEqual(s, -math.sqrt(0.5), places=14)

    def test_givens_extreme_tiny_scale_no_underflow(self):
        G, c, s, r = givens_matrix(2, 0, 1, 1e-200, 1e-200)
        self.assertTrue(np.isfinite(G).all())
        self.assertGreater(r, 0.0)
        np.testing.assert_allclose(G.T @ G, np.eye(2), atol=1e-14)
        np.testing.assert_allclose(G @ np.array([1e-200, 1e-200]), [r, 0.0], rtol=1e-14, atol=1e-14 * r)


class TestTier2HouseholderCancellationAndZeroVector(unittest.TestCase):
    """Tier 2 Boundary 4: Householder Sign Choice, Cancellation & Zero-Vector Guards."""

    def test_householder_zero_vector_returns_identity(self):
        H = householder_reflector(np.zeros(4), sign_aware=True)
        np.testing.assert_allclose(H, np.eye(4), atol=1e-15)

    def test_householder_cancellation_at_1e_minus_9_naive_fails_safe_succeeds(self):
        x = np.array([1.0, 1e-9], dtype=float)
        H_bad = householder_reflector(x, sign_aware=False)
        H_good = householder_reflector(x, sign_aware=True)
        y_bad = H_bad @ x
        y_good = H_good @ x
        self.assertGreater(abs(float(y_bad[1])), 1e-10)
        self.assertLess(abs(float(y_good[1])), 1e-15)
        self.assertAlmostEqual(float(y_good[0]), -1.0, places=14)

    def test_householder_cancellation_negative_x1_sign_match(self):
        x = np.array([-1.0, 1e-11], dtype=float)
        H_good = householder_reflector(x, sign_aware=True)
        y_good = H_good @ x
        self.assertLess(abs(float(y_good[1])), 1e-15)
        self.assertAlmostEqual(float(y_good[0]), 1.0, places=14)

    def test_householder_already_aligned_e1_stays_exact_reflection(self):
        e1 = np.array([3.0, 0.0, 0.0], dtype=float)
        H = householder_reflector(e1, sign_aware=True)
        np.testing.assert_allclose(H @ e1, [-3.0, 0.0, 0.0], atol=1e-14)
        np.testing.assert_allclose(H.T @ H, np.eye(3), atol=1e-14)

    def test_householder_1d_boundary(self):
        H = householder_reflector(np.array([5.0]), sign_aware=True)
        np.testing.assert_allclose(H, np.array([[-1.0]]), atol=1e-15)


class TestTier2RepeatedMultiplicationDrift(unittest.TestCase):
    """Tier 2 Boundary 5: Repeated Orthogonal Multiplication Q^1000 Drift."""

    def setUp(self):
        self.Q = rot2d(0.37)

    def test_q_power_1000_float64_orthogonality_drift(self):
        P = np.eye(2)
        for _ in range(1000):
            P = P @ self.Q
        err = float(np.linalg.norm(P.T @ P - np.eye(2), "fro"))
        self.assertLess(err, 1e-11)

    def test_q_power_1000_determinant_drift(self):
        P = np.eye(2)
        for _ in range(1000):
            P = P @ self.Q
        self.assertLess(abs(float(np.linalg.det(P)) - 1.0), 1e-11)

    def test_q_power_1000_matches_analytic_angle(self):
        P = np.eye(2)
        for _ in range(1000):
            P = P @ self.Q
        np.testing.assert_allclose(P, rot2d(1000.0 * 0.37), atol=1e-10)

    def test_non_orthogonal_contraction_collapses_after_1000_steps(self):
        M_shrink = 0.99 * self.Q
        P = np.linalg.matrix_power(M_shrink, 1000)
        self.assertLess(float(np.linalg.norm(P, 2)), 1e-4)

    def test_non_orthogonal_expansion_explodes_after_1000_steps(self):
        M_grow = 1.005 * self.Q
        P = np.linalg.matrix_power(M_grow, 1000)
        self.assertGreater(float(np.linalg.norm(P, 2)), 100.0)


# =============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS (5 TESTS)
# =============================================================================

class TestTier3CrossFeatureCombinations(unittest.TestCase):
    """Tier 3: Group Closure, QR Uniqueness, Similarity, Procrustes & Conditioning."""

    def test_orthogonal_group_closure_and_inverse(self):
        rng = np.random.default_rng(831)
        Q1, _ = householder_qr(rng.normal(size=(4, 4)))
        Q2, _ = givens_qr(rng.normal(size=(4, 4)))
        prod = Q1 @ Q2
        inv1 = np.linalg.inv(Q1)
        np.testing.assert_allclose(prod.T @ prod, np.eye(4), atol=1e-13)
        np.testing.assert_allclose(inv1, Q1.T, atol=1e-13)
        self.assertAlmostEqual(float(np.linalg.det(prod)), float(np.linalg.det(Q1) * np.linalg.det(Q2)), places=12)

    def test_householder_vs_givens_qr_uniqueness(self):
        rng = np.random.default_rng(832)
        A = rng.normal(size=(5, 5))
        Q_h, R_h = householder_qr(A)
        Q_g, R_g = givens_qr(A)
        np.testing.assert_allclose(Q_h, Q_g, atol=1e-12)
        np.testing.assert_allclose(R_h, R_g, atol=1e-12)

    def test_orthogonal_similarity_preserves_symmetry_spectrum_and_trace(self):
        rng = np.random.default_rng(833)
        B = rng.normal(size=(4, 4))
        A_sym = B + B.T
        Q, _ = householder_qr(rng.normal(size=(4, 4)))
        A_sim = Q.T @ A_sym @ Q
        np.testing.assert_allclose(A_sim, A_sim.T, atol=1e-13)
        np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(A_sim)), np.sort(np.linalg.eigvalsh(A_sym)), atol=1e-12)
        self.assertAlmostEqual(float(np.trace(A_sim)), float(np.trace(A_sym)), places=12)

    def test_orthogonal_procrustes_svd_recovers_exact_rotation(self):
        rng = np.random.default_rng(834)
        Q_true = rot_z_3d(0.75) @ rot_x_3d(-0.4)
        X = rng.normal(size=(3, 50))
        Y = Q_true @ X
        U, _, Vt = np.linalg.svd(Y @ X.T)
        Q_est = U @ Vt
        np.testing.assert_allclose(Q_est, Q_true, atol=1e-13)

    def test_orthogonal_multiplication_preserves_condition_number(self):
        rng = np.random.default_rng(835)
        A = rng.normal(size=(4, 4))
        Q, _ = householder_qr(rng.normal(size=(4, 4)))
        self.assertAlmostEqual(float(np.linalg.cond(Q @ A)), float(np.linalg.cond(A)), places=10)
        self.assertAlmostEqual(float(np.linalg.cond(A @ Q)), float(np.linalg.cond(A)), places=10)


# =============================================================================
# TIER 4: REAL-WORLD ML & GATE CS/DA SCENARIOS (5 TESTS)
# =============================================================================

class TestTier4RealWorldMLAndGATE(unittest.TestCase):
    """Tier 4: Deep Orthogonal Init, RoPE, QR Least Squares & GATE Shortcuts."""

    def test_ml_orthogonal_init_100_layer_gradient_norm_preservation(self):
        rng = np.random.default_rng(841)
        x = rng.normal(size=32)
        x_orth = x.copy()
        x_gauss = x.copy()
        for _ in range(100):
            W_orth, _ = householder_qr(rng.normal(size=(32, 32)))
            W_gauss = rng.normal(scale=1.25 / math.sqrt(32.0), size=(32, 32))
            x_orth = W_orth @ x_orth
            x_gauss = W_gauss @ x_gauss
        self.assertAlmostEqual(float(np.linalg.norm(x_orth)), float(np.linalg.norm(x)), places=10)
        self.assertGreater(float(np.linalg.norm(x_gauss)), 1e3 * float(np.linalg.norm(x)))

    def test_ml_rope_relative_position_invariance(self):
        rng = np.random.default_rng(842)
        d_pairs = 8
        thetas = 10000.0 ** (-2.0 * np.arange(d_pairs) / (2.0 * d_pairs))
        q = rng.normal(size=2 * d_pairs)
        k = rng.normal(size=2 * d_pairs)

        def apply_rope(vec: np.ndarray, pos: int) -> np.ndarray:
            out = np.empty_like(vec)
            for idx, th in enumerate(thetas):
                out[2 * idx : 2 * idx + 2] = rot2d(pos * float(th)) @ vec[2 * idx : 2 * idx + 2]
            return out

        m, n = 17, 42
        lhs = float(np.dot(apply_rope(q, m), apply_rope(k, n)))
        rhs = float(np.dot(q, apply_rope(k, n - m)))
        self.assertAlmostEqual(lhs, rhs, places=12)

    def test_ml_householder_qr_least_squares(self):
        A = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]], dtype=float)
        b = np.array([1.2, 2.9, 5.1, 6.8], dtype=float)
        Q, R = householder_qr(A)
        # Solve R[:2, :2] x = (Q.T @ b)[:2]
        x_qr = np.linalg.solve(R[:2, :2], (Q.T @ b)[:2])
        x_lstsq, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        np.testing.assert_allclose(x_qr, x_lstsq, atol=1e-13)

    def test_gate_so3_euler_axis_trace_and_kreyszig_ex5(self):
        Q_k = (1.0 / 3.0) * np.array([[2.0, 1.0, 2.0], [-2.0, 2.0, 1.0], [1.0, 2.0, -2.0]])
        self.assertAlmostEqual(float(np.linalg.det(Q_k)), -1.0, places=13)
        eigs_k = np.linalg.eigvals(Q_k)
        np.testing.assert_allclose(np.abs(eigs_k), np.ones(3), atol=1e-13)

        theta = 0.85
        R = rot_y_3d(theta) @ rot_z_3d(0.4) @ rot_y_3d(-theta)  # conjugate still in SO(3)
        eigs_r = np.linalg.eigvals(R)
        np.testing.assert_allclose(np.abs(eigs_r), np.ones(3), atol=1e-13)
        self.assertTrue(np.any(np.isclose(eigs_r, 1.0 + 0j, atol=1e-12)))
        # Trace of rotation by phi=0.4 is 1 + 2 cos(0.4)
        self.assertAlmostEqual(float(np.trace(R)), 1.0 + 2.0 * math.cos(0.4), places=13)

    def test_gate_symmetric_orthogonal_involution_and_inverse_shortcut(self):
        H = reflect2d(0.55)
        np.testing.assert_allclose(H, H.T, atol=1e-14)
        np.testing.assert_allclose(H @ H, np.eye(2), atol=1e-14)
        eigs = np.sort(np.linalg.eigvalsh(H))
        np.testing.assert_allclose(eigs, [-1.0, 1.0], atol=1e-14)
        self.assertAlmostEqual(float(np.trace(H)), 0.0, places=14)


# =============================================================================
# TIER 5: STATIC AST & SYNTAX INTEGRITY (5 TESTS)
# =============================================================================

class TestSprint08ASTAndSyntax(unittest.TestCase):
    """Tier 5: Static AST verification of Sprint 8 scene for ManimCE standards."""

    SCENE_PATH = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "linear_algebra",
            "sprint_08_orthogonal_matrices_rotations",
            "scene_orthogonal_matrices_rotations.py",
        )
    )

    def _get_ast(self):
        if not os.path.exists(self.SCENE_PATH):
            self.skipTest(f"Scene script '{self.SCENE_PATH}' not yet created.")
        with open(self.SCENE_PATH, "r", encoding="utf-8") as f:
            return ast.parse(f.read(), filename=self.SCENE_PATH)

    def test_no_deprecated_manimlib_symbols(self):
        tree = self._get_ast()
        deprecated = {"CONFIG", "ShowCreation", "TexMobject", "TextMobject"}
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in deprecated:
                found.append((node.id, getattr(node, "lineno", "?")))
            elif isinstance(node, ast.Attribute) and node.attr in deprecated:
                found.append((node.attr, getattr(node, "lineno", "?")))
        self.assertEqual(found, [], f"Found deprecated symbols: {found}")

    def test_scene_classes_defined(self):
        tree = self._get_ast()
        class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        expected = {
            "BaseOrthogonalRotationsScene",
            "Act1SpatialPuzzle",
            "Act2DynamicDeformation",
            "Act3AlgebraicMorph",
            "Act4DegenerateStressTest",
            "OrthogonalRotationsScene",
        }
        self.assertTrue(expected.issubset(class_names), f"Missing scene classes: {expected - class_names}")

    def test_hud_panel_and_right_angle_present(self):
        tree = self._get_ast()
        has_hud_func = any(
            isinstance(node, ast.FunctionDef) and node.name == "make_hud_panel"
            for node in ast.walk(tree)
        )
        has_right_angle = any(
            isinstance(node, ast.Name) and node.id == "RightAngle"
            for node in ast.walk(tree)
        )
        self.assertTrue(has_hud_func, "Sprint 8 scene must define 'make_hud_panel'.")
        self.assertTrue(has_right_angle, "Sprint 8 scene must use 'RightAngle' geometric markers.")

    def test_sprint8_constants_declared(self):
        tree = self._get_ast()
        assigned = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        assigned.add(t.id)
        required = {"Q_PYTH", "M_SHEAR", "H_1", "H_2", "R1_90", "R2_90", "X_GIVENS", "X_HH", "X_CANCEL", "Q_DRIFT_N"}
        self.assertTrue(required.issubset(assigned), f"Missing constants: {required - assigned}")

    def test_no_forbidden_execution_calls(self):
        tree = self._get_ast()
        forbidden = {"eval", "exec"}
        found = [
            (node.func.id, getattr(node, "lineno", "?"))
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden
        ]
        self.assertEqual(found, [], f"Forbidden execution calls found: {found}")


if __name__ == "__main__":
    unittest.main()
