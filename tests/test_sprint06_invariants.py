#!/usr/bin/env python3
r"""
Sprint 6 E2E Invariant Test Suite: Orthogonal Projections onto 1D Lines & k-D Subspaces
========================================================================================
Curriculum: Linear Algebra (Geometric Foundations & AI/Systems Invariants)
References:
    - Mathematics for Machine Learning (MML §3.6, §3.8, §9.2.3, §10.3)
    - Introduction to Applied Linear Algebra (VMLS §11.5, §12.1–12.3, §15)
    - Immersive Linear Algebra (ILA Ch. 3 & Ch. 6)
    - Erwin Kreyszig — Advanced Engineering Mathematics 10th Ed. (Ch. 7–8)

Architecture:
    Comprehensive 5-Tier Test Suite implemented with unittest and NumPy (66 tests):
    - Tier 1 (25): Feature Coverage (Projection operator axioms, 1D line projection,
      k-D subspace minimum distance & orthogonality, normal equations & P invariants,
      orthogonal complement I-P & Pythagorean theorem).
    - Tier 2 (25): Boundary & Corner Cases (Zero/full subspace boundaries, affine
      subspace projection, rank-deficient A^+ & Ridge, Lauchli ill-conditioned normal
      equations, oblique vs orthogonal projectors).
    - Tier 3 (5): Cross-Feature Combinations (Spectrum {0,1}, basis invariance A -> AM,
      normal eqs == QR == SVD equivalence, nested projectors, weighted W-projector).
    - Tier 4 (5): Real-World Scenarios (ML: OLS hat matrix leverage, PCA variance capture,
      Ridge shrinkage effective df; GATE CS/DA: P^k=P & rank-1/3D distance, direct sum).
    - Tier 5 (6): Static AST & Syntax Integrity of scene_orthogonal_projections.py
      (zero deprecated symbols, ThreeDScene subclass, add_fixed_in_frame_mobjects,
      all 6 scene classes, make_hud_panel, zero eval/exec).

Execution:
    .venv/bin/pytest tests/test_sprint06_invariants.py -v
    .venv/bin/python -m unittest tests/test_sprint06_invariants.py
"""

from __future__ import annotations

import ast
import math
import os
import unittest
import numpy as np


# =============================================================================
# TIER 1: FEATURE COVERAGE (25 tests)
# =============================================================================

class TestTier1ProjectionOperatorAxioms(unittest.TestCase):
    """Tier 1 Feature 1: Projection Operator Axioms (MML §3.8 Def. 3.10)."""

    def setUp(self):
        self.A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        self.P = self.A @ np.linalg.inv(self.A.T @ self.A) @ self.A.T

    def test_idempotence_pi_squared_equals_pi(self):
        """Feature 1.1: P^2 == P and P(Px) == Px for arbitrary vectors."""
        np.testing.assert_allclose(self.P @ self.P, self.P, atol=1e-12)
        x = np.array([4.0, -2.0, 7.0])
        np.testing.assert_allclose(self.P @ (self.P @ x), self.P @ x, atol=1e-12)

    def test_subspace_fixed_point_property(self):
        """Feature 1.2: Every u in U = C(A) is fixed pointwise: Pu == u."""
        rng = np.random.default_rng(6101)
        for _ in range(10):
            coeffs = rng.standard_normal(2)
            u = self.A @ coeffs
            np.testing.assert_allclose(self.P @ u, u, atol=1e-12)

    def test_linearity_of_projection_map(self):
        """Feature 1.3: pi_U(alpha*x + beta*y) == alpha*pi_U(x) + beta*pi_U(y)."""
        x = np.array([3.0, -1.0, 1.0])
        y = np.array([-2.0, 5.0, 4.0])
        alpha, beta = 2.5, -1.75
        lhs = self.P @ (alpha * x + beta * y)
        rhs = alpha * (self.P @ x) + beta * (self.P @ y)
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)

    def test_nearest_point_optimality(self):
        """Feature 1.4: ||x - Px||_2 <= ||x - u||_2 for all u in U."""
        x = np.array([3.0, -1.0, 1.0])
        min_dist = np.linalg.norm(x - self.P @ x)
        rng = np.random.default_rng(6102)
        for _ in range(50):
            u = self.A @ rng.uniform(-5.0, 5.0, size=2)
            self.assertLessEqual(min_dist, np.linalg.norm(x - u) + 1e-12)

    def test_trivial_projectors_zero_and_identity(self):
        """Feature 1.5: Trivial projectors P=0 and P=I satisfy all projector axioms."""
        for P_triv, expected_rank in [(np.zeros((3, 3)), 0), (np.eye(3), 3)]:
            np.testing.assert_allclose(P_triv @ P_triv, P_triv, atol=1e-12)
            np.testing.assert_allclose(P_triv.T, P_triv, atol=1e-12)
            self.assertAlmostEqual(np.trace(P_triv), float(expected_rank), places=12)
            self.assertEqual(np.linalg.matrix_rank(P_triv), expected_rank)


class TestTier1OneDLineProjection(unittest.TestCase):
    """Tier 1 Feature 2: 1D Line Projection (MML §3.8.1, Ex. 3.10)."""

    def setUp(self):
        self.b = np.array([1.0, 2.0, 2.0])
        self.x_scene = np.array([3.0, 3.0, 0.0])
        self.x_mml = np.array([1.0, 1.0, 1.0])
        self.P1 = np.outer(self.b, self.b) / np.dot(self.b, self.b)

    def test_scalar_coordinate_lambda(self):
        """Feature 2.1: Scalar coordinate lambda = (b^T x) / ||b||^2 matches exact values."""
        lam_scene = np.dot(self.b, self.x_scene) / np.dot(self.b, self.b)
        lam_mml = np.dot(self.b, self.x_mml) / np.dot(self.b, self.b)
        self.assertAlmostEqual(lam_scene, 1.0, places=12)
        self.assertAlmostEqual(lam_mml, 5.0 / 9.0, places=12)

    def test_projected_points_1d(self):
        """Feature 2.2: pi_U(x) = lambda * b matches MML Ex. 3.10 and scene target."""
        np.testing.assert_allclose(self.P1 @ self.x_scene, [1.0, 2.0, 2.0], atol=1e-12)
        np.testing.assert_allclose(self.P1 @ self.x_mml, np.array([5.0, 10.0, 10.0]) / 9.0, atol=1e-12)

    def test_rank_one_projection_matrix_structure(self):
        """Feature 2.3: P1 = b b^T / ||b||^2 matches 1/9 [[1,2,2],[2,4,4],[2,4,4]], rank 1."""
        expected = np.array([[1, 2, 2], [2, 4, 4], [2, 4, 4]], dtype=float) / 9.0
        np.testing.assert_allclose(self.P1, expected, atol=1e-12)
        self.assertEqual(np.linalg.matrix_rank(self.P1), 1)
        self.assertAlmostEqual(np.trace(self.P1), 1.0, places=12)

    def test_residual_orthogonality_to_direction_b(self):
        """Feature 2.4: Residual r = x - pi_U(x) is strictly orthogonal to b."""
        r_scene = self.x_scene - self.P1 @ self.x_scene
        r_mml = self.x_mml - self.P1 @ self.x_mml
        np.testing.assert_allclose(r_scene, [2.0, 1.0, -2.0], atol=1e-12)
        self.assertAlmostEqual(np.dot(r_scene, self.b), 0.0, places=12)
        self.assertAlmostEqual(np.dot(r_mml, self.b), 0.0, places=12)

    def test_basis_scaling_invariance(self):
        """Feature 2.5: Projector P1 is invariant under non-zero rescaling b -> c*b."""
        for c in [-7.5, -1.0, 0.25, 13.0]:
            b_scaled = c * self.b
            P_scaled = np.outer(b_scaled, b_scaled) / np.dot(b_scaled, b_scaled)
            np.testing.assert_allclose(P_scaled, self.P1, atol=1e-12)


class TestTier1SubspaceMinDistanceAndOrthogonality(unittest.TestCase):
    """Tier 1 Feature 3: k-D Subspace Minimum Distance & Orthogonality Principle (VMLS §12.2)."""

    def setUp(self):
        self.A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        self.y = np.array([3.0, -1.0, 1.0])
        self.x_hat = np.array([2.0, -1.0])
        self.y_hat = np.array([2.0, 1.0, 0.0])
        self.r_opt = np.array([1.0, -2.0, 1.0])

    def test_sliding_trajectory_distance_formula(self):
        """Feature 3.1: Sliding path x(s) = [2+s, -1-s]^T yields exact d(s) = sqrt(6 + 2*s^2)."""
        for s in np.linspace(-2.0, 1.5, 36):
            x_s = np.array([2.0 + s, -1.0 - s])
            y_hat_s = self.A @ x_s
            np.testing.assert_allclose(y_hat_s, [2.0 + s, 1.0, -s], atol=1e-12)
            self.assertAlmostEqual(np.linalg.norm(self.y - y_hat_s), math.sqrt(6.0 + 2.0 * s * s), places=12)

    def test_global_minimum_at_s_zero(self):
        """Feature 3.2: Distance achieves strict global minimum sqrt(6) at s = 0."""
        s_grid = np.linspace(-2.0, 1.5, 351)  # includes s=0.0 exactly
        dists = [np.linalg.norm(self.y - self.A @ np.array([2.0 + s, -1.0 - s])) for s in s_grid]
        min_idx = int(np.argmin(dists))
        self.assertAlmostEqual(s_grid[min_idx], 0.0, places=12)
        self.assertAlmostEqual(dists[min_idx], math.sqrt(6.0), places=12)

    def test_vmls_calculus_free_quadratic_identity(self):
        """Feature 3.3: ||y - Ax||^2 == ||y - A x_hat||^2 + ||A(x - x_hat)||^2 (VMLS Eq. 12.7)."""
        rng = np.random.default_rng(6103)
        for _ in range(25):
            x = rng.uniform(-10.0, 10.0, size=2)
            lhs = np.linalg.norm(self.y - self.A @ x) ** 2
            rhs = np.linalg.norm(self.r_opt) ** 2 + np.linalg.norm(self.A @ (x - self.x_hat)) ** 2
            self.assertAlmostEqual(lhs, rhs, places=10)

    def test_column_orthogonality_at_optimum(self):
        """Feature 3.4: Optimal residual satisfies A^T (y - y_hat) == 0."""
        np.testing.assert_allclose(self.A.T @ self.r_opt, np.zeros(2), atol=1e-12)
        self.assertAlmostEqual(np.dot(self.A[:, 0], self.r_opt), 0.0, places=12)
        self.assertAlmostEqual(np.dot(self.A[:, 1], self.r_opt), 0.0, places=12)

    def test_orthogonality_to_arbitrary_subspace_vectors(self):
        """Feature 3.5: (Az)^T r_opt == 0 for any z in R^2 (VMLS Eq. 12.9)."""
        rng = np.random.default_rng(6104)
        Z = rng.uniform(-50.0, 50.0, size=(30, 2))
        dots = (Z @ self.A.T) @ self.r_opt
        np.testing.assert_allclose(dots, np.zeros(30), atol=1e-11)


class TestTier1NormalEquationsAndProjectorMatrix(unittest.TestCase):
    """Tier 1 Feature 4: Normal Equations & Closed-Form Projector P (MML §3.8.2)."""

    def setUp(self):
        self.A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        self.y = np.array([3.0, -1.0, 1.0])
        self.P = self.A @ np.linalg.inv(self.A.T @ self.A) @ self.A.T

    def test_gram_matrix_spd_invertible(self):
        """Feature 4.1: A^T A = [[3, 3], [3, 5]] is symmetric positive definite with det=6."""
        G = self.A.T @ self.A
        np.testing.assert_allclose(G, [[3.0, 3.0], [3.0, 5.0]], atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(G), 6.0, places=12)
        self.assertTrue(np.all(np.linalg.eigvalsh(G) > 0.0))

    def test_normal_equations_solution(self):
        """Feature 4.2: A^T A x_hat = A^T y uniquely solves to x_hat = [2, -1]^T."""
        x_hat = np.linalg.solve(self.A.T @ self.A, self.A.T @ self.y)
        np.testing.assert_allclose(x_hat, [2.0, -1.0], atol=1e-12)

    def test_closed_form_projector_mml_example_3_11(self):
        """Feature 4.3: P = A(A^T A)^{-1} A^T matches MML Eq. (3.64) exactly."""
        expected_P = np.array([
            [5.0, 2.0, -1.0],
            [2.0, 2.0, 2.0],
            [-1.0, 2.0, 5.0],
        ]) / 6.0
        np.testing.assert_allclose(self.P, expected_P, atol=1e-12)

    def test_projector_symmetry_and_idempotence(self):
        """Feature 4.4: Orthogonal projector P satisfies P^2 = P and P^T = P."""
        np.testing.assert_allclose(self.P @ self.P, self.P, atol=1e-12)
        np.testing.assert_allclose(self.P.T, self.P, atol=1e-12)

    def test_trace_equals_rank_equals_subspace_dim(self):
        """Feature 4.5: Tr(P) == rank(P) == dim(C(A)) == 2."""
        self.assertAlmostEqual(np.trace(self.P), 2.0, places=12)
        self.assertEqual(np.linalg.matrix_rank(self.P), 2)
        self.assertEqual(np.linalg.matrix_rank(self.A), 2)


class TestTier1OrthogonalComplementAndPythagorean(unittest.TestCase):
    """Tier 1 Feature 5: Orthogonal Complement I - P & Pythagorean Split (MML §3.6)."""

    def setUp(self):
        self.A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        self.n = np.array([1.0, -2.0, 1.0])
        self.P = self.A @ np.linalg.inv(self.A.T @ self.A) @ self.A.T
        self.Q_comp = np.eye(3) - self.P
        self.y = np.array([3.0, -1.0, 1.0])

    def test_complement_projects_onto_left_nullspace(self):
        """Feature 5.1: I - P equals n n^T / ||n||^2 and maps into N(A^T)."""
        expected_comp = np.outer(self.n, self.n) / np.dot(self.n, self.n)
        np.testing.assert_allclose(self.Q_comp, expected_comp, atol=1e-12)
        np.testing.assert_allclose(self.A.T @ (self.Q_comp @ self.y), np.zeros(2), atol=1e-12)

    def test_complement_annihilation_with_p(self):
        """Feature 5.2: P(I - P) == 0 and (I - P)P == 0."""
        np.testing.assert_allclose(self.P @ self.Q_comp, np.zeros((3, 3)), atol=1e-12)
        np.testing.assert_allclose(self.Q_comp @ self.P, np.zeros((3, 3)), atol=1e-12)

    def test_direct_sum_reconstruction(self):
        """Feature 5.3: Every y decomposes uniquely as Py + (I - P)y with Py perp (I-P)y."""
        y_u = self.P @ self.y
        y_perp = self.Q_comp @ self.y
        np.testing.assert_allclose(y_u + y_perp, self.y, atol=1e-12)
        self.assertAlmostEqual(np.dot(y_u, y_perp), 0.0, places=12)

    def test_pythagorean_theorem_identity(self):
        """Feature 5.4: ||y||^2 = ||Py||^2 + ||(I-P)y||^2 (11 = 5 + 6)."""
        norm_y_sq = np.linalg.norm(self.y) ** 2
        norm_u_sq = np.linalg.norm(self.P @ self.y) ** 2
        norm_p_sq = np.linalg.norm(self.Q_comp @ self.y) ** 2
        self.assertAlmostEqual(norm_y_sq, 11.0, places=12)
        self.assertAlmostEqual(norm_u_sq, 5.0, places=12)
        self.assertAlmostEqual(norm_p_sq, 6.0, places=12)
        self.assertAlmostEqual(norm_y_sq, norm_u_sq + norm_p_sq, places=12)

    def test_complement_dimension_n_minus_k(self):
        """Feature 5.5: Tr(I - P) == rank(I - P) == n - k == 1."""
        self.assertAlmostEqual(np.trace(self.Q_comp), 1.0, places=12)
        self.assertEqual(np.linalg.matrix_rank(self.Q_comp), 1)


# =============================================================================
# TIER 2: BOUNDARY & CORNER CASES (25 tests)
# =============================================================================

class TestTier2ZeroAndFullSubspaceBoundaries(unittest.TestCase):
    """Tier 2 Boundary 1: Zero/Full Subspaces & In/Orthogonal Target Vectors."""

    def setUp(self):
        self.A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        self.P = self.A @ np.linalg.inv(self.A.T @ self.A) @ self.A.T

    def test_zero_subspace_projector(self):
        """Boundary 1.1: 0D subspace {0} has P=0, norm=0, residual=y."""
        P0 = np.zeros((3, 3))
        y = np.array([2.0, -3.0, 4.0])
        np.testing.assert_allclose(P0 @ y, np.zeros(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(y - P0 @ y), np.linalg.norm(y), places=12)

    def test_full_space_projector(self):
        """Boundary 1.2: Full space R^3 (k=3) has P=I_3 and zero residual for all y."""
        B_full = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 1.0]])
        P_full = B_full @ np.linalg.inv(B_full.T @ B_full) @ B_full.T
        np.testing.assert_allclose(P_full, np.eye(3), atol=1e-12)
        y = np.array([7.0, -4.0, 9.0])
        np.testing.assert_allclose(y - P_full @ y, np.zeros(3), atol=1e-12)

    def test_target_already_in_subspace(self):
        """Boundary 1.3: If y in U, Py == y and residual norm is 0."""
        y_in = 3.0 * self.A[:, 0] - 2.0 * self.A[:, 1]
        np.testing.assert_allclose(self.P @ y_in, y_in, atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(y_in - self.P @ y_in), 0.0, places=12)

    def test_target_in_orthogonal_complement(self):
        """Boundary 1.4: If y in U^perp, Py == 0 and residual equals y."""
        y_perp = 4.0 * np.array([1.0, -2.0, 1.0])
        np.testing.assert_allclose(self.P @ y_perp, np.zeros(3), atol=1e-12)
        np.testing.assert_allclose((np.eye(3) - self.P) @ y_perp, y_perp, atol=1e-12)

    def test_zero_target_vector(self):
        """Boundary 1.5: Projecting zero vector yields zero projection and zero residual."""
        z = np.zeros(3)
        np.testing.assert_allclose(self.P @ z, z, atol=1e-12)
        np.testing.assert_allclose((np.eye(3) - self.P) @ z, z, atol=1e-12)


class TestTier2AffineSubspaceProjection(unittest.TestCase):
    """Tier 2 Boundary 2: Affine Subspace Projection (MML §3.8.4)."""

    def setUp(self):
        self.A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        self.P = self.A @ np.linalg.inv(self.A.T @ self.A) @ self.A.T
        self.x0 = np.array([0.0, 0.0, 1.0])
        self.y_aff = np.array([3.0, -1.0, 2.0])

    def pi_L(self, x: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        base = self.x0 if x0 is None else x0
        return base + self.P @ (x - base)

    def test_affine_projection_formula(self):
        """Boundary 2.1: pi_L(y_aff) == [2, 1, 1]^T and lies in x0 + C(A)."""
        proj = self.pi_L(self.y_aff)
        np.testing.assert_allclose(proj, [2.0, 1.0, 1.0], atol=1e-12)
        # Check (proj - x0) satisfies x - 2y + z = 0
        diff = proj - self.x0
        self.assertAlmostEqual(diff[0] - 2.0 * diff[1] + diff[2], 0.0, places=12)

    def test_basepoint_shift_invariance_along_u(self):
        """Boundary 2.2: Shifting x0 by any u in U leaves pi_L(x) unchanged."""
        u_shift = 2.5 * self.A[:, 0] - 1.5 * self.A[:, 1]
        proj1 = self.pi_L(self.y_aff, self.x0)
        proj2 = self.pi_L(self.y_aff, self.x0 + u_shift)
        np.testing.assert_allclose(proj1, proj2, atol=1e-12)

    def test_affine_map_is_non_linear_when_x0_outside_u(self):
        """Boundary 2.3: pi_L(0) != 0 and pi_L(2x) != 2*pi_L(x), though pi_L^2 == pi_L."""
        p0 = self.pi_L(np.zeros(3))
        self.assertGreater(np.linalg.norm(p0), 0.1)
        x = np.array([1.0, 2.0, 3.0])
        self.assertFalse(np.allclose(self.pi_L(2.0 * x), 2.0 * self.pi_L(x)))
        np.testing.assert_allclose(self.pi_L(self.pi_L(x)), self.pi_L(x), atol=1e-12)

    def test_affine_residual_orthogonal_to_u(self):
        """Boundary 2.4: Residual x - pi_L(x) is strictly orthogonal to C(A)."""
        r = self.y_aff - self.pi_L(self.y_aff)
        np.testing.assert_allclose(r, [1.0, -2.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(self.A.T @ r, np.zeros(2), atol=1e-12)

    def test_affine_minimum_distance_equals_shifted_linear_distance(self):
        """Boundary 2.5: d(y_aff, L) == d(y_aff - x0, U) == sqrt(6)."""
        d_aff = np.linalg.norm(self.y_aff - self.pi_L(self.y_aff))
        d_lin = np.linalg.norm((self.y_aff - self.x0) - self.P @ (self.y_aff - self.x0))
        self.assertAlmostEqual(d_aff, d_lin, places=12)
        self.assertAlmostEqual(d_aff, math.sqrt(6.0), places=12)


class TestTier2RankDeficientAndRidgeResolution(unittest.TestCase):
    """Tier 2 Boundary 3: Rank-Deficient A_deg -> Pseudoinverse A^+ & Ridge."""

    def setUp(self):
        self.A_deg = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        self.y = np.array([3.0, -1.0, 1.0])

    def test_singular_gram_matrix_raises_linalgerror(self):
        """Boundary 3.1: A_deg^T A_deg has det == 0 and solve raises LinAlgError."""
        G = self.A_deg.T @ self.A_deg
        self.assertAlmostEqual(np.linalg.det(G), 0.0, places=12)
        with self.assertRaises(np.linalg.LinAlgError):
            np.linalg.solve(G, self.A_deg.T @ self.y)

    def test_non_unique_least_squares_coefficients(self):
        """Boundary 3.2: Adding null vector v=[-2, 1]^T to x_hat gives identical projection."""
        x_ln = np.linalg.pinv(self.A_deg) @ self.y
        v_null = np.array([-2.0, 1.0])
        for alpha in [-10.0, 0.0, 3.5]:
            x_alt = x_ln + alpha * v_null
            np.testing.assert_allclose(self.A_deg @ x_alt, self.A_deg @ x_ln, atol=1e-12)

    def test_moore_penrose_projector_and_min_norm_solution(self):
        """Boundary 3.3: P = A A^+ = 1/3 * 1 1^T and x_LN = [0.2, 0.4]^T."""
        A_pinv = np.linalg.pinv(self.A_deg)
        P_deg = self.A_deg @ A_pinv
        np.testing.assert_allclose(P_deg, np.ones((3, 3)) / 3.0, atol=1e-12)
        x_ln = A_pinv @ self.y
        np.testing.assert_allclose(x_ln, [0.2, 0.4], atol=1e-12)
        self.assertAlmostEqual(np.dot(x_ln, np.array([-2.0, 1.0])), 0.0, places=12)

    def test_ridge_spd_invertibility_for_all_positive_lambda(self):
        """Boundary 3.4: A^T A + lam I is strictly SPD and nonsingular for all lam > 0."""
        G = self.A_deg.T @ self.A_deg
        for lam in [1e-6, 1e-2, 0.5, 10.0]:
            G_reg = G + lam * np.eye(2)
            eigs = np.linalg.eigvalsh(G_reg)
            self.assertTrue(np.all(eigs >= lam - 1e-12))
            x_rls = np.linalg.solve(G_reg, self.A_deg.T @ self.y)
            self.assertEqual(x_rls.shape, (2,))

    def test_ridge_limit_converges_to_min_norm_pseudoinverse(self):
        """Boundary 3.5: As lam -> 0+, Ridge solution converges to A^+ y = [0.2, 0.4]^T."""
        x_ln = np.linalg.pinv(self.A_deg) @ self.y
        x_rls = np.linalg.solve(
            self.A_deg.T @ self.A_deg + 1e-10 * np.eye(2),
            self.A_deg.T @ self.y,
        )
        np.testing.assert_allclose(x_rls, x_ln, atol=1e-6)


class TestTier2IllConditionedNormalEquations(unittest.TestCase):
    """Tier 2 Boundary 4: Lauchli Ill-Conditioned Normal Equations (VMLS §12.3)."""

    def _lauchli(self, eps: float) -> tuple[np.ndarray, np.ndarray]:
        A = np.array([[1.0, 1.0], [eps, 0.0], [0.0, eps]])
        b = np.array([2.0, eps, eps])
        return A, b

    def test_condition_number_squaring(self):
        """Boundary 4.1: kappa_2(A_eps^T A_eps) == kappa_2(A_eps)^2."""
        A, _ = self._lauchli(1e-4)
        k_A = np.linalg.cond(A)
        k_ATA = np.linalg.cond(A.T @ A)
        self.assertAlmostEqual(k_ATA / (k_A ** 2), 1.0, places=4)

    def test_normal_equations_collapse_at_eps_1e_8(self):
        """Boundary 4.2: At eps=1e-8 in float64, A^T A loses rank (1+eps^2 == 1.0)."""
        A, _ = self._lauchli(1e-8)
        G = A.T @ A
        np.testing.assert_array_equal(G, np.ones((2, 2)))
        self.assertEqual(np.linalg.matrix_rank(G), 1)

    def test_qr_and_svd_lstsq_remain_accurate(self):
        """Boundary 4.3: QR and SVD lstsq recover exact x_hat = [1, 1]^T at eps=1e-8."""
        A, b = self._lauchli(1e-8)
        x_lstsq, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        Q, R = np.linalg.qr(A)
        x_qr = np.linalg.solve(R, Q.T @ b)
        self.assertEqual(rank, 2)
        np.testing.assert_allclose(x_lstsq, [1.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(x_qr, [1.0, 1.0], atol=1e-12)

    def test_qr_residual_orthogonality_under_ill_conditioning(self):
        """Boundary 4.4: Residual from QR solution stays exact zero on consistent Lauchli."""
        A, b = self._lauchli(1e-7)
        Q, R = np.linalg.qr(A)
        x_qr = np.linalg.solve(R, Q.T @ b)
        r = b - A @ x_qr
        self.assertLess(np.linalg.norm(r), 1e-14)
        self.assertLess(np.linalg.norm(A.T @ r), 1e-14)

    def test_ridge_stabilizes_singular_lauchli_normal_eqs(self):
        """Boundary 4.5: Ridge restores full rank when normal equations collapse."""
        A, b = self._lauchli(1e-8)
        x_rls = np.linalg.solve(A.T @ A + 1e-6 * np.eye(2), A.T @ b)
        self.assertTrue(np.all(np.isfinite(x_rls)))
        np.testing.assert_allclose(x_rls, [1.0, 1.0], atol=1e-4)


class TestTier2ObliqueVsOrthogonalProjectors(unittest.TestCase):
    """Tier 2 Boundary 5: Oblique vs. Orthogonal Projectors."""

    def setUp(self):
        self.A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        self.P_orth = self.A @ np.linalg.inv(self.A.T @ self.A) @ self.A.T
        self.P_ob = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 2.0, 0.0],
        ])
        self.y_test = np.array([1.0, 1.0, 3.0])

    def test_oblique_idempotence_and_image(self):
        """Boundary 5.1: P_ob^2 == P_ob and P_ob maps every y into U = C(A)."""
        np.testing.assert_allclose(self.P_ob @ self.P_ob, self.P_ob, atol=1e-12)
        y_proj = self.P_ob @ self.y_test
        self.assertAlmostEqual(y_proj[0] - 2.0 * y_proj[1] + y_proj[2], 0.0, places=12)

    def test_oblique_asymmetry(self):
        """Boundary 5.2: Oblique projector breaks symmetry P_ob^T != P_ob."""
        self.assertFalse(np.allclose(self.P_ob.T, self.P_ob))

    def test_oblique_non_contractive_operator_norm(self):
        """Boundary 5.3: ||P_ob||_2 == sqrt(6) > 1 (amplifies orthogonal vectors)."""
        norm_ob = np.linalg.norm(self.P_ob, 2)
        self.assertAlmostEqual(norm_ob, math.sqrt(6.0), places=12)
        self.assertGreater(norm_ob, 1.0)

    def test_oblique_breaks_pythagorean_split(self):
        """Boundary 5.4: (P_ob y)^T (I - P_ob) y != 0, breaking Pythagoras."""
        y_u = self.P_ob @ self.y_test
        y_r = self.y_test - y_u
        self.assertNotAlmostEqual(np.dot(y_u, y_r), 0.0, places=3)
        self.assertNotAlmostEqual(
            np.linalg.norm(self.y_test) ** 2,
            np.linalg.norm(y_u) ** 2 + np.linalg.norm(y_r) ** 2,
            places=3,
        )

    def test_oblique_strictly_larger_projection_error(self):
        """Boundary 5.5: ||y - P_ob y||_2 == 2.0 > ||y - P_orth y||_2 == sqrt(6)/3."""
        err_ob = np.linalg.norm(self.y_test - self.P_ob @ self.y_test)
        err_orth = np.linalg.norm(self.y_test - self.P_orth @ self.y_test)
        self.assertAlmostEqual(err_ob, 2.0, places=12)
        self.assertAlmostEqual(err_orth, math.sqrt(6.0) / 3.0, places=12)
        self.assertGreater(err_ob, err_orth)


# =============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS (5 tests)
# =============================================================================

class TestTier3CrossFeatureCombinations(unittest.TestCase):
    """Tier 3: Spectral, Basis-Invariance, QR/SVD, Nested, and Weighted Projectors."""

    def setUp(self):
        self.A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        self.P = self.A @ np.linalg.inv(self.A.T @ self.A) @ self.A.T

    def test_eigenvalue_spectrum_0_1_and_operator_norm(self):
        """Cross 1: Symmetric projector P has eigenvalues {0, 1, 1} and ||P||_2 == 1."""
        eigs = np.sort(np.linalg.eigvalsh(self.P))
        np.testing.assert_allclose(eigs, [0.0, 1.0, 1.0], atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(self.P, 2), 1.0, places=12)

    def test_basis_invariance_under_invertible_reparameterization(self):
        """Cross 2: Replacing A with A*M (M in GL_2) produces the identical projector P."""
        M = np.array([[2.0, -3.0], [1.0, 4.0]])
        B = self.A @ M
        P_B = B @ np.linalg.inv(B.T @ B) @ B.T
        np.testing.assert_allclose(P_B, self.P, atol=1e-12)

    def test_equivalence_normal_eqs_qr_and_svd(self):
        """Cross 3: A(A^T A)^{-1}A^T == Q Q^T == U_k U_k^T."""
        Q, _ = np.linalg.qr(self.A)
        U, _, _ = np.linalg.svd(self.A, full_matrices=False)
        np.testing.assert_allclose(Q @ Q.T, self.P, atol=1e-12)
        np.testing.assert_allclose(U @ U.T, self.P, atol=1e-12)

    def test_nested_subspace_projectors_commute_and_absorb(self):
        """Cross 4: If U1 subset U2, then P1 P2 == P2 P1 == P1."""
        a1 = self.A[:, [0]]
        P1 = a1 @ np.linalg.inv(a1.T @ a1) @ a1.T
        P2 = self.P
        np.testing.assert_allclose(P1 @ P2, P1, atol=1e-12)
        np.testing.assert_allclose(P2 @ P1, P1, atol=1e-12)

    def test_weighted_inner_product_projector(self):
        """Cross 5: W-orthogonal projector P_W = A(A^T W A)^{-1} A^T W is W-self-adjoint."""
        W = np.diag([1.0, 4.0, 2.0])
        P_W = self.A @ np.linalg.inv(self.A.T @ W @ self.A) @ self.A.T @ W
        np.testing.assert_allclose(P_W @ P_W, P_W, atol=1e-12)
        # W-self-adjoint: W P_W == P_W^T W
        np.testing.assert_allclose(W @ P_W, P_W.T @ W, atol=1e-12)
        # Euclidean symmetry fails unless W = cI
        self.assertFalse(np.allclose(P_W.T, P_W))


# =============================================================================
# TIER 4: REAL-WORLD ML & GATE CS/DA SCENARIOS (5 tests)
# =============================================================================

class TestTier4RealWorldScenarios(unittest.TestCase):
    """Tier 4: ML (OLS Hat Matrix, PCA, Ridge) & GATE CS/DA Applications."""

    def test_ml_ols_hat_matrix_and_leverage(self):
        """Real-World 1 (ML): OLS Hat matrix H = X(X^T X)^{-1}X^T leverage bounds & sum."""
        rng = np.random.default_rng(6401)
        n, p = 40, 4
        X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
        y = X @ np.array([1.0, -2.0, 0.5, 3.0]) + 0.2 * rng.standard_normal(n)
        H = X @ np.linalg.inv(X.T @ X) @ X.T
        h_diag = np.diag(H)
        self.assertTrue(np.all(h_diag >= -1e-12) and np.all(h_diag <= 1.0 + 1e-12))
        self.assertAlmostEqual(float(np.sum(h_diag)), float(p), places=11)
        residuals = y - H @ y
        np.testing.assert_allclose(X.T @ residuals, np.zeros(p), atol=1e-10)

    def test_ml_pca_rank_k_projection_optimality(self):
        """Real-World 2 (ML): PCA projector P_k = B B^T achieves exact eigenvalue residual loss."""
        rng = np.random.default_rng(6402)
        X = rng.standard_normal((150, 5)) @ np.diag([4.0, 3.0, 1.5, 0.4, 0.1])
        Xc = X - X.mean(axis=0)
        cov = (Xc.T @ Xc) / len(Xc)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
        B = eigvecs[:, :2]
        P_pca = B @ B.T
        recon_mse = np.mean(np.sum((Xc - Xc @ P_pca.T) ** 2, axis=1))
        self.assertAlmostEqual(recon_mse, float(np.sum(eigvals[2:])), places=11)

    def test_ml_ridge_shrinkage_and_effective_df(self):
        """Real-World 3 (ML): Ridge hat matrix H_lam has eigenvalues < 1 & Tr(H_lam) = eff df."""
        A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        lam = 0.75
        H_lam = A @ np.linalg.inv(A.T @ A + lam * np.eye(2)) @ A.T
        s = np.linalg.svd(A, compute_uv=False)
        eff_df = sum((si ** 2) / (si ** 2 + lam) for si in s)
        self.assertAlmostEqual(np.trace(H_lam), eff_df, places=12)
        eigs = np.linalg.eigvalsh(H_lam)
        self.assertTrue(np.all(eigs < 1.0))
        self.assertFalse(np.allclose(H_lam @ H_lam, H_lam))

    def test_gate_projector_power_trace_and_distance(self):
        """Real-World 4 (GATE): P^2026 == P, Tr(P) == rank(P), and 3D plane distance."""
        b = np.array([1.0, 2.0, 2.0])
        P1 = np.outer(b, b) / np.dot(b, b)
        P_pow = np.linalg.matrix_power(P1, 2026)
        np.testing.assert_allclose(P_pow, P1, atol=1e-12)
        self.assertAlmostEqual(np.trace(P_pow), float(np.linalg.matrix_rank(P_pow)), places=12)
        y = np.array([3.0, 3.0, 0.0])
        dist = math.sqrt(np.linalg.norm(y) ** 2 - (np.dot(b, y) / np.linalg.norm(b)) ** 2)
        self.assertAlmostEqual(dist, 3.0, places=12)

    def test_gate_inconsistent_least_squares_direct_sum(self):
        """Real-World 5 (GATE): Normal equations solve inconsistent Ax=b and R^3 = U (+) U^perp."""
        A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        b = np.array([3.0, -1.0, 1.0])
        self.assertEqual(np.linalg.matrix_rank(A), 2)
        self.assertEqual(np.linalg.matrix_rank(np.column_stack([A, b])), 3)
        x_hat = np.linalg.solve(A.T @ A, A.T @ b)
        r = b - A @ x_hat
        self.assertAlmostEqual(np.dot(A @ x_hat, r), 0.0, places=12)


# =============================================================================
# TIER 5: STATIC AST & SYNTAX INTEGRITY (6 tests)
# =============================================================================

class TestSprint06ASTAndSyntax(unittest.TestCase):
    """Tier 5: Static AST verification of Sprint 6 scene for ManimCE 3D standards."""

    SCENE_PATH = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "linear_algebra",
            "sprint_06_orthogonal_projections",
            "scene_orthogonal_projections.py",
        )
    )

    def _get_ast(self):
        if not os.path.exists(self.SCENE_PATH):
            self.skipTest(f"Scene script '{self.SCENE_PATH}' not yet created.")
        with open(self.SCENE_PATH, "r", encoding="utf-8") as f:
            return ast.parse(f.read(), filename=self.SCENE_PATH)

    def test_no_deprecated_manimlib_symbols(self):
        """AST 1: Zero deprecated symbols (CONFIG, ShowCreation, TexMobject, TextMobject)."""
        tree = self._get_ast()
        deprecated = {"CONFIG", "ShowCreation", "TexMobject", "TextMobject"}
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in deprecated:
                found.append((node.id, getattr(node, "lineno", "?")))
            elif isinstance(node, ast.Attribute) and node.attr in deprecated:
                found.append((node.attr, getattr(node, "lineno", "?")))
        self.assertEqual(found, [], f"Deprecated symbols found: {found}")

    def test_scene_subclasses_threedscene(self):
        """AST 2: Base scene class inherits from ThreeDScene."""
        tree = self._get_ast()
        threed_subclasses = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if (isinstance(base, ast.Name) and base.id == "ThreeDScene") or (
                        isinstance(base, ast.Attribute) and base.attr == "ThreeDScene"
                    ):
                        threed_subclasses.append(node.name)
        self.assertGreater(len(threed_subclasses), 0, "Must define a ThreeDScene subclass.")

    def test_hud_pinning_add_fixed_in_frame_mobjects_invoked(self):
        """AST 3: Scene invokes add_fixed_in_frame_mobjects for pinned 2D HUD panels."""
        tree = self._get_ast()
        has_hud_call = any(
            isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "add_fixed_in_frame_mobjects")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "add_fixed_in_frame_mobjects")
            )
            for node in ast.walk(tree)
        )
        self.assertTrue(has_hud_call, "Must call add_fixed_in_frame_mobjects.")

    def test_scene_classes_defined(self):
        """AST 4: All 6 required Sprint 6 scene classes are defined."""
        tree = self._get_ast()
        class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        expected = {
            "BaseOrthogonalProjectionsScene",
            "Act1SpatialPuzzle",
            "Act2DynamicDeformation",
            "Act3AlgebraicMorph",
            "Act4DegenerateStressTest",
            "OrthogonalProjectionsScene",
        }
        missing = expected - class_names
        self.assertEqual(missing, set(), f"Missing scene classes: {missing}")

    def test_make_hud_panel_usage(self):
        """AST 5: File defines and invokes make_hud_panel."""
        tree = self._get_ast()
        has_def = any(
            isinstance(node, ast.FunctionDef) and node.name == "make_hud_panel"
            for node in ast.walk(tree)
        )
        has_call = any(
            isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "make_hud_panel")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "make_hud_panel")
            )
            for node in ast.walk(tree)
        )
        self.assertTrue(has_def and has_call, "Must define and invoke make_hud_panel.")

    def test_no_forbidden_execution_calls(self):
        """AST 6: Zero calls to eval() or exec()."""
        tree = self._get_ast()
        forbidden = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
                forbidden.append((node.func.id, getattr(node, "lineno", "?")))
        self.assertEqual(forbidden, [], f"Forbidden calls found: {forbidden}")


if __name__ == "__main__":
    unittest.main()
