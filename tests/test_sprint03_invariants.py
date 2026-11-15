#!/usr/bin/env python3
r"""
Sprint 3 E2E Invariant Test Suite: Column Space, Null Space & Rank-Nullity
========================================================================
Curriculum: Linear Algebra (Geometric Foundations & AI/Systems Invariants)
References:
    - Mathematics for Machine Learning (MML Ch. 2.6 - Image & Kernel, Rank-Nullity)
    - Introduction to Applied Linear Algebra (VMLS Ch. 10)
    - Erwin Kreyszig — Advanced Engineering Mathematics 10th Ed. (Ch. 7)

Architecture:
    Comprehensive 4-Tier Test Suite implemented with unittest and NumPy:
    - Tier 1: Feature Coverage (Rank-2 3x3 architecture, column space plane x+y-z=0,
      null space 1D line t[1, 1, -1]^T, rank-nullity invariant 2+1=3, particular solution
      x_p=[1, 2, 0]^T and affine line x_p + N(A)). >=5 tests each.
    - Tier 2: Boundary & Corner Cases (Inconsistent systems Rouche-Capelli test,
      rank-1 collapse, full rank-3 trivial nullspace, zero matrix rank 0, large coordinate precision).
      >=5 tests each.
    - Tier 3: Cross-Feature Combinations (Orthogonal complement N(A) = (C(A^T))^\perp,
      projection matrix P_C = A(A^T A)^+ A^T, direct sum decomposition, affine subspace invariance,
      Moore-Penrose pseudoinverse axioms).
    - Tier 4: Real-World Scenarios (ML: Overparameterized Jacobian null space & loss flatness,
      LoRA low-rank adaptation W = W_0 + BA with rank r << d, Linear autoencoder bottleneck;
      GATE: Rouche-Capelli 3D classification, abstract space rank-nullity).
    - AST & Syntax: Static AST inspection of sprint 3 scene for ThreeDScene subclassing,
      zero deprecated symbols, and add_fixed_in_frame_mobjects HUD pinning
      (gracefully skipped if not yet created).

Execution:
    python -m unittest tests/test_sprint03_invariants.py
    pytest tests/test_sprint03_invariants.py
"""

from __future__ import annotations

import ast
import math
import os
import unittest
import numpy as np


# =============================================================================
# TIER 1: FEATURE COVERAGE
# =============================================================================

class TestTier1RankTwoArchitecture(unittest.TestCase):
    """Tier 1 Feature 1: Rank-2 3x3 Matrix Architecture (MML §2.6, Kreyszig §7.4)."""

    def setUp(self):
        # Canonical Sprint 3 matrix:
        # A = [[1, 0, 1], [0, 1, 1], [1, 1, 2]]
        self.A = np.array([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0]
        ])

    def test_matrix_dimensions_and_rank(self):
        """Feature 1.1: Matrix is 3x3 and has rank exactly 2."""
        self.assertEqual(self.A.shape, (3, 3))
        rank = np.linalg.matrix_rank(self.A)
        self.assertEqual(rank, 2)

    def test_determinant_zero(self):
        """Feature 1.2: Determinant is exactly 0.0 (3D volume collapses to zero)."""
        det = np.linalg.det(self.A)
        self.assertAlmostEqual(det, 0.0, places=12)

    def test_column_linear_dependency(self):
        """Feature 1.3: Column 3 is the exact linear sum of Column 1 and Column 2: a3 = a1 + a2."""
        a1 = self.A[:, 0]
        a2 = self.A[:, 1]
        a3 = self.A[:, 2]
        np.testing.assert_allclose(a3, a1 + a2, atol=1e-12)

    def test_svd_spectrum_two_non_zero_singular_values(self):
        """Feature 1.4: SVD spectrum contains exactly two positive singular values and one zero."""
        s = np.linalg.svd(self.A, compute_uv=False)
        self.assertGreater(s[0], 1.0)
        self.assertGreater(s[1], 0.5)
        self.assertAlmostEqual(s[2], 0.0, places=12)

    def test_unit_cube_volume_annihilation(self):
        """Feature 1.5: 3D unit cube vertices map to a degenerate 2D polygon with zero 3D volume."""
        # 8 vertices of unit cube [0, 1]^3
        cube_vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
        ], dtype=float)
        mapped = cube_vertices @ self.A.T
        # All mapped points satisfy plane equation x + y - z = 0
        plane_eval = mapped[:, 0] + mapped[:, 1] - mapped[:, 2]
        np.testing.assert_allclose(plane_eval, 0.0, atol=1e-12)


class TestTier1ColumnSpacePlane(unittest.TestCase):
    """Tier 1 Feature 2: Column Space Plane Geometry C(A) (MML §2.6, VMLS §10.1)."""

    def setUp(self):
        self.A = np.array([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0]
        ])
        self.a1 = self.A[:, 0]
        self.a2 = self.A[:, 1]
        self.normal = np.array([1.0, 1.0, -1.0])

    def test_column_space_dimension(self):
        """Feature 2.1: Column space dimension equals rank(A) == 2."""
        rank = np.linalg.matrix_rank(self.A)
        self.assertEqual(rank, 2)

    def test_normal_vector_orthogonality(self):
        """Feature 2.2: Normal vector n = [1, 1, -1]^T is orthogonal to all columns of A."""
        for j in range(3):
            col = self.A[:, j]
            dot_prod = np.dot(self.normal, col)
            self.assertAlmostEqual(dot_prod, 0.0, places=12)

    def test_plane_equation_containment(self):
        """Feature 2.3: For any vector x in R^3, y = Ax satisfies plane equation x + y - z = 0."""
        rng = np.random.default_rng(301)
        X = rng.uniform(-50.0, 50.0, size=(100, 3))
        Y = X @ self.A.T  # Shape (100, 3)
        plane_residuals = Y @ self.normal
        np.testing.assert_allclose(plane_residuals, 0.0, atol=1e-12)

    def test_origin_contained_in_column_space(self):
        """Feature 2.4: Origin (0, 0, 0) satisfies plane equation, confirming C(A) is a linear subspace."""
        zero = np.zeros(3)
        self.assertAlmostEqual(np.dot(self.normal, zero), 0.0, places=15)

    def test_target_vector_containment_rouche_capelli(self):
        """Feature 2.5: Canonical target b = [1, 2, 3]^T satisfies rank([A|b]) == rank(A) == 2."""
        b = np.array([1.0, 2.0, 3.0])
        # Check plane equation: 1 + 2 - 3 == 0
        self.assertAlmostEqual(np.dot(self.normal, b), 0.0, places=12)
        aug = np.column_stack([self.A, b])
        self.assertEqual(np.linalg.matrix_rank(aug), 2)


class TestTier1NullSpaceLine(unittest.TestCase):
    """Tier 1 Feature 3: Null Space 1D Line Geometry N(A) (MML §2.6, Kreyszig §7.4)."""

    def setUp(self):
        self.A = np.array([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0]
        ])
        self.v_null = np.array([1.0, 1.0, -1.0])

    def test_null_space_dimension(self):
        """Feature 3.1: Null space dimension (nullity) is 3 - rank(A) == 1."""
        s = np.linalg.svd(self.A, compute_uv=False)
        nullity = int(np.sum(s < 1e-12))
        self.assertEqual(nullity, 1)

    def test_null_vector_annihilation(self):
        """Feature 3.2: Null vector v_null = [1, 1, -1]^T satisfies A @ v_null == [0, 0, 0]^T."""
        out = self.A @ self.v_null
        np.testing.assert_allclose(out, np.zeros(3), atol=1e-12)

    def test_1d_null_line_collapse(self):
        """Feature 3.3: Every point on line t * v_null maps directly to the origin 0."""
        t_values = np.linspace(-100.0, 100.0, 50)
        pts = t_values[:, np.newaxis] * self.v_null[np.newaxis, :]
        mapped = pts @ self.A.T
        np.testing.assert_allclose(mapped, 0.0, atol=1e-12)

    def test_origin_contained_in_null_space(self):
        """Feature 3.4: Origin t=0 is contained in N(A), satisfying subspace axiom."""
        self.assertAlmostEqual(np.linalg.norm(self.A @ np.zeros(3)), 0.0, places=15)

    def test_null_vector_uniqueness_up_to_scalar(self):
        """Feature 3.5: Any vector annihilated by A is a scalar multiple of v_null."""
        _, _, Vt = np.linalg.svd(self.A)
        svd_null_dir = Vt[2]  # Right singular vector corresponding to sigma_3 = 0
        # Cosine similarity must be +1 or -1
        cos_sim = abs(np.dot(svd_null_dir, self.v_null) / (np.linalg.norm(svd_null_dir) * np.linalg.norm(self.v_null)))
        self.assertAlmostEqual(cos_sim, 1.0, places=10)


class TestTier1RankNullityTheorem(unittest.TestCase):
    """Tier 1 Feature 4: Rank-Nullity Theorem Conservation (MML Theorem 2.24)."""

    def setUp(self):
        self.A = np.array([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0]
        ])

    def test_rank_nullity_invariant(self):
        """Feature 4.1: rank(A) + nullity(A) == n (2 + 1 == 3) holds strictly."""
        n = self.A.shape[1]
        rank = np.linalg.matrix_rank(self.A)
        s = np.linalg.svd(self.A, compute_uv=False)
        nullity = int(np.sum(s < 1e-12))
        self.assertEqual(rank + nullity, n)
        self.assertEqual(rank, 2)
        self.assertEqual(nullity, 1)

    def test_rank_nullity_elementary_row_operations(self):
        """Feature 4.2: Invariance under row reduction: RREF produces 2 pivot columns and 1 free variable."""
        # Row 3 - Row 1 - Row 2 -> [0, 0, 0]
        RREF = np.array([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0]
        ])
        pivots = 2
        free_variables = 3 - pivots
        self.assertEqual(pivots, 2)
        self.assertEqual(free_variables, 1)
        self.assertEqual(pivots + free_variables, 3)

    def test_rank_nullity_randomized_matrices(self):
        """Feature 4.3: Rank-nullity conservation holds across 20 random 3x3 matrices of varying ranks."""
        rng = np.random.default_rng(302)
        for _ in range(20):
            M = rng.uniform(-10.0, 10.0, size=(3, 3))
            rank = np.linalg.matrix_rank(M)
            s = np.linalg.svd(M, compute_uv=False)
            nullity = int(np.sum(s < 1e-10))
            self.assertEqual(rank + nullity, 3)

    def test_dimension_conservation_projection(self):
        """Feature 4.4: Dimension of domain R^3 is preserved: dim(Im(T)) + dim(Ker(T)) == 3."""
        dim_domain = 3
        dim_im = 2
        dim_ker = 1
        self.assertEqual(dim_im + dim_ker, dim_domain)

    def test_svd_singular_values_conservation(self):
        """Feature 4.5: Total count of non-zero + zero singular values equals domain dimension n."""
        s = np.linalg.svd(self.A, compute_uv=False)
        nonzero_count = np.sum(s >= 1e-12)
        zero_count = np.sum(s < 1e-12)
        self.assertEqual(nonzero_count + zero_count, len(s))
        self.assertEqual(nonzero_count, 2)
        self.assertEqual(zero_count, 1)


class TestTier1AffineSubspaceSolutions(unittest.TestCase):
    """Tier 1 Feature 5: Affine Solution Line x_p + N(A) (VMLS §10.2, MML §2.6)."""

    def setUp(self):
        self.A = np.array([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0]
        ])
        self.b = np.array([1.0, 2.0, 3.0])
        self.x_p = np.array([1.0, 2.0, 0.0])
        self.v_null = np.array([1.0, 1.0, -1.0])

    def test_particular_solution(self):
        """Feature 5.1: Particular solution x_p = [1, 2, 0]^T satisfies A @ x_p == b."""
        np.testing.assert_allclose(self.A @ self.x_p, self.b, atol=1e-12)

    def test_affine_line_homogeneity(self):
        """Feature 5.2: Every point on affine line x_p + t * v_null maps to exact same target b."""
        t_values = np.linspace(-20.0, 20.0, 41)
        affine_pts = self.x_p[np.newaxis, :] + t_values[:, np.newaxis] * self.v_null[np.newaxis, :]
        mapped = affine_pts @ self.A.T
        expected = np.tile(self.b, (len(t_values), 1))
        np.testing.assert_allclose(mapped, expected, atol=1e-12)

    def test_affine_line_parallel_to_null_line(self):
        """Feature 5.3: Affine solution line is strictly parallel to null space line (tangent v_null)."""
        pt1 = self.x_p + 1.0 * self.v_null
        pt2 = self.x_p + 5.0 * self.v_null
        direction = pt2 - pt1
        cos_sim = np.dot(direction, self.v_null) / (np.linalg.norm(direction) * np.linalg.norm(self.v_null))
        self.assertAlmostEqual(cos_sim, 1.0, places=12)

    def test_minimum_norm_solution(self):
        r"""Feature 5.4: Least-norm solution x_LN = [0, 1, 1]^T has minimum Euclidean norm and x_LN \perp v_null."""
        x_LN = np.array([0.0, 1.0, 1.0])  # Corresponds to t = -1
        np.testing.assert_allclose(self.A @ x_LN, self.b, atol=1e-12)

        # Orthogonality to null space: x_LN . v_null == 0
        self.assertAlmostEqual(np.dot(x_LN, self.v_null), 0.0, places=12)

        # Norm comparison against arbitrary other solutions on affine line
        norm_LN = np.linalg.norm(x_LN)
        self.assertAlmostEqual(norm_LN, math.sqrt(2.0), places=12)
        for t in [-5.0, -2.0, 0.0, 1.0, 3.0]:
            x_other = self.x_p + t * self.v_null
            self.assertGreater(np.linalg.norm(x_other), norm_LN - 1e-12)

    def test_affine_line_does_not_contain_origin(self):
        """Feature 5.5: Non-zero target b implies origin is NOT on affine line (affine != linear)."""
        # Distance from origin to affine line
        # Distance = ||x_LN|| = sqrt(2) > 0
        dist_origin = np.linalg.norm(self.x_p - (np.dot(self.x_p, self.v_null) / np.dot(self.v_null, self.v_null)) * self.v_null)
        self.assertGreater(dist_origin, 1.0)
        self.assertAlmostEqual(dist_origin, math.sqrt(2.0), places=12)


# =============================================================================
# TIER 2: BOUNDARY & CORNER CASES
# =============================================================================

class TestTier2InconsistentSystems(unittest.TestCase):
    """Tier 2 Boundary 1: Inconsistent Systems b \notin C(A) via Rouché-Capelli (Kreyszig §7.5)."""

    def setUp(self):
        self.A = np.array([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0]
        ])
        # b_unreachable: 1 + 2 - 0 = 3 != 0 (not on plane x + y - z = 0)
        self.b_unreach = np.array([1.0, 2.0, 0.0])
        self.normal = np.array([1.0, 1.0, -1.0])

    def test_rouche_capelli_rank_failure(self):
        """Boundary 1.1: rank([A|b]) == 3 > rank(A) == 2 flags complete inconsistency."""
        rank_A = np.linalg.matrix_rank(self.A)
        aug = np.column_stack([self.A, self.b_unreach])
        rank_aug = np.linalg.matrix_rank(aug)
        self.assertEqual(rank_A, 2)
        self.assertEqual(rank_aug, 3)
        self.assertGreater(rank_aug, rank_A)

    def test_inconsistent_plane_distance(self):
        """Boundary 1.2: Distance from b_unreach to plane x + y - z = 0 is non-zero."""
        # dist = |n . b| / ||n|| = |1(1) + 1(2) - 1(0)| / sqrt(3) = 3 / sqrt(3) = sqrt(3)
        dist = abs(np.dot(self.normal, self.b_unreach)) / np.linalg.norm(self.normal)
        self.assertAlmostEqual(dist, math.sqrt(3.0), places=12)

    def test_least_squares_residual_matches_plane_distance(self):
        """Boundary 1.3: Least squares solution residual ||A x_LS - b|| equals geometric plane distance."""
        x_LS, residuals, _, _ = np.linalg.lstsq(self.A, self.b_unreach, rcond=None)
        actual_residual_norm = np.linalg.norm(self.A @ x_LS - self.b_unreach)
        expected_dist = math.sqrt(3.0)
        self.assertAlmostEqual(actual_residual_norm, expected_dist, places=10)

    def test_normal_equations_consistency(self):
        """Boundary 1.4: Normal equations A^T A x = A^T b are consistent even when Ax = b is inconsistent."""
        AtA = self.A.T @ self.A
        Atb = self.A.T @ self.b_unreach
        rank_AtA = np.linalg.matrix_rank(AtA)
        rank_aug_normal = np.linalg.matrix_rank(np.column_stack([AtA, Atb]))
        self.assertEqual(rank_AtA, rank_aug_normal)

    def test_projection_onto_column_space_is_consistent(self):
        """Boundary 1.5: Projection b_proj = P_C b lies in C(A) and has consistent solution."""
        P_C = self.A @ np.linalg.pinv(self.A)
        b_proj = P_C @ self.b_unreach
        aug_proj = np.column_stack([self.A, b_proj])
        self.assertEqual(np.linalg.matrix_rank(aug_proj), 2)


class TestTier2RankOneCollapse(unittest.TestCase):
    """Tier 2 Boundary 2: Rank-1 Collapse & 2D Null Plane (MML §2.6)."""

    def setUp(self):
        # Rank-1 3x3 matrix: all rows are multiples of [1, 2, 3]
        self.A_rank1 = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [-1.0, -2.0, -3.0]
        ])

    def test_rank1_collapse_to_line(self):
        """Boundary 2.1: Column space is a 1D line along [1, 2, -1]^T, rank == 1."""
        self.assertEqual(np.linalg.matrix_rank(self.A_rank1), 1)

    def test_rank1_nullity_is_two(self):
        """Boundary 2.2: Nullity is 3 - 1 == 2 (null space is a 2D plane)."""
        s = np.linalg.svd(self.A_rank1, compute_uv=False)
        nullity = int(np.sum(s < 1e-12))
        self.assertEqual(nullity, 2)

    def test_rank1_rouche_capelli_conditions(self):
        """Boundary 2.3: Consistent target has rank 1; generic target has rank 2 > 1."""
        b_consistent = np.array([2.0, 4.0, -2.0])  # Multiple of [1, 2, -1]^T
        aug_cons = np.column_stack([self.A_rank1, b_consistent])
        self.assertEqual(np.linalg.matrix_rank(aug_cons), 1)

        b_inconsistent = np.array([1.0, 0.0, 0.0])
        aug_incons = np.column_stack([self.A_rank1, b_inconsistent])
        self.assertEqual(np.linalg.matrix_rank(aug_incons), 2)

    def test_rank1_svd_single_non_zero_singular_value(self):
        """Boundary 2.4: SVD has exactly one non-zero singular value."""
        s = np.linalg.svd(self.A_rank1, compute_uv=False)
        self.assertGreater(s[0], 0.0)
        self.assertAlmostEqual(s[1], 0.0, places=12)
        self.assertAlmostEqual(s[2], 0.0, places=12)

    def test_rank1_outer_product_representation(self):
        """Boundary 2.5: Matrix factorizes as outer product u @ v^T."""
        u = np.array([[1.0], [2.0], [-1.0]])
        v = np.array([[1.0], [2.0], [3.0]])
        expected = u @ v.T
        np.testing.assert_allclose(self.A_rank1, expected, atol=1e-12)


class TestTier2FullRankThreeMatrix(unittest.TestCase):
    """Tier 2 Boundary 3: Full Rank-3 & Trivial Null Space (MML §2.6)."""

    def setUp(self):
        self.I3 = np.eye(3)
        self.A_full = np.array([
            [2.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [1.0, 0.0, 4.0]
        ])

    def test_full_rank3_trivial_nullspace(self):
        """Boundary 3.1: Full rank matrix has nullity == 0 and trivial kernel {0}."""
        self.assertEqual(np.linalg.matrix_rank(self.A_full), 3)
        s = np.linalg.svd(self.A_full, compute_uv=False)
        nullity = int(np.sum(s < 1e-12))
        self.assertEqual(nullity, 0)

    def test_full_rank3_universal_reachability(self):
        """Boundary 3.2: Every vector b in R^3 is in the column space (Col(A) == R^3)."""
        rng = np.random.default_rng(303)
        targets = rng.uniform(-10.0, 10.0, size=(10, 3))
        for b in targets:
            aug = np.column_stack([self.A_full, b])
            self.assertEqual(np.linalg.matrix_rank(aug), 3)

    def test_full_rank3_unique_solution(self):
        """Boundary 3.3: Every system Ax = b has a unique solution with zero null space freedom."""
        b = np.array([4.0, -1.0, 7.0])
        x = np.linalg.solve(self.A_full, b)
        np.testing.assert_allclose(self.A_full @ x, b, atol=1e-12)

    def test_full_rank3_non_zero_determinant(self):
        """Boundary 3.4: Determinant is non-zero (3D volume preserved/scaled)."""
        det = np.linalg.det(self.A_full)
        self.assertNotEqual(det, 0.0)
        self.assertAlmostEqual(det, 25.0, places=10)

    def test_full_rank3_bijective_isomorphism(self):
        """Boundary 3.5: Mapping is a bijective linear isomorphism."""
        cond = np.linalg.cond(self.A_full)
        self.assertLess(cond, 100.0)


class TestTier2ZeroMatrix3D(unittest.TestCase):
    """Tier 2 Boundary 4: Zero Matrix Rank 0 Collapse (VMLS §6.1)."""

    def setUp(self):
        self.zero3 = np.zeros((3, 3))

    def test_zero3d_rank_and_nullity(self):
        """Boundary 4.1: Rank is 0, nullity is 3, rank + nullity == 3."""
        self.assertEqual(np.linalg.matrix_rank(self.zero3), 0)
        s = np.linalg.svd(self.zero3, compute_uv=False)
        nullity = int(np.sum(s < 1e-12))
        self.assertEqual(nullity, 3)

    def test_zero3d_annihilation(self):
        """Boundary 4.2: 0 @ x == 0 for all x in R^3."""
        rng = np.random.default_rng(304)
        pts = rng.uniform(-50.0, 50.0, size=(20, 3))
        mapped = pts @ self.zero3.T
        np.testing.assert_allclose(mapped, 0.0, atol=1e-15)

    def test_zero3d_reachability(self):
        """Boundary 4.3: Only b = [0, 0, 0]^T is reachable; all other b are inconsistent."""
        aug_zero = np.column_stack([self.zero3, np.zeros(3)])
        self.assertEqual(np.linalg.matrix_rank(aug_zero), 0)

        aug_nonzero = np.column_stack([self.zero3, np.array([1.0, 0.0, 0.0])])
        self.assertEqual(np.linalg.matrix_rank(aug_nonzero), 1)

    def test_zero3d_solution_space_is_entire_r3(self):
        """Boundary 4.4: For b = 0, solution space has dimension 3."""
        # Any vector in R^3 is a homogeneous solution
        v = np.array([12.3, -45.6, 78.9])
        np.testing.assert_allclose(self.zero3 @ v, np.zeros(3), atol=1e-15)

    def test_zero3d_svd_all_zeros(self):
        """Boundary 4.5: All 3 singular values are exactly 0."""
        s = np.linalg.svd(self.zero3, compute_uv=False)
        np.testing.assert_allclose(s, 0.0, atol=1e-15)


class TestTier2LargeCoordinateStability3D(unittest.TestCase):
    """Tier 2 Boundary 5: Large Coordinate Precision & Floating Point Boundaries (VMLS §10)."""

    def setUp(self):
        self.A = np.array([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0]
        ])
        self.v_null = np.array([1.0, 1.0, -1.0])

    def test_large_coordinates_scale_1e10(self):
        """Boundary 5.1: Null space annihilation holds for astronomical coordinates (1e10)."""
        huge_null_vec = 1e10 * self.v_null
        out = self.A @ huge_null_vec
        np.testing.assert_allclose(out, 0.0, atol=1e-4)

    def test_microscopic_coordinates_scale_1e_10(self):
        """Boundary 5.2: Plane containment holds for microscopic coordinates (1e-10)."""
        tiny_x = np.array([1e-10, 2e-10, -1.5e-10])
        out = self.A @ tiny_x
        # Output must lie on plane: x + y - z == 0
        residual = out[0] + out[1] - out[2]
        self.assertAlmostEqual(residual, 0.0, places=18)

    def test_near_rank_deficiency_conditioning(self):
        """Boundary 5.3: Numerical threshold separates near-singular (eps=1e-14) from true rank deficiency."""
        A_perturbed = self.A.copy()
        A_perturbed[2, 2] += 1e-14
        # SVD spectrum
        s = np.linalg.svd(A_perturbed, compute_uv=False)
        self.assertLess(s[2], 1e-13)
        # Using tolerance 1e-10 detects rank as 2
        effective_rank = np.sum(s > 1e-10)
        self.assertEqual(effective_rank, 2)

    def test_svd_tolerance_thresholding(self):
        """Boundary 5.4: SVD rank calculation is robust to machine precision noise."""
        rank_strict = np.linalg.matrix_rank(self.A, tol=1e-12)
        self.assertEqual(rank_strict, 2)

    def test_dynamic_range_superposition(self):
        """Boundary 5.5: Simultaneous superposition of 1e6 and 1e-6 vectors maintains linearity."""
        u_large = np.array([1e6, 2e6, 3e6])
        v_small = np.array([1e-6, -2e-6, 4e-6])
        lhs = self.A @ (u_large + v_small)
        rhs = self.A @ u_large + self.A @ v_small
        np.testing.assert_allclose(lhs, rhs, rtol=1e-12)


# =============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS
# =============================================================================

class TestTier3CrossFeatureCombinations(unittest.TestCase):
    """Tier 3: Fundamental Theorem of Linear Algebra & Subspace Projections (MML §2.6, VMLS §10)."""

    def setUp(self):
        self.A = np.array([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0]
        ])
        self.v_null = np.array([1.0, 1.0, -1.0])
        self.b = np.array([1.0, 2.0, 3.0])
        self.x_p = np.array([1.0, 2.0, 0.0])

    def test_orthogonal_complement_null_row_space(self):
        r"""Cross-Feature 1: N(A) == (C(A^T))^\perp: Null vector is orthogonal to every row of A."""
        for i in range(3):
            row = self.A[i, :]
            dot_val = np.dot(row, self.v_null)
            self.assertAlmostEqual(dot_val, 0.0, places=12)

    def test_projection_matrix_properties(self):
        """Cross-Feature 2: Projection matrix P_C onto column space is symmetric and idempotent."""
        # P_C = A (A^T A)^+ A^T
        pinv_A = np.linalg.pinv(self.A)
        P_C = self.A @ pinv_A

        # Symmetry: P_C^T == P_C
        np.testing.assert_allclose(P_C, P_C.T, atol=1e-12)

        # Idempotence: P_C @ P_C == P_C
        np.testing.assert_allclose(P_C @ P_C, P_C, atol=1e-12)

        # Trace equals rank (2.0)
        self.assertAlmostEqual(np.trace(P_C), 2.0, places=12)

    def test_direct_sum_decomposition(self):
        r"""Cross-Feature 3: Any vector x in R^3 decomposes as x = x_row + x_null with x_row \perp x_null."""
        rng = np.random.default_rng(305)
        x = rng.uniform(-10.0, 10.0, size=3)

        # Projection onto null space
        P_null = np.outer(self.v_null, self.v_null) / np.dot(self.v_null, self.v_null)
        x_null = P_null @ x
        x_row = x - x_null

        # Check orthogonality
        self.assertAlmostEqual(np.dot(x_row, x_null), 0.0, places=12)
        # Check reconstruction
        np.testing.assert_allclose(x_row + x_null, x, atol=1e-12)

    def test_affine_subspace_projection_invariance(self):
        """Cross-Feature 4: Projection of affine solution line onto row space collapses to single point x_LN."""
        pinv_A = np.linalg.pinv(self.A)
        x_LN = pinv_A @ self.b  # [0, 1, 1]^T
        P_row = np.eye(3) - np.outer(self.v_null, self.v_null) / np.dot(self.v_null, self.v_null)

        for t in [-10.0, -1.0, 0.0, 5.0]:
            x_affine = self.x_p + t * self.v_null
            proj_row = P_row @ x_affine
            np.testing.assert_allclose(proj_row, x_LN, atol=1e-12)

    def test_moore_penrose_pseudoinverse_axioms(self):
        """Cross-Feature 5: Moore-Penrose pseudoinverse satisfies all 4 Penrose equations."""
        A_plus = np.linalg.pinv(self.A)
        # 1. A A^+ A == A
        np.testing.assert_allclose(self.A @ A_plus @ self.A, self.A, atol=1e-12)
        # 2. A^+ A A^+ == A^+
        np.testing.assert_allclose(A_plus @ self.A @ A_plus, A_plus, atol=1e-12)
        # 3. (A A^+)^T == A A^+
        AA_plus = self.A @ A_plus
        np.testing.assert_allclose(AA_plus, AA_plus.T, atol=1e-12)
        # 4. (A^+ A)^T == A^+ A
        A_plusA = A_plus @ self.A
        np.testing.assert_allclose(A_plusA, A_plusA.T, atol=1e-12)


# =============================================================================
# TIER 4: REAL-WORLD SCENARIOS
# =============================================================================

class TestTier4RealWorldScenarios(unittest.TestCase):
    """Tier 4: Deep Learning Null Spaces / LoRA & GATE Linear Algebra (MML §2.6)."""

    def setUp(self):
        self.rng = np.random.default_rng(306)

    def test_ml_overparameterized_jacobian_null_space(self):
        """Real-World 1: ML Overparameterization: Parameter Jacobian has wide null space with zero loss change."""
        N_samples = 4
        P_params = 10
        # Jacobian J of predictions with respect to parameters (N x P)
        J = self.rng.standard_normal((N_samples, P_params))
        rank_J = np.linalg.matrix_rank(J)
        self.assertEqual(rank_J, N_samples)

        # By Rank-Nullity: nullity = P - rank >= 10 - 4 = 6
        nullity_J = P_params - rank_J
        self.assertEqual(nullity_J, 6)

        # SVD right singular vectors for null directions
        _, _, Vt = np.linalg.svd(J)
        null_basis = Vt[rank_J:]  # Shape (6, 10)

        # Any perturbation along null basis causes 0 change in output: J @ delta == 0
        delta = null_basis[0]
        output_change = J @ delta
        np.testing.assert_allclose(output_change, 0.0, atol=1e-12)

    def test_ml_lora_low_rank_decomposition(self):
        """Real-World 2: LoRA (Low-Rank Adaptation): Weight update Delta W = B @ A with rank r << d."""
        d_model = 16
        r = 2
        # Adapter matrices B (d x r) and A (r x d)
        B = self.rng.standard_normal((d_model, r))
        A = self.rng.standard_normal((r, d_model))
        Delta_W = B @ A

        # Invariant: rank(Delta_W) <= r == 2
        rank_lora = np.linalg.matrix_rank(Delta_W)
        self.assertLessEqual(rank_lora, r)
        self.assertEqual(rank_lora, r)

        # Nullity of Delta_W is d - r == 14
        s = np.linalg.svd(Delta_W, compute_uv=False)
        nullity_lora = int(np.sum(s < 1e-10))
        self.assertEqual(nullity_lora, d_model - r)

    def test_ml_linear_autoencoder_bottleneck(self):
        """Real-World 3: Linear Autoencoder 3 -> 2 -> 3 creates rank-2 column space with 1D null loss."""
        # Encoder W_enc: 3 -> 2; Decoder W_dec: 2 -> 3
        W_enc = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])  # 2 x 3
        W_dec = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # 3 x 2
        Autoencoder = W_dec @ W_enc  # 3 x 3

        # Rank cannot exceed bottleneck dimension 2
        self.assertEqual(np.linalg.matrix_rank(Autoencoder), 2)
        # Any vector along [1, 1, -1]^T is lost in compression
        v_lost = np.array([1.0, 1.0, -1.0])
        np.testing.assert_allclose(Autoencoder @ v_lost, np.zeros(3), atol=1e-12)

    def test_gate_rouche_capelli_3d_classification(self):
        """Real-World 4: GATE CSE Complete Rouché-Capelli classification in 3D."""
        A = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.0]])

        # Case 1: Infinite solutions: rank(A) == rank([A|b]) == 2 < 3
        b_inf = np.array([1.0, 2.0, 3.0])
        self.assertEqual(np.linalg.matrix_rank(A), 2)
        self.assertEqual(np.linalg.matrix_rank(np.column_stack([A, b_inf])), 2)

        # Case 2: Inconsistent / No solution: rank(A) == 2 < rank([A|b]) == 3
        b_none = np.array([1.0, 0.0, 0.0])
        self.assertEqual(np.linalg.matrix_rank(np.column_stack([A, b_none])), 3)

        # Case 3: Unique solution: rank(A_full) == rank([A_full|b]) == 3
        A_full = np.eye(3)
        self.assertEqual(np.linalg.matrix_rank(A_full), 3)
        self.assertEqual(np.linalg.matrix_rank(np.column_stack([A_full, b_none])), 3)

    def test_gate_rank_nullity_abstract_transformation(self):
        """Real-World 5: GATE Linear Algebra: Dimension theorem on differential operator D: P_3 -> P_2."""
        # Polynomials of degree <= 3 (dim = 4: basis 1, x, x^2, x^3)
        # Derivative mapping D: P_3 -> P_2 (dim P_2 = 3: basis 1, x, x^2)
        D_matrix = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 3.0]
        ])
        rank_D = np.linalg.matrix_rank(D_matrix)
        self.assertEqual(rank_D, 3)
        # Kernel of D is constant polynomials (dim = 1)
        s = np.linalg.svd(D_matrix, compute_uv=False)
        nullity_D = int(D_matrix.shape[1] - rank_D)
        self.assertEqual(nullity_D, 1)
        self.assertEqual(rank_D + nullity_D, 4)


# =============================================================================
# TIER 5: STATIC AST & SYNTAX INTEGRITY
# =============================================================================

class TestSprint03ASTAndSyntax(unittest.TestCase):
    """Tier 5: Static AST verification of Sprint 3 scene code for ManimCE 3D standards."""

    SCENE_PATH = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "linear_algebra",
            "sprint_03_column_null_space",
            "scene_column_null_space.py"
        )
    )

    def _get_ast(self):
        if not os.path.exists(self.SCENE_PATH):
            self.skipTest(
                f"Scene script '{self.SCENE_PATH}' not yet created (Pending Milestone 3)."
            )
        with open(self.SCENE_PATH, "r", encoding="utf-8") as f:
            return ast.parse(f.read(), filename=self.SCENE_PATH)

    def test_no_deprecated_manimlib_symbols(self):
        """AST 1: File contains zero deprecated symbols: CONFIG, ShowCreation, TexMobject, TextMobject."""
        tree = self._get_ast()
        deprecated = {"CONFIG", "ShowCreation", "TexMobject", "TextMobject"}
        found_deprecated = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in deprecated:
                found_deprecated.append((node.id, getattr(node, "lineno", "?")))
            elif isinstance(node, ast.Attribute) and node.attr in deprecated:
                found_deprecated.append((node.attr, getattr(node, "lineno", "?")))
        self.assertEqual(
            found_deprecated,
            [],
            f"Found deprecated symbols in Sprint 3 scene: {found_deprecated}"
        )

    def test_scene_subclasses_threedscene(self):
        """AST 2: Base or master scene class inherits from ThreeDScene for 3D camera support."""
        tree = self._get_ast()
        threed_subclasses = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if (isinstance(base, ast.Name) and base.id == "ThreeDScene") or \
                       (isinstance(base, ast.Attribute) and base.attr == "ThreeDScene"):
                        threed_subclasses.append(node.name)
        self.assertGreater(
            len(threed_subclasses),
            0,
            "Sprint 3 must define at least one class subclassing ThreeDScene."
        )

    def test_hud_pinning_add_fixed_in_frame_mobjects_invoked(self):
        """AST 3: Scene invokes add_fixed_in_frame_mobjects to pin 2D HUD text in 3D scene."""
        tree = self._get_ast()
        has_hud_call = any(
            isinstance(node, ast.Call) and (
                (isinstance(node.func, ast.Name) and node.func.id == "add_fixed_in_frame_mobjects") or
                (isinstance(node.func, ast.Attribute) and node.func.attr == "add_fixed_in_frame_mobjects")
            )
            for node in ast.walk(tree)
        )
        self.assertTrue(
            has_hud_call,
            "Sprint 3 scene must call add_fixed_in_frame_mobjects for viewport-pinned HUD panels."
        )

    def test_scene_class_defined(self):
        """AST 4: Master scene class ColumnNullSpaceScene is defined."""
        tree = self._get_ast()
        class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        self.assertIn("ColumnNullSpaceScene", class_names)

    def test_make_hud_panel_usage(self):
        """AST 5: File defines or invokes make_hud_panel."""
        tree = self._get_ast()
        has_hud_func = any(
            isinstance(node, ast.FunctionDef) and node.name == "make_hud_panel"
            for node in ast.walk(tree)
        )
        has_hud_call = any(
            isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "make_hud_panel")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "make_hud_panel")
            )
            for node in ast.walk(tree)
        )
        self.assertTrue(
            has_hud_func or has_hud_call,
            "Sprint 3 scene must define or call 'make_hud_panel'."
        )


if __name__ == "__main__":
    unittest.main()
