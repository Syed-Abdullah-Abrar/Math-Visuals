#!/usr/bin/env python3
"""
Sprint 1 E2E Invariant Test Suite: Vectors, Linear Combinations & Span
====================================================================
Curriculum: Linear Algebra (Geometric Foundations & AI/Systems Invariants)
References:
    - Mathematics for Machine Learning (MML Ch. 2.4-2.6)
    - Applied Linear Algebra (VMLS Ch. 1 & 5)
    - Immersive Linear Algebra (ILA Ch. 1-2)

Architecture:
    Comprehensive 4-Tier Test Suite implemented with unittest and NumPy:
    - Tier 1: Feature Coverage (>=5 test cases per feature for vector addition,
      scalar scaling, linear combinations, 2D basis span, origin anchor invariant).
    - Tier 2: Boundary & Corner Cases (>=5 test cases per category for collinear collapse,
      0D zero vector, negative scalars, orthogonal basis, large coordinates & precision).
    - Tier 3: Cross-Feature Combinations (pairwise feature interactions).
    - Tier 4: Real-World Scenarios (Transformer attention convex combination,
      Word2Vec linear analogy vector arithmetic, GATE CSE linear systems consistency).

Execution:
    python -m unittest tests/test_sprint01_invariants.py
"""

from __future__ import annotations

import math
import unittest
import numpy as np


# =============================================================================
# TIER 1: FEATURE COVERAGE
# =============================================================================

class TestTier1VectorAddition(unittest.TestCase):
    """Tier 1 Feature 1: Vector Addition Axioms & Geometric Invariants (MML §2.4, VMLS §1.1)."""

    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.v1 = np.array([2.0, 1.0])
        self.v2 = np.array([-1.0, 2.0])

    def test_addition_commutativity(self):
        """Feature 1.1: Vector addition is commutative (u + v == v + u)."""
        u = self.v1
        v = self.v2
        np.testing.assert_allclose(u + v, v + u, atol=1e-12)
        # Randomized vector batch commutativity
        batch_u = self.rng.uniform(-100, 100, size=(50, 2))
        batch_v = self.rng.uniform(-100, 100, size=(50, 2))
        np.testing.assert_allclose(batch_u + batch_v, batch_v + batch_u, atol=1e-12)

    def test_addition_associativity(self):
        """Feature 1.2: Vector addition is associative ((u + v) + w == u + (v + w))."""
        u = self.v1
        v = self.v2
        w = np.array([3.0, -4.0])
        lhs = (u + v) + w
        rhs = u + (v + w)
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)

    def test_additive_identity(self):
        """Feature 1.3: Zero vector acts as additive identity (v + 0 == v)."""
        zero = np.zeros(2)
        np.testing.assert_allclose(self.v1 + zero, self.v1, atol=1e-12)
        np.testing.assert_allclose(self.v2 + zero, self.v2, atol=1e-12)

    def test_additive_inverse(self):
        """Feature 1.4: Every vector has an additive inverse (v + (-v) == 0)."""
        zero = np.zeros(2)
        np.testing.assert_allclose(self.v1 + (-self.v1), zero, atol=1e-12)
        np.testing.assert_allclose(self.v2 + (-self.v2), zero, atol=1e-12)

    def test_parallelogram_law(self):
        """Feature 1.5: Parallelogram law holds (||u + v||^2 + ||u - v||^2 == 2*(||u||^2 + ||v||^2))."""
        u = self.v1
        v = self.v2
        sum_norm_sq = np.linalg.norm(u + v) ** 2
        diff_norm_sq = np.linalg.norm(u - v) ** 2
        indep_sum = 2.0 * (np.linalg.norm(u) ** 2 + np.linalg.norm(v) ** 2)
        self.assertAlmostEqual(sum_norm_sq + diff_norm_sq, indep_sum, places=10)

    def test_triangle_inequality(self):
        """Feature 1.6: Triangle inequality holds (||u + v|| <= ||u|| + ||v||)."""
        u = self.v1
        v = self.v2
        norm_sum = np.linalg.norm(u + v)
        sum_norms = np.linalg.norm(u) + np.linalg.norm(v)
        self.assertLessEqual(norm_sum, sum_norms + 1e-12)


class TestTier1ScalarScaling(unittest.TestCase):
    """Tier 1 Feature 2: Scalar Scaling Axioms & Properties (MML §2.4, VMLS §1.2)."""

    def setUp(self):
        self.rng = np.random.default_rng(101)
        self.v = np.array([2.0, 1.0])

    def test_distributivity_over_scalar_addition(self):
        """Feature 2.1: (c + d) * v == c * v + d * v."""
        c, d = 3.5, -1.2
        lhs = (c + d) * self.v
        rhs = c * self.v + d * self.v
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)

    def test_distributivity_over_vector_addition(self):
        """Feature 2.2: c * (u + v) == c * u + c * v."""
        c = 2.5
        u = self.v
        w = np.array([-1.0, 2.0])
        lhs = c * (u + w)
        rhs = c * u + c * w
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)

    def test_multiplicative_associativity(self):
        """Feature 2.3: c * (d * v) == (c * d) * v."""
        c, d = 4.0, -0.5
        lhs = c * (d * self.v)
        rhs = (c * d) * self.v
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)

    def test_multiplicative_identity_and_annihilation(self):
        """Feature 2.4: 1.0 * v == v and 0.0 * v == 0."""
        np.testing.assert_allclose(1.0 * self.v, self.v, atol=1e-12)
        np.testing.assert_allclose(0.0 * self.v, np.zeros(2), atol=1e-12)

    def test_norm_homogeneity(self):
        """Feature 2.5: ||c * v|| == |c| * ||v|| (absolute homogeneity)."""
        scalars = [-5.0, -1.5, 0.0, 2.3, 10.0]
        base_norm = np.linalg.norm(self.v)
        for c in scalars:
            scaled_norm = np.linalg.norm(c * self.v)
            expected_norm = abs(c) * base_norm
            self.assertAlmostEqual(scaled_norm, expected_norm, places=10)


class TestTier1LinearCombinations(unittest.TestCase):
    """Tier 1 Feature 3: Linear Combinations & Geometric Synthesis (MML §2.4, VMLS §1.3)."""

    def setUp(self):
        self.v1 = np.array([2.0, 1.0])
        self.v2 = np.array([-1.0, 2.0])
        self.b_target = np.array([3.0, 4.0])

    def test_act1_puzzle_solution(self):
        """Feature 3.1: Act 1 solution 2*v1 + 1*v2 == [3, 4] exactly."""
        c1, c2 = 2.0, 1.0
        result = c1 * self.v1 + c2 * self.v2
        np.testing.assert_allclose(result, self.b_target, atol=1e-12)

    def test_naive_shortfall_attempt(self):
        """Feature 3.2: Act 1 shortfall attempt 1*v1 + 1*v2 = [1, 3] != [3, 4]."""
        shortfall = 1.0 * self.v1 + 1.0 * self.v2
        np.testing.assert_allclose(shortfall, np.array([1.0, 3.0]), atol=1e-12)
        self.assertFalse(np.allclose(shortfall, self.b_target))

    def test_matrix_vector_linear_combination_equivalence(self):
        """Feature 3.3: A @ c == c1*v1 + c2*v2 for matrix A = [v1, v2]."""
        A = np.column_stack([self.v1, self.v2])
        c = np.array([2.0, 1.0])
        lhs = A @ c
        rhs = c[0] * self.v1 + c[1] * self.v2
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)

    def test_convex_combination_boundedness(self):
        """Feature 3.4: Convex combination alpha*v1 + (1-alpha)*v2 lies on connecting line segment."""
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            p = alpha * self.v1 + (1.0 - alpha) * self.v2
            dist_v1_p = np.linalg.norm(p - self.v1)
            dist_p_v2 = np.linalg.norm(self.v2 - p)
            segment_length = np.linalg.norm(self.v2 - self.v1)
            self.assertAlmostEqual(dist_v1_p + dist_p_v2, segment_length, places=10)

    def test_conic_combination_positivity(self):
        """Feature 3.5: Non-negative coefficients produce conic combination in cone(v1, v2)."""
        c1, c2 = 3.0, 4.0
        v_cone = c1 * self.v1 + c2 * self.v2
        A = np.column_stack([self.v1, self.v2])
        recovered_c = np.linalg.solve(A, v_cone)
        self.assertGreaterEqual(recovered_c[0], 0.0)
        self.assertGreaterEqual(recovered_c[1], 0.0)


class TestTier1BasisSpan2D(unittest.TestCase):
    """Tier 1 Feature 4: 2D Basis Span & Bijectivity (MML §2.6, ILA §2.3)."""

    def setUp(self):
        self.v1 = np.array([2.0, 1.0])
        self.v2 = np.array([-1.0, 2.0])
        self.A = np.column_stack([self.v1, self.v2])
        self.rng = np.random.default_rng(202)

    def test_basis_rank_two(self):
        """Feature 4.1: Basis matrix A = [v1, v2] has matrix_rank == 2."""
        self.assertEqual(np.linalg.matrix_rank(self.A), 2)

    def test_basis_determinant(self):
        """Feature 4.2: Determinant det(A) == 5.0 != 0."""
        det = np.linalg.det(self.A)
        self.assertAlmostEqual(det, 5.0, places=10)

    def test_linear_independence_kernel(self):
        """Feature 4.3: c1*v1 + c2*v2 == 0 has only trivial solution c1=c2=0."""
        zero = np.zeros(2)
        c = np.linalg.solve(self.A, zero)
        np.testing.assert_allclose(c, [0.0, 0.0], atol=1e-12)

    def test_universal_reachability_span(self):
        """Feature 4.4: span(v1, v2) = R^2 (any target point b in R^2 is reachable)."""
        random_targets = self.rng.uniform(-1000.0, 1000.0, size=(100, 2))
        for b in random_targets:
            c = np.linalg.solve(self.A, b)
            reconstructed_b = self.A @ c
            np.testing.assert_allclose(reconstructed_b, b, atol=1e-10)

    def test_bijective_invertibility(self):
        """Feature 4.5: Mapping x -> A x is bijective with well-defined inverse A^{-1}."""
        A_inv = np.linalg.inv(self.A)
        identity = np.eye(2)
        np.testing.assert_allclose(self.A @ A_inv, identity, atol=1e-12)
        np.testing.assert_allclose(A_inv @ self.A, identity, atol=1e-12)


class TestTier1OriginAnchorInvariant(unittest.TestCase):
    """Tier 1 Feature 5: Origin Anchor Invariant (0 in span(S)) (MML §2.4, VMLS §5.1)."""

    def setUp(self):
        self.v1 = np.array([2.0, 1.0])
        self.v2 = np.array([-1.0, 2.0])

    def test_origin_reachability_by_zero_coefficients(self):
        """Feature 5.1: Origin is reached by c1=0, c2=0."""
        origin = 0.0 * self.v1 + 0.0 * self.v2
        np.testing.assert_allclose(origin, np.zeros(2), atol=1e-12)

    def test_subspace_origin_closure(self):
        """Feature 5.2: Every linear subspace must contain the origin vector."""
        # For arbitrary set of vectors, 0 is always in span
        v_any = np.array([7.3, -2.1])
        origin = 0.0 * v_any
        np.testing.assert_allclose(origin, [0.0, 0.0], atol=1e-12)

    def test_origin_projection_invariant(self):
        """Feature 5.3: Orthogonal projection of origin onto any subspace is origin."""
        # Project 0 onto span(v1)
        zero = np.zeros(2)
        proj = (np.dot(zero, self.v1) / np.dot(self.v1, self.v1)) * self.v1
        np.testing.assert_allclose(proj, zero, atol=1e-12)

    def test_origin_norm_minimizer(self):
        """Feature 5.4: Origin has strictly minimum norm 0 in any subspace."""
        zero = np.zeros(2)
        self.assertEqual(np.linalg.norm(zero), 0.0)
        # Any non-zero linear combination has strictly positive norm
        v_combo = 0.1 * self.v1 - 0.2 * self.v2
        self.assertGreater(np.linalg.norm(v_combo), 0.0)

    def test_subspace_intersection_contains_origin(self):
        """Feature 5.5: Intersection of any two linear subspaces contains {0}."""
        # Line 1: span(v1), Line 2: span(v2). Since v1, v2 independent, intersection is {0}.
        zero = np.zeros(2)
        # 0 is in span(v1) and 0 is in span(v2)
        self.assertTrue(np.allclose(0.0 * self.v1, zero))
        self.assertTrue(np.allclose(0.0 * self.v2, zero))


# =============================================================================
# TIER 2: BOUNDARY & CORNER CASES
# =============================================================================

class TestTier2CollinearCollapse(unittest.TestCase):
    """Tier 2 Boundary 1: Collinear Collapse & Rank Deficiency (MML §2.4, VMLS §5.2)."""

    def setUp(self):
        self.v1 = np.array([2.0, 1.0])
        self.v2_collinear = 2.0 * self.v1  # [4, 2]
        self.A_collinear = np.column_stack([self.v1, self.v2_collinear])
        self.b_unreachable = np.array([1.0, 4.0])

    def test_collinear_rank_deficiency(self):
        """Boundary 1.1: Collinear basis matrix has rank 1 (dimension collapse)."""
        self.assertEqual(np.linalg.matrix_rank(self.A_collinear), 1)

    def test_collinear_zero_determinant(self):
        """Boundary 1.2: Collinear basis has det(A) == 0.0."""
        det = np.linalg.det(self.A_collinear)
        self.assertAlmostEqual(det, 0.0, places=10)

    def test_collinear_subspace_1d_constraint(self):
        """Boundary 1.3: Reachable vectors satisfy y = 0.5 * x."""
        scalars = [-3.0, -1.0, 0.0, 1.5, 4.0]
        for c1 in scalars:
            for c2 in scalars:
                v = c1 * self.v1 + c2 * self.v2_collinear
                # x = 2*c1 + 4*c2, y = 1*c1 + 2*c2 => y == 0.5 * x
                self.assertAlmostEqual(v[1], 0.5 * v[0], places=10)

    def test_rouche_capelli_inconsistency_unreachable_point(self):
        """Boundary 1.4: Rouché-Capelli rank failure: rank(A) = 1 != rank([A|b]) = 2."""
        aug_matrix = np.column_stack([self.A_collinear, self.b_unreachable])
        rank_A = np.linalg.matrix_rank(self.A_collinear)
        rank_aug = np.linalg.matrix_rank(aug_matrix)
        self.assertEqual(rank_A, 1)
        self.assertEqual(rank_aug, 2)
        # System is inconsistent (0 solutions)
        self.assertNotEqual(rank_A, rank_aug)

    def test_collinear_infinite_solutions_for_points_on_line(self):
        """Boundary 1.5: Points on line have rank(A) == rank([A|b]) == 1 < n=2 (infinitely many solutions)."""
        b_on_line = np.array([6.0, 3.0])  # 3 * v1
        aug_matrix = np.column_stack([self.A_collinear, b_on_line])
        rank_A = np.linalg.matrix_rank(self.A_collinear)
        rank_aug = np.linalg.matrix_rank(aug_matrix)
        self.assertEqual(rank_A, 1)
        self.assertEqual(rank_aug, 1)
        # Multiple distinct coefficient pairs give the exact same point
        c_sol1 = (3.0, 0.0)      # 3*v1 + 0*v2 = [6, 3]
        c_sol2 = (1.0, 1.0)      # 1*v1 + 1*v2 = [2, 1] + [4, 2] = [6, 3]
        c_sol3 = (-1.0, 2.0)     # -1*v1 + 2*v2 = [-2, -1] + [8, 4] = [6, 3]
        for c1, c2 in [c_sol1, c_sol2, c_sol3]:
            res = c1 * self.v1 + c2 * self.v2_collinear
            np.testing.assert_allclose(res, b_on_line, atol=1e-12)


class TestTier2ZeroVectorSubspace(unittest.TestCase):
    """Tier 2 Boundary 2: 0D Zero Vector Trivial Subspace (MML §2.4, VMLS §5.1)."""

    def setUp(self):
        self.v_zero = np.zeros(2)
        self.A_zero = np.zeros((2, 2))

    def test_zero_matrix_rank_zero(self):
        """Boundary 2.1: Matrix of zero vectors has rank 0."""
        self.assertEqual(np.linalg.matrix_rank(self.A_zero), 0)

    def test_zero_vector_span_is_origin_only(self):
        """Boundary 2.2: span({0}) contains only the origin {0}."""
        for c in [-10.0, -1.0, 0.0, 2.5, 100.0]:
            p = c * self.v_zero
            np.testing.assert_allclose(p, np.zeros(2), atol=1e-12)

    def test_zero_vector_unreachability_of_nonzero_target(self):
        """Boundary 2.3: Any non-zero target b has rank([0|b]) == 1 > rank(0) == 0."""
        b = np.array([1.0, 2.0])
        aug = np.column_stack([self.A_zero, b])
        self.assertEqual(np.linalg.matrix_rank(self.A_zero), 0)
        self.assertEqual(np.linalg.matrix_rank(aug), 1)

    def test_trivial_subspace_dimension(self):
        """Boundary 2.4: Dimension of trivial subspace {0} is 0."""
        # Rank of zero subspace basis is 0
        dim = np.linalg.matrix_rank(np.column_stack([self.v_zero]))
        self.assertEqual(dim, 0)

    def test_single_zero_vector_with_non_zero_vector(self):
        """Boundary 2.5: System [v1, 0] has rank 1 and collapses along second axis."""
        v1 = np.array([2.0, 1.0])
        A = np.column_stack([v1, self.v_zero])
        self.assertEqual(np.linalg.matrix_rank(A), 1)
        self.assertAlmostEqual(np.linalg.det(A), 0.0, places=10)


class TestTier2NegativeScalars(unittest.TestCase):
    """Tier 2 Boundary 3: Negative Scalars & Direction Inversion (ILA §1.2, VMLS §1.2)."""

    def setUp(self):
        self.v1 = np.array([2.0, 1.0])
        self.v2 = np.array([-1.0, 2.0])
        self.A = np.column_stack([self.v1, self.v2])

    def test_direction_inversion_pi_radians(self):
        """Boundary 3.1: v and -v have relative angle exactly pi radians."""
        dot_prod = np.dot(self.v1, -self.v1)
        cos_theta = dot_prod / (np.linalg.norm(self.v1) * np.linalg.norm(-self.v1))
        self.assertAlmostEqual(cos_theta, -1.0, places=10)
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        # Note: arccos derivative diverges near -1 (d/dx ~ (1-x^2)^(-1/2)), producing 1e-8 rad uncertainty from float eps
        self.assertAlmostEqual(angle, math.pi, places=7)

    def test_span_reaches_quadrant_3(self):
        """Boundary 3.2: Negative coefficients reach Quadrant III (x < 0, y < 0)."""
        target_q3 = np.array([-3.0, -4.0])
        c = np.linalg.solve(self.A, target_q3)
        np.testing.assert_allclose(c, [-2.0, -1.0], atol=1e-12)
        self.assertLess(c[0], 0.0)
        self.assertLess(c[1], 0.0)

    def test_span_reaches_all_four_quadrants(self):
        """Boundary 3.3: Signs of (c1, c2) can reach points across all 4 Cartesian quadrants."""
        quadrant_targets = {
            "Q1": np.array([3.0, 4.0]),     # c = [2, 1]
            "Q2": np.array([-4.0, 3.0]),    # c = [-1, 2]
            "Q3": np.array([-3.0, -4.0]),   # c = [-2, -1]
            "Q4": np.array([4.0, -3.0]),    # c = [1, -2]
        }
        for q, target in quadrant_targets.items():
            c = np.linalg.solve(self.A, target)
            reconstructed = self.A @ c
            np.testing.assert_allclose(reconstructed, target, atol=1e-12)

    def test_norm_scaling_with_negative_scalar(self):
        """Boundary 3.4: ||-c * v|| == c * ||v|| for negative scalars."""
        for c in [-0.1, -1.0, -7.5, -100.0]:
            scaled = c * self.v1
            self.assertAlmostEqual(np.linalg.norm(scaled), abs(c) * np.linalg.norm(self.v1), places=10)

    def test_negative_scalar_continuity_through_origin(self):
        """Boundary 3.5: Scaling along c in [-1, 1] passes through 0 continuously."""
        steps = np.linspace(-1.0, 1.0, 21)
        norms = [np.linalg.norm(s * self.v1) for s in steps]
        min_idx = np.argmin(norms)
        self.assertEqual(steps[min_idx], 0.0)
        self.assertAlmostEqual(norms[min_idx], 0.0, places=12)


class TestTier2OrthogonalBasis(unittest.TestCase):
    """Tier 2 Boundary 4: Orthogonal Basis Decoupling & Conditioning (MML §2.6, VMLS §5.3)."""

    def setUp(self):
        self.v1 = np.array([2.0, 1.0])
        self.v2 = np.array([-1.0, 2.0])
        self.A = np.column_stack([self.v1, self.v2])

    def test_strictly_orthogonal_inner_product(self):
        """Boundary 4.1: v1 and v2 are strictly orthogonal (<v1, v2> == 0)."""
        inner = np.dot(self.v1, self.v2)
        self.assertAlmostEqual(inner, 0.0, places=12)

    def test_condition_number_one(self):
        """Boundary 4.2: Scaled orthogonal basis has condition number == 1.0."""
        # Both v1 and v2 have norm sqrt(5), so singular values are identical
        cond = np.linalg.cond(self.A)
        self.assertAlmostEqual(cond, 1.0, places=10)

    def test_pythagorean_decoupling(self):
        """Boundary 4.3: ||c1*v1 + c2*v2||^2 == (c1*||v1||)^2 + (c2*||v2||)^2."""
        c1, c2 = 3.0, -4.0
        v_combo = c1 * self.v1 + c2 * self.v2
        lhs = np.linalg.norm(v_combo) ** 2
        rhs = (c1 * np.linalg.norm(self.v1)) ** 2 + (c2 * np.linalg.norm(self.v2)) ** 2
        self.assertAlmostEqual(lhs, rhs, places=10)

    def test_projection_coordinate_recovery(self):
        """Boundary 4.4: Orthogonal coordinates can be recovered via scalar projections c_i = (b . v_i) / ||v_i||^2."""
        b = np.array([3.0, 4.0])
        c1_proj = np.dot(b, self.v1) / np.dot(self.v1, self.v1)
        c2_proj = np.dot(b, self.v2) / np.dot(self.v2, self.v2)
        c_solve = np.linalg.solve(self.A, b)
        self.assertAlmostEqual(c1_proj, c_solve[0], places=10)
        self.assertAlmostEqual(c2_proj, c_solve[1], places=10)
        self.assertAlmostEqual(c1_proj, 2.0, places=10)
        self.assertAlmostEqual(c2_proj, 1.0, places=10)

    def test_gram_schmidt_null_correction(self):
        """Boundary 4.5: Gram-Schmidt projection of v2 onto v1 produces zero correction."""
        proj_v2_on_v1 = (np.dot(self.v2, self.v1) / np.dot(self.v1, self.v1)) * self.v1
        np.testing.assert_allclose(proj_v2_on_v1, np.zeros(2), atol=1e-12)
        v2_ortho = self.v2 - proj_v2_on_v1
        np.testing.assert_allclose(v2_ortho, self.v2, atol=1e-12)


class TestTier2LargeCoordinatesPrecision(unittest.TestCase):
    """Tier 2 Boundary 5: High Dynamic Range & Numerical Precision (MML §2.4, Python/ML rules)."""

    def setUp(self):
        self.v1 = np.array([2.0, 1.0])
        self.v2 = np.array([-1.0, 2.0])
        self.A = np.column_stack([self.v1, self.v2])

    def test_large_coordinates_stability(self):
        """Boundary 5.1: Solves accurately for large target b = 1e12 * [3, 4]."""
        scale = 1e12
        b_large = scale * np.array([3.0, 4.0])
        c = np.linalg.solve(self.A, b_large)
        expected_c = scale * np.array([2.0, 1.0])
        np.testing.assert_allclose(c, expected_c, rtol=1e-12)

    def test_microscopic_coordinates_stability(self):
        """Boundary 5.2: Solves accurately for tiny target b = 1e-12 * [3, 4]."""
        scale = 1e-12
        b_small = scale * np.array([3.0, 4.0])
        c = np.linalg.solve(self.A, b_small)
        expected_c = scale * np.array([2.0, 1.0])
        np.testing.assert_allclose(c, expected_c, rtol=1e-12)

    def test_ill_conditioned_almost_collinear_detection(self):
        """Boundary 5.3: Ill-conditioned basis (angle 1e-6 rad) exhibits condition number blowup."""
        theta = 1e-6
        v_perturbed = np.array([2.0 * np.cos(theta) - 1.0 * np.sin(theta),
                                2.0 * np.sin(theta) + 1.0 * np.cos(theta)])
        # Almost parallel to v1
        v1_norm = self.v1 / np.linalg.norm(self.v1)
        A_ill = np.column_stack([v1_norm, v_perturbed / np.linalg.norm(v_perturbed)])
        cond = np.linalg.cond(A_ill)
        self.assertGreater(cond, 1e5)

    def test_no_nan_or_inf_in_broad_sweep(self):
        """Boundary 5.4: Continuous grid sweep produces zero NaN or Inf values."""
        grid = np.linspace(-1e6, 1e6, 50)
        c1, c2 = np.meshgrid(grid, grid)
        vectors = c1.ravel()[:, None] * self.v1 + c2.ravel()[:, None] * self.v2
        self.assertFalse(np.isnan(vectors).any())
        self.assertFalse(np.isinf(vectors).any())

    def test_high_dynamic_range_superposition(self):
        """Boundary 5.5: Superposition of 1e8 * v1 and 1e-8 * v2 maintains component accuracy."""
        c1, c2 = 1e8, 1e-8
        b = c1 * self.v1 + c2 * self.v2
        # Verify c1 can be reconstructed via projection along v1
        recovered_c1 = np.dot(b, self.v1) / np.dot(self.v1, self.v1)
        self.assertAlmostEqual(recovered_c1, c1, places=3)


# =============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS
# =============================================================================

class TestTier3CrossFeatureCombinations(unittest.TestCase):
    """Tier 3: Pairwise Cross-Feature Interactions & Mathematical Dualities."""

    def setUp(self):
        self.v1 = np.array([2.0, 1.0])
        self.v2 = np.array([-1.0, 2.0])
        self.A = np.column_stack([self.v1, self.v2])
        self.rng = np.random.default_rng(303)

    def test_scaling_and_addition_distributivity_under_orthogonal_basis(self):
        """Cross 1: Scalar scaling distributive property over linear combinations in orthogonal basis."""
        alpha, beta = 2.5, -3.2
        c1, c2 = 1.2, -0.8
        u = c1 * self.v1 + c2 * self.v2
        w = -0.5 * self.v1 + 2.1 * self.v2
        lhs = alpha * (u + w)
        rhs = alpha * u + alpha * w
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)

    def test_collinear_collapse_under_linear_transformation(self):
        """Cross 2: Linear transformation of collinear system preserves rank deficiency."""
        A_collinear = np.column_stack([self.v1, 2.0 * self.v1])
        T = np.array([[3.0, 1.0], [-1.0, 2.0]])  # Non-singular transformation
        transformed_system = T @ A_collinear
        self.assertEqual(np.linalg.matrix_rank(transformed_system), 1)
        self.assertAlmostEqual(np.linalg.det(transformed_system), 0.0, places=10)

    def test_origin_anchor_invariance_under_affine_shift(self):
        """Cross 3: Affine shift (span(S) + d) contains origin iff d is in span(S)."""
        # For full 2D span, any d in R^2 means origin is always reached with -d
        d = np.array([5.0, -7.0])
        c_to_reach_neg_d = np.linalg.solve(self.A, -d)
        reached_origin = (self.A @ c_to_reach_neg_d) + d
        np.testing.assert_allclose(reached_origin, np.zeros(2), atol=1e-12)

    def test_basis_rotation_preserves_span_and_quadrant_coverage(self):
        """Cross 4: SO(2) rotation of basis preserves full rank 2 and quadrant reachability."""
        theta = math.pi / 4.0  # 45 deg
        R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        rotated_A = R @ self.A
        self.assertEqual(np.linalg.matrix_rank(rotated_A), 2)
        self.assertAlmostEqual(abs(np.linalg.det(rotated_A)), 5.0, places=10)

    def test_rank_invariance_under_arbitrary_nonzero_scalar_rescaling(self):
        """Cross 5: Rescaling basis vectors by arbitrary non-zero scalars preserves span dimension."""
        for s1, s2 in [(10.0, 0.1), (-2.0, 5.0), (1e3, 1e-3)]:
            rescaled_A = np.column_stack([s1 * self.v1, s2 * self.v2])
            self.assertEqual(np.linalg.matrix_rank(rescaled_A), 2)


# =============================================================================
# TIER 4: REAL-WORLD SCENARIOS
# =============================================================================

class TestTier4RealWorldScenarios(unittest.TestCase):
    """Tier 4: Frontier Real-World Scenarios (Transformer Attention, Word2Vec, GATE CSE)."""

    def setUp(self):
        self.rng = np.random.default_rng(404)

    def test_transformer_attention_convex_hull_containment(self):
        """Real-World 1: Transformer attention output is strictly a convex combination of Value vectors."""
        d_k = 4
        num_keys = 5
        # Simulated Q, K, V
        q = self.rng.standard_normal((1, d_k))
        k = self.rng.standard_normal((num_keys, d_k))
        v = self.rng.standard_normal((num_keys, 2))  # 2D values for geometric inspection

        # Scaled dot-product attention logits
        logits = (q @ k.T) / math.sqrt(d_k)
        # Numerically stable softmax
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        attention_weights = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # 1. Partition of unity: weights sum to 1
        self.assertAlmostEqual(float(np.sum(attention_weights)), 1.0, places=10)
        # 2. Non-negativity
        self.assertTrue(np.all(attention_weights >= 0.0))

        # Context vector output
        context_vector = attention_weights @ v  # Shape (1, 2)
        c_vec = context_vector[0]

        # Invariant: context vector lies within bounding box / convex coordinates of V
        min_v = np.min(v, axis=0)
        max_v = np.max(v, axis=0)
        self.assertTrue(np.all(c_vec >= min_v - 1e-12))
        self.assertTrue(np.all(c_vec <= max_v + 1e-12))

    def test_transformer_attention_temperature_limits(self):
        """Real-World 2: Attention limits: T -> infty gives uniform barycenter; T -> 0 gives argmax."""
        logits = np.array([[1.0, 5.0, 2.0]])
        v = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])

        # High temperature (uniform barycenter)
        t_high = 1e5
        weights_high = np.exp(logits / t_high) / np.sum(np.exp(logits / t_high))
        out_high = weights_high @ v
        expected_barycenter = np.mean(v, axis=0)
        np.testing.assert_allclose(out_high[0], expected_barycenter, atol=1e-3)

        # Low temperature (hard argmax on index 1)
        t_low = 1e-4
        scaled = logits / t_low
        scaled_max = np.max(scaled)
        weights_low = np.exp(scaled - scaled_max) / np.sum(np.exp(scaled - scaled_max))
        out_low = weights_low @ v
        expected_argmax = v[1]
        np.testing.assert_allclose(out_low[0], expected_argmax, atol=1e-3)

    def test_word2vec_linear_analogy_parallelogram(self):
        """Real-World 3: Word2Vec linear semantic offset: king - man + woman ~ queen."""
        # Simulated semantic embedding space with clear gender and royalty axes
        v_man = np.array([1.0, 0.0, 0.0, 0.0])
        v_woman = np.array([1.0, 1.0, 0.0, 0.0])
        v_king = np.array([1.0, 0.0, 5.0, 0.0])
        v_queen = np.array([1.0, 1.0, 5.0, 0.0])

        # Parallelogram relation: (v_king - v_man) == (v_queen - v_woman)
        offset_royalty_1 = v_king - v_man
        offset_royalty_2 = v_queen - v_woman
        np.testing.assert_allclose(offset_royalty_1, offset_royalty_2, atol=1e-12)

        # Analogy vector prediction: v_pred = v_king - v_man + v_woman
        v_pred = v_king - v_man + v_woman
        np.testing.assert_allclose(v_pred, v_queen, atol=1e-12)

        # Cosine similarity is exactly 1.0
        cos_sim = np.dot(v_pred, v_queen) / (np.linalg.norm(v_pred) * np.linalg.norm(v_queen))
        self.assertAlmostEqual(cos_sim, 1.0, places=10)

    def test_gate_cse_rouche_capelli_consistency_classification(self):
        """Real-World 4: GATE CSE Linear Algebra: Complete Rouché-Capelli classification."""
        # Case A: Unique solution (rank(A) == rank([A|b]) == n)
        A_unique = np.array([[2.0, 1.0], [-1.0, 2.0]])
        b_unique = np.array([3.0, 4.0])
        aug_unique = np.column_stack([A_unique, b_unique])
        self.assertEqual(np.linalg.matrix_rank(A_unique), 2)
        self.assertEqual(np.linalg.matrix_rank(aug_unique), 2)

        # Case B: Infinitely many solutions (rank(A) == rank([A|b]) < n)
        A_infinite = np.array([[2.0, 4.0], [1.0, 2.0]])
        b_infinite = np.array([6.0, 3.0])
        aug_infinite = np.column_stack([A_infinite, b_infinite])
        self.assertEqual(np.linalg.matrix_rank(A_infinite), 1)
        self.assertEqual(np.linalg.matrix_rank(aug_infinite), 1)

        # Case C: Inconsistent / No solution (rank(A) < rank([A|b]))
        A_inconsistent = np.array([[2.0, 4.0], [1.0, 2.0]])
        b_inconsistent = np.array([1.0, 4.0])
        aug_inconsistent = np.column_stack([A_inconsistent, b_inconsistent])
        self.assertEqual(np.linalg.matrix_rank(A_inconsistent), 1)
        self.assertEqual(np.linalg.matrix_rank(aug_inconsistent), 2)

    def test_gate_cse_rank_nullity_theorem(self):
        """Real-World 5: Rank-Nullity Theorem: dim(Null(A)) + rank(A) == n."""
        # Test across n=3 matrices of various ranks (0, 1, 2, 3)
        n = 3
        test_matrices = [
            np.eye(3),                                    # rank 3, nullity 0
            np.diag([1.0, 2.0, 0.0]),                     # rank 2, nullity 1
            np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]]), # rank 1, nullity 2
            np.zeros((3, 3)),                             # rank 0, nullity 3
        ]
        for A in test_matrices:
            rank = np.linalg.matrix_rank(A)
            # SVD decomposition to compute nullity (number of zero singular values)
            s = np.linalg.svd(A, compute_uv=False)
            nullity = int(np.sum(s < 1e-10))
            self.assertEqual(rank + nullity, n)


if __name__ == "__main__":
    unittest.main()
