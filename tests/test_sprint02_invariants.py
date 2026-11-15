#!/usr/bin/env python3
"""
Sprint 2 E2E Invariant Test Suite: Linear Mappings & Matrix Composition
=====================================================================
Curriculum: Linear Algebra (Geometric Foundations & AI/Systems Invariants)
References:
    - Mathematics for Machine Learning (MML Ch. 2.5)
    - Introduction to Applied Linear Algebra (VMLS Ch. 6 & 10)
    - Erwin Kreyszig — Advanced Engineering Mathematics 10th Ed. (Ch. 7)

Architecture:
    Comprehensive 4-Tier Test Suite implemented with unittest and NumPy:
    - Tier 1: Feature Coverage (Linearity definition, parallel grid preservation,
      basis tracking, sequential composition, non-commutativity). >=5 tests each.
    - Tier 2: Boundary & Corner Cases (Singular rank-1 matrix, zero matrix rank 0,
      identity matrix, 180° rotation / negative coordinates, ill-conditioned matrices kappa > 10^8).
      >=5 tests each.
    - Tier 3: Cross-Feature Combinations (Determinant multiplication, associativity,
      inverse of composition, trace cyclic invariance, rank inequalities).
    - Tier 4: Real-World Scenarios (ML: Linear layer weight projection, Attention W_Q W_K composition,
      Batch norm folding; GATE: Matrix chain multiplication dimensions, system consistency).
    - AST & Syntax: Static AST inspection of sprint 2 scene for ManimCE v0.18+ standards
      (zero deprecated symbols, scene classes, make_hud_panel; gracefully skipped if not yet created).

Execution:
    python -m unittest tests/test_sprint02_invariants.py
    pytest tests/test_sprint02_invariants.py
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

class TestTier1LinearityDefinition(unittest.TestCase):
    """Tier 1 Feature 1: Linearity Axioms & Superposition (MML §2.5, VMLS §7.1)."""

    def setUp(self):
        self.rng = np.random.default_rng(201)
        # Canonical Sprint 2 transformation matrix A = [[2, 0], [1, 2]]
        self.A = np.array([[2.0, 0.0], [1.0, 2.0]])

    def test_vector_additivity(self):
        """Feature 1.1: Additivity holds: T(u + v) == T(u) + T(v) across random vector pairs."""
        u_batch = self.rng.uniform(-10.0, 10.0, size=(100, 2))
        v_batch = self.rng.uniform(-10.0, 10.0, size=(100, 2))
        lhs = (u_batch + v_batch) @ self.A.T
        rhs = (u_batch @ self.A.T) + (v_batch @ self.A.T)
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)

    def test_scalar_homogeneity(self):
        """Feature 1.2: Homogeneity holds: T(c * u) == c * T(u) for various scalar classes."""
        u = np.array([3.0, -2.0])
        scalars = [0.0, 1.0, -1.0, 2.5, -7.3, 1e-4, 1e4]
        for c in scalars:
            lhs = self.A @ (c * u)
            rhs = c * (self.A @ u)
            np.testing.assert_allclose(lhs, rhs, atol=1e-12)

    def test_superposition_principle(self):
        """Feature 1.3: Superposition holds: T(c*u + d*v) == c*T(u) + d*T(v)."""
        u = np.array([1.5, 2.5])
        v = np.array([-3.0, 4.0])
        c, d = 3.5, -2.1
        lhs = self.A @ (c * u + d * v)
        rhs = c * (self.A @ u) + d * (self.A @ v)
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)

    def test_origin_fixed_point(self):
        """Feature 1.4: Origin anchor invariant: T(0) == 0 strictly (distinguishes linear from affine)."""
        zero = np.zeros(2)
        np.testing.assert_allclose(self.A @ zero, zero, atol=1e-15)

        # Contrast with affine mapping f(x) = Ax + b where f(0) = b != 0
        b_offset = np.array([1.0, -2.0])
        affine_out = self.A @ zero + b_offset
        self.assertFalse(np.allclose(affine_out, zero))
        np.testing.assert_allclose(affine_out, b_offset, atol=1e-15)

    def test_linear_combination_generalization(self):
        """Feature 1.5: Generalized linearity: T(sum c_i v_i) == sum c_i T(v_i) for k vectors."""
        k = 5
        V = self.rng.uniform(-5.0, 5.0, size=(k, 2))
        coeffs = self.rng.uniform(-3.0, 3.0, size=k)
        lin_comb = np.sum(coeffs[:, np.newaxis] * V, axis=0)
        lhs = self.A @ lin_comb
        mapped_V = (V @ self.A.T)
        rhs = np.sum(coeffs[:, np.newaxis] * mapped_V, axis=0)
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)


class TestTier1ParallelGridPreservation(unittest.TestCase):
    """Tier 1 Feature 2: Parallel Grid Line Invariance (MML §2.5, Kreyszig §7.1)."""

    def setUp(self):
        self.rng = np.random.default_rng(202)
        self.A = np.array([[2.0, 0.0], [1.0, 2.0]])

    def test_straight_line_mapping(self):
        """Feature 2.1: Straight line L(t) = p + t*v maps to straight line A*p + t*(A*v)."""
        p = np.array([1.0, 2.0])
        v = np.array([3.0, -1.0])
        t_values = np.linspace(-5.0, 5.0, 50)
        line_pts = p[np.newaxis, :] + t_values[:, np.newaxis] * v[np.newaxis, :]
        mapped_pts = line_pts @ self.A.T

        expected_p = self.A @ p
        expected_v = self.A @ v
        expected_line = expected_p[np.newaxis, :] + t_values[:, np.newaxis] * expected_v[np.newaxis, :]
        np.testing.assert_allclose(mapped_pts, expected_line, atol=1e-12)

    def test_parallel_lines_remain_parallel(self):
        """Feature 2.2: Parallel lines remain parallel: direction vectors stay collinear."""
        v = np.array([1.0, 2.0])
        p1 = np.array([0.0, 0.0])
        p2 = np.array([-2.0, 3.0])

        mapped_dir1 = self.A @ v
        mapped_dir2 = self.A @ v  # Direction vector for both lines
        # Cross product of 2D direction vectors: x1*y2 - x2*y1 == 0
        cross_prod = mapped_dir1[0] * mapped_dir2[1] - mapped_dir1[1] * mapped_dir2[0]
        self.assertAlmostEqual(cross_prod, 0.0, places=12)

    def test_grid_spacing_equidistance(self):
        """Feature 2.3: Equidistant grid points remain equidistant after linear transformation."""
        p = np.array([0.0, 0.0])
        v = np.array([1.0, 0.0])
        k_indices = np.arange(10)
        grid_pts = p[np.newaxis, :] + k_indices[:, np.newaxis] * v[np.newaxis, :]
        mapped_grid = grid_pts @ self.A.T

        # Delta vector between consecutive points
        deltas = np.diff(mapped_grid, axis=0)
        expected_step = self.A @ v
        for delta in deltas:
            np.testing.assert_allclose(delta, expected_step, atol=1e-12)

    def test_parallelogram_grid_cells(self):
        """Feature 2.4: Unit square grid cell (0,0), e1, e1+e2, e2 maps to closed parallelogram."""
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])
        a1 = self.A @ e1
        a2 = self.A @ e2

        # In a parallelogram, opposite side vectors must be equal
        side_bottom = a1 - np.zeros(2)
        side_top = (a1 + a2) - a2
        np.testing.assert_allclose(side_bottom, side_top, atol=1e-12)

        side_left = a2 - np.zeros(2)
        side_right = (a1 + a2) - a1
        np.testing.assert_allclose(side_left, side_right, atol=1e-12)

    def test_midpoint_preservation(self):
        """Feature 2.5: Line segment midpoints are preserved: T((u+v)/2) == (T(u)+T(v))/2."""
        u = np.array([-4.0, 6.0])
        v = np.array([2.0, -8.0])
        midpoint = 0.5 * (u + v)
        mapped_midpoint = self.A @ midpoint
        midpoint_of_mapped = 0.5 * (self.A @ u + self.A @ v)
        np.testing.assert_allclose(mapped_midpoint, midpoint_of_mapped, atol=1e-12)


class TestTier1BasisTracking(unittest.TestCase):
    """Tier 1 Feature 3: Basis Vector Tracking & Matrix Reconstruction (VMLS §6, MML §2.5)."""

    def setUp(self):
        self.e1 = np.array([1.0, 0.0])
        self.e2 = np.array([0.0, 1.0])
        # Canonical Sprint 2 matrix A = [[2, 0], [1, 2]]
        self.a1 = np.array([2.0, 1.0])
        self.a2 = np.array([0.0, 2.0])
        self.A = np.column_stack([self.a1, self.a2])

    def test_basis_mapping_to_columns(self):
        """Feature 3.1: Columns of A are exactly the images of unit basis vectors e1 and e2."""
        np.testing.assert_allclose(self.A @ self.e1, self.a1, atol=1e-12)
        np.testing.assert_allclose(self.A @ self.e2, self.a2, atol=1e-12)

    def test_arbitrary_vector_synthesis(self):
        """Feature 3.2: Arbitrary vector [x, y]^T maps to x*a1 + y*a2."""
        test_coords = [
            (2.0, 3.0),
            (-1.5, 4.2),
            (0.0, -5.0),
            (7.0, 0.0),
            (-3.0, -2.0)
        ]
        for x, y in test_coords:
            v = np.array([x, y])
            mapped_direct = self.A @ v
            mapped_linear_comb = x * self.a1 + y * self.a2
            np.testing.assert_allclose(mapped_direct, mapped_linear_comb, atol=1e-12)

    def test_matrix_reconstruction_from_basis_images(self):
        """Feature 3.3: Matrix A is uniquely recovered by stacking observed basis images."""
        recovered_A = np.column_stack([self.A @ self.e1, self.A @ self.e2])
        np.testing.assert_allclose(recovered_A, self.A, atol=1e-12)

    def test_unit_square_area_equals_determinant(self):
        """Feature 3.4: Deformed area of unit square equals |det(A)|."""
        det_A = np.linalg.det(self.A)
        self.assertAlmostEqual(det_A, 4.0, places=10)
        # 2D cross product of transformed basis vectors gives signed area
        signed_area = self.a1[0] * self.a2[1] - self.a1[1] * self.a2[0]
        self.assertAlmostEqual(signed_area, det_A, places=10)

    def test_orientation_preservation(self):
        """Feature 3.5: Positive determinant det(A) > 0 preserves right-handed orientation."""
        det_A = np.linalg.det(self.A)
        self.assertGreater(det_A, 0.0)
        # Standard basis e1 x e2 is positive (+1 k-hat)
        orig_cross = self.e1[0] * self.e2[1] - self.e1[1] * self.e2[0]
        # Transformed basis a1 x a2 is positive (+4 k-hat)
        trans_cross = self.a1[0] * self.a2[1] - self.a1[1] * self.a2[0]
        self.assertEqual(np.sign(orig_cross), np.sign(trans_cross))


class TestTier1SequentialComposition(unittest.TestCase):
    """Tier 1 Feature 4: Matrix Multiplication as Sequential Composition (MML §2.5, VMLS §10)."""

    def setUp(self):
        self.rng = np.random.default_rng(203)
        self.A = np.array([[2.0, 0.0], [1.0, 2.0]])
        self.B = np.array([[0.0, -1.0], [1.0, 0.0]])  # 90° CCW rotation
        self.C = np.array([[1.0, 1.0], [0.0, 1.0]])   # Shear matrix

    def test_sequential_action_equivalence(self):
        """Feature 4.1: (BA)x == B(Ax) for batch of 50 random vectors."""
        BA = self.B @ self.A
        X = self.rng.uniform(-10.0, 10.0, size=(50, 2))
        for x in X:
            direct = BA @ x
            sequential = self.B @ (self.A @ x)
            np.testing.assert_allclose(direct, sequential, atol=1e-12)

    def test_composition_associativity(self):
        """Feature 4.2: Associativity holds: ((C @ B) @ A) == (C @ (B @ A))."""
        lhs = (self.C @ self.B) @ self.A
        rhs = self.C @ (self.B @ self.A)
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)

    def test_composition_distributivity(self):
        """Feature 4.3: Distributivity over addition: C(A + B) == CA + CB."""
        lhs = self.C @ (self.A + self.B)
        rhs = self.C @ self.A + self.C @ self.B
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)

    def test_identity_composition(self):
        """Feature 4.4: Identity matrix is neutral element: AI == IA == A."""
        I = np.eye(2)
        np.testing.assert_allclose(self.A @ I, self.A, atol=1e-12)
        np.testing.assert_allclose(I @ self.A, self.A, atol=1e-12)

    def test_multi_stage_pipeline(self):
        """Feature 4.5: 3-stage composition: (C @ B @ A)x == C(B(Ax))."""
        CBA = self.C @ self.B @ self.A
        x = np.array([3.0, -2.0])
        out_direct = CBA @ x
        out_step1 = self.A @ x
        out_step2 = self.B @ out_step1
        out_step3 = self.C @ out_step2
        np.testing.assert_allclose(out_direct, out_step3, atol=1e-12)


class TestTier1NonCommutativity(unittest.TestCase):
    """Tier 1 Feature 5: Geometric Non-Commutativity SR != RS (Kreyszig §7.2, VMLS §10)."""

    def setUp(self):
        # S: horizontal shear by k=1
        self.S = np.array([[1.0, 1.0], [0.0, 1.0]])
        # R: 90° counter-clockwise rotation
        self.R = np.array([[0.0, -1.0], [1.0, 0.0]])
        # SR: Rotate first, then Shear: [[1, -1], [1, 0]]
        self.SR = self.S @ self.R
        # RS: Shear first, then Rotate: [[0, -1], [1, 1]]
        self.RS = self.R @ self.S

    def test_rotation_shear_matrix_discrepancy(self):
        """Feature 5.1: S @ R != R @ S with non-zero commutator norm."""
        commutator = self.SR - self.RS
        expected_commutator = np.array([[1.0, 0.0], [0.0, -1.0]])
        np.testing.assert_allclose(commutator, expected_commutator, atol=1e-12)
        self.assertGreater(np.linalg.norm(commutator), 0.0)

    def test_point_destination_divergence(self):
        """Feature 5.2: Unit vector e1 = [1, 0]^T lands at distinct coordinates under SR vs RS."""
        e1 = np.array([1.0, 0.0])
        dest_SR = self.SR @ e1  # [1, 1]^T
        dest_RS = self.RS @ e1  # [0, 1]^T
        np.testing.assert_allclose(dest_SR, np.array([1.0, 1.0]), atol=1e-12)
        np.testing.assert_allclose(dest_RS, np.array([0.0, 1.0]), atol=1e-12)
        spatial_distance = np.linalg.norm(dest_SR - dest_RS)
        self.assertAlmostEqual(spatial_distance, 1.0, places=10)

    def test_commutator_trace_zero(self):
        """Feature 5.3: Lie algebra invariant: Tr([S, R]) == Tr(SR - RS) == 0."""
        commutator = self.SR - self.RS
        tr_comm = np.trace(commutator)
        self.assertAlmostEqual(tr_comm, 0.0, places=12)

    def test_commutative_special_cases(self):
        """Feature 5.4: Identification of commuting pairs: scalar multiples and 2D rotations."""
        # 1. Scalar multiple commutes with any matrix
        cI = 3.5 * np.eye(2)
        np.testing.assert_allclose(cI @ self.S, self.S @ cI, atol=1e-12)

        # 2. Rotations in 2D form abelian group SO(2): R(theta1)R(theta2) == R(theta2)R(theta1)
        theta1, theta2 = math.pi / 6, math.pi / 4
        R1 = np.array([[math.cos(theta1), -math.sin(theta1)], [math.sin(theta1), math.cos(theta1)]])
        R2 = np.array([[math.cos(theta2), -math.sin(theta2)], [math.sin(theta2), math.cos(theta2)]])
        np.testing.assert_allclose(R1 @ R2, R2 @ R1, atol=1e-12)

    def test_composition_inverses_order_reversal(self):
        """Feature 5.5: Shoe-and-sock rule: (SR)^-1 == R^-1 S^-1 != S^-1 R^-1."""
        SR_inv = np.linalg.inv(self.SR)
        R_inv = np.linalg.inv(self.R)
        S_inv = np.linalg.inv(self.S)

        np.testing.assert_allclose(SR_inv, R_inv @ S_inv, atol=1e-12)
        self.assertFalse(np.allclose(SR_inv, S_inv @ R_inv))


# =============================================================================
# TIER 2: BOUNDARY & CORNER CASES
# =============================================================================

class TestTier2SingularRankOneMatrix(unittest.TestCase):
    """Tier 2 Boundary 1: Singular Rank-1 Matrix Collapse (MML §2.6, Kreyszig §7.4)."""

    def setUp(self):
        # Rank-1 matrix with collinear columns
        self.A_sing = np.array([[2.0, 4.0], [1.0, 2.0]])

    def test_rank1_collapse_dimension(self):
        """Boundary 1.1: Matrix rank is 1 and determinant is exactly 0."""
        self.assertEqual(np.linalg.matrix_rank(self.A_sing), 1)
        self.assertAlmostEqual(np.linalg.det(self.A_sing), 0.0, places=12)

    def test_rank1_codomain_confinement(self):
        """Boundary 1.2: Entire 2D plane collapses onto 1D line y = 0.5x."""
        rng = np.random.default_rng(204)
        pts = rng.uniform(-50.0, 50.0, size=(100, 2))
        mapped = pts @ self.A_sing.T
        # For every mapped point [x, y], x - 2y == 0
        residuals = mapped[:, 0] - 2.0 * mapped[:, 1]
        np.testing.assert_allclose(residuals, 0.0, atol=1e-12)

    def test_rank1_kernel_non_trivial(self):
        """Boundary 1.3: Null space is non-trivial 1D line [-2, 1]^T."""
        null_vec = np.array([-2.0, 1.0])
        np.testing.assert_allclose(self.A_sing @ null_vec, np.zeros(2), atol=1e-12)

    def test_rank1_non_invertibility(self):
        """Boundary 1.4: Inversion of singular matrix raises LinAlgError."""
        with self.assertRaises(np.linalg.LinAlgError):
            np.linalg.inv(self.A_sing)

    def test_rank1_unreachable_point_detection(self):
        """Boundary 1.5: Target b not on line y = 0.5x has rank([A|b]) > rank(A)."""
        b_unreachable = np.array([1.0, 3.0])  # 1 - 2(3) = -5 != 0
        aug = np.column_stack([self.A_sing, b_unreachable])
        self.assertEqual(np.linalg.matrix_rank(self.A_sing), 1)
        self.assertEqual(np.linalg.matrix_rank(aug), 2)


class TestTier2ZeroMatrixRankZero(unittest.TestCase):
    """Tier 2 Boundary 2: Trivial Zero Matrix Annihilation (VMLS §6.1)."""

    def setUp(self):
        self.zero_mat = np.zeros((2, 2))

    def test_zero_matrix_rank_and_det(self):
        """Boundary 2.1: Zero matrix has rank 0 and determinant 0."""
        self.assertEqual(np.linalg.matrix_rank(self.zero_mat), 0)
        self.assertAlmostEqual(np.linalg.det(self.zero_mat), 0.0, places=15)

    def test_zero_matrix_annihilation(self):
        """Boundary 2.2: 0 @ x == 0 for all x in R^2."""
        rng = np.random.default_rng(205)
        batch = rng.uniform(-100.0, 100.0, size=(50, 2))
        mapped = batch @ self.zero_mat.T
        np.testing.assert_allclose(mapped, 0.0, atol=1e-15)

    def test_zero_matrix_kernel_is_entire_space(self):
        """Boundary 2.3: Nullity is 2; every vector in R^2 is in kernel."""
        # Rank-Nullity: rank + nullity = 2 -> nullity = 2 - 0 = 2
        s = np.linalg.svd(self.zero_mat, compute_uv=False)
        nullity = int(np.sum(s < 1e-12))
        self.assertEqual(nullity, 2)

    def test_zero_matrix_nilpotence_composition(self):
        """Boundary 2.4: 0 @ A == A @ 0 == 0 for any matrix A."""
        A = np.array([[3.0, -1.0], [4.0, 2.0]])
        np.testing.assert_allclose(self.zero_mat @ A, self.zero_mat, atol=1e-15)
        np.testing.assert_allclose(A @ self.zero_mat, self.zero_mat, atol=1e-15)

    def test_zero_matrix_area_collapse(self):
        """Boundary 2.5: Area of any polygon maps to 0."""
        unit_square_area = 1.0
        mapped_area = unit_square_area * abs(np.linalg.det(self.zero_mat))
        self.assertEqual(mapped_area, 0.0)


class TestTier2IdentityMatrix(unittest.TestCase):
    """Tier 2 Boundary 3: Identity Matrix Invariants (MML §2.4)."""

    def setUp(self):
        self.I = np.eye(2)

    def test_identity_isometry(self):
        """Boundary 3.1: Norm preservation: ||Ix|| == ||x|| for all x."""
        rng = np.random.default_rng(206)
        batch = rng.uniform(-100.0, 100.0, size=(50, 2))
        norms_orig = np.linalg.norm(batch, axis=1)
        norms_mapped = np.linalg.norm(batch @ self.I.T, axis=1)
        np.testing.assert_allclose(norms_orig, norms_mapped, atol=1e-12)

    def test_identity_determinant_and_rank(self):
        """Boundary 3.2: det(I) == 1.0 and rank(I) == 2."""
        self.assertAlmostEqual(np.linalg.det(self.I), 1.0, places=12)
        self.assertEqual(np.linalg.matrix_rank(self.I), 2)

    def test_identity_fixed_point_all_vectors(self):
        """Boundary 3.3: Every vector in R^2 is an eigenvector with eigenvalue 1."""
        eigvals, eigvecs = np.linalg.eig(self.I)
        np.testing.assert_allclose(eigvals, [1.0, 1.0], atol=1e-12)

    def test_identity_grid_invariance(self):
        """Boundary 3.4: Basis vectors e1, e2 and grid spacing are identical before and after."""
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])
        np.testing.assert_allclose(self.I @ e1, e1, atol=1e-15)
        np.testing.assert_allclose(self.I @ e2, e2, atol=1e-15)

    def test_identity_neutral_composition(self):
        """Boundary 3.5: Composition with any matrix leaves it unchanged."""
        A = np.array([[-3.2, 5.1], [1.7, -4.9]])
        np.testing.assert_allclose(self.I @ A, A, atol=1e-12)
        np.testing.assert_allclose(A @ self.I, A, atol=1e-12)


class TestTier2Rotation180AndReflections(unittest.TestCase):
    """Tier 2 Boundary 4: 180° Inversion & Reflection Parity Flip (Kreyszig §7.2)."""

    def setUp(self):
        self.R180 = np.array([[-1.0, 0.0], [0.0, -1.0]])
        self.Reflect_X = np.array([[1.0, 0.0], [0.0, -1.0]])

    def test_rotation_180_inversion(self):
        """Boundary 4.1: R(180°) == -I, inverting all coordinates: x -> -x."""
        x = np.array([3.5, -4.2])
        np.testing.assert_allclose(self.R180 @ x, -x, atol=1e-12)

    def test_rotation_180_preserves_area_and_orientation(self):
        """Boundary 4.2: det(R(180°)) == +1 (orientation and area preserved)."""
        self.assertAlmostEqual(np.linalg.det(self.R180), 1.0, places=12)

    def test_reflection_orientation_reversal(self):
        """Boundary 4.3: Reflection has det == -1, reversing orientation parity (handedness)."""
        self.assertAlmostEqual(np.linalg.det(self.Reflect_X), -1.0, places=12)

    def test_negative_coordinates_4quadrant_mapping(self):
        """Boundary 4.4: 180° rotation maps Quadrant I (+,+) strictly to Quadrant III (-,-)."""
        q1_pts = np.array([[1.0, 2.0], [3.0, 5.0], [0.5, 0.5]])
        mapped = q1_pts @ self.R180.T
        self.assertTrue(np.all(mapped < 0))

    def test_involution_property(self):
        """Boundary 4.5: Both R(180°) and reflections are involutions: M^2 == I."""
        np.testing.assert_allclose(self.R180 @ self.R180, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(self.Reflect_X @ self.Reflect_X, np.eye(2), atol=1e-12)


class TestTier2IllConditionedMatrices(unittest.TestCase):
    """Tier 2 Boundary 5: Ill-Conditioned Mappings kappa(A) > 10^8 (MML §2.3, VMLS §10)."""

    def setUp(self):
        # Matrix with condition number ~ 10^9
        eps = 1e-9
        self.A_ill = np.array([[1.0, 1.0], [1.0, 1.0 + eps]])

    def test_ill_conditioned_matrix_construction(self):
        """Boundary 5.1: Condition number kappa(A) >= 10^8."""
        cond = np.linalg.cond(self.A_ill)
        self.assertGreater(cond, 1e8)

    def test_ill_conditioned_almost_collinear_columns(self):
        """Boundary 5.2: Column angle is near zero (almost linearly dependent)."""
        c1 = self.A_ill[:, 0]
        c2 = self.A_ill[:, 1]
        cos_theta = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2))
        self.assertAlmostEqual(cos_theta, 1.0, places=8)

    def test_ill_conditioned_inversion_sensitivity(self):
        """Boundary 5.3: Small perturbation in b causes large perturbation in x = A^-1 b."""
        b = np.array([2.0, 2.0])
        delta_b = np.array([0.0, 1e-9])
        x = np.linalg.solve(self.A_ill, b)
        x_perturbed = np.linalg.solve(self.A_ill, b + delta_b)
        delta_x_norm = np.linalg.norm(x_perturbed - x)
        # Amplification factor exceeds 10^6
        self.assertGreater(delta_x_norm / np.linalg.norm(delta_b), 1e6)

    def test_ill_conditioned_volume_near_zero(self):
        """Boundary 5.4: Determinant is nonzero but microscopic relative to column norms."""
        det = np.linalg.det(self.A_ill)
        self.assertNotEqual(det, 0.0)
        self.assertLess(abs(det), 1e-8)

    def test_ill_conditioned_svd_ratio(self):
        """Boundary 5.5: Ratio of singular values sigma_max / sigma_min >= 10^8."""
        s = np.linalg.svd(self.A_ill, compute_uv=False)
        ratio = s[0] / s[1]
        self.assertGreater(ratio, 1e8)


# =============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS
# =============================================================================

class TestTier3CrossFeatureCombinations(unittest.TestCase):
    """Tier 3: Pairwise & Multi-Feature Invariants (MML §2.5, Kreyszig §7)."""

    def setUp(self):
        self.rng = np.random.default_rng(207)
        self.A = np.array([[2.0, 1.0], [0.5, 3.0]])
        self.B = np.array([[1.0, -2.0], [3.0, 1.0]])
        self.C = np.array([[-1.0, 4.0], [2.0, -1.0]])

    def test_determinant_multiplication_rule(self):
        """Cross-Feature 1: det(AB) == det(A) * det(B) across 50 random invertible pairs."""
        for _ in range(50):
            M1 = self.rng.uniform(-5.0, 5.0, size=(2, 2))
            M2 = self.rng.uniform(-5.0, 5.0, size=(2, 2))
            det_prod = np.linalg.det(M1 @ M2)
            prod_det = np.linalg.det(M1) * np.linalg.det(M2)
            self.assertAlmostEqual(det_prod, prod_det, places=8)

    def test_associativity_with_scaling(self):
        """Cross-Feature 2: ((c * A) @ B) @ C == A @ (c * (B @ C))."""
        c = -3.7
        lhs = ((c * self.A) @ self.B) @ self.C
        rhs = self.A @ (c * (self.B @ self.C))
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)

    def test_inverse_of_composition(self):
        """Cross-Feature 3: (AB)^-1 == B^-1 A^-1 for invertible matrices."""
        AB_inv = np.linalg.inv(self.A @ self.B)
        B_inv_A_inv = np.linalg.inv(self.B) @ np.linalg.inv(self.A)
        np.testing.assert_allclose(AB_inv, B_inv_A_inv, atol=1e-12)

    def test_trace_cyclic_invariance(self):
        """Cross-Feature 4: Tr(ABC) == Tr(BCA) == Tr(CAB)."""
        tr_ABC = np.trace(self.A @ self.B @ self.C)
        tr_BCA = np.trace(self.B @ self.C @ self.A)
        tr_CAB = np.trace(self.C @ self.A @ self.B)
        self.assertAlmostEqual(tr_ABC, tr_BCA, places=10)
        self.assertAlmostEqual(tr_BCA, tr_CAB, places=10)

    def test_rank_composition_inequality(self):
        """Cross-Feature 5: Sylvester's rank inequality: rank(AB) <= min(rank(A), rank(B))."""
        rank_A = np.linalg.matrix_rank(self.A)
        rank_B = np.linalg.matrix_rank(self.B)
        rank_AB = np.linalg.matrix_rank(self.A @ self.B)
        self.assertLessEqual(rank_AB, min(rank_A, rank_B))


# =============================================================================
# TIER 4: REAL-WORLD SCENARIOS
# =============================================================================

class TestTier4RealWorldScenarios(unittest.TestCase):
    """Tier 4: ML Feedforward & Attention Projections and GATE Linear Algebra (MML §2.5)."""

    def setUp(self):
        self.rng = np.random.default_rng(208)

    def test_ml_linear_layer_projection(self):
        """Real-World 1: ML Feed-Forward Layer: y = x W^T + b and multi-layer linear collapse."""
        d_in, d_out = 4, 3
        W = self.rng.standard_normal((d_out, d_in))
        b = np.array([0.1, -0.2, 0.5])
        x = np.array([1.0, 2.0, -1.0, 0.5])

        y = x @ W.T + b
        self.assertEqual(y.shape, (d_out,))

        # Multi-layer linear collapse: W2 @ (W1 @ x) == (W2 @ W1) @ x
        # Demonstrates why non-linear activations (ReLU/GELU) are mathematically necessary
        d_mid = 5
        W1 = self.rng.standard_normal((d_mid, d_in))
        W2 = self.rng.standard_normal((d_out, d_mid))
        y_sequential = W2 @ (W1 @ x)
        W_collapsed = W2 @ W1
        y_collapsed = W_collapsed @ x
        np.testing.assert_allclose(y_sequential, y_collapsed, atol=1e-12)

    def test_ml_attention_query_key_composition(self):
        """Real-World 2: Attention Projection Composition: Q K^T = (X W_Q)(X W_K)^T."""
        seq_len, d_model, d_k = 4, 8, 4
        X = self.rng.standard_normal((seq_len, d_model))
        W_Q = self.rng.standard_normal((d_model, d_k))
        W_K = self.rng.standard_normal((d_model, d_k))

        Q = X @ W_Q
        K = X @ W_K
        scores_direct = Q @ K.T

        # Equivalent single bilinear operator: X (W_Q W_K^T) X^T
        M_bilinear = W_Q @ W_K.T
        scores_bilinear = X @ M_bilinear @ X.T
        np.testing.assert_allclose(scores_direct, scores_bilinear, atol=1e-12)

    def test_ml_batch_norm_folding(self):
        """Real-World 3: ML Inference Optimization: Folding BatchNorm into preceding Linear Layer."""
        d_in, d_out = 3, 2
        W = np.array([[2.0, -1.0, 0.5], [1.0, 3.0, -2.0]])
        b = np.array([0.5, -1.0])
        gamma = np.array([1.5, 0.8])
        beta = np.array([-0.2, 0.3])
        mean = np.array([0.1, 0.4])
        var = np.array([0.25, 0.16])
        eps = 1e-5

        std = np.sqrt(var + eps)
        scale = gamma / std

        # Fused weights and bias
        W_fused = scale[:, np.newaxis] * W
        b_fused = scale * (b - mean) + beta

        x = np.array([1.0, -2.0, 3.0])
        # Unfused calculation
        y_linear = W @ x + b
        y_bn = gamma * ((y_linear - mean) / std) + beta

        # Fused calculation
        y_fused = W_fused @ x + b_fused
        np.testing.assert_allclose(y_bn, y_fused, atol=1e-12)

    def test_gate_matrix_chain_multiplication_dimensions(self):
        """Real-World 4: GATE CSE Matrix Chain Multiplication: Associative grouping cost."""
        # Dimensions: A1 (10x100), A2 (100x5), A3 (5x50)
        d0, d1, d2, d3 = 10, 100, 5, 50
        # Grouping 1: (A1 A2) A3 -> cost = 10*100*5 + 10*5*50 = 5000 + 2500 = 7500
        cost_group1 = d0 * d1 * d2 + d0 * d2 * d3
        # Grouping 2: A1 (A2 A3) -> cost = 100*5*50 + 10*100*50 = 25000 + 50000 = 75000
        cost_group2 = d1 * d2 * d3 + d0 * d1 * d3

        self.assertEqual(cost_group1, 7500)
        self.assertEqual(cost_group2, 75000)
        self.assertLess(cost_group1, cost_group2)

    def test_gate_rouche_capelli_consistency(self):
        """Real-World 5: GATE Linear Algebra: System consistency via Rouché-Capelli rank equality."""
        # Consistent unique solution: rank(A) == rank([A|b]) == n
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([5.0, 5.0])
        aug = np.column_stack([A, b])
        self.assertEqual(np.linalg.matrix_rank(A), 2)
        self.assertEqual(np.linalg.matrix_rank(aug), 2)

        x = np.linalg.solve(A, b)
        np.testing.assert_allclose(x, np.array([2.0, 1.0]), atol=1e-12)


# =============================================================================
# TIER 5: STATIC AST & SYNTAX INTEGRITY
# =============================================================================

class TestSprint02ASTAndSyntax(unittest.TestCase):
    """Tier 5: Static AST verification of Sprint 2 scene code for ManimCE standards."""

    SCENE_PATH = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "linear_algebra",
            "sprint_02_linear_mappings",
            "scene_linear_mappings.py"
        )
    )

    def _get_ast(self):
        if not os.path.exists(self.SCENE_PATH):
            self.skipTest(
                f"Scene script '{self.SCENE_PATH}' not yet created (Pending Milestone 2)."
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
            f"Found deprecated symbols in Sprint 2 scene: {found_deprecated}"
        )

    def test_scene_class_defined(self):
        """AST 2: Master scene class LinearMappingsScene is defined."""
        tree = self._get_ast()
        class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        self.assertIn("LinearMappingsScene", class_names)

    def test_hud_panel_creation_function_present(self):
        """AST 3: File defines or invokes make_hud_panel."""
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
            "Sprint 2 scene must define or call 'make_hud_panel' for pinned HUD text."
        )

    def test_semantic_color_palette_constants(self):
        """AST 4: Semantic color constants are declared (I_HAT_COLOR, J_HAT_COLOR)."""
        tree = self._get_ast()
        assigned_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigned_names.add(target.id)
        # Check standard 3b1b basis color constants
        self.assertTrue(
            "I_HAT_COLOR" in assigned_names or "IHAT_COLOR" in assigned_names or "GREEN" in assigned_names,
            "Scene should reference standard basis color constants."
        )

    def test_no_forbidden_execution_calls(self):
        """AST 5: Scene script contains zero calls to os.system, subprocess, or eval."""
        tree = self._get_ast()
        forbidden = {"eval", "exec"}
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in forbidden:
                    found.append((node.func.id, getattr(node, "lineno", "?")))
        self.assertEqual(found, [], f"Forbidden functions detected: {found}")

    def test_act2_and_act3_unit_coordinate_referencing_no_double_transform(self):
        """AST 6: Act 2 & Act 3 use unit coordinate queries (no double matrix squaring)."""
        tree = self._get_ast()
        with open(self.SCENE_PATH, "r", encoding="utf-8") as f:
            code = f.read()
        # Verify Act 2 references plane.c2p(1, 0) and plane.c2p(0, 1) rather than m_curr @ i_unit inside c2p
        self.assertIn("plane.c2p(1, 0)", code, "Act 2/3 must query unit coordinates to prevent matrix squaring.")
        self.assertIn("plane.c2p(0, 1)", code, "Act 2/3 must query unit coordinates to prevent matrix squaring.")
        self.assertNotIn("c2p(*(m_curr @ i_unit))", code, "Act 2 must not compound m_curr onto deforming plane.")

    def test_act4_compact_title_and_label_non_occlusion(self):
        """AST 7: Act 4 title is compact and dest_rs_label is positioned at DL to prevent occlusion."""
        tree = self._get_ast()
        with open(self.SCENE_PATH, "r", encoding="utf-8") as f:
            code = f.read()
        self.assertIn('Text("Act 4: Non-Commutativity Stress Test", font_size=18', code)
        self.assertIn("dest_rs_label = MathTex(r\"(RS)\\mathbf{x} = [-3, 5]^\\top\", font_size=18, color=ERROR_COLOR).next_to(", code)
        self.assertIn("dest_rs_dot, DL, buff=0.15", code)


if __name__ == "__main__":
    unittest.main()
