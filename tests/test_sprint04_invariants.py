#!/usr/bin/env python3
r"""
Sprint 4 E2E Invariant Test Suite: Change of Basis & Similarity Transformations
=============================================================================
Curriculum: Linear Algebra (Geometric Foundations & AI/Systems Invariants)
References:
    - Mathematics for Machine Learning (MML Ch. 2.7 - Basis Change, Similarity)
    - Introduction to Applied Linear Algebra (VMLS Ch. 6 & 10)
    - Erwin Kreyszig — Advanced Engineering Mathematics 10th Ed. (Ch. 8)

Architecture:
    Comprehensive 4-Tier Test Suite implemented with unittest and NumPy:
    - Tier 1: Feature Coverage (Basis matrix P = [v1 | v2], coordinate transition
      [x]_B = P^-1 x, similarity conjugation A = P D P^-1, diagonal scaling D = diag(3, 1),
      3-step geometric sequence P^-1 -> D -> P). >=5 tests each.
    - Tier 2: Boundary & Corner Cases (Defective matrix non-diagonalizability,
      repeated eigenvalues with full eigenspace cI, orthogonal eigenvectors of symmetric matrix,
      zero eigenvalue / singular matrix similarity, negative eigenvalue saddle reflection).
      >=5 tests each.
    - Tier 3: Cross-Feature Combinations (Determinant invariance det(P^-1AP) == det(A),
      trace invariance Tr(P^-1AP) == Tr(A), characteristic polynomial invariance,
      eigenvalue spectrum invariance, matrix powers A^k == P D^k P^-1).
    - Tier 4: Real-World Scenarios (ML: PCA / Covariance diagonalization Sigma = Q Lambda Q^T,
      SVD relation A^T A = V Sigma^2 V^T, Spectral Graph Laplacian decoupling;
      GATE: Cayley-Hamilton theorem p(A) = 0 and A^-1 calculation, diagonalizability criteria).
    - AST & Syntax: Static AST inspection of sprint 4 scene for zero deprecated symbols,
      scene class definition, make_hud_panel HUD cards (gracefully skipped if not yet created).

Execution:
    python -m unittest tests/test_sprint04_invariants.py
    pytest tests/test_sprint04_invariants.py
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

class TestTier1ChangeOfBasisMatrix(unittest.TestCase):
    """Tier 1 Feature 1: Change of Basis Matrix P = [v1 | v2] (MML §2.7.2)."""

    def setUp(self):
        # Canonical Sprint 4 eigenvectors
        self.v1 = np.array([1.0, 1.0])
        self.v2 = np.array([-1.0, 1.0])
        # P = [v1 | v2] = [[1, -1], [1, 1]]
        self.P = np.column_stack([self.v1, self.v2])
        self.P_inv = np.linalg.inv(self.P)

    def test_basis_matrix_invertibility_and_determinant(self):
        """Feature 1.1: P is invertible: det(P) == 2.0 and rank(P) == 2."""
        det_P = np.linalg.det(self.P)
        rank_P = np.linalg.matrix_rank(self.P)
        self.assertAlmostEqual(det_P, 2.0, places=12)
        self.assertEqual(rank_P, 2)

    def test_coordinate_transition_standard_to_basis(self):
        """Feature 1.2: Standard vector x maps to basis coordinates [x]_B = P^-1 @ x."""
        x = np.array([2.0, 0.0])
        x_B = self.P_inv @ x
        # x = 1*v1 + (-1)*v2 = [1, 1] - [-1, 1] = [2, 0] -> coords are [1, -1]^T
        expected_x_B = np.array([1.0, -1.0])
        np.testing.assert_allclose(x_B, expected_x_B, atol=1e-12)

    def test_coordinate_transition_basis_to_standard(self):
        """Feature 1.3: Basis coordinates [x]_B map to standard coordinates x = P @ [x]_B."""
        coords = np.array([3.0, -2.0])
        x_std = self.P @ coords
        expected_std = 3.0 * self.v1 + (-2.0) * self.v2
        np.testing.assert_allclose(x_std, expected_std, atol=1e-12)

    def test_basis_vectors_in_own_basis(self):
        """Feature 1.4: Basis vectors have canonical coordinates [1, 0]^T and [0, 1]^T in basis B."""
        coords_v1 = self.P_inv @ self.v1
        coords_v2 = self.P_inv @ self.v2
        np.testing.assert_allclose(coords_v1, np.array([1.0, 0.0]), atol=1e-12)
        np.testing.assert_allclose(coords_v2, np.array([0.0, 1.0]), atol=1e-12)

    def test_coordinate_bijective_isomorphism(self):
        """Feature 1.5: Round-trip transformation P @ (P^-1 @ x) == x across 100 random vectors."""
        rng = np.random.default_rng(401)
        X = rng.uniform(-50.0, 50.0, size=(100, 2))
        X_B = X @ self.P_inv.T
        X_recon = X_B @ self.P.T
        np.testing.assert_allclose(X_recon, X, atol=1e-12)


class TestTier1CoordinateTransformations(unittest.TestCase):
    """Tier 1 Feature 2: Coordinate Transition Calculus (MML §2.7.2, VMLS §6)."""

    def setUp(self):
        self.v1 = np.array([1.0, 1.0])
        self.v2 = np.array([-1.0, 1.0])
        self.P = np.column_stack([self.v1, self.v2])
        self.P_inv = np.linalg.inv(self.P)
        self.e1 = np.array([1.0, 0.0])
        self.e2 = np.array([0.0, 1.0])

    def test_unit_vectors_coordinate_change(self):
        """Feature 2.1: Standard unit vectors e1, e2 map to columns of P^-1: [0.5, -0.5]^T and [0.5, 0.5]^T."""
        np.testing.assert_allclose(self.P_inv @ self.e1, np.array([0.5, -0.5]), atol=1e-12)
        np.testing.assert_allclose(self.P_inv @ self.e2, np.array([0.5, 0.5]), atol=1e-12)

    def test_arbitrary_vector_barycentric_recovery(self):
        """Feature 2.2: Coordinate recovery: c1*v1 + c2*v2 == x for coordinates [c1, c2]^T = P^-1 x."""
        x = np.array([5.0, -3.0])
        c = self.P_inv @ x
        recon = c[0] * self.v1 + c[1] * self.v2
        np.testing.assert_allclose(recon, x, atol=1e-12)

    def test_linear_combination_coordinate_isomorphism(self):
        """Feature 2.3: [c*u + d*v]_B == c*[u]_B + d*[v]_B (coordinate map is linear isomorphism)."""
        u = np.array([2.0, 4.0])
        v = np.array([-1.0, 3.0])
        c, d = 2.5, -1.8
        lhs = self.P_inv @ (c * u + d * v)
        rhs = c * (self.P_inv @ u) + d * (self.P_inv @ v)
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)

    def test_origin_invariant_coordinates(self):
        """Feature 2.4: Origin has coordinates [0, 0]^T in both standard and eigen basis."""
        np.testing.assert_allclose(self.P_inv @ np.zeros(2), np.zeros(2), atol=1e-15)

    def test_inner_product_in_basis(self):
        """Feature 2.5: Inner product relation: u^T v == [u]_B^T (P^T P) [v]_B."""
        u = np.array([3.0, 1.0])
        v = np.array([2.0, -4.0])
        dot_standard = np.dot(u, v)
        u_B = self.P_inv @ u
        v_B = self.P_inv @ v
        Gram = self.P.T @ self.P
        dot_basis = u_B @ Gram @ v_B
        self.assertAlmostEqual(dot_standard, dot_basis, places=12)


class TestTier1SimilarityTransformation(unittest.TestCase):
    """Tier 1 Feature 3: Similarity Transformation Conjugation A = P D P^-1 (MML §2.7.2)."""

    def setUp(self):
        # Canonical matrix A = [[2, 1], [1, 2]]
        self.A = np.array([[2.0, 1.0], [1.0, 2.0]])
        self.v1 = np.array([1.0, 1.0])
        self.v2 = np.array([-1.0, 1.0])
        self.P = np.column_stack([self.v1, self.v2])
        self.P_inv = np.linalg.inv(self.P)
        self.D = np.diag([3.0, 1.0])

    def test_similarity_conjugation_equality(self):
        """Feature 3.1: A == P @ D @ P^-1 holds exactly."""
        reconstructed_A = self.P @ self.D @ self.P_inv
        np.testing.assert_allclose(reconstructed_A, self.A, atol=1e-12)

    def test_diagonal_form_extraction(self):
        """Feature 3.2: D == P^-1 @ A @ P extracts exact eigenvalues on diagonal [3, 1]."""
        extracted_D = self.P_inv @ self.A @ self.P
        np.testing.assert_allclose(extracted_D, self.D, atol=1e-12)

    def test_eigenvector_action(self):
        """Feature 3.3: A @ v1 == 3*v1 and A @ v2 == 1*v2."""
        np.testing.assert_allclose(self.A @ self.v1, 3.0 * self.v1, atol=1e-12)
        np.testing.assert_allclose(self.A @ self.v2, 1.0 * self.v2, atol=1e-12)

    def test_similarity_reflexivity_and_identity(self):
        """Feature 3.4: I^-1 @ A @ I == A (every matrix is similar to itself)."""
        I = np.eye(2)
        np.testing.assert_allclose(I @ self.A @ I, self.A, atol=1e-12)

    def test_similarity_transitivity(self):
        """Feature 3.5: Similarity is an equivalence relation: Q^-1 (P^-1 A P) Q == (PQ)^-1 A (PQ)."""
        Q = np.array([[1.0, 2.0], [0.0, 1.0]])
        Q_inv = np.linalg.inv(Q)
        lhs = Q_inv @ (self.P_inv @ self.A @ self.P) @ Q
        PQ = self.P @ Q
        PQ_inv = np.linalg.inv(PQ)
        rhs = PQ_inv @ self.A @ PQ
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)


class TestTier1DiagonalScalingAction(unittest.TestCase):
    """Tier 1 Feature 4: Decoupled Diagonal Axis Scaling (MML §2.7.2, Kreyszig §8.4)."""

    def setUp(self):
        self.D = np.diag([3.0, 1.0])
        self.v1 = np.array([1.0, 1.0])
        self.v2 = np.array([-1.0, 1.0])

    def test_decoupled_diagonal_action(self):
        """Feature 4.1: D scales coordinates independently without cross-axis coupling: D [c1, c2]^T == [3c1, c2]^T."""
        coords = np.array([2.5, -4.0])
        scaled = self.D @ coords
        expected = np.array([3.0 * 2.5, 1.0 * (-4.0)])
        np.testing.assert_allclose(scaled, expected, atol=1e-12)

    def test_first_eigenaxis_stretch(self):
        """Feature 4.2: Component along first eigenaxis stretches threefold (lambda1 = 3)."""
        coords = np.array([1.0, 0.0])
        np.testing.assert_allclose(self.D @ coords, np.array([3.0, 0.0]), atol=1e-12)

    def test_second_eigenaxis_invariance(self):
        """Feature 4.3: Component along second eigenaxis remains completely invariant (lambda2 = 1)."""
        coords = np.array([0.0, 1.0])
        np.testing.assert_allclose(self.D @ coords, np.array([0.0, 1.0]), atol=1e-12)

    def test_diagonal_area_scaling(self):
        """Feature 4.4: Area scaling under diagonal matrix equals det(D) == lambda1 * lambda2 == 3.0."""
        det_D = np.linalg.det(self.D)
        self.assertAlmostEqual(det_D, 3.0, places=12)

    def test_diagonal_matrix_powers(self):
        """Feature 4.5: D^k == diag(3^k, 1^k) evaluates in O(1) scalar arithmetic."""
        k = 4
        D_k = np.linalg.matrix_power(self.D, k)
        expected = np.diag([3.0**k, 1.0**k])
        np.testing.assert_allclose(D_k, expected, atol=1e-12)


class TestTier1ThreeStepSequence(unittest.TestCase):
    """Tier 1 Feature 5: 3-Step Geometric Dance P^-1 -> D -> P (MML §2.7.2)."""

    def setUp(self):
        self.A = np.array([[2.0, 1.0], [1.0, 2.0]])
        self.v1 = np.array([1.0, 1.0])
        self.v2 = np.array([-1.0, 1.0])
        self.P = np.column_stack([self.v1, self.v2])
        self.P_inv = np.linalg.inv(self.P)
        self.D = np.diag([3.0, 1.0])
        self.x = np.array([2.0, 0.0])

    def test_three_step_canonical_point(self):
        """Feature 5.1: Vector x = [2, 0]^T maps to [4, 2]^T identically via direct Ax and 3-step sequence."""
        direct_out = self.A @ self.x
        seq_out = self.P @ (self.D @ (self.P_inv @ self.x))
        np.testing.assert_allclose(direct_out, np.array([4.0, 2.0]), atol=1e-12)
        np.testing.assert_allclose(seq_out, np.array([4.0, 2.0]), atol=1e-12)

    def test_step1_to_eigen_frame(self):
        """Feature 5.2: Step 1 (P^-1): Translates [2, 0]^T into eigen-coordinates [1, -1]^T."""
        x_eigen = self.P_inv @ self.x
        np.testing.assert_allclose(x_eigen, np.array([1.0, -1.0]), atol=1e-12)

    def test_step2_diagonal_stretch(self):
        """Feature 5.3: Step 2 (D): Decoupled stretching maps [1, -1]^T to [3, -1]^T."""
        x_eigen = self.P_inv @ self.x
        stretched_eigen = self.D @ x_eigen
        np.testing.assert_allclose(stretched_eigen, np.array([3.0, -1.0]), atol=1e-12)

    def test_step3_back_to_standard_frame(self):
        """Feature 5.4: Step 3 (P): Restores [3, -1]^T to standard coordinates [4, 2]^T."""
        stretched_eigen = np.array([3.0, -1.0])
        x_final = self.P @ stretched_eigen
        np.testing.assert_allclose(x_final, np.array([4.0, 2.0]), atol=1e-12)

    def test_random_vector_three_step_equivalence(self):
        """Feature 5.5: A @ x == P @ (D @ (P^-1 @ x)) holds across 100 random vectors."""
        rng = np.random.default_rng(402)
        batch = rng.uniform(-100.0, 100.0, size=(100, 2))
        direct_batch = batch @ self.A.T
        seq_batch = ((batch @ self.P_inv.T) @ self.D.T) @ self.P.T
        np.testing.assert_allclose(direct_batch, seq_batch, atol=1e-12)


# =============================================================================
# TIER 2: BOUNDARY & CORNER CASES
# =============================================================================

class TestTier2DefectiveMatrixNonDiagonalizable(unittest.TestCase):
    """Tier 2 Boundary 1: Defective Matrix & Geometric Multiplicity Deficiency (Kreyszig §8.4)."""

    def setUp(self):
        # Jordan block: algebraic multiplicity 2, geometric multiplicity 1
        self.M_def = np.array([[2.0, 1.0], [0.0, 2.0]])

    def test_defective_matrix_characteristic_equation(self):
        """Boundary 1.1: Characteristic polynomial has repeated root lambda = 2 with algebraic multiplicity 2."""
        eigvals = np.linalg.eigvals(self.M_def)
        np.testing.assert_allclose(eigvals, [2.0, 2.0], atol=1e-12)

    def test_defective_matrix_geometric_multiplicity(self):
        """Boundary 1.2: Eigenspace dimension (nullity of M - 2I) is 1 < 2."""
        M_minus_2I = self.M_def - 2.0 * np.eye(2)
        rank = np.linalg.matrix_rank(M_minus_2I)
        self.assertEqual(rank, 1)
        # nullity = 2 - rank = 1
        nullity = 2 - rank
        self.assertEqual(nullity, 1)

    def test_defective_matrix_single_eigenspace(self):
        """Boundary 1.3: All eigenvectors lie on 1D horizontal line span([1, 0]^T)."""
        _, eigvecs = np.linalg.eig(self.M_def)
        # Both columns of eigvecs must be collinear with [1, 0]^T
        for j in range(2):
            v = eigvecs[:, j]
            self.assertAlmostEqual(v[1], 0.0, places=10)

    def test_defective_matrix_non_diagonalizability(self):
        """Boundary 1.4: Matrix cannot be diagonalized: eigenvector matrix is singular."""
        _, eigvecs = np.linalg.eig(self.M_def)
        det_eigvecs = np.linalg.det(eigvecs)
        self.assertAlmostEqual(det_eigvecs, 0.0, places=10)

    def test_jordan_block_powers(self):
        """Boundary 1.5: Jordan block power formula: M^k == [[2^k, k*2^(k-1)], [0, 2^k]]."""
        k = 5
        M_k = np.linalg.matrix_power(self.M_def, k)
        expected = np.array([
            [2.0**k, k * (2.0**(k - 1))],
            [0.0, 2.0**k]
        ])
        np.testing.assert_allclose(M_k, expected, atol=1e-12)


class TestTier2RepeatedEigenvaluesScalar(unittest.TestCase):
    """Tier 2 Boundary 2: Scalar Matrix Full Eigenspace c*I (MML §2.7)."""

    def setUp(self):
        self.c = 4.0
        self.A_scalar = self.c * np.eye(2)

    def test_scalar_matrix_full_eigenspace(self):
        """Boundary 2.1: Algebraic multiplicity == 2 and geometric multiplicity == 2."""
        rank_A_minus_cI = np.linalg.matrix_rank(self.A_scalar - self.c * np.eye(2))
        self.assertEqual(rank_A_minus_cI, 0)
        nullity = 2 - rank_A_minus_cI
        self.assertEqual(nullity, 2)

    def test_scalar_matrix_every_vector_is_eigenvector(self):
        """Boundary 2.2: Every non-zero vector in R^2 is an eigenvector with eigenvalue c."""
        rng = np.random.default_rng(403)
        pts = rng.uniform(-10.0, 10.0, size=(20, 2))
        for v in pts:
            np.testing.assert_allclose(self.A_scalar @ v, self.c * v, atol=1e-12)

    def test_scalar_matrix_similarity_invariance(self):
        """Boundary 2.3: P^-1 (c*I) P == c*I for any invertible matrix P."""
        P_random = np.array([[2.0, 5.0], [-1.0, 3.0]])
        P_inv = np.linalg.inv(P_random)
        conjugated = P_inv @ self.A_scalar @ P_random
        np.testing.assert_allclose(conjugated, self.A_scalar, atol=1e-12)

    def test_scalar_matrix_isotropic_scaling(self):
        """Boundary 2.4: Isotropic transformation preserves circular aspect ratio without shearing."""
        theta = np.linspace(0, 2 * math.pi, 20)
        circle_pts = np.column_stack([np.cos(theta), np.sin(theta)])
        mapped = circle_pts @ self.A_scalar.T
        radii = np.linalg.norm(mapped, axis=1)
        np.testing.assert_allclose(radii, self.c, atol=1e-12)

    def test_scalar_matrix_determinant_and_trace(self):
        """Boundary 2.5: det(c*I) == c^2 == 16.0 and Tr(c*I) == 2c == 8.0."""
        self.assertAlmostEqual(np.linalg.det(self.A_scalar), self.c**2, places=12)
        self.assertAlmostEqual(np.trace(self.A_scalar), 2.0 * self.c, places=12)


class TestTier2OrthogonalEigenvectorsSymmetric(unittest.TestCase):
    """Tier 2 Boundary 3: Symmetric Matrix Orthogonal Eigenspaces (MML §2.7.2)."""

    def setUp(self):
        self.A = np.array([[2.0, 1.0], [1.0, 2.0]])

    def test_symmetric_matrix_eigenvectors_orthogonal(self):
        """Boundary 3.1: Distinct eigenvectors of symmetric matrix are strictly orthogonal: v1 . v2 == 0."""
        v1 = np.array([1.0, 1.0])
        v2 = np.array([-1.0, 1.0])
        self.assertAlmostEqual(np.dot(v1, v2), 0.0, places=12)

    def test_normalized_eigenbasis_orthogonal_matrix(self):
        """Boundary 3.2: Orthonormal eigenvector matrix Q satisfies Q^T Q == I and Q^-1 == Q^T."""
        q1 = np.array([1.0, 1.0]) / math.sqrt(2.0)
        q2 = np.array([-1.0, 1.0]) / math.sqrt(2.0)
        Q = np.column_stack([q1, q2])
        np.testing.assert_allclose(Q.T @ Q, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(np.linalg.inv(Q), Q.T, atol=1e-12)

    def test_spectral_theorem_outer_product_decomposition(self):
        """Boundary 3.3: Spectral decomposition A == lambda1 * q1 q1^T + lambda2 * q2 q2^T."""
        q1 = np.array([1.0, 1.0]) / math.sqrt(2.0)
        q2 = np.array([-1.0, 1.0]) / math.sqrt(2.0)
        lambda1, lambda2 = 3.0, 1.0
        spectral_A = lambda1 * np.outer(q1, q1) + lambda2 * np.outer(q2, q2)
        np.testing.assert_allclose(spectral_A, self.A, atol=1e-12)

    def test_orthogonal_similarity_conjugation(self):
        """Boundary 3.4: Q^T A Q == diag(3, 1)."""
        q1 = np.array([1.0, 1.0]) / math.sqrt(2.0)
        q2 = np.array([-1.0, 1.0]) / math.sqrt(2.0)
        Q = np.column_stack([q1, q2])
        D = Q.T @ self.A @ Q
        np.testing.assert_allclose(D, np.diag([3.0, 1.0]), atol=1e-12)

    def test_symmetric_real_eigenvalues(self):
        """Boundary 3.5: Eigenvalues of real symmetric matrix are strictly real."""
        eigvals = np.linalg.eigvals(self.A)
        self.assertTrue(np.all(np.isreal(eigvals)))


class TestTier2ZeroEigenvalueSingularSimilarity(unittest.TestCase):
    """Tier 2 Boundary 4: Zero Eigenvalue / Singular Matrix Similarity (Kreyszig §8.4)."""

    def setUp(self):
        # Singular symmetric matrix with eigenvalues 2 and 0
        self.A_sing = np.array([[1.0, 1.0], [1.0, 1.0]])

    def test_zero_eigenvalue_spectrum(self):
        """Boundary 4.1: Eigenvalues are lambda1 = 2.0 and lambda2 = 0.0."""
        eigvals = np.sort(np.linalg.eigvals(self.A_sing))
        np.testing.assert_allclose(eigvals, [0.0, 2.0], atol=1e-12)

    def test_singular_matrix_diagonalizability(self):
        """Boundary 4.2: Singular matrix is diagonalizable: A_sing == P @ diag(2, 0) @ P^-1."""
        v1 = np.array([1.0, 1.0])   # lambda = 2
        v2 = np.array([-1.0, 1.0])  # lambda = 0 (null space)
        P = np.column_stack([v1, v2])
        D = np.diag([2.0, 0.0])
        recon = P @ D @ np.linalg.inv(P)
        np.testing.assert_allclose(recon, self.A_sing, atol=1e-12)

    def test_zero_eigenvalue_nullspace_collapse(self):
        """Boundary 4.3: Eigenvector v2 corresponding to lambda = 0 spans the null space N(A)."""
        v2 = np.array([-1.0, 1.0])
        np.testing.assert_allclose(self.A_sing @ v2, np.zeros(2), atol=1e-12)

    def test_zero_eigenvalue_determinant_zero(self):
        """Boundary 4.4: det(A_sing) == lambda1 * lambda2 == 2.0 * 0.0 == 0.0."""
        self.assertAlmostEqual(np.linalg.det(self.A_sing), 0.0, places=12)

    def test_zero_eigenvalue_rank_conservation(self):
        """Boundary 4.5: Matrix rank equals count of non-zero eigenvalues (1)."""
        rank = np.linalg.matrix_rank(self.A_sing)
        self.assertEqual(rank, 1)


class TestTier2NegativeEigenvaluesOrientationReversal(unittest.TestCase):
    """Tier 2 Boundary 5: Negative Eigenvalue & Hyperbolic Reflection (MML §2.7)."""

    def setUp(self):
        # Saddle / hyperbolic scaling: stretches x by 2, inverts y by -1
        self.D_neg = np.diag([2.0, -1.0])
        self.P = np.array([[1.0, 1.0], [0.0, 1.0]])  # Shear basis
        self.A_neg = self.P @ self.D_neg @ np.linalg.inv(self.P)

    def test_negative_eigenvalue_determinant(self):
        """Boundary 5.1: det(A) == lambda1 * lambda2 == 2 * (-1) == -2.0 (orientation inverted)."""
        self.assertAlmostEqual(np.linalg.det(self.A_neg), -2.0, places=12)

    def test_negative_eigenvalue_axis_reflection(self):
        """Boundary 5.2: Negative eigenvalue reverses direction along corresponding eigenvector: A v == -v."""
        v2 = self.P[:, 1]  # Second column of P
        np.testing.assert_allclose(self.A_neg @ v2, -1.0 * v2, atol=1e-12)

    def test_negative_eigenvalue_trace_and_spectrum(self):
        """Boundary 5.3: Tr(A) == lambda1 + lambda2 == 2 + (-1) == 1.0."""
        self.assertAlmostEqual(np.trace(self.A_neg), 1.0, places=12)

    def test_negative_eigenvalue_parity_inversion(self):
        """Boundary 5.4: Basis orientation parity (2D cross product) changes sign under A_neg."""
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])
        u = self.A_neg @ e1
        v = self.A_neg @ e2
        cross_out = u[0] * v[1] - u[1] * v[0]
        self.assertLess(cross_out, 0.0)
        self.assertAlmostEqual(cross_out, -2.0, places=12)

    def test_even_and_odd_powers_orientation(self):
        """Boundary 5.5: Odd powers have det < 0 (orientation inverted); even powers have det > 0."""
        A_sq = self.A_neg @ self.A_neg
        A_cube = A_sq @ self.A_neg
        self.assertGreater(np.linalg.det(A_sq), 0.0)
        self.assertLess(np.linalg.det(A_cube), 0.0)


# =============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS
# =============================================================================

class TestTier3CrossFeatureCombinations(unittest.TestCase):
    """Tier 3: Invariant Quantities Under Similarity & Spectral Invariants (Kreyszig §8.4)."""

    def setUp(self):
        self.A = np.array([[2.0, 1.0], [1.0, 2.0]])
        self.rng = np.random.default_rng(404)

    def test_determinant_similarity_invariant(self):
        """Cross-Feature 1: det(P^-1 A P) == det(A) == 3.0 across 50 random change-of-basis matrices P."""
        det_A = np.linalg.det(self.A)
        for _ in range(50):
            P = self.rng.uniform(-5.0, 5.0, size=(2, 2))
            while abs(np.linalg.det(P)) < 0.1:
                P = self.rng.uniform(-5.0, 5.0, size=(2, 2))
            P_inv = np.linalg.inv(P)
            conjugated = P_inv @ self.A @ P
            self.assertAlmostEqual(np.linalg.det(conjugated), det_A, places=8)

    def test_trace_similarity_invariant(self):
        """Cross-Feature 2: Tr(P^-1 A P) == Tr(A) == 4.0 across 50 random change-of-basis matrices P."""
        tr_A = np.trace(self.A)
        for _ in range(50):
            P = self.rng.uniform(-5.0, 5.0, size=(2, 2))
            while abs(np.linalg.det(P)) < 0.1:
                P = self.rng.uniform(-5.0, 5.0, size=(2, 2))
            P_inv = np.linalg.inv(P)
            conjugated = P_inv @ self.A @ P
            self.assertAlmostEqual(np.trace(conjugated), tr_A, places=8)

    def test_characteristic_polynomial_similarity_invariant(self):
        """Cross-Feature 3: det(lambda*I - P^-1 A P) == det(lambda*I - A) across 10 sample lambda values."""
        P = np.array([[3.0, 1.0], [-2.0, 4.0]])
        P_inv = np.linalg.inv(P)
        A_tilde = P_inv @ self.A @ P
        for lam in np.linspace(-5.0, 5.0, 10):
            poly_orig = np.linalg.det(lam * np.eye(2) - self.A)
            poly_tilde = np.linalg.det(lam * np.eye(2) - A_tilde)
            self.assertAlmostEqual(poly_orig, poly_tilde, places=10)

    def test_eigenvalue_spectrum_invariance(self):
        """Cross-Feature 4: Eigenvalues of P^-1 A P match eigenvalues of A exactly."""
        P = np.array([[-1.0, 2.0], [3.0, -1.0]])
        P_inv = np.linalg.inv(P)
        A_tilde = P_inv @ self.A @ P
        eigs_orig = np.sort(np.linalg.eigvals(self.A))
        eigs_tilde = np.sort(np.linalg.eigvals(A_tilde))
        np.testing.assert_allclose(eigs_orig, eigs_tilde, atol=1e-12)

    def test_matrix_powers_via_diagonalization(self):
        """Cross-Feature 5: Fast exponentiation: A^k == P @ D^k @ P^-1 for k in {2, 3, 5, 10}."""
        v1 = np.array([1.0, 1.0])
        v2 = np.array([-1.0, 1.0])
        P = np.column_stack([v1, v2])
        P_inv = np.linalg.inv(P)
        D = np.diag([3.0, 1.0])
        for k in [2, 3, 5, 10]:
            A_k_direct = np.linalg.matrix_power(self.A, k)
            A_k_diag = P @ np.diag([3.0**k, 1.0**k]) @ P_inv
            np.testing.assert_allclose(A_k_direct, A_k_diag, atol=1e-10)


# =============================================================================
# TIER 4: REAL-WORLD SCENARIOS
# =============================================================================

class TestTier4RealWorldScenarios(unittest.TestCase):
    """Tier 4: ML PCA & Graph Laplacians, and GATE Linear Algebra (MML §2.7)."""

    def setUp(self):
        self.rng = np.random.default_rng(405)

    def test_ml_pca_covariance_diagonalization(self):
        """Real-World 1: Principal Component Analysis (PCA): Covariance Sigma = Q Lambda Q^T decorrelates features."""
        N, d = 500, 2
        # Correlated 2D data
        true_cov = np.array([[4.0, 2.0], [2.0, 3.0]])
        L = np.linalg.cholesky(true_cov)
        X = self.rng.standard_normal((N, d)) @ L.T
        X_centered = X - np.mean(X, axis=0)

        # Sample covariance
        Sigma = (X_centered.T @ X_centered) / (N - 1)

        # Eigendecomposition: Sigma == Q @ Lambda @ Q^T
        eigvals, Q = np.linalg.eigh(Sigma)
        # Transform data to principal components
        Z = X_centered @ Q
        # Covariance of transformed data must be diagonal
        Sigma_Z = (Z.T @ Z) / (N - 1)
        off_diag = abs(Sigma_Z[0, 1])
        self.assertAlmostEqual(off_diag, 0.0, places=10)
        np.testing.assert_allclose(np.diag(Sigma_Z), eigvals, atol=1e-10)

    def test_ml_svd_relation_to_covariance(self):
        """Real-World 2: SVD connection to Gram matrix: X^T X == V Sigma_svd^2 V^T."""
        X = self.rng.standard_normal((10, 3))
        Gram = X.T @ X
        U, s, Vt = np.linalg.svd(X)
        V = Vt.T

        recon_Gram = V @ np.diag(s**2) @ V.T
        np.testing.assert_allclose(recon_Gram, Gram, atol=1e-12)

    def test_ml_spectral_graph_laplacian_decoupling(self):
        """Real-World 3: Graph Convolution: Normalized Graph Laplacian eigendecomposition decouples graph signals."""
        # Adjacency matrix of 3-node path graph: 0 - 1 - 2
        Adj = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        Deg = np.diag([1.0, 2.0, 1.0])
        Laplacian = Deg - Adj

        # Eigendecomposition
        eigvals, U = np.linalg.eigh(Laplacian)
        # Graph Laplacian is positive semi-definite: all eigenvalues >= 0
        self.assertTrue(np.all(eigvals >= -1e-12))
        # Smallest eigenvalue is 0 (connected component indicator)
        self.assertAlmostEqual(eigvals[0], 0.0, places=12)

        # Decoupling property: U^T L U == diag(eigvals)
        np.testing.assert_allclose(U.T @ Laplacian @ U, np.diag(eigvals), atol=1e-12)

    def test_gate_cayley_hamilton_theorem(self):
        """Real-World 4: GATE CSE Cayley-Hamilton Theorem: p(A) == A^2 - 4A + 3I == 0 and inverse computation."""
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        I = np.eye(2)
        # Characteristic polynomial: p(lambda) = lambda^2 - 4*lambda + 3
        # By Cayley-Hamilton: A^2 - 4*A + 3*I == 0
        p_A = A @ A - 4.0 * A + 3.0 * I
        np.testing.assert_allclose(p_A, 0.0, atol=1e-12)

        # Fast inverse formula: A^-1 = (1/3) * (4*I - A)
        A_inv_formula = (1.0 / 3.0) * (4.0 * I - A)
        A_inv_direct = np.linalg.inv(A)
        np.testing.assert_allclose(A_inv_formula, A_inv_direct, atol=1e-12)

    def test_gate_diagonalizability_criteria(self):
        """Real-World 5: GATE Linear Algebra: Complete diagonalizability criteria verification."""
        # Case A: Distinct eigenvalues -> always diagonalizable
        A_distinct = np.array([[1.0, 2.0], [0.0, 3.0]])
        self.assertEqual(len(set(np.linalg.eigvals(A_distinct))), 2)

        # Case B: Symmetric matrix -> always orthogonally diagonalizable
        A_sym = np.array([[3.0, -1.0], [-1.0, 3.0]])
        np.testing.assert_allclose(A_sym, A_sym.T)

        # Case C: Defective matrix -> not diagonalizable (alg mult 2 != geom mult 1)
        A_defective = np.array([[5.0, 1.0], [0.0, 5.0]])
        rank_diff = np.linalg.matrix_rank(A_defective - 5.0 * np.eye(2))
        geom_mult = 2 - rank_diff
        self.assertEqual(geom_mult, 1)


# =============================================================================
# TIER 5: STATIC AST & SYNTAX INTEGRITY
# =============================================================================

class TestSprint04ASTAndSyntax(unittest.TestCase):
    """Tier 5: Static AST verification of Sprint 4 scene code for ManimCE standards."""

    SCENE_PATH = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "linear_algebra",
            "sprint_04_change_of_basis",
            "scene_change_of_basis.py"
        )
    )

    def _get_ast(self):
        if not os.path.exists(self.SCENE_PATH):
            self.skipTest(
                f"Scene script '{self.SCENE_PATH}' not yet created (Pending Milestone 4)."
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
            f"Found deprecated symbols in Sprint 4 scene: {found_deprecated}"
        )

    def test_scene_class_defined(self):
        """AST 2: Master scene class ChangeOfBasisScene is defined."""
        tree = self._get_ast()
        class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        self.assertIn("ChangeOfBasisScene", class_names)

    def test_hud_panel_creation_present(self):
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
            "Sprint 4 scene must define or call 'make_hud_panel'."
        )

    def test_eigen_basis_constants_defined(self):
        """AST 4: Eigenvector constants or coordinate plane variables are declared."""
        tree = self._get_ast()
        assigned_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigned_names.add(target.id)
        # Check standard Sprint 4 naming
        has_expected_var = any(name in assigned_names for name in ["V1", "V2", "EIGEN_V1", "EIGEN_V2", "P_MATRIX", "A_MATRIX"])
        self.assertTrue(
            has_expected_var or len(assigned_names) > 5,
            "Scene should declare matrix / eigenvector configuration constants."
        )

    def test_no_forbidden_execution_calls(self):
        """AST 5: Scene script contains zero calls to eval or exec."""
        tree = self._get_ast()
        forbidden = {"eval", "exec"}
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in forbidden:
                    found.append((node.func.id, getattr(node, "lineno", "?")))
        self.assertEqual(found, [], f"Forbidden functions detected: {found}")


if __name__ == "__main__":
    unittest.main()
