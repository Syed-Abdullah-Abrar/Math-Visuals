#!/usr/bin/env python3
"""
Mathematical Invariant & Structural Verification Suite for Sprint 5
==================================================================
Curriculum: Norms (L1, L2, L_inf), Inner Products & Cauchy-Schwarz
References: MML Ch. 3.1-3.4 & 3.8.1, VMLS Ch. 3-4, ILA Ch. 3, Kreyszig Ch. 7-8

Tiers:
- Tier 1: Core mathematical definitions & canonical invariants (25 tests)
- Tier 2: Boundary, degenerate & numerical stability cases (25 tests)
- Tier 3: Cross-feature combinations & dualities (5 tests)
- Tier 4: Frontier ML & GATE CS/DA real-world scenarios (5 tests)
- Tier 5: Static AST & ManimCE syntax integrity (7 tests)
Total: 67 tests.
"""

from __future__ import annotations

import ast
import math
import os
import unittest
import numpy as np


def _overflow_safe_lp(x: np.ndarray, p: float) -> float:
    """Compute ||x||_p safely without overflow/underflow."""
    abs_x = np.abs(x).astype(np.float64)
    m = float(np.max(abs_x))
    if m == 0.0:
        return 0.0
    if np.isinf(p):
        return m
    scaled = abs_x / m
    return float(m * (np.sum(scaled ** p) ** (1.0 / p)))


def _quasi_lp(x: np.ndarray, p: float) -> float:
    """Compute quasi-norm (sum |x_i|^p)^(1/p) for 0 < p < 1."""
    return float(np.sum(np.abs(x) ** p) ** (1.0 / p))


def _rms(x: np.ndarray) -> float:
    return float(np.linalg.norm(x, ord=2) / np.sqrt(len(x)))


# =============================================================================
# TIER 1: CORE MATHEMATICAL FEATURES (25 tests)
# =============================================================================


class TestTier1LpNormFamily(unittest.TestCase):
    """Tier 1.1: Lp norm definitions, axioms, ordering, and unit-ball geometry."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(5101)

    def test_canonical_puzzle_and_vec_norms(self):
        x_puzzle = np.array([3.0, 4.0])
        self.assertAlmostEqual(np.linalg.norm(x_puzzle, ord=1), 7.0)
        self.assertAlmostEqual(np.linalg.norm(x_puzzle, ord=2), 5.0)
        self.assertAlmostEqual(np.linalg.norm(x_puzzle, ord=np.inf), 4.0)

        x_vec = np.array([1.0, 2.0])
        y_vec = np.array([2.0, 1.0])
        self.assertAlmostEqual(np.linalg.norm(x_vec, ord=1), 3.0)
        self.assertAlmostEqual(np.linalg.norm(x_vec, ord=2), math.sqrt(5.0))
        self.assertAlmostEqual(np.linalg.norm(y_vec, ord=np.inf), 2.0)

    def test_norm_axioms_l1_l2_linf(self):
        X = self.rng.normal(size=(50, 5))
        Y = self.rng.normal(size=(50, 5))
        lam = -3.75
        for p in (1, 2, np.inf):
            nx = np.linalg.norm(X, ord=p, axis=1)
            ny = np.linalg.norm(Y, ord=p, axis=1)
            nxy = np.linalg.norm(X + Y, ord=p, axis=1)
            nlam = np.linalg.norm(lam * X, ord=p, axis=1)
            self.assertTrue(np.all(nx > 0.0))
            self.assertTrue(np.allclose(nlam, abs(lam) * nx))
            self.assertTrue(np.all(nxy <= nx + ny + 1e-12))

    def test_monotonic_lp_ordering(self):
        X = self.rng.normal(size=(100, 6))
        linf = np.linalg.norm(X, ord=np.inf, axis=1)
        l2 = np.linalg.norm(X, ord=2, axis=1)
        l1 = np.linalg.norm(X, ord=1, axis=1)
        self.assertTrue(np.all(linf <= l2 + 1e-12))
        self.assertTrue(np.all(l2 <= l1 + 1e-12))
        self.assertTrue(np.all(l1 <= np.sqrt(6.0) * l2 + 1e-12))
        self.assertTrue(np.all(l2 <= np.sqrt(6.0) * linf + 1e-12))

    def test_unit_ball_diagonal_vertex_sweep(self):
        radii = []
        for p in (1.0, 1.5, 2.0, 4.0, 20.0):
            coord = 2.0 ** (-1.0 / p)
            pt = np.array([coord, coord])
            self.assertAlmostEqual(float(np.linalg.norm(pt, ord=p)), 1.0, places=12)
            radii.append(coord)
        self.assertTrue(all(radii[i] < radii[i + 1] for i in range(len(radii) - 1)))

    def test_unit_ball_axis_vertices_invariant(self):
        axes = [np.array([1.0, 0.0]), np.array([-1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, -1.0])]
        for v in axes:
            for p in (1, 2, 5, 50, np.inf):
                self.assertAlmostEqual(float(np.linalg.norm(v, ord=p)), 1.0, places=12)


class TestTier1InducedMetricsAndRMS(unittest.TestCase):
    """Tier 1.2: Induced metrics, RMS norm, standardization, and Chebyshev bound."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(5102)

    def test_metric_axioms_across_norms(self):
        x = self.rng.normal(size=7)
        y = self.rng.normal(size=7)
        z = self.rng.normal(size=7)
        for p in (1, 2, np.inf):
            dxy = float(np.linalg.norm(x - y, ord=p))
            dyx = float(np.linalg.norm(y - x, ord=p))
            dxz = float(np.linalg.norm(x - z, ord=p))
            dzy = float(np.linalg.norm(z - y, ord=p))
            self.assertGreater(dxy, 0.0)
            self.assertAlmostEqual(dxy, dyx)
            self.assertLessEqual(dxy, dxz + dzy + 1e-12)
            self.assertAlmostEqual(float(np.linalg.norm(x - x, ord=p)), 0.0)

    def test_rms_and_pythagorean_variance_split(self):
        # VMLS §3.3 vector [1, -2, 3, 2]: exact std = sqrt(3.5) (not VMLS printed 1.872 typo)
        x = np.array([1.0, -2.0, 3.0, 2.0])
        avg_x = float(np.mean(x))
        std_x = _rms(x - avg_x)
        rms_x = _rms(x)
        self.assertAlmostEqual(avg_x, 1.0)
        self.assertAlmostEqual(std_x, math.sqrt(3.5), places=12)
        self.assertAlmostEqual(rms_x ** 2, avg_x ** 2 + std_x ** 2, places=12)

    def test_z_score_standardization_sphere(self):
        x = self.rng.normal(loc=5.0, scale=3.0, size=12)
        z = (x - np.mean(x)) / _rms(x - np.mean(x))
        self.assertAlmostEqual(float(np.mean(z)), 0.0, places=12)
        self.assertAlmostEqual(_rms(z), 1.0, places=12)
        self.assertAlmostEqual(float(np.linalg.norm(z)), math.sqrt(len(x)), places=12)

    def test_chebyshev_fraction_and_converse_bound(self):
        x = self.rng.normal(size=200)
        rms_x = _rms(x)
        for a in (0.5, 1.0, 2.0, 3.0):
            frac = float(np.mean(np.abs(x) >= a))
            self.assertLessEqual(frac, (rms_x / a) ** 2 + 1e-12)
        self.assertGreaterEqual(float(np.max(np.abs(x))), rms_x - 1e-12)

    def test_vmls_distance_and_reverse_triangle(self):
        u = np.array([1.8, 2.0, -3.7, 4.7])
        v = np.array([0.6, 2.1, 1.9, -1.4])
        w = np.array([2.0, 1.9, -4.0, 4.6])
        self.assertLessEqual(np.linalg.norm(u - v), np.linalg.norm(u - w) + np.linalg.norm(w - v))
        for p in (1, 2, np.inf):
            lhs = abs(np.linalg.norm(u, ord=p) - np.linalg.norm(v, ord=p))
            rhs = np.linalg.norm(u - v, ord=p)
            self.assertLessEqual(lhs, rhs + 1e-12)


class TestTier1DotProductAndProjection(unittest.TestCase):
    """Tier 1.3: Euclidean dot product, ruler projection, and orthogonal residual."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(5103)

    def test_dot_product_bilinearity_and_symmetry(self):
        x = self.rng.normal(size=5)
        y = self.rng.normal(size=5)
        z = self.rng.normal(size=5)
        a, b = 2.5, -1.3
        self.assertAlmostEqual(float(np.dot(x, y)), float(np.dot(y, x)))
        self.assertAlmostEqual(float(np.dot(a * x + b * z, y)), a * float(np.dot(x, y)) + b * float(np.dot(z, y)))

    def test_canonical_ruler_projections(self):
        x_p, y_p = np.array([3.0, 4.0]), np.array([2.0, 1.0])
        proj_p = (np.dot(x_p, y_p) / np.dot(y_p, y_p)) * y_p
        resid_p = x_p - proj_p
        self.assertTrue(np.allclose(proj_p, np.array([4.0, 2.0])))
        self.assertTrue(np.allclose(resid_p, np.array([-1.0, 2.0])))
        self.assertAlmostEqual(float(np.dot(resid_p, y_p)), 0.0)

        x_v, y_v = np.array([1.0, 2.0]), np.array([2.0, 1.0])
        proj_v = (np.dot(x_v, y_v) / np.dot(y_v, y_v)) * y_v
        resid_v = x_v - proj_v
        self.assertTrue(np.allclose(proj_v, np.array([1.6, 0.8])))
        self.assertTrue(np.allclose(resid_v, np.array([-0.6, 1.2])))
        self.assertAlmostEqual(float(np.dot(resid_v, y_v)), 0.0)

    def test_signed_scalar_projection_relation(self):
        y_ruler = np.array([1.6, 1.2])
        u_ruler = y_ruler / np.linalg.norm(y_ruler)
        for _ in range(30):
            x = self.rng.normal(size=2)
            signed_shadow = float(np.dot(x, u_ruler))
            self.assertAlmostEqual(float(np.dot(x, y_ruler)), signed_shadow * float(np.linalg.norm(y_ruler)))

    def test_canonical_345_angle(self):
        x = np.array([1.0, 2.0])
        y = np.array([2.0, 1.0])
        cos_w = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
        self.assertAlmostEqual(cos_w, 0.8, places=12)
        self.assertAlmostEqual(math.sin(math.acos(cos_w)), 0.6, places=12)

    def test_correlation_coefficient_equivalence(self):
        a = self.rng.normal(size=25)
        b = 2.0 * a + self.rng.normal(scale=0.5, size=25)
        a_tilde = a - np.mean(a)
        b_tilde = b - np.mean(b)
        rho = float(np.dot(a_tilde, b_tilde) / (np.linalg.norm(a_tilde) * np.linalg.norm(b_tilde)))
        self.assertAlmostEqual(rho, float(np.corrcoef(a, b)[0, 1]), places=12)
        self.assertGreaterEqual(rho, -1.0 - 1e-12)
        self.assertLessEqual(rho, 1.0 + 1e-12)


class TestTier1GeneralSPDInnerProducts(unittest.TestCase):
    """Tier 1.4: SPD inner product <x,y>_A = x^T A y, Cholesky whitening, A-ellipsoid."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(5104)
        self.A_spd = np.array([[2.0, 1.0], [1.0, 2.0]])

    def test_a_spd_spectrum_and_symmetry(self):
        evals = np.linalg.eigvalsh(self.A_spd)
        self.assertTrue(np.allclose(self.A_spd, self.A_spd.T))
        self.assertTrue(np.allclose(evals, np.array([1.0, 3.0])))
        self.assertTrue(np.all(evals > 0.0))

    def test_spd_inner_product_axioms(self):
        X = self.rng.normal(size=(40, 2))
        Y = self.rng.normal(size=(40, 2))
        ip_xy = np.sum((X @ self.A_spd) * Y, axis=1)
        ip_yx = np.sum((Y @ self.A_spd) * X, axis=1)
        quad_x = np.sum((X @ self.A_spd) * X, axis=1)
        self.assertTrue(np.allclose(ip_xy, ip_yx))
        self.assertTrue(np.all(quad_x > 0.0))

    def test_cholesky_whitening_isometry(self):
        L = np.linalg.cholesky(self.A_spd)
        self.assertTrue(np.allclose(L @ L.T, self.A_spd))
        X = self.rng.normal(size=(50, 2))
        norm_A = np.sqrt(np.sum((X @ self.A_spd) * X, axis=1))
        norm_whitened = np.linalg.norm(X @ L, axis=1)  # ||L^T x||_2
        self.assertTrue(np.allclose(norm_A, norm_whitened, atol=1e-12))

    def test_a_ellipsoid_level_set(self):
        evals, evecs = np.linalg.eigh(self.A_spd)
        thetas = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
        circ = np.column_stack([np.cos(thetas), np.sin(thetas)])
        pts = (circ / np.sqrt(evals)[None, :]) @ evecs.T
        quad = np.einsum("ij,jk,ik->i", pts, self.A_spd, pts)
        self.assertTrue(np.allclose(quad, 1.0, atol=1e-12))

    def test_a_orthogonality_vs_euclidean_non_orthogonality(self):
        u = np.array([1.0, 0.0])
        v = np.array([-1.0, 2.0])
        self.assertAlmostEqual(float(u @ v), -1.0)
        self.assertAlmostEqual(float(u @ self.A_spd @ v), 0.0)
        # Also verify MML Ex. 3.7
        A_mml = np.diag([2.0, 1.0])
        x_m, y_m = np.array([1.0, 1.0]), np.array([-1.0, 1.0])
        self.assertAlmostEqual(float(x_m @ y_m), 0.0)
        cos_a = float((x_m @ A_mml @ y_m) / (math.sqrt(x_m @ A_mml @ x_m) * math.sqrt(y_m @ A_mml @ y_m)))
        self.assertAlmostEqual(cos_a, -1.0 / 3.0, places=12)


class TestTier1CauchySchwarzAndOrthogonality(unittest.TestCase):
    """Tier 1.5: Cauchy-Schwarz, angle trap, Pythagorean split, triangle inequality."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(5105)

    def test_euclidean_cauchy_schwarz_batch(self):
        X = self.rng.normal(size=(500, 7))
        Y = self.rng.normal(size=(500, 7))
        dots = np.abs(np.sum(X * Y, axis=1))
        bounds = np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1)
        self.assertTrue(np.all(dots <= bounds + 1e-12))

    def test_spd_cauchy_schwarz_batch(self):
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        X = self.rng.normal(size=(500, 2))
        Y = self.rng.normal(size=(500, 2))
        ips = np.abs(np.sum((X @ A) * Y, axis=1))
        nx = np.sqrt(np.sum((X @ A) * X, axis=1))
        ny = np.sqrt(np.sum((Y @ A) * Y, axis=1))
        self.assertTrue(np.all(ips <= nx * ny + 1e-12))

    def test_equality_iff_collinear(self):
        x = np.array([1.0, 2.0, -3.0])
        for s in (0.5, 4.0, -2.5):
            y = s * x
            self.assertAlmostEqual(abs(float(x @ y)), float(np.linalg.norm(x) * np.linalg.norm(y)))
        y_non = np.array([1.0, 2.0, -2.9])
        self.assertLess(abs(float(x @ y_non)), float(np.linalg.norm(x) * np.linalg.norm(y_non)))

    def test_pythagorean_theorem_euclidean_and_spd(self):
        x_p, y_p = np.array([3.0, 4.0]), np.array([2.0, 1.0])
        proj = (np.dot(x_p, y_p) / np.dot(y_p, y_p)) * y_p
        resid = x_p - proj
        self.assertAlmostEqual(float(np.linalg.norm(x_p) ** 2), float(np.linalg.norm(proj) ** 2 + np.linalg.norm(resid) ** 2))

        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        u, v = np.array([1.0, 0.0]), np.array([-1.0, 2.0])
        self.assertAlmostEqual(float((u + v) @ A @ (u + v)), float(u @ A @ u + v @ A @ v))

    def test_triangle_inequality_via_cauchy_schwarz(self):
        X = self.rng.normal(size=(100, 4))
        Y = self.rng.normal(size=(100, 4))
        lhs = np.linalg.norm(X + Y, axis=1) ** 2
        rhs = (np.linalg.norm(X, axis=1) + np.linalg.norm(Y, axis=1)) ** 2
        self.assertTrue(np.all(lhs <= rhs + 1e-12))


# =============================================================================
# TIER 2: BOUNDARY, DEGENERATE & STABILITY CASES (25 tests)
# =============================================================================


class TestTier2QuasiNormSubOneFailure(unittest.TestCase):
    """Tier 2.1: Quasi-norms 0 < p < 1 break convexity and triangle inequality."""

    def test_p05_triangle_inequality_failure(self):
        u = np.array([1.0, 0.0])
        v = np.array([0.0, 1.0])
        self.assertAlmostEqual(_quasi_lp(u, 0.5), 1.0)
        self.assertAlmostEqual(_quasi_lp(v, 0.5), 1.0)
        self.assertAlmostEqual(_quasi_lp(u + v, 0.5), 4.0)
        self.assertGreater(_quasi_lp(u + v, 0.5), _quasi_lp(u, 0.5) + _quasi_lp(v, 0.5))

    def test_p05_non_convex_midpoint_escapes_unit_ball(self):
        mid = 0.5 * (np.array([1.0, 0.0]) + np.array([0.0, 1.0]))
        self.assertAlmostEqual(_quasi_lp(mid, 0.5), 2.0)
        self.assertGreater(_quasi_lp(mid, 0.5), 1.0)

    def test_p025_stronger_triangle_violation(self):
        u, v = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        self.assertAlmostEqual(_quasi_lp(u + v, 0.25), 16.0)

    def test_p05_still_satisfies_homogeneity(self):
        x = np.array([2.0, -3.0])
        for lam in (-2.5, 0.5, 4.0):
            self.assertAlmostEqual(_quasi_lp(lam * x, 0.5), abs(lam) * _quasi_lp(x, 0.5), places=12)

    def test_p1_boundary_restores_convexity(self):
        u, v = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        mid = 0.5 * (u + v)
        self.assertLessEqual(float(np.linalg.norm(mid, ord=1)), 1.0 + 1e-12)
        self.assertLessEqual(float(np.linalg.norm(u + v, ord=1)), float(np.linalg.norm(u, ord=1) + np.linalg.norm(v, ord=1)))


class TestTier2ParallelogramLawBoundaries(unittest.TestCase):
    """Tier 2.2: Jordan-von Neumann parallelogram law separates inner-product norms."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(5202)
        self.u = np.array([1.0, 0.0])
        self.v = np.array([0.0, 1.0])

    def test_l2_passes_parallelogram(self):
        X = self.rng.normal(size=(50, 4))
        Y = self.rng.normal(size=(50, 4))
        lhs = np.linalg.norm(X + Y, axis=1) ** 2 + np.linalg.norm(X - Y, axis=1) ** 2
        rhs = 2.0 * (np.linalg.norm(X, axis=1) ** 2 + np.linalg.norm(Y, axis=1) ** 2)
        self.assertTrue(np.allclose(lhs, rhs))

    def test_spd_norm_passes_parallelogram(self):
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        X = self.rng.normal(size=(50, 2))
        Y = self.rng.normal(size=(50, 2))
        nA2 = lambda Z: np.sum((Z @ A) * Z, axis=1)
        lhs = nA2(X + Y) + nA2(X - Y)
        rhs = 2.0 * (nA2(X) + nA2(Y))
        self.assertTrue(np.allclose(lhs, rhs))

    def test_l1_fails_parallelogram(self):
        lhs = np.linalg.norm(self.u + self.v, ord=1) ** 2 + np.linalg.norm(self.u - self.v, ord=1) ** 2
        rhs = 2.0 * (np.linalg.norm(self.u, ord=1) ** 2 + np.linalg.norm(self.v, ord=1) ** 2)
        self.assertAlmostEqual(lhs, 8.0)
        self.assertAlmostEqual(rhs, 4.0)
        self.assertNotAlmostEqual(lhs, rhs)

    def test_linf_fails_parallelogram(self):
        lhs = np.linalg.norm(self.u + self.v, ord=np.inf) ** 2 + np.linalg.norm(self.u - self.v, ord=np.inf) ** 2
        rhs = 2.0 * (np.linalg.norm(self.u, ord=np.inf) ** 2 + np.linalg.norm(self.v, ord=np.inf) ** 2)
        self.assertAlmostEqual(lhs, 2.0)
        self.assertAlmostEqual(rhs, 4.0)
        self.assertNotAlmostEqual(lhs, rhs)

    def test_polarization_identity_consistency(self):
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        x, y = np.array([1.5, -2.0]), np.array([0.7, 3.1])
        nxpy = float((x + y) @ A @ (x + y))
        nxmy = float((x - y) @ A @ (x - y))
        self.assertAlmostEqual(0.25 * (nxpy - nxmy), float(x @ A @ y))

        # Fails additivity for L1 polarization: phi(x+z, y) != phi(x, y) + phi(z, y)
        phi_l1 = lambda a, b: 0.25 * (np.linalg.norm(a + b, 1) ** 2 - np.linalg.norm(a - b, 1) ** 2)
        x_l1, z_l1, y_l1 = np.array([1.0, 2.0]), np.array([2.0, -1.0]), np.array([1.0, 1.0])
        self.assertNotAlmostEqual(phi_l1(x_l1 + z_l1, y_l1), phi_l1(x_l1, y_l1) + phi_l1(z_l1, y_l1))


class TestTier2IndefiniteAndSingularForms(unittest.TestCase):
    """Tier 2.3: Indefinite, non-symmetric, and singular forms fail inner-product axioms."""

    def test_indefinite_negative_energy(self):
        A_indef = np.array([[1.0, 0.0], [0.0, -1.0]])
        x_indef = np.array([1.0, 2.0])
        self.assertAlmostEqual(float(x_indef @ A_indef @ x_indef), -3.0)
        self.assertLess(float(x_indef @ A_indef @ x_indef), 0.0)

    def test_indefinite_nonzero_null_vector(self):
        A_indef = np.array([[1.0, 0.0], [0.0, -1.0]])
        z = np.array([1.0, 1.0])
        self.assertGreater(np.linalg.norm(z), 0.0)
        self.assertAlmostEqual(float(z @ A_indef @ z), 0.0)

    def test_indefinite_cauchy_schwarz_failure(self):
        A_indef = np.array([[1.0, 0.0], [0.0, -1.0]])
        z = np.array([1.0, 1.0])
        w = np.array([1.0, 0.0])
        # |<z, w>| = 1 > sqrt(|<z,z>|)*sqrt(|<w,w>|) = 0
        self.assertGreater(abs(float(z @ A_indef @ w)), math.sqrt(abs(float(z @ A_indef @ z)) * abs(float(w @ A_indef @ w))))

    def test_rank1_psd_seminorm_fails_definiteness(self):
        A_psd = np.array([[1.0, 1.0], [1.0, 1.0]])
        v = np.array([-1.0, 1.0])
        self.assertGreater(np.linalg.norm(v), 0.0)
        self.assertAlmostEqual(float(v @ A_psd @ v), 0.0)

    def test_sylvester_and_nonsymmetric_failure(self):
        # MML Ex. 3.4 A2 = [[9, 6], [6, 3]]: delta_1 = 9 > 0, delta_2 = -9 < 0
        A2 = np.array([[9.0, 6.0], [6.0, 3.0]])
        self.assertLess(float(np.linalg.det(A2)), 0.0)
        x = np.array([2.0, -3.0])
        self.assertAlmostEqual(float(x @ A2 @ x), -9.0)
        # MML Ex. 3.2 non-symmetric
        B = np.array([[2.0, 0.0], [1.0, 2.0]])
        self.assertFalse(np.allclose(B, B.T))


class TestTier2ZeroAndCollinearDegeneracies(unittest.TestCase):
    """Tier 2.4: Zero vector, constant vector standardization, and +/-1 angle saturation."""

    def test_zero_vector_norms_are_zero(self):
        z = np.zeros(5)
        for p in (1, 2, 10, np.inf):
            self.assertEqual(float(np.linalg.norm(z, ord=p)), 0.0)

    def test_zero_vector_orthogonal_to_all(self):
        z = np.zeros(4)
        y = np.array([3.0, -1.0, 4.0, 2.0])
        self.assertEqual(float(np.dot(z, y)), 0.0)
        # Angle denominator is zero
        self.assertEqual(float(np.linalg.norm(z) * np.linalg.norm(y)), 0.0)

    def test_cauchy_schwarz_trivial_zero_saturation(self):
        z = np.zeros(3)
        y = np.array([1.0, 2.0, 3.0])
        self.assertEqual(abs(float(np.dot(z, y))), float(np.linalg.norm(z) * np.linalg.norm(y)))

    def test_collinear_aligned_omega_zero(self):
        x = np.array([3.0, 4.0])
        y = 2.5 * x
        cos_w = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
        self.assertAlmostEqual(cos_w, 1.0, places=12)

    def test_collinear_antiparallel_omega_pi(self):
        x = np.array([3.0, 4.0])
        y = -2.5 * x
        cos_w = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
        self.assertAlmostEqual(cos_w, -1.0, places=12)


class TestTier2ExtremeScalesAndConditioning(unittest.TestCase):
    """Tier 2.5: Overflow/underflow safety, extreme p -> inf, and ill-conditioned SPD."""

    def test_overflow_safe_l2_at_1e200(self):
        x = np.array([3.0e200, 4.0e200])
        self.assertAlmostEqual(_overflow_safe_lp(x, 2.0), 5.0e200, delta=1e187)

    def test_underflow_safe_l2_at_1e_minus_200(self):
        x = np.array([3.0e-200, 4.0e-200])
        self.assertAlmostEqual(_overflow_safe_lp(x, 2.0), 5.0e-200, delta=1e-213)

    def test_high_p_convergence_to_linf(self):
        x = np.array([1.5, -4.0, 3.9, 0.2])
        l200 = _overflow_safe_lp(x, 200.0)
        linf = _overflow_safe_lp(x, np.inf)
        self.assertTrue(math.isfinite(l200))
        self.assertAlmostEqual(l200, linf, delta=0.05)

    def test_ill_conditioned_spd_cholesky_and_cs(self):
        A = np.diag([1.0, 1e8])
        theta = 0.4
        Q = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        A_rot = Q @ A @ Q.T
        L = np.linalg.cholesky(A_rot)
        self.assertTrue(np.allclose(L @ L.T, A_rot, rtol=1e-12))
        rng = np.random.default_rng(5205)
        X = rng.normal(size=(50, 2))
        Y = rng.normal(size=(50, 2))
        ips = np.abs(np.sum((X @ A_rot) * Y, axis=1))
        nx = np.sqrt(np.sum((X @ A_rot) * X, axis=1))
        ny = np.sqrt(np.sum((Y @ A_rot) * Y, axis=1))
        self.assertTrue(np.all(ips <= nx * ny + 1e-9))

    def test_high_dim_n1000_norm_equivalence(self):
        rng = np.random.default_rng(5206)
        x = rng.normal(size=1000)
        linf = float(np.linalg.norm(x, ord=np.inf))
        l2 = float(np.linalg.norm(x, ord=2))
        l1 = float(np.linalg.norm(x, ord=1))
        self.assertLessEqual(linf, l2)
        self.assertLessEqual(l2, l1)
        self.assertLessEqual(l1, math.sqrt(1000.0) * l2)
        self.assertLessEqual(l2, math.sqrt(1000.0) * linf)


# =============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS (5 tests)
# =============================================================================


class TestTier3CrossFeatureCombinations(unittest.TestCase):
    """Tier 3: Multi-concept dualities and structural invariants."""

    def test_norm_equivalence_saturation_vectors(self):
        n = 9
        e1 = np.zeros(n)
        e1[0] = 1.0
        self.assertAlmostEqual(np.linalg.norm(e1, np.inf), np.linalg.norm(e1, 2))
        self.assertAlmostEqual(np.linalg.norm(e1, 2), np.linalg.norm(e1, 1))

        ones = np.ones(n)
        self.assertAlmostEqual(np.linalg.norm(ones, np.inf), 1.0)
        self.assertAlmostEqual(np.linalg.norm(ones, 2), math.sqrt(n))
        self.assertAlmostEqual(np.linalg.norm(ones, 1), float(n))

    def test_cholesky_whitening_turns_a_orth_into_90_deg(self):
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        L = np.linalg.cholesky(A)
        u = np.array([1.0, 0.0])
        v = np.array([-1.0, 2.0])
        u_w = L.T @ u
        v_w = L.T @ v
        self.assertAlmostEqual(float(u_w @ v_w), 0.0, places=12)
        self.assertAlmostEqual(float(np.linalg.norm(u_w) ** 2), float(u @ A @ u), places=12)

    def test_projection_minimizes_distance_to_line(self):
        x = np.array([3.0, 4.0])
        y = np.array([2.0, 1.0])
        t_star = float(np.dot(x, y) / np.dot(y, y))
        min_dist = float(np.linalg.norm(x - t_star * y))
        cos_w = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
        sin_w = math.sqrt(max(0.0, 1.0 - cos_w ** 2))
        self.assertAlmostEqual(min_dist, float(np.linalg.norm(x)) * sin_w, places=12)
        for dt in (-0.5, -0.1, 0.1, 0.5):
            self.assertLess(min_dist, float(np.linalg.norm(x - (t_star + dt) * y)))

    def test_joint_parallelogram_and_polarization_spd(self):
        A = np.array([[3.0, -1.0], [-1.0, 2.0]])
        x, y = np.array([1.2, -0.8]), np.array([2.1, 1.4])
        n2 = lambda z: float(z @ A @ z)
        self.assertAlmostEqual(n2(x + y) + n2(x - y), 2.0 * (n2(x) + n2(y)), places=12)
        self.assertAlmostEqual(0.25 * (n2(x + y) - n2(x - y)), float(x @ A @ y), places=12)

    def test_holder_inequality_duality(self):
        rng = np.random.default_rng(5305)
        x = rng.normal(size=10)
        y = rng.normal(size=10)
        for p in (1.5, 2.0, 3.0, 5.0):
            q = p / (p - 1.0)
            lhs = abs(float(np.dot(x, y)))
            rhs = float(np.linalg.norm(x, ord=p) * np.linalg.norm(y, ord=q))
            self.assertLessEqual(lhs, rhs + 1e-12)


# =============================================================================
# TIER 4: FRONTIER ML & GATE CS/DA SCENARIOS (5 tests)
# =============================================================================


class TestTier4RealWorldScenarios(unittest.TestCase):
    """Tier 4: ML embeddings/regularization/RMSNorm/attention & GATE CS/DA shortcuts."""

    def test_ml_cosine_similarity_and_normalized_distance(self):
        x = np.array([1.0, 2.0, -1.0])
        y = np.array([2.0, 1.0, 0.5])
        x_hat = x / np.linalg.norm(x)
        y_hat = y / np.linalg.norm(y)
        cos_w = float(np.dot(x_hat, y_hat))
        self.assertAlmostEqual(float(np.linalg.norm(x_hat - y_hat) ** 2), 2.0 * (1.0 - cos_w), places=12)
        # Positive scale invariance
        cos_scaled = float(np.dot(4.2 * x, 0.3 * y) / (np.linalg.norm(4.2 * x) * np.linalg.norm(0.3 * y)))
        self.assertAlmostEqual(cos_w, cos_scaled, places=12)

    def test_ml_lasso_diamond_sparsity_vs_ridge_circle(self):
        beta_ols = np.array([2.0, 0.6])
        # Closest point on L1 unit ball is exact sparse vertex [1, 0]
        # Verify among dense boundary sample on B1
        thetas = np.linspace(0.0, 2.0 * np.pi, 4001)
        b1_pts = np.column_stack([np.cos(thetas), np.sin(thetas)])
        b1_pts /= np.linalg.norm(b1_pts, ord=1, axis=1, keepdims=True)
        idx_l1 = int(np.argmin(np.linalg.norm(b1_pts - beta_ols, axis=1)))
        self.assertTrue(np.allclose(b1_pts[idx_l1], np.array([1.0, 0.0]), atol=1e-2))

        # Closest point on L2 unit ball is beta_ols / ||beta_ols||_2 (dense)
        b2_star = beta_ols / np.linalg.norm(beta_ols)
        self.assertGreater(abs(b2_star[1]), 0.25)

    def test_ml_rmsnorm_and_mahalanobis_metric(self):
        x = np.array([2.0, -4.0, 1.0, 3.0])
        rms_norm = x / _rms(x)
        self.assertAlmostEqual(float(np.linalg.norm(rms_norm)), math.sqrt(len(x)), places=12)
        self.assertTrue(np.allclose((5.0 * x) / _rms(5.0 * x), rms_norm))

        # Mahalanobis via inverse covariance
        Sigma = np.array([[2.0, 0.8], [0.8, 1.5]])
        diff = np.array([1.5, -0.5])
        L = np.linalg.cholesky(Sigma)
        d_m_inv = math.sqrt(float(diff @ np.linalg.solve(Sigma, diff)))
        d_m_chol = float(np.linalg.norm(np.linalg.solve(L, diff)))
        self.assertAlmostEqual(d_m_inv, d_m_chol, places=12)

    def test_ml_scaled_dot_product_attention_variance(self):
        rng = np.random.default_rng(5404)
        dk = 64
        q = rng.normal(size=(5000, dk))
        k = rng.normal(size=(5000, dk))
        raw_dots = np.sum(q * k, axis=1)
        scaled_dots = raw_dots / math.sqrt(dk)
        self.assertGreater(float(np.var(raw_dots)), 50.0)
        self.assertAlmostEqual(float(np.var(scaled_dots)), 1.0, delta=0.1)

    def test_gate_cs_extremum_sylvester_and_titu(self):
        # 1. GATE CS-S ellipsoid extremum: max_{x^T A x <= 1} c^T x = sqrt(c^T A^-1 c)
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        c = np.array([3.0, -1.0])
        bound = math.sqrt(float(c @ np.linalg.solve(A, c)))
        x_star = np.linalg.solve(A, c) / bound
        self.assertAlmostEqual(float(x_star @ A @ x_star), 1.0, places=12)
        self.assertAlmostEqual(float(c @ x_star), bound, places=12)

        # 2. GATE Titu's lemma: sum x_i = 1 => sum x_i^2 >= 1/n
        n = 5
        x = np.array([0.4, -0.1, 0.3, 0.25, 0.15])
        self.assertAlmostEqual(float(np.sum(x)), 1.0)
        self.assertGreaterEqual(float(np.sum(x ** 2)), 1.0 / n)


# =============================================================================
# TIER 5: STATIC AST & SYNTAX INTEGRITY (7 tests)
# =============================================================================


class TestSprint05ASTAndSyntax(unittest.TestCase):
    """Tier 5: Static AST verification of Sprint 5 scene code for ManimCE standards."""

    SCENE_PATH = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "linear_algebra",
            "sprint_05_norms_inner_products",
            "scene_norms_inner_products.py",
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
        self.assertEqual(found, [], f"Deprecated symbols detected: {found}")

    def test_scene_class_defined(self):
        tree = self._get_ast()
        class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        expected = {
            "BaseNormsInnerProductsScene",
            "Act1SpatialPuzzle",
            "Act2DynamicDeformation",
            "Act3AlgebraicMorph",
            "Act4DegenerateStressTest",
            "NormsInnerProductsScene",
        }
        self.assertTrue(expected.issubset(class_names), f"Missing classes: {expected - class_names}")

    def test_hud_panel_creation_present(self):
        tree = self._get_ast()
        has_def = any(isinstance(n, ast.FunctionDef) and n.name == "make_hud_panel" for n in ast.walk(tree))
        has_call = any(
            isinstance(n, ast.Call)
            and (
                (isinstance(n.func, ast.Name) and n.func.id == "make_hud_panel")
                or (isinstance(n.func, ast.Attribute) and n.func.attr == "make_hud_panel")
            )
            for n in ast.walk(tree)
        )
        self.assertTrue(has_def and has_call, "make_hud_panel must be defined and called.")

    def test_semantic_palette_and_math_constants_defined(self):
        tree = self._get_ast()
        assigned = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigned.add(target.id)
        required_palette = {
            "I_HAT_COLOR",
            "J_HAT_COLOR",
            "VEC_COLOR",
            "TARGET_COLOR",
            "SUBSPACE_COLOR",
            "LINE_COLOR",
            "BG_COLOR",
            "PANEL_BG",
        }
        required_math = {
            "X_PUZZLE",
            "Y_PUZZLE",
            "X_VEC",
            "Y_VEC",
            "Y_RULER",
            "X_SWEEP_NORM",
            "A_SPD",
            "U_A_ORTH",
            "V_A_ORTH",
            "U_DEG",
            "V_DEG",
            "QUASI_P",
            "A_INDEF",
            "X_INDEF",
        }
        self.assertTrue(required_palette.issubset(assigned), f"Missing palette: {required_palette - assigned}")
        self.assertTrue(required_math.issubset(assigned), f"Missing constants: {required_math - assigned}")

    def test_no_forbidden_execution_calls(self):
        tree = self._get_ast()
        forbidden = {"eval", "exec"}
        found = [
            (node.func.id, getattr(node, "lineno", "?"))
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden
        ]
        self.assertEqual(found, [], f"Forbidden execution calls: {found}")

    def test_valuetracker_and_always_redraw_present(self):
        tree = self._get_ast()
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        self.assertIn("ValueTracker", names)
        self.assertIn("always_redraw", names)

    def test_right_angle_and_decimal_number_present(self):
        tree = self._get_ast()
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        self.assertIn("RightAngle", names)
        self.assertIn("DecimalNumber", names)


if __name__ == "__main__":
    unittest.main()
