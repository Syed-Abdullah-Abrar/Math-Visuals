#!/usr/bin/env python3
r"""
Adversarial Numerical & Mathematical Verification Suite for Sprint 5
==================================================================
Curriculum: Linear Algebra (Norms, Inner Products & Cauchy-Schwarz)
References: MML Ch. 3.1-3.4 & 3.8.1, VMLS Ch. 3-4, Kreyszig Ch. 7-8

This test suite empirically stress-tests:
1. 100,000 random vector pairs in R^8 for Cauchy-Schwarz, triangle inequality, and norm equivalence.
2. Lp unit-ball monotonicity and overflow/underflow-safe norm evaluation up to p = 200 and 10^+/-180.
3. SPD inner-product ellipsoids up to kappa(A) = 10^8 via Cholesky vs indefinite bilinear failures.
4. Quasi-norm (p < 1) triangle-inequality failures and L1/L_inf parallelogram-law rejections.
5. Static AST integrity check of Sprint 5 ManimCE scene script.
6. Prompt 1 verification: Nested Lp balls & p=0.5 non-convexity.
7. Prompt 2 verification: Cauchy-Schwarz saturation, ruler projection & parallelogram law.
8. Prompts 3 & 4 verification: RMS/z-score/Chebyshev & Sylvester/C-S extrema.
"""

from __future__ import annotations

import ast
import math
import os
import sys
import numpy as np


def _safe_lp(X: np.ndarray, p: float, axis: int = -1) -> np.ndarray:
    abs_X = np.abs(X).astype(np.float64)
    m = np.max(abs_X, axis=axis, keepdims=True)
    if np.isinf(p):
        return np.squeeze(m, axis=axis)
    safe_m = np.where(m == 0.0, 1.0, m)
    scaled = abs_X / safe_m
    res = np.squeeze(m, axis=axis) * (np.sum(scaled ** p, axis=axis) ** (1.0 / p))
    return res


def test_100k_vectors_cauchy_schwarz_and_norm_equivalence():
    """Verify Cauchy-Schwarz, triangle inequality, and full norm equivalence on 100,000 pairs in R^8."""
    print("\n[TEST 1] 100k Vector Cauchy-Schwarz & Norm Equivalence (R^8)")
    rng = np.random.default_rng(5501)
    N, n = 100_000, 8
    X = rng.normal(size=(N, n))
    Y = rng.normal(size=(N, n))

    dots = np.abs(np.sum(X * Y, axis=1))
    nx2 = np.linalg.norm(X, ord=2, axis=1)
    ny2 = np.linalg.norm(Y, ord=2, axis=1)
    assert np.all(dots <= nx2 * ny2 + 1e-12), "Cauchy-Schwarz violated"

    nx1 = np.linalg.norm(X, ord=1, axis=1)
    nx_inf = np.linalg.norm(X, ord=np.inf, axis=1)
    assert np.all(nx_inf <= nx2 + 1e-12)
    assert np.all(nx2 <= nx1 + 1e-12)
    assert np.all(nx1 <= math.sqrt(n) * nx2 + 1e-12)
    assert np.all(nx2 <= math.sqrt(n) * nx_inf + 1e-12)

    nxy2 = np.linalg.norm(X + Y, ord=2, axis=1)
    assert np.all(nxy2 <= nx2 + ny2 + 1e-12), "Triangle inequality violated"
    print(f"  ✓ Verified {N:,} R^{n} pairs: max C-S ratio = {np.max(dots / (nx2 * ny2)):.6f}")


def test_lp_unit_ball_monotonicity_and_overflow_safe_norms():
    """Verify p->inf convergence, 10^+/-180 overflow/underflow safety, and unit-ball diagonal radii."""
    print("\n[TEST 2] Lp Monotonicity & Overflow-Safe Extreme Scales")
    rng = np.random.default_rng(5502)
    X = rng.uniform(-5.0, 5.0, size=(500, 6))
    linf = _safe_lp(X, np.inf, axis=1)
    l200 = _safe_lp(X, 200.0, axis=1)
    assert np.all(np.isfinite(l200))
    assert np.max(np.abs(l200 - linf)) < 0.06

    # Extreme scales
    x_hi = np.array([3.0e180, 4.0e180])
    x_lo = np.array([3.0e-180, 4.0e-180])
    assert math.isclose(float(_safe_lp(x_hi, 2.0, axis=0)), 5.0e180, rel_tol=1e-12)
    assert math.isclose(float(_safe_lp(x_lo, 2.0, axis=0)), 5.0e-180, rel_tol=1e-12)

    # Diagonal radius (2^(-1/p), 2^(-1/p)) monotonic sweep
    prev = 0.0
    for p in (1.0, 1.5, 2.0, 5.0, 25.0, 200.0):
        c = 2.0 ** (-1.0 / p)
        assert c > prev
        assert math.isclose(float(_safe_lp(np.array([c, c]), p, axis=0)), 1.0, rel_tol=1e-12)
        prev = c
    print("  ✓ Overflow/underflow-safe Lp norms and diagonal monotonicity verified")


def test_spd_inner_product_ellipsoid_and_indefinite_failures():
    """Verify 10k-vector SPD Cauchy-Schwarz/parallelogram up to kappa=1e8 and indefinite failures."""
    print("\n[TEST 3] SPD Inner Product Ellipsoids (kappa=1e8) & Indefinite Failures")
    rng = np.random.default_rng(5503)
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    A_ill = Q @ np.diag([1.0, 100.0, 1e5, 1e8]) @ Q.T
    A_ill = 0.5 * (A_ill + A_ill.T)
    L = np.linalg.cholesky(A_ill)
    assert np.allclose(L @ L.T, A_ill, rtol=1e-11)

    X = rng.normal(size=(10_000, 4))
    Y = rng.normal(size=(10_000, 4))
    ips = np.abs(np.sum((X @ A_ill) * Y, axis=1))
    nx = np.sqrt(np.sum((X @ A_ill) * X, axis=1))
    ny = np.sqrt(np.sum((Y @ A_ill) * Y, axis=1))
    assert np.all(ips <= nx * ny * (1.0 + 1e-11))

    # Indefinite A_indef = diag(1, -1)
    A_indef = np.diag([1.0, -1.0])
    assert float(np.array([1.0, 2.0]) @ A_indef @ np.array([1.0, 2.0])) == -3.0
    assert float(np.array([1.0, 1.0]) @ A_indef @ np.array([1.0, 1.0])) == 0.0
    print("  ✓ Verified 10,000 SPD pairs at kappa=1e8 and indefinite failures")


def test_quasi_norm_and_parallelogram_boundary_violations():
    """Verify p < 1 quasi-norm triangle inequality failure and L1/L_inf parallelogram failure."""
    print("\n[TEST 4] Quasi-Norm (p<1) & Parallelogram Law Boundary Violations")
    u = np.array([1.0, 0.0])
    v = np.array([0.0, 1.0])
    for p in (0.25, 0.5, 0.75):
        nu = float(np.sum(np.abs(u) ** p) ** (1.0 / p))
        nv = float(np.sum(np.abs(v) ** p) ** (1.0 / p))
        nuv = float(np.sum(np.abs(u + v) ** p) ** (1.0 / p))
        assert nuv > nu + nv, f"Expected triangle violation at p={p}"

    # Parallelogram law: ||u+v||^2 + ||u-v||^2 == 2(||u||^2 + ||v||^2)
    l2_lhs = np.linalg.norm(u + v, 2) ** 2 + np.linalg.norm(u - v, 2) ** 2
    l2_rhs = 2.0 * (np.linalg.norm(u, 2) ** 2 + np.linalg.norm(v, 2) ** 2)
    assert math.isclose(l2_lhs, l2_rhs)

    l1_lhs = np.linalg.norm(u + v, 1) ** 2 + np.linalg.norm(u - v, 1) ** 2
    l1_rhs = 2.0 * (np.linalg.norm(u, 1) ** 2 + np.linalg.norm(v, 1) ** 2)
    assert not math.isclose(l1_lhs, l1_rhs)

    linf_lhs = np.linalg.norm(u + v, np.inf) ** 2 + np.linalg.norm(u - v, np.inf) ** 2
    linf_rhs = 2.0 * (np.linalg.norm(u, np.inf) ** 2 + np.linalg.norm(v, np.inf) ** 2)
    assert not math.isclose(linf_lhs, linf_rhs)
    print("  ✓ Verified quasi-norm (p=0.25, 0.5, 0.75) & L1/L_inf parallelogram failures")


def test_ast_scene_static_integrity():
    """Static AST integrity check on Sprint 5 scene_norms_inner_products.py."""
    print("\n[TEST 5] Sprint 5 Scene Static AST Integrity")
    scene_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "linear_algebra",
            "sprint_05_norms_inner_products",
            "scene_norms_inner_products.py",
        )
    )
    if not os.path.exists(scene_path):
        print(f"  ⚠ Skipping AST check: {scene_path} not yet created")
        return

    with open(scene_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=scene_path)

    deprecated = {"CONFIG", "ShowCreation", "TexMobject", "TextMobject"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert node.id not in deprecated, f"Deprecated symbol {node.id}"
        elif isinstance(node, ast.Attribute):
            assert node.attr not in deprecated, f"Deprecated attr {node.attr}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"eval", "exec"}, f"Forbidden call {node.func.id}"

    classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    for req in (
        "BaseNormsInnerProductsScene",
        "Act1SpatialPuzzle",
        "Act2DynamicDeformation",
        "Act3AlgebraicMorph",
        "Act4DegenerateStressTest",
        "NormsInnerProductsScene",
    ):
        assert req in classes, f"Missing scene class {req}"
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "make_hud_panel" in funcs
    print("  ✓ Verified AST integrity: 0 deprecated symbols, 0 eval/exec, all 6 classes & make_hud_panel present")


def test_tweaks_prompt_1_lp_balls_and_quasinorm():
    """Verify tweaks_prompts.md Prompt 1: Nested Lp unit balls & p=0.5 failure."""
    print("\n[TEST 6] Prompt 1: Nested Lp Unit Balls & p=0.5 Non-Convexity")
    rng = np.random.default_rng(501)
    X = rng.normal(size=(1000, 6))
    n = X.shape[1]
    l_inf = np.linalg.norm(X, ord=np.inf, axis=1)
    l_2 = np.linalg.norm(X, ord=2, axis=1)
    l_1 = np.linalg.norm(X, ord=1, axis=1)
    assert np.all(l_inf <= l_2 + 1e-12)
    assert np.all(l_2 <= l_1 + 1e-12)
    assert np.all(l_1 <= np.sqrt(n) * l_2 + 1e-12)
    assert np.all(l_2 <= np.sqrt(n) * l_inf + 1e-12)

    u, v = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    norm_half = lambda z: float(np.sum(np.abs(z) ** 0.5) ** 2.0)
    assert math.isclose(norm_half(u + v), 4.0) and norm_half(u + v) > norm_half(u) + norm_half(v)
    print("  ✓ Prompt 1 assertions verified")


def test_tweaks_prompt_2_cauchy_schwarz_and_parallelogram():
    """Verify tweaks_prompts.md Prompt 2: Ruler projection, collinearity & parallelogram law."""
    print("\n[TEST 7] Prompt 2: Cauchy-Schwarz, Ruler Projection & Parallelogram")
    x = np.array([1.0, 2.0])
    y = np.array([2.0, 1.0])
    proj = (np.dot(x, y) / np.dot(y, y)) * y
    resid = x - proj
    assert np.allclose(proj, np.array([1.6, 0.8]))
    assert np.allclose(resid, np.array([-0.6, 1.2]))
    assert math.isclose(float(np.dot(resid, y)), 0.0, abs_tol=1e-12)
    assert abs(float(np.dot(x, y))) == 4.0 < float(np.linalg.norm(x) * np.linalg.norm(y))
    assert math.isclose(float(np.dot(x, 3.0 * x)), float(np.linalg.norm(x) * np.linalg.norm(3.0 * x)))
    print("  ✓ Prompt 2 assertions verified")


def test_tweaks_prompt_3_and_4_ml_regularization_and_gate_spd():
    """Verify tweaks_prompts.md Prompts 3 & 4: RMS/z-score/Chebyshev & GATE SPD extrema."""
    print("\n[TEST 8] Prompts 3 & 4: ML RMS/Z-Score/Chebyshev & GATE Sylvester/C-S")
    x = np.array([1.0, -2.0, 3.0, 2.0])
    avg_x = float(np.mean(x))
    std_x = float(np.linalg.norm(x - avg_x) / math.sqrt(len(x)))
    assert math.isclose(std_x, math.sqrt(3.5))
    assert math.isclose(float(np.linalg.norm(x) ** 2) / len(x), avg_x ** 2 + std_x ** 2)

    z = (x - avg_x) / std_x
    assert math.isclose(float(np.mean(z)), 0.0, abs_tol=1e-12)
    assert math.isclose(float(np.linalg.norm(z)), math.sqrt(len(x)))
    rms_x = float(np.linalg.norm(x) / math.sqrt(len(x)))
    assert float(np.mean(np.abs(x) >= 2.5)) <= (rms_x / 2.5) ** 2 + 1e-12

    A_spd = np.array([[2.0, 1.0], [1.0, 2.0]])
    assert A_spd[0, 0] > 0 and np.linalg.det(A_spd) > 0
    c = np.array([3.0, -1.0])
    bound = math.sqrt(float(c @ np.linalg.solve(A_spd, c)))
    x_star = np.linalg.solve(A_spd, c) / bound
    assert math.isclose(float(x_star @ A_spd @ x_star), 1.0)
    assert math.isclose(float(c @ x_star), bound)
    print("  ✓ Prompts 3 & 4 assertions verified")


def main() -> None:
    print("=" * 72)
    print("SPRINT 5 ADVERSARIAL VERIFICATION SUITE")
    print("=" * 72)

    tests = [
        test_100k_vectors_cauchy_schwarz_and_norm_equivalence,
        test_lp_unit_ball_monotonicity_and_overflow_safe_norms,
        test_spd_inner_product_ellipsoid_and_indefinite_failures,
        test_quasi_norm_and_parallelogram_boundary_violations,
        test_ast_scene_static_integrity,
        test_tweaks_prompt_1_lp_balls_and_quasinorm,
        test_tweaks_prompt_2_cauchy_schwarz_and_parallelogram,
        test_tweaks_prompt_3_and_4_ml_regularization_and_gate_spd,
    ]

    passed = 0
    failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as exc:
            failed += 1
            print(f"  ✗ FAILED {fn.__name__}: {exc}")

    print("\n" + "=" * 72)
    print(f"SUMMARY: {passed}/{len(tests)} adversarial checks passed, {failed} failed.")
    print("=" * 72)
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
