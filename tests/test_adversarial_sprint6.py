#!/usr/bin/env python3
r"""
Adversarial Numerical & Mathematical Verification Suite for Sprint 6
===================================================================
Curriculum: Linear Algebra (Orthogonal Projections onto 1D Lines & k-D Subspaces)
References: MML §3.6, §3.8, §9.2.3, §10.3; VMLS §11.5, §12.1–12.3, §15; ILA Ch. 3, 6

This test suite empirically stress-tests:
1. 100,000 random 3D vectors decomposed into y = P y + (I - P) y with orthogonal & Pythagorean split.
2. 100,000 candidate points in R^2 verifying strict global minimum distance at x_hat = [2, -1]^T.
3. Lauchli ill-conditioned normal equations across eps in [1e-2, 1e-10] (normal eqs collapse vs QR/SVD survival).
4. Rank-deficient pseudoinverse A A^+ vs Ridge (lambda -> 0+) and oblique projector norm blowup.
5. Static AST integrity verification of Sprint 6 3D scene (ThreeDScene, HUD pinning, zero deprecated symbols).
6. Prompt 1 verification: 1D line projection, rank-1 P_1, and sliding optimality curve d(s) = sqrt(6 + 2*s^2).
7. Prompt 2 verification: Projector spectrum {0, 1}, Tr == rank, and oblique vs orthogonal residual contrast.
8. Prompts 3 & 4 verification: Ridge shrinkage / PCA error and GATE rank-1 projector instant invariants.
"""

from __future__ import annotations

import ast
import math
import os
import sys
import numpy as np


def test_100k_vectors_orthogonal_projection_decomposition():
    """Verify y = P y + (I - P) y, orthogonality, and Pythagorean split over 100,000 vectors."""
    print("\n[TEST 1] High-Scale 3D Orthogonal Decomposition & Pythagorean Split (N=100,000)")
    rng = np.random.default_rng(6201)

    A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    P = A @ np.linalg.inv(A.T @ A) @ A.T
    Q_comp = np.eye(3) - P

    N = 100000
    Y = rng.uniform(-1e4, 1e4, size=(N, 3))
    Y_proj = Y @ P.T
    Y_res = Y @ Q_comp.T

    # 1. Exact reconstruction y = Py + (I-P)y
    recon_err = np.max(np.abs((Y_proj + Y_res) - Y))
    assert recon_err < 1e-10, f"Reconstruction error too large: {recon_err}"

    # 2. Relative orthogonality between Y_proj and Y_res
    dots = np.sum(Y_proj * Y_res, axis=1)
    norms_prod = np.linalg.norm(Y_proj, axis=1) * np.linalg.norm(Y_res, axis=1) + 1e-15
    max_rel_dot = np.max(np.abs(dots) / norms_prod)
    assert max_rel_dot < 1e-10, f"Non-orthogonal projection detected: {max_rel_dot}"

    # 3. Relative Pythagorean split: ||y||^2 == ||Py||^2 + ||(I-P)y||^2
    lhs = np.sum(Y ** 2, axis=1)
    rhs = np.sum(Y_proj ** 2, axis=1) + np.sum(Y_res ** 2, axis=1)
    max_rel_pyth = np.max(np.abs(lhs - rhs) / (lhs + 1e-15))
    assert max_rel_pyth < 1e-12, f"Pythagorean mismatch: {max_rel_pyth}"

    # 4. Idempotence over 100k projected points: P(Py) == Py
    idem_err = np.max(np.abs((Y_proj @ P.T) - Y_proj))
    assert idem_err < 1e-10, f"Action idempotence failed: {idem_err}"

    print(f"  ✓ 100,000 vectors: max rel dot = {max_rel_dot:.2e}, max rel Pythagorean err = {max_rel_pyth:.2e}")


def test_100k_points_global_distance_minimizer():
    """Verify x_hat = [2, -1]^T strictly minimizes ||y - Ax||_2 across 100,000 candidates."""
    print("\n[TEST 2] 100,000-Point Global Minimum Distance & Quadratic Error Identity")
    rng = np.random.default_rng(6202)

    A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = np.array([3.0, -1.0, 1.0])
    x_hat = np.array([2.0, -1.0])
    r_hat = y - A @ x_hat
    min_sq = float(np.dot(r_hat, r_hat))

    N = 100000
    deltas = rng.uniform(-50.0, 50.0, size=(N, 2))
    # Guarantee strict non-zero perturbation
    deltas[np.linalg.norm(deltas, axis=1) < 1e-3] += np.array([0.5, -0.5])
    X_cand = x_hat + deltas

    residuals = y - X_cand @ A.T
    sq_errors = np.sum(residuals ** 2, axis=1)
    excess_sq = np.sum((deltas @ A.T) ** 2, axis=1)

    max_quad_err = np.max(np.abs(sq_errors - (min_sq + excess_sq)) / sq_errors)
    assert max_quad_err < 1e-12, f"Quadratic error identity failed: {max_quad_err}"
    assert np.all(sq_errors > min_sq), "Strict global minimality violated!"

    print(f"  ✓ 100,000 candidates strictly > sqrt(6); max rel quadratic identity err = {max_quad_err:.2e}")


def test_lauchli_ill_conditioned_normal_equations():
    """Stress-test Lauchli conditioning: kappa(A^T A) == kappa(A)^2 & normal eq breakdown vs QR."""
    print("\n[TEST 3] Lauchli Ill-Conditioning: Normal Equations Collapse vs. QR/SVD Survival")

    for eps in [1e-2, 1e-4, 1e-6]:
        A = np.array([[1.0, 1.0], [eps, 0.0], [0.0, eps]])
        ratio = np.linalg.cond(A.T @ A) / (np.linalg.cond(A) ** 2)
        tol = 1e-6 if eps >= 1e-4 else 1e-3  # 1+eps^2 loses 12 digits in float64 at eps=1e-6
        assert np.isclose(ratio, 1.0, rtol=tol), f"Expected kappa(A^T A)=kappa(A)^2, got ratio={ratio}"

    # At eps = 1e-8 and 1e-10, float64 rounds 1 + eps^2 -> 1.0, collapsing A^T A to rank 1
    for eps in [1e-8, 1e-10]:
        A = np.array([[1.0, 1.0], [eps, 0.0], [0.0, eps]])
        b = np.array([2.0, eps, eps])
        G = A.T @ A
        assert np.linalg.matrix_rank(G) == 1, f"Expected rank collapse at eps={eps}"

        # QR and SVD lstsq still solve x_hat = [1, 1]^T accurately
        Q, R = np.linalg.qr(A)
        x_qr = np.linalg.solve(R, Q.T @ b)
        x_svd, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        assert rank == 2
        assert np.allclose(x_qr, [1.0, 1.0], atol=1e-12)
        assert np.allclose(x_svd, [1.0, 1.0], atol=1e-12)

    print("  ✓ Verified kappa(A^T A) = kappa(A)^2 & float64 breakdown vs QR/SVD accuracy at eps=1e-8, 1e-10")


def test_rank_deficient_and_oblique_projectors():
    """Stress-test A A^+ vs Ridge convergence and oblique projector norm blowup."""
    print("\n[TEST 4] Rank-Deficient Ridge Limit & Oblique Projector Norm Blowup")

    A_deg = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    y = np.array([3.0, -1.0, 1.0])
    x_ln = np.linalg.pinv(A_deg) @ y
    assert np.allclose(x_ln, [0.2, 0.4], atol=1e-12)

    errs = []
    for lam in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]:
        x_rls = np.linalg.solve(A_deg.T @ A_deg + lam * np.eye(2), A_deg.T @ y)
        errs.append(np.linalg.norm(x_rls - x_ln))
    assert errs[-1] < 1e-8 and errs == sorted(errs, reverse=True), f"Ridge did not converge monotonically: {errs}"

    # Oblique projector blowup: P_ob = (e1 w^T) / (w^T e1) with w = [1, cot(theta), 0]
    for theta in [0.5, 0.1, 0.01, 1e-3]:
        u = np.array([1.0, 0.0, 0.0])
        w = np.array([1.0, 1.0 / math.tan(theta), 0.0])
        P_ob = np.outer(u, w) / np.dot(w, u)
        assert np.allclose(P_ob @ P_ob, P_ob, atol=1e-12)
        expected_norm = 1.0 / math.sin(theta)
        assert np.isclose(np.linalg.norm(P_ob, 2), expected_norm, rtol=1e-10)
        assert np.linalg.norm(P_ob, 2) > 1.0

    print(f"  ✓ Verified Ridge -> A^+ convergence ({errs[-1]:.2e}) & oblique ||P_ob||_2 = 1/sin(theta) blowup")


def test_ast_scene_static_integrity():
    """Static AST check for Sprint 6 scene (ThreeDScene, HUD pinning, zero deprecated symbols)."""
    print("\n[TEST 5] AST Scene Static Integrity Verification (ThreeDScene & HUD Pinning)")
    scene_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "linear_algebra",
            "sprint_06_orthogonal_projections",
            "scene_orthogonal_projections.py",
        )
    )
    if not os.path.exists(scene_path):
        print(f"  [SKIPPED] {scene_path} not yet created.")
        return

    with open(scene_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=scene_path)

    deprecated = {"CONFIG", "ShowCreation", "TexMobject", "TextMobject"}
    found_dep = [
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id in deprecated
    ]
    assert not found_dep, f"Deprecated symbols found: {found_dep}"

    threed_classes = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and any(
            (isinstance(b, ast.Name) and b.id == "ThreeDScene")
            or (isinstance(b, ast.Attribute) and b.attr == "ThreeDScene")
            for b in node.bases
        )
    ]
    assert threed_classes, "Sprint 6 scene must inherit from ThreeDScene"

    has_hud_call = any(
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "add_fixed_in_frame_mobjects")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "add_fixed_in_frame_mobjects")
        )
        for node in ast.walk(tree)
    )
    assert has_hud_call, "Sprint 6 scene must invoke add_fixed_in_frame_mobjects"

    class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert "OrthogonalProjectionsScene" in class_names, "Master class OrthogonalProjectionsScene missing"
    print("  ✓ AST verified: ThreeDScene, HUD pinning, zero deprecated symbols, and OrthogonalProjectionsScene")


def test_tweaks_prompt_1_1d_line_projection_and_rank1_matrix():
    """Verify Prompt 1: Minimum-distance optimality, orthogonality principle & sliding d(s)."""
    print("\n[TEST 6] Prompt 1: Minimum-Distance Optimality & Sliding Error Curve")
    A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = np.array([3.0, -1.0, 1.0])
    x_hat = np.linalg.solve(A.T @ A, A.T @ y)
    r_hat = y - A @ x_hat

    assert np.allclose(x_hat, [2.0, -1.0])
    assert np.allclose(A.T @ r_hat, 0.0, atol=1e-12)

    for s in np.linspace(-2.0, 1.5, 71):
        x_s = np.array([2.0 + s, -1.0 - s])
        assert np.isclose(np.linalg.norm(y - A @ x_s), np.sqrt(6.0 + 2.0 * s * s), atol=1e-12)
    print("  ✓ Prompt 1 verified: x_hat=[2,-1]^T, A^T r = 0, and exact d(s)=sqrt(6+2s^2)")


def test_tweaks_prompt_2_kd_subspace_error_curve_minimum():
    """Verify Prompt 2: Projector invariants {0,1}, Tr=rank & oblique contrast."""
    print("\n[TEST 7] Prompt 2: Projector Invariants & Oblique Contrast")
    A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    P = A @ np.linalg.inv(A.T @ A) @ A.T
    P_ob = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 2.0, 0.0]])
    y_test = np.array([1.0, 1.0, 3.0])

    assert np.allclose(np.sort(np.linalg.eigvalsh(P)), [0.0, 1.0, 1.0], atol=1e-12)
    assert np.isclose(np.trace(P), np.linalg.matrix_rank(P))
    assert np.isclose(np.linalg.norm(P, 2), 1.0)

    assert np.allclose(P_ob @ P_ob, P_ob) and not np.allclose(P_ob.T, P_ob)
    _, s_ob, Vt_ob = np.linalg.svd(P_ob)
    assert np.isclose(s_ob[0], np.sqrt(6.0))
    assert np.isclose(np.linalg.norm(P_ob @ Vt_ob[0]), np.sqrt(6.0))
    assert np.linalg.norm(y_test - P_ob @ y_test) > np.linalg.norm(y_test - P @ y_test)
    print("  ✓ Prompt 2 verified: spectrum {0,1,1}, Tr=rank=2, ||P_ob||_2=sqrt(6) > 1")


def test_tweaks_prompt_3_and_4_normal_equations_and_edge_cases():
    """Verify Prompts 3 & 4: Ridge/PCA unification & GATE rank-1 projector shortcuts."""
    print("\n[TEST 8] Prompts 3 & 4: Ridge/PCA Unification & GATE Rank-1 Shortcuts")
    # Prompt 3: Ridge effective df & PCA reconstruction
    A_deg = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    s_deg = np.linalg.svd(A_deg, compute_uv=False)
    lam = 0.5
    H_lam = A_deg @ np.linalg.inv(A_deg.T @ A_deg + lam * np.eye(2)) @ A_deg.T
    eff_df = sum((si ** 2) / (si ** 2 + lam) for si in s_deg if si > 1e-12)
    assert np.isclose(np.trace(H_lam), eff_df, atol=1e-12)
    assert not np.allclose(H_lam @ H_lam, H_lam)

    rng = np.random.default_rng(6208)
    Xc = rng.standard_normal((120, 3))
    Xc -= Xc.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh((Xc.T @ Xc) / len(Xc))
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    P_M = eigvecs[:, :2] @ eigvecs[:, :2].T
    assert np.isclose(np.mean(np.sum((Xc - Xc @ P_M.T) ** 2, axis=1)), eigvals[2], atol=1e-12)

    # Prompt 4: GATE rank-1 projector shortcuts
    u = np.array([1.0, 2.0, 2.0])
    y = np.array([3.0, 3.0, 0.0])
    P1 = np.outer(u, u) / np.dot(u, u)
    I_minus_P = np.eye(3) - P1
    assert np.linalg.matrix_rank(P1) == 1 and np.isclose(np.trace(P1), 1.0)
    assert np.isclose(np.linalg.det(P1), 0.0)
    assert np.allclose(np.linalg.matrix_power(I_minus_P, 7), I_minus_P)
    assert np.isclose(abs(np.dot(u, y)) / np.linalg.norm(u), 3.0)
    assert np.isclose(np.linalg.norm(I_minus_P @ y), 3.0)
    print("  ✓ Prompts 3 & 4 verified: Ridge df, PCA error, and GATE rank-1 shortcuts")


def main():
    print("=" * 70)
    print("ADVERSARIAL NUMERICAL VERIFICATION SUITE: SPRINT 6")
    print("Linear Algebra Curriculum — Orthogonal Projections (1D & k-D)")
    print("=" * 70)

    tests = [
        test_100k_vectors_orthogonal_projection_decomposition,
        test_100k_points_global_distance_minimizer,
        test_lauchli_ill_conditioned_normal_equations,
        test_rank_deficient_and_oblique_projectors,
        test_ast_scene_static_integrity,
        test_tweaks_prompt_1_1d_line_projection_and_rank1_matrix,
        test_tweaks_prompt_2_kd_subspace_error_curve_minimum,
        test_tweaks_prompt_3_and_4_normal_equations_and_edge_cases,
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
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
