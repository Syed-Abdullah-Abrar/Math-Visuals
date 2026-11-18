#!/usr/bin/env python3
r"""
Adversarial Numerical & Mathematical Verification Suite for Sprint 7
===================================================================
Curriculum: Linear Algebra (Gram-Schmidt Orthonormalization & QR Decomposition)
References: MML §§3.5, 3.8.2-3.8.3; VMLS §§5.3-5.4, 10.4, 11.3, 11.5, 12.1-12.3; Kreyszig §20.9

This test suite empirically stress-tests:
1. 500 random tall/square matrices + 100,000 random vectors verifying QR & orthonormal expansion.
2. Lauchli (eps = 1e-2..1e-8) & Hilbert (H_4..H_10) orthogonality hierarchy: CGS << MGS << Householder/CGS2.
3. Rank-deficient, zero-column, and near-singular stability with early termination.
4. QR uniqueness up to sign diagonal D = diag(+-1) across 500 random invertible matrices.
5. Static AST syntax integrity of Sprint 7 ManimCE 3D scene.
6. Prompt 1 verification: Sequential orthogonal stripping & nested prefix span invariance.
7. Prompt 2 verification: Upper-triangular R, uniqueness (R_ii > 0), A^T A = R^T R & least squares.
8. Prompt 3 & 4 verification: CGS/Normal-Eq collapse at kappa=1e8 vs QR, |det A| = prod R_ii & R_33 = 0.
"""

from __future__ import annotations

import ast
import os
import sys

import numpy as np


def _cgs_qr(A: np.ndarray, tol: float = 1e-14) -> tuple[np.ndarray, np.ndarray, int]:
    m, n = A.shape
    Q = np.zeros((m, n), dtype=float)
    R = np.zeros((n, n), dtype=float)
    rank = 0
    for j in range(n):
        v = A[:, j].astype(float).copy()
        for i in range(j):
            R[i, j] = float(np.dot(Q[:, i], A[:, j]))
            v = v - R[i, j] * Q[:, i]
        nv = float(np.linalg.norm(v))
        if nv <= tol:
            R[j, j] = 0.0
            break
        R[j, j] = nv
        Q[:, j] = v / nv
        rank += 1
    return Q, R, rank


def _mgs_qr(A: np.ndarray, tol: float = 1e-14) -> tuple[np.ndarray, np.ndarray, int]:
    m, n = A.shape
    V = A.astype(float).copy()
    Q = np.zeros((m, n), dtype=float)
    R = np.zeros((n, n), dtype=float)
    rank = 0
    for i in range(n):
        nv = float(np.linalg.norm(V[:, i]))
        if nv <= tol:
            R[i, i] = 0.0
            break
        R[i, i] = nv
        Q[:, i] = V[:, i] / nv
        rank += 1
        for j in range(i + 1, n):
            R[i, j] = float(np.dot(Q[:, i], V[:, j]))
            V[:, j] = V[:, j] - R[i, j] * Q[:, i]
    return Q, R, rank


def _canon_qr(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Q, R = np.linalg.qr(A, mode="reduced")
    D = np.diag(np.where(np.diag(R) >= 0.0, 1.0, -1.0))
    return Q @ D, D @ R


def _back_sub(R: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = R.shape[0]
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - float(np.dot(R[i, i + 1 :], x[i + 1 :]))) / R[i, i]
    return x


def test_large_scale_randomized_qr():
    """Verify A = QR, Q^T Q = I, R upper-triangular (R_ii > 0) & 100k vector expansion."""
    print("\n[TEST 1] Large-Scale Randomized QR & 100k Orthonormal Expansion")
    rng = np.random.default_rng(7001)
    max_recon_err = 0.0
    max_ortho_err = 0.0

    for _ in range(500):
        m = int(rng.integers(3, 21))
        n = int(rng.integers(2, m + 1))
        A = rng.standard_normal((m, n))
        Q, R, rank = _mgs_qr(A)
        assert rank == n
        assert np.all(np.diag(R) > 0.0)
        assert np.allclose(np.tril(R, -1), 0.0, atol=1e-12)
        max_recon_err = max(max_recon_err, float(np.max(np.abs(Q @ R - A))))
        max_ortho_err = max(max_ortho_err, float(np.max(np.abs(Q.T @ Q - np.eye(n)))))

    assert max_recon_err < 1e-11, f"Reconstruction error too high: {max_recon_err}"
    assert max_ortho_err < 1e-11, f"Orthogonality error too high: {max_ortho_err}"

    # 100,000 random 3D vectors expanded in canonical orthonormal frame Q
    Q3 = np.array([[2.0, -2.0, -1.0], [1.0, 2.0, -2.0], [2.0, 1.0, 2.0]]) / 3.0
    X = rng.uniform(-1e4, 1e4, size=(100000, 3))
    coords = X @ Q3
    X_recon = coords @ Q3.T
    exp_err = float(np.max(np.abs(X_recon - X)))
    assert exp_err < 1e-11, f"100k orthonormal expansion failed: {exp_err}"
    print(f"  ✓ 500 random QR max recon={max_recon_err:.2e}, ortho={max_ortho_err:.2e}")
    print(f"  ✓ 100,000-vector orthonormal expansion max error={exp_err:.2e}")


def test_lauchli_and_hilbert_cgs_vs_mgs_vs_householder():
    """Verify strict orthogonality hierarchy CGS << MGS << Householder on Lauchli & Hilbert."""
    print("\n[TEST 2] Lauchli & Hilbert CGS vs MGS vs Householder Stability Sweep")
    for eps in (1e-2, 1e-4, 1e-6, 1e-8):
        A = np.array([[1.0, 1.0, 1.0], [eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]])
        Qc, _, _ = _cgs_qr(A, tol=1e-15)
        Qm, _, _ = _mgs_qr(A, tol=1e-15)
        Qh, _ = _canon_qr(A)
        ec = float(np.linalg.norm(Qc.T @ Qc - np.eye(3), "fro"))
        em = float(np.linalg.norm(Qm.T @ Qm - np.eye(3), "fro"))
        eh = float(np.linalg.norm(Qh.T @ Qh - np.eye(3), "fro"))
        assert ec > em > eh, f"Expected ec > em > eh at eps={eps}, got {ec}, {em}, {eh}"
        if eps == 1e-8:
            assert ec > 0.1 and em < 1e-7 and eh < 1e-14

    for n in (4, 6, 8, 10):
        idx = np.arange(n, dtype=float)
        H = 1.0 / (idx[:, None] + idx[None, :] + 1.0)
        Qc, _, _ = _cgs_qr(H, tol=1e-16)
        Qm, _, _ = _mgs_qr(H, tol=1e-16)
        Qh, _ = _canon_qr(H)
        ec = float(np.linalg.norm(Qc.T @ Qc - np.eye(n), "fro"))
        em = float(np.linalg.norm(Qm.T @ Qm - np.eye(n), "fro"))
        eh = float(np.linalg.norm(Qh.T @ Qh - np.eye(n), "fro"))
        assert ec > em and eh < 1e-14, f"Hilbert H_{n} violated hierarchy: {ec}, {em}, {eh}"

    print("  ✓ Lauchli (1e-2..1e-8) & Hilbert (H_4..H_10) orthogonality hierarchy confirmed")


def test_rank_deficient_and_zero_column_stability():
    """Verify zero columns, dependent columns, rank-1 matrices, and near-singular perturbations."""
    print("\n[TEST 3] Rank-Deficient, Zero-Column & Perturbation Stability")
    A = np.array([[2.0, 0.0, -1.0], [1.0, 2.0, -1.0], [2.0, 2.0, 3.0]])
    a3_deg = 2.0 * A[:, 0] - 3.0 * A[:, 1]
    A_deg = np.column_stack([A[:, 0], A[:, 1], a3_deg])
    Q, R, rank = _mgs_qr(A_deg, tol=1e-12)
    assert rank == 2 and R[2, 2] == 0.0
    assert not np.any(np.isnan(Q)) and not np.any(np.isnan(R))

    # Zero-column matrix
    A_zero = np.column_stack([np.zeros(4), np.ones(4), np.arange(4, dtype=float)])
    Qz, Rz, rz = _mgs_qr(A_zero, tol=1e-12)
    assert rz == 0 and Rz[0, 0] == 0.0
    assert not np.any(np.isnan(Qz))

    # Rank-1 matrix
    u = np.array([1.0, -2.0, 3.0, -4.0])
    A_r1 = np.column_stack([u, 5.0 * u, -2.0 * u])
    _, R_r1, r_r1 = _mgs_qr(A_r1, tol=1e-12)
    assert r_r1 == 1 and R_r1[1, 1] == 0.0 and R_r1[2, 2] == 0.0

    # Near-singular perturbation delta=1e-14 vs 1e-5
    _, _, r_sub = _mgs_qr(np.column_stack([u, u + np.array([1e-14, 0, 0, 0])]), tol=1e-12)
    A_ind = np.column_stack([u, u + np.array([1e-5, 0, 0, 0])])
    Q_ind, _, r_ind = _mgs_qr(A_ind, tol=1e-12)
    Qh_ind, _ = _canon_qr(A_ind)
    assert r_sub == 1 and r_ind == 2
    assert np.allclose(Q_ind.T @ Q_ind, np.eye(2), atol=1e-9)
    assert np.allclose(Qh_ind.T @ Qh_ind, np.eye(2), atol=1e-14)
    print("  ✓ Rank-deficient, zero-column, rank-1 & perturbation boundary checks passed")


def test_qr_uniqueness_up_to_sign_diagonal():
    """Verify QR uniqueness when R_ii > 0 and sign-diagonal relation D = diag(+-1) across 500 matrices."""
    print("\n[TEST 4] QR Uniqueness & Sign-Diagonal D = diag(+-1) (500 matrices)")
    rng = np.random.default_rng(7004)
    max_diff = 0.0
    for _ in range(500):
        n = int(rng.integers(2, 10))
        A = rng.standard_normal((n, n)) + 3.0 * np.eye(n)
        Qm, Rm, rank = _mgs_qr(A)
        assert rank == n
        Qh, Rh = _canon_qr(A)
        max_diff = max(max_diff, float(np.max(np.abs(Qm - Qh))), float(np.max(np.abs(Rm - Rh))))

        signs = rng.choice([-1.0, 1.0], size=n)
        D = np.diag(signs)
        Q2, R2 = Qm @ D, D @ Rm
        assert np.allclose(Q2 @ R2, A, atol=1e-12)
        assert np.allclose(Qm.T @ Q2, D, atol=1e-12)

    assert max_diff < 1e-11, f"Canonical QR uniqueness mismatch: {max_diff}"
    print(f"  ✓ 500 invertible matrices verified unique to max_diff={max_diff:.2e}")


def test_ast_scene_static_integrity():
    """Static AST check for Sprint 7 3D scene (gracefully skipped if scene not yet created)."""
    print("\n[TEST 5] AST Scene Static Integrity Verification (ThreeDScene & 3D Helpers)")
    scene_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "linear_algebra",
            "sprint_07_gram_schmidt_qr",
            "scene_gram_schmidt_qr.py",
        )
    )
    if not os.path.exists(scene_path):
        print(f"  [SKIPPED] {scene_path} not yet created.")
        return

    with open(scene_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=scene_path)

    deprecated = {"CONFIG", "ShowCreation", "TexMobject", "TextMobject"}
    found_dep = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in deprecated]
    assert not found_dep, f"Deprecated symbols found: {found_dep}"

    forbidden = [
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {"eval", "exec"}
    ]
    assert not forbidden, f"Forbidden eval/exec found: {forbidden}"

    class_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    required_classes = {
        "BaseGramSchmidtQRScene",
        "Act1SpatialPuzzle",
        "Act2DynamicDeformation",
        "Act3AlgebraicMorph",
        "Act4DegenerateStressTest",
        "GramSchmidtQRScene",
    }
    assert required_classes.issubset(class_names), f"Missing classes: {required_classes - class_names}"

    threed = [
        n.name
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef)
        and any(
            (isinstance(b, ast.Name) and b.id == "ThreeDScene")
            or (isinstance(b, ast.Attribute) and b.attr == "ThreeDScene")
            for b in n.bases
        )
    ]
    assert threed, "Must define a ThreeDScene subclass"

    has_fixed_hud = any(
        isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Name) and n.func.id == "add_fixed_in_frame_mobjects")
            or (isinstance(n.func, ast.Attribute) and n.func.attr == "add_fixed_in_frame_mobjects")
        )
        for n in ast.walk(tree)
    )
    assert has_fixed_hud, "Must invoke add_fixed_in_frame_mobjects"
    print("  ✓ AST verified: ThreeDScene, all 6 classes, HUD pinning & zero deprecated/eval symbols")


def test_tweaks_prompt_1_sequential_stripping():
    """Verify Prompt 1: Sequential stripping, orthogonality & nested prefix span invariance."""
    print("\n[TEST 6] Prompt 1: 3D Orthogonal Stripping & Nested Span Invariance")
    A = np.array([[2.0, 0.0, -1.0], [1.0, 2.0, -1.0], [2.0, 2.0, 3.0]])
    Q, R, rank = _mgs_qr(A)
    assert rank == 3
    for k in (1, 2, 3):
        Ak = A[:, :k]
        Qk = Q[:, :k]
        PA = Ak @ np.linalg.inv(Ak.T @ Ak) @ Ak.T
        PQ = Qk @ Qk.T
        assert np.allclose(PA, PQ, atol=1e-12)
        assert np.isclose(np.sum(A[:, k - 1] ** 2), np.sum(R[:k, k - 1] ** 2), atol=1e-12)
    print("  ✓ Prompt 1 verified: prefix projectors & Pythagorean R-column norms match")


def test_tweaks_prompt_2_qr_factorization_and_least_squares():
    """Verify Prompt 2: A = QR, R_ij = q_i^T a_j, A^T A = R^T R & least squares back-sub."""
    print("\n[TEST 7] Prompt 2: Upper-Triangular R, Gram Identity & QR Least Squares")
    A = np.array([[2.0, 0.0, -1.0], [1.0, 2.0, -1.0], [2.0, 2.0, 3.0]])
    b = np.array([1.0, -2.0, 3.0])
    Q, R, _ = _mgs_qr(A)
    assert np.allclose(Q.T @ A, R, atol=1e-12)
    assert np.allclose(A.T @ A, R.T @ R, atol=1e-12)
    x_qr = _back_sub(R, Q.T @ b)
    x_ref, *_ = np.linalg.lstsq(A, b, rcond=None)
    assert np.allclose(x_qr, x_ref, atol=1e-12)
    assert np.allclose(x_qr, [1.0, -1.0, 1.0], atol=1e-12)
    print("  ✓ Prompt 2 verified: R = Q^T A, A^T A = R^T R & back-sub == lstsq")


def test_tweaks_prompt_3_and_4_cgs_drift_and_rank_deficiency():
    """Verify Prompt 3 (Lauchli Normal Eq collapse vs QR) & Prompt 4 (|det A| & R_33 = 0)."""
    print("\n[TEST 8] Prompts 3 & 4: Lauchli Conditioning, |det A| = prod R_ii & R_33 = 0")
    eps = 1e-8
    A_l = np.array([[1.0, 1.0, 1.0], [eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]])
    x_true = np.array([1.0, -2.0, 1.0])
    b_l = A_l @ x_true
    Qh, Rh = _canon_qr(A_l)
    x_qr = _back_sub(Rh, Qh.T @ b_l)
    assert np.linalg.norm(x_qr - x_true) / np.linalg.norm(x_true) < 1e-7

    ne_collapsed = False
    try:
        x_ne = np.linalg.solve(A_l.T @ A_l, A_l.T @ b_l)
        if np.linalg.norm(x_ne - x_true) > 1e-2:
            ne_collapsed = True
    except np.linalg.LinAlgError:
        ne_collapsed = True
    assert ne_collapsed, "Normal equations must collapse at Lauchli eps=1e-8"

    # Prompt 4: |det A| = prod R_ii (including det(Q) = -1 reflection case) & R_33 = 0
    A = np.array([[2.0, 0.0, -1.0], [1.0, 2.0, -1.0], [2.0, 2.0, 3.0]])
    Q, R, _ = _mgs_qr(A)
    assert np.isclose(abs(np.linalg.det(A)), np.prod(np.diag(R)), atol=1e-12)
    Q_refl = Q @ np.diag([1.0, 1.0, -1.0])
    A_refl = Q_refl @ R
    assert np.isclose(np.linalg.det(A_refl), -np.prod(np.diag(R)), atol=1e-12)

    a3_deg = A[:, 1] - A[:, 0]
    _, R_deg, rank_deg = _mgs_qr(np.column_stack([A[:, :2], a3_deg]), tol=1e-12)
    assert rank_deg == 2 and np.isclose(R_deg[2, 2], 0.0, atol=1e-12)
    print("  ✓ Prompt 3 verified: QR survives Lauchli eps=1e-8 while Normal Eq collapses")
    print("  ✓ Prompt 4 verified: |det A| = prod R_ii (with det Q = +-1) & R_33 = 0")


def main():
    print("=" * 70)
    print("ADVERSARIAL NUMERICAL VERIFICATION SUITE: SPRINT 7")
    print("Linear Algebra Curriculum — Gram-Schmidt Orthonormalization & QR")
    print("=" * 70)

    tests = [
        test_large_scale_randomized_qr,
        test_lauchli_and_hilbert_cgs_vs_mgs_vs_householder,
        test_rank_deficient_and_zero_column_stability,
        test_qr_uniqueness_up_to_sign_diagonal,
        test_ast_scene_static_integrity,
        test_tweaks_prompt_1_sequential_stripping,
        test_tweaks_prompt_2_qr_factorization_and_least_squares,
        test_tweaks_prompt_3_and_4_cgs_drift_and_rank_deficiency,
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
