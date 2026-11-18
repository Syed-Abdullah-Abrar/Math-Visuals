# Sprint 8: Orthogonal Matrices, Rotations $SO(n)$ & Givens/Householder

## Overview & Curriculum Context

Sprint 8 delivers the visual and numerical foundation for **Orthogonal Matrices, Proper Rotations $SO(n)$, Improper Reflections, Givens Rotations, and Householder Reflectors**. Rather than presenting $Q^\top Q = I$ as a purely algebraic identity, this module makes its geometric invariance unmistakable: orthogonal matrices are rigid-body isometries that preserve the Euclidean unit circle $S^1$, vector lengths $\|Q\mathbf{x}\|_2 = \|\mathbf{x}\|_2$, and pairwise angles—even when contrasted against area-preserving non-orthogonal shears ($\det M = 1$) that distort $S^1$ into a tilted ellipse.

Furthermore, this sprint visually unpacks:
1. **The Two Disconnected Halves of $O(n)$**: Proper rotations $SO(n)$ ($\det Q = +1$, orientation-preserving) versus improper reflections $O(n)\setminus SO(n)$ ($\det Q = -1$, orientation-reversing) separated by the singular wall $\det = 0$, together with the 2D **Cartan–Dieudonné composition** $H(\alpha_2)H(\alpha_1) = R(2\Delta\alpha)$ and **3D $SO(3)$ non-commutativity** ($R_1 R_2 \ne R_2 R_1$).
2. **Structured Orthogonal Zeroing Primitives**: Comparing **Givens rotations** $G(i, j, \theta) \in SO(n)$ (circular planar glide zeroing one coordinate while preserving matrix sparsity) against **Householder reflectors** $H = I - 2\mathbf{v}\mathbf{v}^\top/\|\mathbf{v}\|_2^2$ (perpendicular bounce across $\mathbf{v}^\perp$ zeroing entire subcolumns in one step).
3. **Numerical Stability & Edge Cases**: Demonstrating Householder $+\operatorname{sgn}(x_1)$ catastrophic cancellation avoidance at $\mathbf{x} = [1, 10^{-9}]^\top$, Givens `hypot` zero-pair and extreme-scale guards, and $Q^{1000}$ repeated multiplication stability ($\mathcal{O}(n\epsilon_{\text{mach}})$ drift vs. exponential blowup/collapse).

This module grounds core concepts across canonical literature:
- **Mathematics for Machine Learning (MML)**: Angles, orthogonality, orthogonal matrices, 2D/3D rotations, and Givens rotations (§3.4 & §3.9).
- **Introduction to Applied Linear Algebra (VMLS - Boyd & Vandenberghe)**: Geometric rotations/reflections and QR factorization (§7.1 & §10.4).
- **Immersive Linear Algebra (ILA - Ström, Åström et al.)**: Orthonormal bases, rigid transformations, and 3D rotations (Ch. 6).
- **Advanced Engineering Mathematics (Kreyszig 10th Ed.)**: Orthogonal matrices, unit-circle eigenvalues, Householder reflectors, and Givens QR factorization (§8.3 & §20.9).

---

## The 4-Part Pedagogical Protocol

```
linear_algebra/sprint_08_orthogonal_matrices_rotations/
├── README.md                                 # Master companion guide (this document)
├── brief.md                                  # Part 1: Geometric Consensus Brief (4 lenses)
├── storyboard.md                             # Part 2: 4-Act Visual Storyboard & Choreography
├── scene_orthogonal_matrices_rotations.py    # Part 3: Production ManimCE v0.18+ Animation Script
└── tweaks_prompts.md                         # Part 4: Interactive Tweaks & Intuition Stress-Test Prompts
```

| Part | Document | Primary Role |
|---|---|---|
| **Part 1** | [`brief.md`](./brief.md) | **Geometric Consensus Brief**: Synthesizes Geometric Action, Geometric Invariants, Algebraic-to-Visual Dictionary, and Dual Bridge (Frontier ML + GATE CS/DA). |
| **Part 2** | [`storyboard.md`](./storyboard.md) | **4-Act Visual Storyboard**: Frame-by-frame spatial choreography detailing timing, isotropic grids, isometric 3D callouts, and pinned HUD states. |
| **Part 3** | [`scene_orthogonal_matrices_rotations.py`](./scene_orthogonal_matrices_rotations.py) | **Production ManimCE Code**: Executable animation script adhering to all 8 ManimCE coding rules (zero overlap, `ValueTracker`, exact NumPy math). |
| **Part 4** | [`tweaks_prompts.md`](./tweaks_prompts.md) | **Interactive Tweaks & Intuition Prompts**: Parameter modification matrix and 4 rigorous conceptual stress-test problems with runnable NumPy proofs. |

---

## Canonical Textbook Cross-Reference Matrix

| Topic / Invariant | MML Reference | VMLS Reference | Kreyszig Reference | Visual / Algorithmic Manifestation |
|---|---|---|---|---|
| **Orthogonal Group $O(n)$** | §3.4 Def. 3.8 | §10.4 (pp. 189–191) | §8.3 eq. (3) | $Q^\top Q = Q Q^\top = I_n \iff Q^{-1} = Q^\top$ |
| **Length & Angle Isometry** | §3.4 eqs. (3.31)–(3.32) | §10.4 | §8.3 Thm 2 | $\|Q\mathbf{x}\|_2 = \|\mathbf{x}\|_2$, $(Q\mathbf{x})^\top(Q\mathbf{y}) = \mathbf{x}^\top\mathbf{y}$; $S^1$ preserved |
| **Determinant & Unit Spectrum** | §3.4 | §10.4 | §8.3 Thms 4–5 | $\det(Q) = \pm 1$, $|\lambda_i(Q)| = 1$, $\sigma_i(Q) = 1$, $\kappa_2(Q) = 1$ |
| **2D Rotation $R(\theta) \in SO(2)$** | §3.9.1 eq. (3.76) | §7.1 eq. (7.1) | §8.3 Prob. 1 | CCW rigid sweep ($\det = +1$), Abelian in 2D |
| **Reflection & Cartan–Dieudonné** | §3.9 | §7.1 (p. 130) | §8.3 | $H(\alpha) \in O(2)\setminus SO(2)$, $H(\alpha_2)H(\alpha_1) = R(2\Delta\alpha)$ |
| **3D $SO(3)$ Non-Commutativity** | §3.9.2–3.9.4 eqs. (3.77)–(3.79) | §7.1 | §8.3 | Side-by-side 3D cubes: $R_1(90^\circ)R_2(90^\circ)\mathbf{e}_1 \ne R_2(90^\circ)R_1(90^\circ)\mathbf{e}_1$ |
| **Givens Rotation $G(i,j,\theta)$** | §3.9.3 Def. 3.11 | §10.4 | §20.9 eqs. (7)–(11) | Circular planar glide zeroing $x_j \to 0$; `hypot` $(0,0)$ guard |
| **Householder Reflector $H$** | §3.8.3 / §3.9 | §10.4 | §20.9 eqs. (1)–(4) | $H = I - 2\mathbf{v}\mathbf{v}^\top/\|\mathbf{v}\|_2^2$, $+\operatorname{sgn}(x_1)$ cancellation guard |
| **ML: Orthogonal Init & RoPE** | §3.4, §3.9 | §10.4 | §8.3 | Isometric 100-layer gradient flow; RoPE relative offset $(n-m)$ |
| **GATE: Trace, Euler Axis & Powers** | §3.9 | §10.4 | §8.3 Ex. 5, Prob. 12 | $SO(3)$ axis $\lambda = +1$, $\operatorname{Tr}(R) = 1 + 2\cos\theta$, $Q^{1000}$ stability |

---

## Scene Manifest & Architecture

```
scene_orthogonal_matrices_rotations.py
├── BaseOrthogonalRotationsScene (Isotropic planes, 3D isometric projector, HUD factory, safe RightAngle/Arc)
├── Act1SpatialPuzzle (~18s)             [Scene]
├── Act2DynamicDeformation (~28s)        [Scene]
├── Act3AlgebraicMorph (~26s)            [Scene]
├── Act4DegenerateStressTest (~24s)      [Scene]
└── OrthogonalRotationsScene (~96s)      [Master Combined Scene]
```

### Scene Descriptions
1. **`Act1SpatialPuzzle`**: Contrasts area-preserving shear $M_{\text{shear}} = \begin{bmatrix} 1 & 1.2 \\ 0 & 1 \end{bmatrix}$ ($\det = 1$, yet warps $S^1$ into an ellipse and collapses $90^\circ \to 39.81^\circ$) against orthogonal $Q_{\text{pyth}} = \begin{bmatrix} 0.6 & -0.8 \\ 0.8 & 0.6 \end{bmatrix} \in SO(2)$ locking $S^1$, $\|\mathbf{w}\|_2$, and `RightAngle`.
2. **`Act2DynamicDeformation`**: Sweeps $R(\theta) \in SO(2)$, crosses the $\det = 0$ wall into an improper reflection ($\det = -1$, orientation flip), proves Cartan–Dieudonné double reflection $H(\alpha_2)H(\alpha_1) = R(2\Delta\alpha)$, and demonstrates 3D $SO(3)$ non-commutativity on side-by-side projected 3D cubes.
3. **`Act3AlgebraicMorph`**: Visualizes $Q^\top Q = I \implies Q^{-1} = Q^\top$, then compares Givens circular zeroing ($[1.2, 1.6]^\top \to [2.0, 0.0]^\top$) with Householder hyperplane reflection ($[1.2, 1.6]^\top \to [-2.0, 0.0]^\top$ across $\mathbf{v}^\perp$).
4. **`Act4DegenerateStressTest`**: Tests Householder $+\operatorname{sgn}(x_1)$ cancellation avoidance at $\mathbf{x} = [1, 10^{-9}]^\top$, Givens $(0, 0)$ safe guard, $Q^{1000}$ repeated multiplication drift vs. exponential blowup/collapse, and Givens vs. Householder $4 \times 4$ sparsity tradeoffs.
5. **`OrthogonalRotationsScene`**: Master composite scene orchestrating Acts 1–4 with clean updater teardown and `FadeOut` transitions.

---

## Key Concepts & Insights Summary

- **$\det(A) = 1$ Does Not Imply Orthogonality**: Area/volume preservation constrains only the product of singular values ($\prod \sigma_i = 1$). Only $Q^\top Q = I_n$ forces all singular values to unity ($\sigma_i = 1$), locking circles, lengths, and angles.
- **Two Disconnected Topological Components**: $SO(n)$ ($\det = +1$) and $O(n)\setminus SO(n)$ ($\det = -1$) cannot be continuously deformed into one another without passing through singular matrices ($\det = 0$). Composing any two reflections jumps back into $SO(n)$.
- **2D Rotations Commute, 3D Rotations Do Not**: In $\mathbb{R}^2$, all rotations share the same perpendicular axis; in $\mathbb{R}^3$, applying $R_1$ moves the axis of $R_2$, producing $R_1 R_2 \ne R_2 R_1$.
- **Givens vs. Householder Engineering Tradeoff**: Givens rotations act locally on two rows ($\mathcal{O}(n)$ per zero, ideal for sparse/banded matrices and streaming updates), while Householder reflectors zero entire subcolumns via rank-1 updates ($2\times$ fewer FLOPs for dense $QR$).
- **Sign-Aware Householder & Repeated Power Conditioning**: Choosing $\mathbf{v} = \mathbf{x} + \operatorname{sgn}(x_1)\|\mathbf{x}\|_2\mathbf{e}_1$ eliminates catastrophic cancellation on near-axis vectors, and $|\lambda_i(Q)| = 1$ bounds $Q^{1000}$ rounding drift to $\mathcal{O}(n\epsilon_{\text{mach}}) \sim 10^{-13}$ in `float64`.

---

## CLI Render Commands

Run all rendering commands from the project root using the project virtual environment:

```bash
# 1. Interactive / Low-Quality Preview (480p15) — Master Scene
.venv/bin/manim -pql linear_algebra/sprint_08_orthogonal_matrices_rotations/scene_orthogonal_matrices_rotations.py OrthogonalRotationsScene
.venv/bin/manim -ql linear_algebra/sprint_08_orthogonal_matrices_rotations/scene_orthogonal_matrices_rotations.py OrthogonalRotationsScene

# 2. Individual Modular Act Previews (480p15)
.venv/bin/manim -ql linear_algebra/sprint_08_orthogonal_matrices_rotations/scene_orthogonal_matrices_rotations.py Act1SpatialPuzzle
.venv/bin/manim -ql linear_algebra/sprint_08_orthogonal_matrices_rotations/scene_orthogonal_matrices_rotations.py Act2DynamicDeformation
.venv/bin/manim -ql linear_algebra/sprint_08_orthogonal_matrices_rotations/scene_orthogonal_matrices_rotations.py Act3AlgebraicMorph
.venv/bin/manim -ql linear_algebra/sprint_08_orthogonal_matrices_rotations/scene_orthogonal_matrices_rotations.py Act4DegenerateStressTest

# 3. High-Quality Production Render (1080p60 — Interactive -pqh and Headless -qh)
.venv/bin/manim -pqh linear_algebra/sprint_08_orthogonal_matrices_rotations/scene_orthogonal_matrices_rotations.py OrthogonalRotationsScene
.venv/bin/manim -qh --media_dir /tmp/manim_render_s08 linear_algebra/sprint_08_orthogonal_matrices_rotations/scene_orthogonal_matrices_rotations.py OrthogonalRotationsScene
```

---

## Programmatic Verification

Sprint 8 is covered by 5-tier invariant unit tests and 8-target adversarial stress tests in `tests/`:

```bash
# Run full 5-tier invariant test suite (including static AST checks)
.venv/bin/pytest tests/test_sprint08_invariants.py -v

# Run 8-target adversarial stress suite (100k vectors, Q^1000 drift, cancellation sweep, Prompts 1-4)
.venv/bin/pytest tests/test_adversarial_sprint8.py -v
.venv/bin/python tests/test_adversarial_sprint8.py
```
