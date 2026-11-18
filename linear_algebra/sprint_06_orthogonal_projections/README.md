# Sprint 6: Orthogonal Projections onto 1D Lines & $k$-D Subspaces

## Overview & Curriculum Context
Sprint 6 delivers a 3D visual intuition and numerical verification module for orthogonal projections onto 1D lines and $k$-dimensional subspaces, inspired by Grant Sanderson (3Blue1Brown). Subclassing ManimCE's `ThreeDScene`, the scene moves from projecting onto a 1D line (MML Example 3.10) to sweeping a live candidate $\hat{\mathbf{y}}(s) = A\mathbf{x}(s)$ across a 2D subspace plane $U = \mathcal{C}(A) \subset \mathbb{R}^3$ (MML Example 3.11) while plotting reconstruction error $\|\mathbf{y} - \hat{\mathbf{y}}(s)\|_2$ in a pinned 2D corner graph—demonstrating that global minimum distance occurs precisely at orthogonality $\mathbf{y} - \hat{\mathbf{y}} \perp \mathcal{C}(A)$.

The module then derives the normal equations $A^\top A\hat{\mathbf{x}} = A^\top\mathbf{y}$, the symmetric projection matrix $P = A(A^\top A)^{-1}A^\top$, its orthogonal complement $I - P$, and stress-tests affine flats, rank-deficient normal equations (pseudoinverse $AA^+$ and Ridge), Lauchli $\kappa^2$ ill-conditioning, and asymmetric oblique projectors.

---

## The 4-Part Pedagogical Protocol
```
linear_algebra/sprint_06_orthogonal_projections/
├── README.md                         # Curriculum overview, math specs, CLI & key insights
├── brief.md                          # Part 1: Geometric Consensus Brief & Dual Bridge
├── storyboard.md                     # Part 2: 4-Act Visual Storyboard (92s 3D choreography)
├── scene_orthogonal_projections.py   # Part 3: Production ManimCE v0.21.0 3D script
└── tweaks_prompts.md                 # Part 4: Parameter matrix & 4 NumPy proof prompts
```

| Deliverable | File | Description |
|---|---|---|
| **Part 1: Brief** | [`brief.md`](brief.md) | Geometric Action, Geometric Invariants, Algebraic-to-Visual Dictionary, and Frontier ML / GATE CS/DA Bridges. |
| **Part 2: Storyboard** | [`storyboard.md`](storyboard.md) | 70s 4-Act Visual Storyboard with 3D camera specs, HUD pinning layout, and 5-column choreography tables. |
| **Part 3: Production Scene** | [`scene_orthogonal_projections.py`](scene_orthogonal_projections.py) | Runnable ManimCE v0.21.0 `ThreeDScene` script following all 8 coding rules and pinned 2D corner graph updates. |
| **Part 4: Tweaks & Prompts** | [`tweaks_prompts.md`](tweaks_prompts.md) | Interactive parameter configuration matrix + 4 intuition stress-test prompts with executable NumPy `assert` proofs. |

---

## Canonical Textbook Cross-Reference Matrix

| Canonical Reference | Chapter / Section | Core Mathematical Concept | Visual Manifestation in Sprint 6 |
|---|---|---|---|
| **Mathematics for Machine Learning (MML)** | §3.8.1 (pp. 82–85, Ex. 3.10) | 1D line projection $\lambda = \frac{\mathbf{b}^\top\mathbf{x}}{\|\mathbf{b}\|^2}$, $P_1 = \frac{\mathbf{b}\mathbf{b}^\top}{\|\mathbf{b}\|^2}$ | Act 1 sliding coordinate $\lambda\mathbf{b}$ & idempotence $\pi^2=\pi$ |
| **Mathematics for Machine Learning (MML)** | §3.8.2 (pp. 85–88, Ex. 3.11) | Normal equations $A^\top A\hat{\mathbf{x}} = A^\top\mathbf{y}$, $P = A(A^\top A)^{-1}A^\top$ | Act 2 plane $U = \mathcal{C}(A)$ & Act 3 algebraic morph |
| **Mathematics for Machine Learning (MML)** | §3.6 & §3.8.4 (pp. 79–90) | Orthogonal complement $I - P$, affine $\pi_L(\mathbf{x}) = \mathbf{x}_0 + P(\mathbf{x}-\mathbf{x}_0)$ | Act 3 Pythagorean split & Act 4 shifted yellow plane $L$ |
| **Mathematics for Machine Learning (MML)** | §9.2.3 & §10.3 | Ridge $(A^\top A + \lambda I)^{-1}A^\top$ & PCA reconstruction $P_M = BB^\top$ | Act 4 Ridge rescue & Prompt 3 ML unification |
| **Applied Linear Algebra (VMLS - Boyd)** | §11.5, §12.1–12.3, §15 | Direct optimality proof `(12.7)`, Orthogonality Principle `(12.9)`, QR solve | Act 2 convex error curve $d(s) = \sqrt{6+2s^2}$ & Lauchli QR check |
| **Immersive Linear Algebra (ILA)** | Ch. 3 & Ch. 6 | Dot-product projection & orthogonal projections onto subspaces | 3D right-angle elbow markers & interactive shadow intuition |
| **Kreyszig / GATE CS/DA** | Kreyszig Ch. 7–8, GATE LA | Spectrum $\{0, 1\}$, $\operatorname{Tr}(P) = \operatorname{rank}(P)$, rank-1 outer products | Act 3 invariants card & Prompt 4 instant spectral readout |

---

## Mathematical Specification & Canonical Constants

All scene coordinates remain strictly inside `[-4, 4]^3` with exact integer projections and residuals:

- **Act 1 — 1D Line Projection (MML Example 3.10)**:
  $$\mathbf{b} = \begin{bmatrix} 1 \\ 2 \\ 2 \end{bmatrix}, \quad \mathbf{x}_{1D} = \begin{bmatrix} 3 \\ 3 \\ 0 \end{bmatrix} \implies \lambda = \frac{\mathbf{b}^\top \mathbf{x}_{1D}}{\|\mathbf{b}\|_2^2} = \frac{9}{9} = 1, \quad \hat{\mathbf{y}}_{1D} = \begin{bmatrix} 1 \\ 2 \\ 2 \end{bmatrix}, \quad \mathbf{r}_{1D} = \begin{bmatrix} 2 \\ 1 \\ -2 \end{bmatrix}$$
  $$P_1 = \frac{\mathbf{b}\mathbf{b}^\top}{\|\mathbf{b}\|_2^2} = \frac{1}{9}\begin{bmatrix} 1 & 2 & 2 \\ 2 & 4 & 4 \\ 2 & 4 & 4 \end{bmatrix}, \qquad \|\hat{\mathbf{y}}_{1D}\|_2^2 + \|\mathbf{r}_{1D}\|_2^2 = 9 + 9 = 18 = \|\mathbf{x}_{1D}\|_2^2$$

- **Acts 2 & 3 — 2D Subspace Projection & Normal Equations (MML Example 3.11)**:
  $$A = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{bmatrix}, \quad U = \mathcal{C}(A) = \{ \mathbf{v} \in \mathbb{R}^3 \mid x - 2y + z = 0 \}, \quad \mathbf{y} = \begin{bmatrix} 3 \\ -1 \\ 1 \end{bmatrix}$$
  $$A^\top A = \begin{bmatrix} 3 & 3 \\ 3 & 5 \end{bmatrix} \ (\det = 6), \quad \hat{\mathbf{x}} = \begin{bmatrix} 2 \\ -1 \end{bmatrix}, \quad \hat{\mathbf{y}}^* = A\hat{\mathbf{x}} = \begin{bmatrix} 2 \\ 1 \\ 0 \end{bmatrix}, \quad \mathbf{r}^* = \mathbf{y} - \hat{\mathbf{y}}^* = \begin{bmatrix} 1 \\ -2 \\ 1 \end{bmatrix}$$
  $$P = A(A^\top A)^{-1}A^\top = \frac{1}{6}\begin{bmatrix} 5 & 2 & -1 \\ 2 & 2 & 2 \\ -1 & 2 & 5 \end{bmatrix}, \qquad I - P = \frac{1}{6}\begin{bmatrix} 1 & -2 & 1 \\ -2 & 4 & -2 \\ 1 & -2 & 1 \end{bmatrix}$$
  - **Act 2 Live Sliding Path**: $\mathbf{x}(s) = [2 + s, -1 - s]^\top$ for $s \in [-2.0, 1.5]$ gives $\hat{\mathbf{y}}(s) = [2 + s, 1, -s]^\top$ and exact error curve $d(s) = \|\mathbf{y} - A\mathbf{x}(s)\|_2 = \sqrt{6 + 2s^2}$, minimized at $s = 0$ ($d(0) = \sqrt{6}$).

- **Act 4 — Degenerate & Extension Cases**:
  - **Affine Flat**: $L = \mathbf{x}_0 + U$ with $\mathbf{x}_0 = [0, 0, 1]^\top$, $\mathbf{y}_{\text{aff}} = [3, -1, 2]^\top \mapsto \pi_L(\mathbf{y}_{\text{aff}}) = [2, 1, 1]^\top$ ($\pi_L(\mathbf{0}) \ne \mathbf{0}$).
  - **Rank-Deficient & Ridge**: $A_{\text{deg}} = [\mathbf{a}_1 \mid 2\mathbf{a}_1]$ ($\det(A_{\text{deg}}^\top A_{\text{deg}}) = 0$) resolved via $P = A_{\text{deg}} A_{\text{deg}}^+ = \frac{1}{3}\mathbf{1}\mathbf{1}^\top$ and Ridge $(A_{\text{deg}}^\top A_{\text{deg}} + \lambda I)^{-1}A_{\text{deg}}^\top$.
  - **Lauchli Ill-Conditioning**: $A_\varepsilon = \begin{bmatrix} 1 & 1 \\ \varepsilon & 0 \\ 0 & \varepsilon \end{bmatrix}$ ($\varepsilon = 10^{-8}$) collapses `A.T @ A` to rank 1 in float64 while QR/SVD `lstsq` remains accurate.
  - **Oblique Contrast**: $P_{\text{ob}} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ -1 & 2 & 0 \end{bmatrix}$ satisfies $P_{\text{ob}}^2 = P_{\text{ob}}$ with $P_{\text{ob}}^\top \ne P_{\text{ob}}$ and $\|P_{\text{ob}}\|_2 = \sqrt{6} > 1$.

---

## Scene Manifest & Architecture

`scene_orthogonal_projections.py` implements the 6-class modular architecture:
- `BaseOrthogonalProjectionsScene(ThreeDScene)`: Configures `#1C1C24`, `ThreeDAxes`, $\phi = 70^\circ, \theta = -45^\circ$, `rate = 0.08`, `pin_hud`/`unpin_hud`, custom 3D right-angle elbow builder, and `run_act1`..`run_act4`.
- `Act1SpatialPuzzle`: 1D line projection, scalar coordinate $\lambda$, and $\pi^2 = \pi$ idempotence.
- `Act2DynamicDeformation`: Live 3D sliding candidate $\hat{\mathbf{y}}(s)$ with pinned 2D corner error curve $d(s) = \sqrt{6 + 2s^2}$.
- `Act3AlgebraicMorph`: Normal equations morph to $P = A(A^\top A)^{-1}A^\top$, $I - P$, Pythagorean split, and invariants.
- `Act4DegenerateStressTest`: Affine shift, rank-deficient $A^+$ & Ridge, Lauchli $\kappa^2$, and oblique vs orthogonal contrast.
- `OrthogonalProjectionsScene`: Full 4-Act composite master scene (`linear_algebra/animations/sprint_06_OrthogonalProjectionsScene_1080p60.mp4`).

---

## CLI Execution & Render Commands

```bash
# 1. Automated Invariant & Adversarial Test Suites
.venv/bin/pytest tests/test_sprint06_invariants.py tests/test_adversarial_sprint6.py -v
.venv/bin/python tests/test_adversarial_sprint6.py

# 2. Preview Low-Resolution Renders (-pql with window / -ql headless, 480p15)
.venv/bin/manim -pql linear_algebra/sprint_06_orthogonal_projections/scene_orthogonal_projections.py OrthogonalProjectionsScene
.venv/bin/manim -ql linear_algebra/sprint_06_orthogonal_projections/scene_orthogonal_projections.py OrthogonalProjectionsScene
.venv/bin/manim -ql linear_algebra/sprint_06_orthogonal_projections/scene_orthogonal_projections.py Act2DynamicDeformation

# 3. Production High-Resolution 1080p60 Renders (-pqh with preview / -qh headless)
.venv/bin/manim -pqh linear_algebra/sprint_06_orthogonal_projections/scene_orthogonal_projections.py OrthogonalProjectionsScene
.venv/bin/manim -qh linear_algebra/sprint_06_orthogonal_projections/scene_orthogonal_projections.py OrthogonalProjectionsScene
```

---

## Mathematical Invariants & Verification Checklist
- [x] Idempotence $\pi^2 = \pi$ ($P^2 = P$) and fixed-point containment $P\mathbf{u} = \mathbf{u}$ for $\mathbf{u} \in U$
- [x] Symmetry $P^\top = P$ separating orthogonal projectors ($\|P\|_2 = 1$) from oblique projectors ($\|P_{\text{ob}}\|_2 > 1$)
- [x] Normal equations $A^\top A\hat{\mathbf{x}} = A^\top\mathbf{y}$ equivalent to residual orthogonality $A^\top(\mathbf{y} - A\hat{\mathbf{x}}) = \mathbf{0}$
- [x] Spectral & dimension invariants: $\operatorname{spec}(P) \subseteq \{0, 1\}$, $\operatorname{Tr}(P) = \operatorname{rank}(P) = \dim(U)$
- [x] Orthogonal complement $I - P$ onto $U^\perp = \mathcal{N}(A^\top)$ and Pythagorean split $\|\mathbf{y}\|_2^2 = \|P\mathbf{y}\|_2^2 + \|(I - P)\mathbf{y}\|_2^2$
- [x] Affine shift $\pi_L(\mathbf{x}) = \mathbf{x}_0 + P(\mathbf{x} - \mathbf{x}_0)$, rank-deficient $A^+$ / Ridge stability, and Lauchli $\kappa(A^\top A) = \kappa(A)^2$ QR survival

---

## Key Concepts & Insights Summary
1. **Orthogonality Is the Exact Equivalent of Minimum Euclidean Distance**: Watching the live corner error graph $d(s) = \|\mathbf{y} - A\mathbf{x}(s)\|_2$ descend along a parabola to its unique minimum precisely when the 3D right-angle elbow locks into place at $\hat{\mathbf{y}}^*$ makes the Orthogonality Principle visually self-evident.
2. **Symmetry Distinguishes Minimum Distance from Mere Idempotence**: Every idempotent matrix ($P^2 = P$) freezes its image subspace and has eigenvalues in $\{0, 1\}$ with $\operatorname{Tr}(P) = \operatorname{rank}(P)$, yet an asymmetric projector ($P_{\text{ob}}^\top \ne P_{\text{ob}}$) casts skewed shadows that amplify orthogonal components ($\|P_{\text{ob}}\|_2 = 1/\sin\theta_{\min} > 1$) and yield strictly suboptimal approximations.
3. **Never Square Condition Numbers Unnecessarily**: Forming normal equations $A^\top A$ squares the condition number ($\kappa(A^\top A) = \kappa(A)^2$), wiping out small singular values in IEEE-754 float64 on matrices like Lauchli ($1 + \varepsilon^2 \to 1.0$ at $\varepsilon = 10^{-8}$). QR factorization ($P = QQ^\top$, $\hat{\mathbf{x}} = R^{-1}Q^\top\mathbf{y}$) and SVD pseudoinverses preserve conditioning, while Ridge regularization $(A^\top A + \lambda I)^{-1}A^\top$ restores invertibility at the controlled cost of shrinkage ($H_\lambda^2 \ne H_\lambda$).
