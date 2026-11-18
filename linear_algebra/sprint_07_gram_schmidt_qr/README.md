# Sprint 7: Gram-Schmidt Orthonormalization & QR Decomposition

## Overview & Curriculum Context

Sprint 7 develops the geometric and numerical foundations of **Gram-Schmidt Orthonormalization** and **$QR$ Decomposition** in 3D space ($\mathbb{R}^3$) following the 3Blue1Brown visual standard. Starting from a skewed, full-rank matrix $A = [\mathbf{a}_1 \mid \mathbf{a}_2 \mid \mathbf{a}_3]$ whose non-orthogonal columns couple coordinates ($\mathbf{a}_i^\top \mathbf{a}_j \ne 0$), the module animates sequential orthogonal stripping onto the unit sphere $S^2$—converting $\{\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3\}$ into an orthonormal frame $Q = [\mathbf{q}_1 \mid \mathbf{q}_2 \mid \mathbf{q}_3]$ ($Q^\top Q = I_3$) while accumulating the upper-triangular coordinate recipe $R$ ($A = QR$).

Rather than presenting $A = QR$ as a static matrix algorithm, Sprint 7 ties every algebraic entry $R_{ij} = \mathbf{q}_i^\top \mathbf{a}_j$ and pivot $R_{ii} = \|\mathbf{u}_i\|_2 > 0$ directly to 3D projection shadows, perpendicular altitudes, and nested subspace invariance. It then bridges directly to $QR$ least squares ($R\hat{\mathbf{x}} = Q^\top\mathbf{b}$), linear-dependence early termination ($R_{33} = 0$), and the dramatic floating-point stability contrast between Classical Gram-Schmidt (CGS), Modified Gram-Schmidt (MGS), and Householder QR.

This module grounds core concepts across canonical literature:
- **Mathematics for Machine Learning (MML — Deisenroth, Faisal & Ong)**: Orthonormal bases (§3.5, Def. 3.9), inversion-free projection coordinates (§3.8.2, Eq. 3.66), and Gram-Schmidt orthogonalization (§3.8.3, Eqs. 3.67–3.71).
- **Introduction to Applied Linear Algebra (VMLS — Boyd & Vandenberghe)**: Orthonormal vectors & expansion (§5.3), Gram-Schmidt Algorithm 5.1 & early termination (§5.4), Thin/Full $QR$ factorization (§10.4), back-substitution & QR linear solves (§11.3, Alg. 11.1–11.2), pseudoinverse $A^\dagger = R^{-1}Q^\top$ (§11.5), and QR least squares (§12.1–12.3, Alg. 12.1).
- **Advanced Engineering Mathematics (Kreyszig 10th Ed.)**: Householder/Givens reductions and iterative $QR$ similarity eigenvalues (§20.9).

---

## The 4-Part Pedagogical Protocol

Sprint 7 strictly implements the repository's 4-Part Pedagogical Protocol:

```
linear_algebra/sprint_07_gram_schmidt_qr/
├── README.md                    # Master companion guide (this document)
├── brief.md                     # Part 1: Geometric Consensus Brief (4 lenses)
├── storyboard.md                # Part 2: 4-Act 3D Visual Storyboard & Choreography
├── scene_gram_schmidt_qr.py     # Part 3: Production ManimCE ThreeDScene Script
└── tweaks_prompts.md            # Part 4: Parameter Tweaks & Intuition Stress-Test Prompts
```

| Part | Document | Primary Role |
|---|---|---|
| **Part 1** | [`brief.md`](./brief.md) | **Geometric Consensus Brief**: Synthesizes Geometric Action, Geometric Invariants, Algebraic-to-Visual Dictionary, and Dual Bridge (Muon / Orthogonal Init & Lauchli QR Conditioning; GATE back-substitution, $|\det A| = \prod R_{ii}$, rank from $R_{ii}=0$). |
| **Part 2** | [`storyboard.md`](./storyboard.md) | **4-Act Visual Storyboard**: Frame-by-frame 3D choreography on isotropic `ThreeDAxes`, detailing camera orientations, HUD pinning, semantic colors, and invariants across Acts 1–4. |
| **Part 3** | [`scene_gram_schmidt_qr.py`](./scene_gram_schmidt_qr.py) | **Production ManimCE Code**: Runnable `ThreeDScene` implementation adhering to all 8 ManimCE coding rules, custom 3D `Polygon` right-angle markers, and viewport-pinned HUD panels. |
| **Part 4** | [`tweaks_prompts.md`](./tweaks_prompts.md) | **Interactive Tweaks & Intuition Prompts**: Parameter matrix and 4 rigorous conceptual stress-test prompts with executable NumPy proofs. |

---

## Canonical Textbook Cross-Reference Matrix

| Topic / Invariant | MML Reference | VMLS Reference | Kreyszig Reference | Visual / Algorithmic Manifestation |
|---|---|---|---|---|
| **Orthonormal Basis & Expansion** | §3.5 (Def 3.9, Eq. 3.33–3.35), §3.8.2 (Eq. 3.66) | §5.3 (Eq. 5.4–5.5) | §7.9 / §20.9 | $Q^\top Q = I_3$, $\mathbf{b} = Q(Q^\top\mathbf{b})$ without inversion |
| **Sequential GS Stripping** | §3.8.3 (Eq. 3.67–3.71) | §5.4 (Alg 5.1, pp. 97–102) | §20.9 | $\mathbf{u}_k = \mathbf{a}_k - \sum_{i<k}(\mathbf{q}_i^\top\mathbf{a}_k)\mathbf{q}_i$ animated with `ValueTracker` |
| **Nested Prefix Span Invariance** | §3.8.3 | §5.4 (Invariants 1–4, p. 99) | — | $\operatorname{span}(\mathbf{a}_1,\dots,\mathbf{a}_k) = \operatorname{span}(\mathbf{q}_1,\dots,\mathbf{q}_k)$, $P_k = Q_k Q_k^\top$ |
| **Thin/Full $A=QR$ & Uniqueness** | §3.8.3 (basis view) | §10.4 (pp. 189–191) | §20.9 | $R_{ij} = \mathbf{q}_i^\top\mathbf{a}_j$ ($i \le j$), $R_{ii}>0$ fixes sign diagonal $D=I$ |
| **Gram Isometry & Volume** | — | §10.4 (Ex 10.39–10.40) | — | $A^\top A = R^\top R$, $|\det A| = \prod R_{ii}$ ($\det Q = \pm 1$) |
| **QR Back-Substitution & LS** | §3.8 | §11.3 (Alg 11.1–11.2), §11.5, §12.1–12.3 (Alg 12.1) | §20.5 | $R\hat{\mathbf{x}} = Q^\top\mathbf{b}$ in $n^2$ flops; $A^\dagger = R^{-1}Q^\top$ |
| **Early Termination & Rank** | §3.8.3 | §5.4 (Alg 5.1 Step 2, p. 100) | — | Coplanar $\mathbf{a}_3^{\text{deg}} \in \mathcal{S}_2 \implies \mathbf{u}_3 = \mathbf{0}, R_{33}=0$ |
| **CGS vs. MGS vs. Householder** | — | §5.4 (p. 102) | §20.9 | Lauchli $\epsilon=10^{-8}$ drift: CGS $\mathcal{O}(\epsilon\kappa^2)$ vs MGS $\mathcal{O}(\epsilon\kappa)$ vs HH $\mathcal{O}(\epsilon)$ |

---

## Mathematical Specification & Canonical Constants

Sprint 7 uses a canonical $3 \times 3$ matrix built from Pythagorean quadruples so every column of $Q$, entry of $R$, projection shadow, and back-substitution solution is exact rational/integer arithmetic:

$$A = \begin{bmatrix} 2 & 0 & -1 \\ 1 & 2 & -1 \\ 2 & 2 & 3 \end{bmatrix} = \underbrace{\frac{1}{3}\begin{bmatrix} 2 & -2 & -1 \\ 1 & 2 & -2 \\ 2 & 1 & 2 \end{bmatrix}}_{Q} \underbrace{\begin{bmatrix} 3 & 2 & 1 \\ 0 & 2 & 1 \\ 0 & 0 & 3 \end{bmatrix}}_{R}$$

- **Step 1**: $\mathbf{u}_1 = [2, 1, 2]^\top$, $R_{11} = \|\mathbf{u}_1\|_2 = 3$, $\mathbf{q}_1 = \frac{1}{3}[2, 1, 2]^\top$.
- **Step 2**: $R_{12} = \mathbf{q}_1^\top \mathbf{a}_2 = 2$, $\mathbf{u}_2 = \mathbf{a}_2 - 2\mathbf{q}_1 = \frac{2}{3}[-2, 2, 1]^\top$, $R_{22} = \|\mathbf{u}_2\|_2 = 2$, $\mathbf{q}_2 = \frac{1}{3}[-2, 2, 1]^\top$.
- **Step 3**: $R_{13} = 1, R_{23} = 1$, planar shadow $\mathbf{p}_{12,3} = \mathbf{q}_1 + \mathbf{q}_2 = [0, 1, 1]^\top$, $\mathbf{u}_3 = \mathbf{a}_3 - \mathbf{p}_{12,3} = [-1, -2, 2]^\top$, $R_{33} = \|\mathbf{u}_3\|_2 = 3$, $\mathbf{q}_3 = \frac{1}{3}[-1, -2, 2]^\top$.
- **Exact Expansion & Solve**: Target $\mathbf{b} = [1, -2, 3]^\top \implies Q^\top \mathbf{b} = [2, -1, 3]^\top \implies R\hat{\mathbf{x}} = Q^\top\mathbf{b}$ gives $\hat{\mathbf{x}} = [1, -1, 1]^\top$, and $\det(Q) = +1 \implies \det(A) = R_{11}R_{22}R_{33} = 18$.
- **Degenerate Probe**: $\mathbf{a}_3^{\text{deg}} = \mathbf{a}_2 - \mathbf{a}_1 = [-2, 1, 0]^\top = -1\mathbf{q}_1 + 2\mathbf{q}_2 \in \operatorname{span}(\mathbf{q}_1, \mathbf{q}_2) \implies \mathbf{u}_3^{\text{deg}} = \mathbf{0}, R_{33} = 0$.

---

## Scene Manifest & Architecture

```
scene_gram_schmidt_qr.py
├── BaseGramSchmidtQRScene(ThreeDScene)   # Isotropic ThreeDAxes, HUD pin/unpin, 3D Polygon elbows, unit sphere
├── Act1SpatialPuzzle                     # Skewed 3D basis + parallelepiped + target b
├── Act2DynamicDeformation                # Unit sphere S^2 + ValueTracker stripping a_1,a_2,a_3 -> q_1,q_2,q_3
├── Act3AlgebraicMorph                    # Tip-to-tail A=QR, expansion b=Q(Q^T b), back-sub & |det A|=18
├── Act4DegenerateStressTest              # Coplanar u_3 -> 0 collapse (R_33=0) & CGS vs MGS vs HH drift card
└── GramSchmidtQRScene                    # Master composite 4-act 1080p60 production scene
```

### 8 ManimCE Coding Rules Compliance
1. **3b1b Semantic Palette**: Exact hex constants (`#83C167`, `#FC6255`, `#58C4DD`, `#2A9D8F`, `#F7D96F`, `#F9E498`, `#F7768E`, `#1C1C24`).
2. **Isotropic 3D Viewport Clamping**: `ThreeDAxes` uses `5.6 x 5.6 x 5.6` on `[-4, 4]^3` (`0.7` units/tick) so $S^2$ and $90^\circ$ corners render without z-axis warping.
3. **Zero Text Overlap & Fixed HUD Pinning**: Every card uses `make_hud_panel` + `add_fixed_in_frame_mobjects`, and `unpin_hud` clears both `fixed_in_frame_mobjects` and `self.mobjects`.
4. **Dynamic Continuous Tracking**: `ValueTracker` + `always_redraw` arrows + re-pinned `DecimalNumber(..., mob_class=Text)` track $\mathbf{q}_i^\top \mathbf{v}_k(t) \to 0.00$ live.
5. **3D Geometric Invariant Markers**: Custom `create_right_angle_3d` (`Polygon` + `VMobject`) enforces 3D perpendicularity markers (avoiding `RightAngle`'s xy-plane restriction).
6. **NumPy Precision**: Import-time `assert np.allclose(...)` checks validate all canonical vectors, orthogonality, $A=QR$, back-substitution, and degenerate collapse.
7. **Pacing & 1080p60 Render Budget**: Targeted camera sweeps pause ambient rotation during static holds for clean frame caching.
8. **Modern ManimCE Syntax**: Zero `CONFIG`, `ShowCreation`, `TexMobject`, `TextMobject`, `eval`, or `exec`.

---

## CLI Render Commands

Run all commands from the repository root (`/home/uni-kernel/dev/math_cs`) using the project virtual environment:

```bash
# 1. Low-Quality Preview (480p15) — Interactive (-pql) or Headless (-ql)
.venv/bin/manim -pql linear_algebra/sprint_07_gram_schmidt_qr/scene_gram_schmidt_qr.py GramSchmidtQRScene
.venv/bin/manim -ql linear_algebra/sprint_07_gram_schmidt_qr/scene_gram_schmidt_qr.py GramSchmidtQRScene

# 2. Individual Modular Acts (480p15)
.venv/bin/manim -ql linear_algebra/sprint_07_gram_schmidt_qr/scene_gram_schmidt_qr.py Act1SpatialPuzzle
.venv/bin/manim -ql linear_algebra/sprint_07_gram_schmidt_qr/scene_gram_schmidt_qr.py Act2DynamicDeformation
.venv/bin/manim -ql linear_algebra/sprint_07_gram_schmidt_qr/scene_gram_schmidt_qr.py Act3AlgebraicMorph
.venv/bin/manim -ql linear_algebra/sprint_07_gram_schmidt_qr/scene_gram_schmidt_qr.py Act4DegenerateStressTest

# 3. High-Quality Production Render (1080p60) — Interactive (-pqh) or Headless (-qh)
.venv/bin/manim -pqh linear_algebra/sprint_07_gram_schmidt_qr/scene_gram_schmidt_qr.py GramSchmidtQRScene
.venv/bin/manim -qh linear_algebra/sprint_07_gram_schmidt_qr/scene_gram_schmidt_qr.py GramSchmidtQRScene
```

The final 1080p60 production render is stored at:
`linear_algebra/animations/sprint_07_GramSchmidtQRScene_1080p60.mp4`

---

## Programmatic Verification & Test Results

Sprint 7 is verified by a 5-tier invariant suite and an 8-part adversarial stress suite in `tests/`:

```bash
# Run 5-Tier Invariant Suite (67 tests incl. Tier 5 AST & Syntax checks)
.venv/bin/pytest tests/test_sprint07_invariants.py -v

# Run 8-Part Adversarial Numerical & Prompt Verification Suite
.venv/bin/python tests/test_adversarial_sprint7.py
.venv/bin/pytest tests/test_adversarial_sprint7.py -v
```

---

## Key Concepts & Insights Summary

1. **Gram-Schmidt Is Sequential Subspace Decoupling**: Removing $\operatorname{proj}_{\operatorname{span}(\mathbf{q}_1,\dots,\mathbf{q}_{k-1})}(\mathbf{a}_k)$ strips away every component parallel to previously built axes without ever altering $\operatorname{span}(\mathbf{a}_1,\dots,\mathbf{a}_k)$, turning an oblique parallelepiped into a unit-sphere orthonormal tripod $\{\mathbf{q}_1, \dots, \mathbf{q}_k\}$.
2. **Upper-Triangular $R$ Is a Causal Coordinate Ledger**: Entry $R_{ij} = \mathbf{q}_i^\top \mathbf{a}_j$ records how much of $\mathbf{q}_i$ was stripped from $\mathbf{a}_j$, while $R_{ii} = \|\mathbf{u}_i\|_2 > 0$ records the surviving perpendicular altitude. Because $\mathbf{a}_j$ never depends on future $\mathbf{q}_{i>j}$, $R$ is strictly upper triangular, $A^\top A = R^\top R$ preserves all norms and angles, and enforcing $R_{ii} > 0$ resolves the $2^k$ sign ambiguity $D = \operatorname{diag}(\pm 1)$ to make thin $QR$ unique.
3. **Inversion-Free Coordinates, Volume & Back-Substitution**: Orthonormality collapses $B(B^\top B)^{-1}B^\top \mathbf{x}$ to $QQ^\top\mathbf{x}$, reduces least squares to $R\hat{\mathbf{x}} = Q^\top\mathbf{b}$ ($n^2$ back-substitution flops), gives $|\det(A)| = \prod R_{ii}$ (distinguishing $\det Q = \pm 1$), and immediately detects linear dependence when a residual collapses ($\mathbf{u}_k = \mathbf{0} \iff R_{kk} = 0$).
4. **Why Algorithm Choice Matters Numerically**: Classical GS, Modified GS, and Householder QR are identical in exact arithmetic, yet on ill-conditioned inputs (e.g., Lauchli $\epsilon=10^{-8}$, $\kappa \approx 1.73\times 10^8$), CGS loses orthogonality at $\mathcal{O}(\epsilon_{\text{mach}}\kappa^2) \sim 0.71$, MGS improves to $\mathcal{O}(\epsilon_{\text{mach}}\kappa) \sim 10^{-8}$, and Householder QR stays at $\mathcal{O}(\epsilon_{\text{mach}}) \sim 10^{-16}$ while avoiding the $\kappa^2$ Normal Equations collapse ($A^\top A$ rounding to singular in `float64`).
