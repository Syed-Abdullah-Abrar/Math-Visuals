# Sprint 5: Norms ($L_1, L_2, L_\infty$), Inner Products & Cauchy-Schwarz

## Overview & Curriculum Context

Sprint 5 builds the geometric foundation for distance, angle, and curvature in $\mathbb{R}^n$ following the 3Blue1Brown visual intuition standard. Rather than memorizing norm formulas and dot-product rules in isolation, this module reveals how norms and inner products sculpt space:
1. **$L_p$ Unit Ball Morph ($p=1 \to 2 \to \infty$)**: Varying $p$ continuously inflates the unit ball $\|\mathbf{x}\|_p = 1$ from the sharp $L_1$ taxicab diamond through the rotation-invariant $L_2$ Euclidean circle to the $L_\infty$ Chebyshev square while keeping axis vertices pinned at $\pm\mathbf{e}_i$.
2. **Dot Product as Ruler Projection**: $\mathbf{x}^\top\mathbf{y} = \|\mathbf{x}\|_2\|\mathbf{y}\|_2\cos\omega$ is animated directly as dropping a perpendicular shadow from $\mathbf{x}$ onto $\mathbf{y}$'s ruler line and scaling by $\|\mathbf{y}\|_2$.
3. **Cauchy-Schwarz & SPD Inner Products**: $|\langle\mathbf{x},\mathbf{y}\rangle| \le \|\mathbf{x}\|\|\mathbf{y}\|$ traps $\cos\omega \in [-1, 1]$ and saturates exclusively at collinearity, while symmetric positive-definite matrices $A = A^\top \succ 0$ warp the Euclidean circle into an $A$-ellipsoid and redefine orthogonality ($\mathbf{x}^\top A\mathbf{y} = 0$).
4. **Axiom Stress-Testing**: Four degenerate regimes—non-convex $p=0.5$ quasi-norm stars, $L_1$/$L_\infty$ parallelogram law failures, $\mathbf{x}=\mathbf{0}$ undefined angles, and indefinite $A_{\text{indef}} = \operatorname{diag}(1,-1)$ hyperbolas—pinpoint the exact mathematical boundaries separating valid norms and inner products from invalid forms.

Canonical textbook grounding:
- **Mathematics for Machine Learning (MML — Deisenroth, Faisal & Ong)**: Norms (§3.1), Inner Products & SPD Theorem 3.5 (§3.2), Lengths/Metrics & Cauchy-Schwarz (§3.3), Angles & Orthogonality (§3.4), 1D Ruler Projection (§3.8.1), Cholesky (§4.3).
- **Introduction to Applied Linear Algebra (VMLS — Boyd & Vandenberghe)**: Norms, RMS & Chebyshev (§3.1), Distance (§3.2), Standard Deviation & Z-scores (§3.3), Angle, Cauchy-Schwarz Proof & Parallelogram Law (§3.4, Ch. 4).
- **Immersive Linear Algebra (ILA — Ström, Åström & Akenine-Möller)**: Geometric dot product & orthogonal projections (Ch. 3).
- **Advanced Engineering Mathematics (Kreyszig 10th Ed.)**: Inner product spaces, Cauchy-Schwarz, positive-definite quadratic forms & Sylvester's criterion (Ch. 7–8).

---

## The 4-Part Pedagogical Protocol

```
linear_algebra/sprint_05_norms_inner_products/
├── README.md                        # Master companion guide (this document)
├── brief.md                         # Part 1: Geometric Consensus Brief (4 lenses)
├── storyboard.md                    # Part 2: 4-Act Visual Storyboard & Choreography
├── scene_norms_inner_products.py    # Part 3: Production ManimCE v0.18+ Animation Script
└── tweaks_prompts.md                # Part 4: Interactive Tweaks & Intuition Stress-Test Prompts
```

| Part | Document | Primary Role |
|---|---|---|
| **Part 1** | [`brief.md`](./brief.md) | **Geometric Consensus Brief**: Synthesizes Geometric Action, Geometric Invariants, Algebraic-to-Visual Dictionary, and Dual Bridge (Frontier ML + GATE CS/DA shortcuts). |
| **Part 2** | [`storyboard.md`](./storyboard.md) | **4-Act Visual Storyboard**: Frame-by-frame spatial choreography detailing timing (~94s), coordinate layout, color palette, and pinned HUD states. |
| **Part 3** | [`scene_norms_inner_products.py`](./scene_norms_inner_products.py) | **Production ManimCE Code**: Executable animation script adhering to the 8 ManimCE coding rules (`make_hud_panel`, `ValueTracker`, `RightAngle`, exact NumPy assertions). |
| **Part 4** | [`tweaks_prompts.md`](./tweaks_prompts.md) | **Interactive Tweaks & Intuition Prompts**: Parameter modification matrix and 4 rigorous conceptual stress-test problems with executable NumPy proofs. |

---

## Canonical Textbook Cross-Reference Matrix

| Topic / Invariant | MML Reference | VMLS Reference | ILA / Kreyszig Reference | Visual / Algorithmic Manifestation |
|---|---|---|---|---|
| **Norm Axioms & Convexity** | §3.1 (Def. 3.1, Eqs. 3.1–3.2) | §3.1 (pp. 45–47) | Kreyszig §7.9 | Positive definiteness, homogeneity, triangle inequality $\iff$ convex unit ball |
| **$L_1, L_2, L_\infty$ Unit Balls** | §3.1 (Exs. 3.1–3.2, Fig. 3.3) | Ex. 3.5 | ILA Ch. 3 | Continuous morph $p=1$ (diamond) $\to p=2$ (circle) $\to p=\infty$ (square) |
| **Norm Ordering & Equivalence** | §3.1 | §3.1 & Ex. 3.5 | — | $\|\mathbf{x}\|_\infty \le \|\mathbf{x}\|_2 \le \|\mathbf{x}\|_1 \le \sqrt{n}\|\mathbf{x}\|_2 \le n\|\mathbf{x}\|_\infty$; nested balls $B_1 \subseteq B_2 \subseteq B_\infty$ |
| **RMS, Z-Score & Chebyshev** | — | §3.1 (Eq. 3.2), §3.3 (Eqs. 3.4–3.5) | — | $\operatorname{rms}(\mathbf{x}) = \|\mathbf{x}\|_2/\sqrt{n}$; $\operatorname{rms}^2 = \operatorname{avg}^2 + \operatorname{std}^2$; $k/n \le (\operatorname{rms}/a)^2$ |
| **Dot Product & Ruler Projection** | §3.2.1, §3.8.1 (Eq. 3.44) | §3.4 (pp. 56–59) | ILA Ch. 3 | Signed projection onto $\mathbf{y}$'s ruler with `RightAngle` residual |
| **SPD Inner Products ($\mathbf{x}^\top A\mathbf{y}$)** | §3.2.2–3.2.3 (Thm. 3.5, Exs. 3.3–3.7) | Ex. 3.28 | Kreyszig §8.4 | Tilted $A$-ellipsoid ($\mathbf{x}^\top A\mathbf{x}=1$) & altered $A$-orthogonality |
| **Cauchy-Schwarz Inequality** | §3.3 (Eq. 3.17), §3.4 (Eqs. 3.24–3.25) | §3.4 (pp. 56–58) | Kreyszig §7.9 | $|\langle\mathbf{x},\mathbf{y}\rangle| \le \|\mathbf{x}\|\|\mathbf{y}\|$; $\cos\omega \in [-1,1]$ gauge; equality iff collinear |
| **Parallelogram Law Test** | §3.3 (p. 75) | Ex. 3.4b | — | $L_2$ & SPD pass; $L_1$ ($8\ne 4$) & $L_\infty$ ($2\ne 4$) fail |
| **ML: Attention, Cosine, Lasso/Ridge, RMSNorm** | §6.5, §8.2.3 (Ex. 8.3) | §3.4 (Ex. 3.24b), Ch. 4 | — | Variance scaling $1/\sqrt{d_k}$; $L_1$ corner sparsity vs $L_2$ circle shrinkage; Mahalanobis |
| **GATE: Sylvester & C-S Extrema** | §3.2.3, Ex. 3.9 | §3.4 | Kreyszig §8.4 | Leading principal minors $>0$; $\max_{\mathbf{x}^\top A\mathbf{x}\le 1}\mathbf{c}^\top\mathbf{x} = \sqrt{\mathbf{c}^\top A^{-1}\mathbf{c}}$ |

---

## Scene Manifest & Architecture

The production script `scene_norms_inner_products.py` provides 1 shared base utility class and 5 modular `Scene` classes for standalone act debugging or end-to-end composite rendering:

```
scene_norms_inner_products.py
├── BaseNormsInnerProductsScene (Scene)  [Shared palette, HUD factory, factored Lp ball, SPD ellipse, safe_right_angle]
├── Act1SpatialPuzzle (~20s)             [Scene]
├── Act2DynamicDeformation (~26s)        [Scene]
├── Act3AlgebraicMorph (~22s)            [Scene]
├── Act4DegenerateStressTest (~26s)      [Scene]
└── NormsInnerProductsScene (~94s)       [Master Combined Scene]
```

### Scene Descriptions
1. **`Act1SpatialPuzzle`**: Compares $L_1=7$, $L_2=5$, and $L_\infty=4$ on $\mathbf{x}_{\text{puzzle}} = [3, 4]^\top$, establishes $\|\mathbf{x}\|_\infty \le \|\mathbf{x}\|_2 \le \|\mathbf{x}\|_1$, and previews $\mathbf{x}^\top\mathbf{y}_{\text{puzzle}} = 10$ via orthogonal projection onto $\mathbf{y}_{\text{puzzle}} = [2, 1]^\top$.
2. **`Act2DynamicDeformation`**: Tracks $\alpha = 1/p \in [0, 1]$ with `ValueTracker` and `always_redraw` to morph the unit ball continuously from $L_1$ diamond $\to L_2$ circle $\to L_\infty$ square, then sweeps $\mathbf{x}(\omega)$ around tilted ruler $\mathbf{y}_{\text{ruler}} = [1.6, 1.2]^\top$ to animate signed dot products and the 3-4-5 canonical right triangle ($\cos\omega = 0.8$).
3. **`Act3AlgebraicMorph`**: Pins a live $\cos\omega \in [-1, 1]$ Cauchy-Schwarz gauge showing equality saturation at collinearity ($\omega \in \{0, \pi\}$), then homotopically warps the $L_2$ unit circle into the $A_{\text{SPD}} = [[2,1],[1,2]]$ unit ellipsoid, turning Euclidean obtuse vectors $\mathbf{u}_A = [1,0]^\top, \mathbf{v}_A = [-1,2]^\top$ strictly $A$-orthogonal.
4. **`Act4DegenerateStressTest`**: Demonstrates $p=0.5$ concave quasi-norm triangle inequality failure ($4 > 2$), $L_1$ parallelogram failure ($8 \ne 4$), $\mathbf{x}=\mathbf{0}$ undefined angle ($0/0$), and indefinite $A_{\text{indef}} = \operatorname{diag}(1, -1)$ hyperbolas before presenting the master summary card.
5. **`NormsInnerProductsScene`**: Composite master scene running Acts 1–4 seamlessly with clean updater teardown and HUD transitions.

---

## CLI Render Commands

Run all rendering commands from the repository root (`/home/uni-kernel/dev/math_cs`):

```bash
# 1. Fast Low-Quality Preview (480p15) — Interactive Preview (-pql) or Headless (-ql)
.venv/bin/manim -pql linear_algebra/sprint_05_norms_inner_products/scene_norms_inner_products.py NormsInnerProductsScene
.venv/bin/manim -ql  linear_algebra/sprint_05_norms_inner_products/scene_norms_inner_products.py NormsInnerProductsScene

# 2. Render Individual Modular Acts (480p15)
.venv/bin/manim -ql linear_algebra/sprint_05_norms_inner_products/scene_norms_inner_products.py Act1SpatialPuzzle
.venv/bin/manim -ql linear_algebra/sprint_05_norms_inner_products/scene_norms_inner_products.py Act2DynamicDeformation
.venv/bin/manim -ql linear_algebra/sprint_05_norms_inner_products/scene_norms_inner_products.py Act3AlgebraicMorph
.venv/bin/manim -ql linear_algebra/sprint_05_norms_inner_products/scene_norms_inner_products.py Act4DegenerateStressTest

# 3. High-Quality Production Render (1080p60) — Interactive Preview (-pqh) or Headless (-qh)
.venv/bin/manim -pqh linear_algebra/sprint_05_norms_inner_products/scene_norms_inner_products.py NormsInnerProductsScene
.venv/bin/manim -qh  linear_algebra/sprint_05_norms_inner_products/scene_norms_inner_products.py NormsInnerProductsScene
```

Staged 1080p60 animation output: `linear_algebra/animations/sprint_05_NormsInnerProductsScene_1080p60.mp4`.

---

## Programmatic Verification & Checklist

Sprint 5 is verified by two automated test suites in `tests/`:
- **5-Tier Invariant Suite**: `.venv/bin/pytest tests/test_sprint05_invariants.py -v` (67 tests covering $L_p$ norms, RMS/Chebyshev, ruler projections, SPD inner products, Cauchy-Schwarz, $p<1$/indefinite/parallelogram/zero degeneracies, ML/GATE bridges, and AST static integrity).
- **Adversarial & Prompt Stress Suite**: `.venv/bin/pytest tests/test_adversarial_sprint5.py -v` and `.venv/bin/python tests/test_adversarial_sprint5.py` (8 checks including 100k-vector Cauchy-Schwarz/norm-equivalence sweeps, overflow-safe extreme norms, $\kappa(A)=10^8$ SPD stability, and `tweaks_prompts.md` proofs).

---

## Key Concepts & Insights Summary

1. **Norms Encode Prior Geometry via Unit Balls**: Choosing $L_1$, $L_2$, or $L_\infty$ does not merely change a scalar number—it reshapes the geometry of distance. Sharp $L_1$ diamond corners explain why Lasso yields exact coordinate sparsity, while the unique circular symmetry of $L_2$ makes Euclidean distance rotation-invariant.
2. **Convexity Is Equivalent to the Triangle Inequality**: Every valid norm has a convex unit ball. At $p = 0.5 < 1$, pinching the unit ball inward breaks convexity and immediately breaks subadditivity ($\|\mathbf{e}_1+\mathbf{e}_2\|_{0.5} = 4 > 2$).
3. **Inner Products Induce Geometry (and $L_1$/$L_\infty$ Cannot)**: By the Jordan–von Neumann parallelogram law, only norms satisfying $\|\mathbf{x}+\mathbf{y}\|^2 + \|\mathbf{x}-\mathbf{y}\|^2 = 2(\|\mathbf{x}\|^2+\|\mathbf{y}\|^2)$ arise from an inner product. Every inner product on $\mathbb{R}^n$ is $\mathbf{x}^\top A\mathbf{y}$ for a unique SPD matrix $A = L L^\top \succ 0$ (MML Theorem 3.5), which is literally Euclidean geometry after Cholesky whitening $\tilde{\mathbf{x}} = L^\top\mathbf{x}$.
4. **Cauchy-Schwarz Grounds Angles, Attention & Optimization**: Trapping $|\langle\mathbf{x},\mathbf{y}\rangle| \le \|\mathbf{x}\|\|\mathbf{y}\|$ makes $\cos\omega \in [-1,1]$ well-defined, ties cosine similarity on normalized embeddings directly to Euclidean distance ($\|\hat{\mathbf{x}}-\hat{\mathbf{y}}\|_2^2 = 2(1-\cos\omega)$), and yields closed-form GATE extrema ($\max_{\mathbf{x}^\top A\mathbf{x} \le 1}\mathbf{c}^\top\mathbf{x} = \sqrt{\mathbf{c}^\top A^{-1}\mathbf{c}}$).
5. **Precision Note on Textbook Calculations**: Automated verification caught a minor rounding typo in VMLS §3.3 (p. 53), which prints $\operatorname{std}([1,-2,3,2]) = 1.872$; the exact value is $\sqrt{3.5} \approx 1.870829$ (rounding to `1.871`). All Sprint 5 assertions use exact NumPy expressions.
