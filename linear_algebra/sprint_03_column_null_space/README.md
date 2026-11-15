# Sprint 3: Column Space, Null Space & Rank-Nullity Theorem in 3D

## Curriculum Overview
Sprint 3 delivers a master-level interactive 3D visual module on the fundamental subspaces of linear transformations, modeling the pedagogical intuition of Grant Sanderson (3Blue1Brown). Subclassing Manim's 3D engine (`ThreeDScene`), this module visualizes how a rank-2 $3 \times 3$ matrix collapses 3D continuous space into a 2D planar sheet $\mathcal{C}(A)$ while simultaneously marching a 1D line of null vectors $\mathcal{N}(A)$ straight into the origin $\mathbf{0}$.

The module bridges theoretical linear algebra with **Frontier Machine Learning** (overparameterization, zero-loss valleys, and LoRA low-rank adaptation) and **GATE CS/DA** (the Rouché-Capelli consistency theorem and system solvability classification).

---

## 4-Part Output Protocol Deliverables

| Deliverable | File | Description |
|---|---|---|
| **Part 1: Brief** | [`brief.md`](brief.md) | ~350-word 3b1b Geometric Consensus Brief covering Geometric Action, Geometric Invariants, Algebraic-to-Visual Dictionary, and Dual Bridges. |
| **Part 2: Storyboard** | [`storyboard.md`](storyboard.md) | 60–120s 4-Act Visual Storyboard with detailed choreography tables across 3D camera orbits, canvas mobjects, and viewport-pinned HUD panels. |
| **Part 3: Production Scene** | [`scene_column_null_space.py`](scene_column_null_space.py) | Production ManimCE v0.18+ script subclassing `ThreeDScene`, implementing the 8 ManimCE coding rules with dark-backed HUD panel pinning (`add_fixed_in_frame_mobjects`). |
| **Part 4: Tweaks & Prompts** | [`tweaks_prompts.md`](tweaks_prompts.md) | Interactive parameter configuration matrix + 4 intuition stress-test prompts with self-contained executable NumPy verification proofs. |

---

## Canonical Textbook Cross-Reference Matrix

| Canonical Reference | Chapter / Section | Core Mathematical Concept | Visual Manifestation in Sprint 3 |
|---|---|---|---|
| **Mathematics for Machine Learning (MML)** | Ch. 2.6 (pp. 56–58) | Image $\text{Im}(T)$ & Kernel $\ker(T)$, Def 2.23 | 2D Planar Sheet $\mathcal{C}(A)$ & 1D Null Line $\mathcal{N}(A)$ |
| **Mathematics for Machine Learning (MML)** | Ch. 2.6 (Thm 2.24, p. 58) | Rank-Nullity Theorem ($\text{rk} + \text{null} = n$) | Volume Conservation Badge: $2 + 1 = 3$ |
| **Mathematics for Machine Learning (MML)** | Ch. 2.6 (Def 2.25, p. 59) | Affine Subspace $L = \mathbf{x}_0 + U$ | Affine Solution Line $\mathbf{x}_p + \mathcal{N}(A) \to \mathbf{b}$ |
| **Applied Linear Algebra (VMLS - Boyd)** | Ch. 10 (pp. 175–198) | System Solvability & Least-Norm Solution | Unique Minimal Solution $\mathbf{x}_{\text{LN}} = A^+ \mathbf{b} \perp \mathcal{N}(A)$ |
| **Advanced Eng. Mathematics (Kreyszig)** | Ch. 7.4–7.5 (pp. 272–287) | Fundamental Theorem of Linear Systems | Orthogonal Complement $\mathcal{N}(A) = (\mathcal{C}(A^\top))^\perp$ |
| **Higher Eng. Mathematics (B.S. Grewal)** | Ch. 2.7 | Rouché-Capelli Consistency Theorem | Augmented Matrix Rank Test: $\text{rank}(A) = 2 < \text{rank}([A \mid \mathbf{b}]) = 3$ |

---

## Mathematical Specification & Canonical Constants

The module operates on the rank-2 $3 \times 3$ linear mapping:
$$A = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix}$$

- **Column Vectors**:
  $$\mathbf{a}_1 = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}, \qquad \mathbf{a}_2 = \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}, \qquad \mathbf{a}_3 = \begin{bmatrix} 1 \\ 1 \\ 2 \end{bmatrix} = \mathbf{a}_1 + \mathbf{a}_2$$
- **Column Space $\mathcal{C}(A)$**:
  2D planar sheet passing through origin $(0, 0, 0)$ with normal $\mathbf{n} = \mathbf{a}_1 \times \mathbf{a}_2 = [-1, -1, 1]^\top \sim [1, 1, -1]^\top$:
  $$\mathcal{C}(A) = \left\{ \mathbf{y} \in \mathbb{R}^3 \;\middle|\; x + y - z = 0 \right\}, \qquad \dim(\mathcal{C}(A)) = \text{rank}(A) = 2$$
- **Null Space $\mathcal{N}(A)$**:
  1D line along $\mathbf{v}_{\text{null}} = [1, 1, -1]^\top$ since $1\mathbf{a}_1 + 1\mathbf{a}_2 - 1\mathbf{a}_3 = \mathbf{0}$:
  $$\mathcal{N}(A) = \left\{ t \begin{bmatrix} 1 \\ 1 \\ -1 \end{bmatrix} \;\middle|\; t \in \mathbb{R} \right\}, \qquad \dim(\mathcal{N}(A)) = \text{nullity}(A) = 1$$
- **Rank-Nullity Theorem**:
  $$\text{rank}(A) + \text{nullity}(A) = 2 + 1 = 3 = \dim(\text{Domain } \mathbb{R}^3)$$
- **Reachable Target & Particular Solution**:
  Target $\mathbf{b} = [1, 2, 3]^\top$ satisfies $1 + 2 - 3 = 0 \implies \mathbf{b} \in \mathcal{C}(A)$.
  Particular solution $\mathbf{x}_p = [1, 2, 0]^\top$ satisfies $A\mathbf{x}_p = \mathbf{b}$.
  General solution forms the affine line $\mathbf{x}_p + t \mathbf{v}_{\text{null}}$.
- **Least-Norm Solution**:
  $$\mathbf{x}_{\text{LN}} = A^+ \mathbf{b} = \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}, \qquad \mathbf{x}_{\text{LN}} \perp \mathbf{v}_{\text{null}}, \quad \|\mathbf{x}_{\text{LN}}\| = \sqrt{2} \approx 1.4142$$
- **Inconsistent Target (Rouché-Capelli Failure)**:
  $\mathbf{b}_{\text{unreach}} = [1, 0, 0]^\top$ hovers off the plane ($1 + 0 - 0 = 1 \ne 0$):
  $$\text{rank}(A) = 2 < \text{rank}([A \mid \mathbf{b}_{\text{unreach}}]) = 3 \implies 0 \text{ Solutions}$$

---

## 3D ManimCE Implementation & Coding Rules Compliance

The production script strictly adheres to the 8 ManimCE coding rules:
1. **Semantic Color Palette**: Green $\mathbf{a}_1$ (`#83C167`), Red $\mathbf{a}_2$ (`#FC6255`), Blue $\mathbf{a}_3$ (`#58C4DD`), Teal Column Space (`#2A9D8F`), Orange Null Space (`#FF9E64`), Yellow Affine Line (`#F7D96F`), and White Origin Anchor (`#FFFFFF`).
2. **3D Coordinate Engine**: Subclasses `ThreeDScene` with `ThreeDAxes`, camera orientation $\phi = 70^\circ, \theta = -45^\circ$, and ambient rotation `rate=0.08`.
3. **Critical 3D HUD Pinning**: All 2D text, formulas, and status cards invoke `self.add_fixed_in_frame_mobjects(...)` inside dark translucent containers (`make_hud_panel`, `fill_opacity=0.92`, `z_index=10`). 3D axes and surfaces never occlude or bleed through typography.
4. **Dynamic Updaters & Sliders**: Null vector march and affine many-to-one collapse animated with pacing and easing curves (`rate_func=rush_into`).
5. **Invariant Markers**: Origin white dot anchor $\mathbf{0}$ and right-angle orthogonal complement indicators.
6. **Numerical Precision**: 100% exact NumPy assertions execute pre-flight at the top of the scene file.
7. **Pacing Rhythm**: 4-Act modular breakdown with individual executable scene classes.
8. **No Deprecated Symbols**: Zero occurrences of `CONFIG`, `ShowCreation`, `TexMobject`, or `TextMobject`.

---

## CLI Execution & Render Commands

```bash
# Automated Test Suite (Fast <0.2s Invariant & AST Static Tests)
.venv/bin/pytest tests/test_sprint03_invariants.py -v

# Adversarial Stress Suite (100,000 points, condition numbers, LoRA)
.venv/bin/python tests/test_adversarial_sprint3.py

# Preview Rendering (-ql: 480p15, fast visual check)
.venv/bin/manim -ql linear_algebra/sprint_03_column_null_space/scene_column_null_space.py Act1SpatialPuzzle
.venv/bin/manim -ql linear_algebra/sprint_03_column_null_space/scene_column_null_space.py ColumnNullSpaceScene

# Production Rendering (-qh: 1080p60 full release)
.venv/bin/manim -qh linear_algebra/sprint_03_column_null_space/scene_column_null_space.py ColumnNullSpaceScene
```

---

## Verification & Test Results
- **Pytest Suite (`tests/test_sprint03_invariants.py`)**: 65/65 tests passed (100% success rate across Tier 1 Feature Coverage, Tier 2 Boundaries, Tier 3 Cross-Feature, Tier 4 Real-World ML/GATE, and Tier 5 Static AST).
- **Adversarial Suite (`tests/test_adversarial_sprint3.py`)**: 8/8 stress tests passed (100,000 points orthogonal decomposition, projection idempotence, Rouché-Capelli classification, affine least-norm optimizer, and prompts 1–4 proofs).
- **ManimCE Render Verification**: Clean exit code 0 on both `Act1SpatialPuzzle` and `ColumnNullSpaceScene`.
