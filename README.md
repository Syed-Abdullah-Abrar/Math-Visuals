# Mathematical & Computational Foundations

Intuitive, first-principles repository combining geometric visualizations, formal mathematical models, and code implementations across Mathematics and Computer Science.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Linear Algebra Curriculum](#1-linear-algebra-curriculum)
   - [Sprint 1: Vectors, Linear Combinations & Span](#sprint-1-vectors-linear-combinations--span)
   - [Sprint 2: Linear Mappings & Matrix Multiplication as Composition](#sprint-2-linear-mappings--matrix-multiplication-as-composition)
   - [Sprint 3: Column Space, Null Space & Rank-Nullity Theorem (3D)](#sprint-3-column-space-null-space--rank-nullity-theorem-3d)
   - [Sprint 4: Change of Basis & Similarity Transformations](#sprint-4-change-of-basis--similarity-transformations)
   - [Sprint 5: Norms ($L_1, L_2, L_\infty$), Inner Products & Cauchy-Schwarz](#sprint-5-norms-l_1-l_2-l_infty-inner-products--cauchy-schwarz)
   - [Sprint 6: Orthogonal Projections onto Lines & Subspaces](#sprint-6-orthogonal-projections-onto-1d-lines--k-d-subspaces)
   - [Sprint 7: Gram-Schmidt Orthonormalization & $QR$ Decomposition](#sprint-7-gram-schmidt-orthonormalization--qr-decomposition)
   - [Sprint 8: Orthogonal Matrices, Rotations ($SO(n)$) & Givens/Householder](#sprint-8-orthogonal-matrices-rotations-son--givenshouseholder)
   - [Curated 1080p60 Production Animations](#curated-1080p60-production-animations)
   - [Key Concepts & Geometric Insights Across Sprints 5–8](#key-concepts--geometric-insights-across-sprints-58)
   - [Render & Verification Quickstart](#render--verification-quickstart)
   - [Curriculum Roadmap](#curriculum-roadmap)
3. [Master Study Guide: How to Study Sprints 1–8](#2-master-study-guide-how-to-study-sprints-18)
4. [Scratchpad & Prototyping](#3-scratchpad--prototyping)
5. [Foundational Philosophy & Blueprint](#4-foundational-philosophy--blueprint)
   - [Part 1: The Core Philosophy (The Unified Field)](#part-1-the-core-philosophy-the-unified-field)
   - [Part 2: The Geometric Translation of Technical Landscapes](#part-2-the-geometric-translation-of-technical-landscapes)
   - [Part 3: The Focus Matrix (Prioritizing via Cognitive Constraints)](#part-3-the-focus-matrix-prioritizing-via-cognitive-constraints)
   - [Part 4: The 2-Year Execution Pipeline](#part-4-the-2-year-execution-pipeline)
   - [Part 5: The System Integration (Your Obsidian Engine)](#part-5-the-system-integration-your-obsidian-engine)

---

## Repository Structure

```
math_cs/
├── README.md                                  # Master blueprint, study guide, and curriculum index
├── .gitignore                                 # Python, virtualenv, and Manim cache rules
├── linear_algebra/                            # 3Blue1Brown-modeled Geometric Linear Algebra curriculum
│   ├── README.md                              # Curriculum module index & sprint roadmap
│   ├── animations/                            # Curated production 1080p60 MP4 animations (Sprints 1–8)
│   ├── sprint_01_vectors_span/                # Sprint 1: Vectors, Linear Combinations & Span
│   │   ├── README.md                          # Master companion guide
│   │   ├── brief.md                           # Part 1: Geometric Consensus Brief
│   │   ├── storyboard.md                      # Part 2: 4-Act Visual Storyboard
│   │   ├── scene_vectors_span.py              # Part 3: Production ManimCE script
│   │   └── tweaks_prompts.md                  # Part 4: Interactive Tweaks & Intuition Prompts
│   ├── sprint_02_linear_mappings/             # Sprint 2: Linear Mappings & Matrix Composition
│   │   ├── README.md                          # Master companion guide
│   │   ├── brief.md                           # Part 1: Geometric Consensus Brief
│   │   ├── storyboard.md                      # Part 2: 4-Act Visual Storyboard
│   │   ├── scene_linear_mappings.py           # Part 3: Production ManimCE script
│   │   └── tweaks_prompts.md                  # Part 4: Interactive Tweaks & Intuition Prompts
│   ├── sprint_03_column_null_space/           # Sprint 3: Column Space, Null Space & Rank-Nullity (3D)
│   │   ├── README.md                          # Master companion guide
│   │   ├── brief.md                           # Part 1: Geometric Consensus Brief
│   │   ├── storyboard.md                      # Part 2: 4-Act Visual Storyboard
│   │   ├── scene_column_null_space.py         # Part 3: Production 3D ManimCE script (ThreeDScene)
│   │   └── tweaks_prompts.md                  # Part 4: Interactive Tweaks & Intuition Prompts
│   ├── sprint_04_change_of_basis/             # Sprint 4: Change of Basis & Similarity (P^-1 A P)
│   │   ├── README.md                          # Master companion guide
│   │   ├── brief.md                           # Part 1: Geometric Consensus Brief
│   │   ├── storyboard.md                      # Part 2: 4-Act Visual Storyboard
│   │   ├── scene_change_of_basis.py           # Part 3: Production ManimCE script
│   │   └── tweaks_prompts.md                  # Part 4: Interactive Tweaks & Intuition Prompts
│   ├── sprint_05_norms_inner_products/        # Sprint 5: Norms (L1, L2, L_inf), Inner Products & Cauchy-Schwarz
│   │   ├── README.md                          # Master companion guide
│   │   ├── brief.md                           # Part 1: Geometric Consensus Brief
│   │   ├── storyboard.md                      # Part 2: 4-Act Visual Storyboard
│   │   ├── scene_norms_inner_products.py      # Part 3: Production ManimCE script
│   │   └── tweaks_prompts.md                  # Part 4: Interactive Tweaks & Intuition Prompts
│   ├── sprint_06_orthogonal_projections/      # Sprint 6: Orthogonal Projections onto Lines & Subspaces
│   │   ├── README.md                          # Master companion guide
│   │   ├── brief.md                           # Part 1: Geometric Consensus Brief
│   │   ├── storyboard.md                      # Part 2: 4-Act Visual Storyboard
│   │   ├── scene_orthogonal_projections.py    # Part 3: Production 3D ManimCE script (ThreeDScene)
│   │   └── tweaks_prompts.md                  # Part 4: Interactive Tweaks & Intuition Prompts
│   ├── sprint_07_gram_schmidt_qr/             # Sprint 7: Gram-Schmidt Orthonormalization & QR Decomposition
│   │   ├── README.md                          # Master companion guide
│   │   ├── brief.md                           # Part 1: Geometric Consensus Brief
│   │   ├── storyboard.md                      # Part 2: 4-Act Visual Storyboard
│   │   ├── scene_gram_schmidt_qr.py           # Part 3: Production 3D ManimCE script (ThreeDScene)
│   │   └── tweaks_prompts.md                  # Part 4: Interactive Tweaks & Intuition Prompts
│   └── sprint_08_orthogonal_matrices_rotations/# Sprint 8: Orthogonal Matrices, Rotations SO(n) & Givens/Householder
│       ├── README.md                          # Master companion guide
│       ├── brief.md                           # Part 1: Geometric Consensus Brief
│       ├── storyboard.md                      # Part 2: 4-Act Visual Storyboard
│       ├── scene_orthogonal_matrices_rotations.py # Part 3: Production ManimCE script
│       └── tweaks_prompts.md                  # Part 4: Interactive Tweaks & Intuition Prompts
├── tests/                                     # 587-test automated invariant & adversarial verification suite
│   ├── test_sprint01_invariants.py            # Sprint 1 invariant & AST tests
│   ├── test_adversarial_sprint1.py            # Sprint 1 numerical stress tests
│   ├── test_sprint02_invariants.py            # Sprint 2 invariant & AST tests
│   ├── test_adversarial_sprint2.py            # Sprint 2 commutator & composition stress tests
│   ├── test_sprint03_invariants.py            # Sprint 3 3D subspace & HUD pinning AST tests
│   ├── test_adversarial_sprint3.py            # Sprint 3 rank-nullity & least-norm stress tests
│   ├── test_sprint04_invariants.py            # Sprint 4 similarity & eigen-basis invariant tests
│   ├── test_adversarial_sprint4.py            # Sprint 4 Jordan block & conditioning stress tests
│   ├── test_sprint05_invariants.py            # Sprint 5 norm ball convexity & Cauchy-Schwarz tests
│   ├── test_adversarial_sprint5.py            # Sprint 5 p < 1 quasi-norm & indefinite form stress tests
│   ├── test_sprint06_invariants.py            # Sprint 6 projection idempotence & normal equations tests
│   ├── test_adversarial_sprint6.py            # Sprint 6 collinear rank failure & oblique stress tests
│   ├── test_sprint07_invariants.py            # Sprint 7 QR decomposition & span invariance tests
│   ├── test_adversarial_sprint7.py            # Sprint 7 CGS vs MGS drift & Lauchli matrix stress tests
│   ├── test_sprint08_invariants.py            # Sprint 8 isometry & Cartan-Dieudonné theorem tests
│   └── test_adversarial_sprint8.py            # Sprint 8 3D non-commutativity & Householder sign tests
└── scratchpad_codes/                          # Prototyping & numerical verification scripts
    └── matrix_expansion.py                    # Quadratic form expansions (symmetric vs general)
```

---

## 1. Linear Algebra Curriculum

> **Status**: Active Curriculum (Sprints 1–4 Completed & Certified)  
> **Primary References**:  
> - *Mathematics for Machine Learning* (**MML** Ch. 2.4–2.7, Ch. 3–4, Ch. 10) — Deisenroth, Faisal, Ong (`/home/uni-kernel/brain/03_RESOURCES/Books/AI-ML/Mathematics/Mathematics for Machine Learning.pdf`)  
> - *Introduction to Applied Linear Algebra* (**VMLS** Ch. 1, 5, 6, 7, 10) — Boyd, Vandenberghe (`/home/uni-kernel/brain/03_RESOURCES/Books/Applied mathematics/Algebra/VMLS.pdf`)  
> - *Immersive Linear Algebra* (**ILA** Ch. 1–2, 6) — Ström, Åström, Akenine-Möller (`https://immersivemath.com/ila/tableofcontents.html`)  
> - *Advanced Engineering Mathematics* (**Kreyszig** 10th Ed. Ch. 7–8) & *Higher Engineering Mathematics* (**B.S. Grewal** Ch. 2) (`/home/uni-kernel/brain/03_RESOURCES/Books/`)

Linear algebra operates through spatial transformations. We treat matrices not as static tables of numbers, but as continuous coordinate deformations that stretch, rotate, shear, or project space while keeping the origin fixed. Every sprint follows a strict **4-Part Output Protocol**:
1. **Part 1 (`brief.md`)**: ~350-word 3Blue1Brown Geometric Consensus Brief across four lenses (Geometric Action, Geometric Invariant, Algebraic-to-Visual Dictionary, and Dual Bridge to Frontier ML + GATE CS/DA).
2. **Part 2 (`storyboard.md`)**: 60–120s 4-Act Visual Storyboard (`Act 1: Concrete Spatial Puzzle` $\to$ `Act 2: Dynamic Deformation & Ghost Space` $\to$ `Act 3: Color-Coded Algebraic Morph` $\to$ `Act 4: Degenerate / Edge-Case Stress Test`).
3. **Part 3 (`scene_*.py`)**: Production ManimCE v0.18+ Python script enforcing the 8 coding rules (3b1b color palette, zero text overlap via `make_hud_panel`, `ValueTracker` updaters, exact `numpy.linalg` arithmetic).
4. **Part 4 (`tweaks_prompts.md`)**: Interactive parameter modifications and 4 intuition stress-test prompts with executable NumPy verification scripts.

---

### Sprint 1: Vectors, Linear Combinations & Span
- **Location**: [`linear_algebra/sprint_01_vectors_span/`](linear_algebra/sprint_01_vectors_span/) ([`README.md`](linear_algebra/sprint_01_vectors_span/README.md), [`brief.md`](linear_algebra/sprint_01_vectors_span/brief.md), [`storyboard.md`](linear_algebra/sprint_01_vectors_span/storyboard.md), [`scene_vectors_span.py`](linear_algebra/sprint_01_vectors_span/scene_vectors_span.py), [`tweaks_prompts.md`](linear_algebra/sprint_01_vectors_span/tweaks_prompts.md))
- **References**: MML §2.4 | ILA Ch. 1–2 | VMLS Ch. 1, 5
- **Geometric Core**:
  - Reaching target vector $\mathbf{b} = [3, 4]^\top$ via basis vectors $\mathbf{v}_1 = [2, 1]^\top$ and $\mathbf{v}_2 = [-1, 2]^\top$.
  - Sweeping scalars $c_1, c_2 \in [-2.5, 2.5]$ to tile the 2D plane into an oblique parallelogram grid.
  - Degenerate stress test: Rotating $\mathbf{v}_2$ into collinearity with $\mathbf{v}_1$ collapses the 2D span sheet into a 1D line ($\det = 0, \text{rank} = 1$).
  - **ML / GATE Bridge**: Transformer self-attention values $\sum \alpha_i \mathbf{v}_i$ as convex hull combinations inside $\text{span}(\{\mathbf{v}_i\})$; Rouché–Capelli consistency ($A\mathbf{x}=\mathbf{b} \iff \text{rank}(A) = \text{rank}([A \mid \mathbf{b}])$).

---

### Sprint 2: Linear Mappings & Matrix Multiplication as Composition
- **Location**: [`linear_algebra/sprint_02_linear_mappings/`](linear_algebra/sprint_02_linear_mappings/) ([`README.md`](linear_algebra/sprint_02_linear_mappings/README.md), [`brief.md`](linear_algebra/sprint_02_linear_mappings/brief.md), [`storyboard.md`](linear_algebra/sprint_02_linear_mappings/storyboard.md), [`scene_linear_mappings.py`](linear_algebra/sprint_02_linear_mappings/scene_linear_mappings.py), [`tweaks_prompts.md`](linear_algebra/sprint_02_linear_mappings/tweaks_prompts.md))
- **References**: MML §2.5, §2.7 | ILA Ch. 6 | VMLS Ch. 6–7, 10
- **Geometric Core**:
  - Why linearity ($T(c\mathbf{u} + d\mathbf{v}) = cT(\mathbf{u}) + dT(\mathbf{v})$) forces grid lines to remain **parallel and evenly spaced** with $\mathbf{0}$ fixed.
  - Basis tracking: Columns of $A$ are the physical landing coordinates of $\hat{\imath} \to \mathbf{a}_1 = [2, 1]^\top$ and $\hat{\jmath} \to \mathbf{a}_2 = [0, 2]^\top$.
  - Matrix multiplication $BA$ as right-to-left sequential spatial composition ($A$ warps space first, then $B$ acts on the deformed space).
  - Non-commutativity proof ($SR \neq RS$): Contrasting horizontal shear $S = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$ and $90^\circ$ rotation $R = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$ on probe $\mathbf{x} = [2, 3]^\top$, visualizing the commutator gap $\|\Delta\| = \sqrt{13}$.
  - **ML / GATE Bridge**: Deep neural network layer collapse without non-linear activations ($W_L \cdots W_1 = W_{\text{eff}}$); multiplicative determinant $\det(BA) = \det(B)\det(A)$ and cyclic trace invariance $\text{Tr}(SR - RS) = 0$.

---

### Sprint 3: Column Space, Null Space & Rank-Nullity Theorem (3D)
- **Location**: [`linear_algebra/sprint_03_column_null_space/`](linear_algebra/sprint_03_column_null_space/) ([`README.md`](linear_algebra/sprint_03_column_null_space/README.md), [`brief.md`](linear_algebra/sprint_03_column_null_space/brief.md), [`storyboard.md`](linear_algebra/sprint_03_column_null_space/storyboard.md), [`scene_column_null_space.py`](linear_algebra/sprint_03_column_null_space/scene_column_null_space.py), [`tweaks_prompts.md`](linear_algebra/sprint_03_column_null_space/tweaks_prompts.md))
- **References**: MML §2.5–2.7 | Kreyszig §7.3–7.5 | B.S. Grewal Ch. 2 | VMLS Ch. 10
- **Geometric Core**:
  - Full `ThreeDScene` animation of rank-2 matrix $A = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix}$ crushing 3D space onto a 2D planar sheet $\mathcal{C}(A) = \{x + y - z = 0\}$ ($\text{rank}(A) = 2$).
  - Null space $\mathcal{N}(A) = \text{span}([1, 1, -1]^\top)$ ($\text{nullity}(A) = 1$) marching straight into the origin $\mathbf{0}$, visually locking in $\text{rank}(A) + \text{nullity}(A) = 2 + 1 = 3$.
  - Complete solution fibers to $A\mathbf{x} = \mathbf{b}$ for $\mathbf{b} = [1, 2, 3]^\top$ shown as the translated affine line $\mathbf{x}_p + \mathcal{N}(A)$ collapsing to the single point $\mathbf{b}$, with the minimum-norm solution $\mathbf{x}_{\text{LN}} = A^+\mathbf{b} \perp \mathcal{N}(A)$.
  - **ML / GATE Bridge**: LoRA low-rank weight updates $\Delta W = BA$ constraining parameter updates to a low-dimensional column subspace; overparameterized null-space loss fibers and GATE rank-nullity / Rouché–Capelli shortcuts.

---

### Sprint 4: Change of Basis & Similarity Transformations
- **Location**: [`linear_algebra/sprint_04_change_of_basis/`](linear_algebra/sprint_04_change_of_basis/) ([`README.md`](linear_algebra/sprint_04_change_of_basis/README.md), [`brief.md`](linear_algebra/sprint_04_change_of_basis/brief.md), [`storyboard.md`](linear_algebra/sprint_04_change_of_basis/storyboard.md), [`scene_change_of_basis.py`](linear_algebra/sprint_04_change_of_basis/scene_change_of_basis.py), [`tweaks_prompts.md`](linear_algebra/sprint_04_change_of_basis/tweaks_prompts.md))
- **References**: MML §2.7.2 | ILA Ch. 6 | VMLS Ch. 6, 10 | Kreyszig Ch. 8
- **Geometric Core**:
  - Overlaying the standard Cartesian grid with the tilted eigen-grid $P = \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}$ for symmetric transformation $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$.
  - Animating the 3-step similarity factorization $A = PDP^{-1}$ on $\mathbf{x} = [2, 0]^\top$: (1) $P^{-1}$ translates into eigen-coordinates $[1, -1]_B^\top$, (2) $D = \text{diag}(3, 1)$ scales purely along the eigen-axes, and (3) $P$ translates back to standard coordinates $[4, 2]^\top$.
  - Defective edge-case stress test: Jordan shear block $M_{\text{def}} = \begin{bmatrix} 2 & 1 \\ 0 & 2 \end{bmatrix}$ where algebraic multiplicity ($2$) exceeds geometric multiplicity ($1$), collapsing the eigen-grid into a 1D line ($\det(P) = 0$).
  - **ML / GATE Bridge**: PCA covariance diagonalization $\Sigma = Q\Lambda Q^\top$ decorrelating features into orthogonal eigen-coordinates; GATE similarity invariants ($\text{Tr}(P^{-1}AP) = \text{Tr}(A)$, $\det(P^{-1}AP) = \det(A)$, $A^k = PD^kP^{-1}$).

---

### Sprint 5: Norms ($L_1, L_2, L_\infty$), Inner Products & Cauchy-Schwarz
- **Location**: [`linear_algebra/sprint_05_norms_inner_products/`](linear_algebra/sprint_05_norms_inner_products/) ([`README.md`](linear_algebra/sprint_05_norms_inner_products/README.md), [`brief.md`](linear_algebra/sprint_05_norms_inner_products/brief.md), [`storyboard.md`](linear_algebra/sprint_05_norms_inner_products/storyboard.md), [`scene_norms_inner_products.py`](linear_algebra/sprint_05_norms_inner_products/scene_norms_inner_products.py), [`tweaks_prompts.md`](linear_algebra/sprint_05_norms_inner_products/tweaks_prompts.md))
- **References**: MML §3.1–3.3 | ILA Ch. 3 | VMLS Ch. 3–4
- **Geometric Core**:
  - Continuous morphing of the unit ball $B_p = \{\mathbf{x} : \|\mathbf{x}\|_p \le 1\}$ from the sharp $L_1$ diamond ($p=1$) to the isotropic $L_2$ Euclidean circle ($p=2$) to the axis-aligned $L_\infty$ square ($p=\infty$).
  - Why $p < 1$ pinches into a non-convex star, breaking the triangle inequality ($\|\mathbf{e}_1+\mathbf{e}_2\|_{0.5} = 4 > 2$).
  - Dot product $\mathbf{x}^\top\mathbf{y}$ visualized as dropping a perpendicular shadow of $\mathbf{x}$ onto $\mathbf{y}$'s unit ruler and multiplying by $\|\mathbf{y}\|_2$.
  - Cauchy-Schwarz $|\mathbf{x}^\top\mathbf{y}| \le \|\mathbf{x}\|_2\|\mathbf{y}\|_2$ as the geometric shadow trapped inside the vector radius circle.
  - SPD inner products $\mathbf{x}^\top A\mathbf{y}$ warping circular unit balls into tilted ellipsoids via Cholesky factor $L$ ($A = LL^\top$).
  - **ML / GATE Bridge**: Softmax saturation avoidance via $1/\sqrt{d_k}$ attention scaling; $L_1$ lasso coordinate-axis sparsity vs $L_2$ ridge shrinkage; RMSNorm activation spheres; Sylvester's principal minors criterion.

---

### Sprint 6: Orthogonal Projections onto 1D Lines & $k$-D Subspaces
- **Location**: [`linear_algebra/sprint_06_orthogonal_projections/`](linear_algebra/sprint_06_orthogonal_projections/) ([`README.md`](linear_algebra/sprint_06_orthogonal_projections/README.md), [`brief.md`](linear_algebra/sprint_06_orthogonal_projections/brief.md), [`storyboard.md`](linear_algebra/sprint_06_orthogonal_projections/storyboard.md), [`scene_orthogonal_projections.py`](linear_algebra/sprint_06_orthogonal_projections/scene_orthogonal_projections.py), [`tweaks_prompts.md`](linear_algebra/sprint_06_orthogonal_projections/tweaks_prompts.md))
- **References**: MML §3.8 | ILA Ch. 3, 6 | VMLS Ch. 11–12
- **Geometric Core**:
  - 1D line projection $\pi_U(\mathbf{x}) = \frac{\mathbf{b}^\top\mathbf{x}}{\|\mathbf{b}\|^2}\mathbf{b}$ and idempotence $\pi^2 = \pi$.
  - Sliding a candidate point $\hat{\mathbf{y}} = A\mathbf{x}(s)$ across a 2D plane $U = \mathcal{C}(A)$ in 3D while plotting $\|\mathbf{y} - \hat{\mathbf{y}}\|_2$ in a live pinned 2D corner graph; hitting the global minimum distance at the exact moment $\mathbf{y} - \hat{\mathbf{y}} \perp \mathcal{C}(A)$.
  - Normal equations derivation $A^\top(\mathbf{y} - A\hat{\mathbf{x}}) = \mathbf{0} \implies P = A(A^\top A)^{-1}A^\top$ and orthogonal complement $I - P$.
  - Pythagorean decomposition $\|\mathbf{y}\|_2^2 = \|P\mathbf{y}\|_2^2 + \|(I - P)\mathbf{y}\|_2^2$ and spectral invariant $\operatorname{Tr}(P) = \operatorname{rank}(P) = k$.
  - Degenerate stress tests: affine flat shift $L = \mathbf{x}_0 + U$, collinear feature failure rescued by Ridge ($A^\top A + \lambda I$), and oblique projection error explosion ($\|P_{\text{ob}}\|_2 > 1$).
  - **ML / GATE Bridge**: Ordinary least squares hat matrix; Ridge regression conditioning; PCA subspace reconstruction; instant rank-1 projector readouts ($P = \mathbf{u}\mathbf{u}^\top / \|\mathbf{u}\|^2 \implies \operatorname{Tr}=1, \operatorname{rank}=1, \det=0$).

---

### Sprint 7: Gram-Schmidt Orthonormalization & $QR$ Decomposition
- **Location**: [`linear_algebra/sprint_07_gram_schmidt_qr/`](linear_algebra/sprint_07_gram_schmidt_qr/) ([`README.md`](linear_algebra/sprint_07_gram_schmidt_qr/README.md), [`brief.md`](linear_algebra/sprint_07_gram_schmidt_qr/brief.md), [`storyboard.md`](linear_algebra/sprint_07_gram_schmidt_qr/storyboard.md), [`scene_gram_schmidt_qr.py`](linear_algebra/sprint_07_gram_schmidt_qr/scene_gram_schmidt_qr.py), [`tweaks_prompts.md`](linear_algebra/sprint_07_gram_schmidt_qr/tweaks_prompts.md))
- **References**: MML §3.8.3 | VMLS Ch. 5.4, 10.4, 11.4 | Kreyszig §20.9
- **Geometric Core**:
  - Three skewed vectors $\{\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3\}$ forming an oblique 3D parallelepiped, stripped sequentially of their parallel components:
    $\mathbf{u}_2 = \mathbf{a}_2 - (\mathbf{q}_1^\top\mathbf{a}_2)\mathbf{q}_1$ and $\mathbf{u}_3 = \mathbf{a}_3 - (\mathbf{q}_1^\top\mathbf{a}_3)\mathbf{q}_1 - (\mathbf{q}_2^\top\mathbf{a}_3)\mathbf{q}_2$.
  - Normalizing each residual onto the unit sphere $S^2$ to produce a rigid orthonormal frame $Q$ ($Q^\top Q = I_3$).
  - Revealing upper-triangular $R$ as the causal reconstruction coordinates ($R_{ii} = \|\mathbf{u}_i\|_2 > 0$).
  - Parallelepiped volume conservation: $|\det(A)| = \prod_{i=1}^n R_{ii}$.
  - Degenerate stress tests: coplanar collapse $\mathbf{a}_3 \in \operatorname{span}(\mathbf{a}_1, \mathbf{a}_2)$ certifying rank deficiency ($R_{33} = 0$), and numerical orthogonality loss of Classical Gram-Schmidt (CGS) vs Modified Gram-Schmidt (MGS) vs Householder $QR$ on ill-conditioned Lauchli matrices.
  - **ML / GATE Bridge**: Muon optimizer polar factor orthonormalization ($UV^\top$); Haar-uniform orthogonal weight initialization; least-squares back-substitution in $n^2$ flops without matrix inversion.

---

### Sprint 8: Orthogonal Matrices, Rotations ($SO(n)$) & Givens/Householder
- **Location**: [`linear_algebra/sprint_08_orthogonal_matrices_rotations/`](linear_algebra/sprint_08_orthogonal_matrices_rotations/) ([`README.md`](linear_algebra/sprint_08_orthogonal_matrices_rotations/README.md), [`brief.md`](linear_algebra/sprint_08_orthogonal_matrices_rotations/brief.md), [`storyboard.md`](linear_algebra/sprint_08_orthogonal_matrices_rotations/storyboard.md), [`scene_orthogonal_matrices_rotations.py`](linear_algebra/sprint_08_orthogonal_matrices_rotations/scene_orthogonal_matrices_rotations.py), [`tweaks_prompts.md`](linear_algebra/sprint_08_orthogonal_matrices_rotations/tweaks_prompts.md))
- **References**: MML §3.9 | ILA Ch. 6 | Kreyszig §8.3
- **Geometric Core**:
  - Rigid isometries: transformations preserving vector lengths ($\|Q\mathbf{x}\|_2 = \|\mathbf{x}\|_2$), pairwise distances, and angles ($Q^\top Q = I \iff Q^{-1} = Q^\top$).
  - Topology of $O(n)$: proper rotations $SO(n)$ ($\det = +1$, preserving handedness) vs improper reflections $O(n)\setminus SO(n)$ ($\det = -1$, reversing handedness) separated by the $\det = 0$ wall.
  - Cartan–Dieudonné composition: two reflections across lines with angular gap $\Delta\alpha$ yield a pure rotation $2\Delta\alpha$.
  - 3D non-commutativity: contrasting $R_x R_y \ne R_y R_x$ on 3D reference cubes.
  - Structured orthogonal elimination: surgical 2D Givens rotations $G(i, j, \theta)$ vs rank-1 Householder reflectors $H = I - 2\mathbf{v}\mathbf{v}^\top / \|\mathbf{v}\|^2$, using the sign rule $\mathbf{v} = \mathbf{x} + \operatorname{sgn}(x_1)\|\mathbf{x}\|\mathbf{e}_1$ to prevent catastrophic IEEE-754 subtraction.
  - **ML / GATE Bridge**: Rotary Position Embeddings (RoPE) relative displacement invariance; isometric deep neural network signal propagation across 100+ layers without vanishing/exploding gradients; 3D rotation trace identity $\operatorname{Tr}(R) = 1 + 2\cos\theta$.

---

## Curated 1080p60 Production Animations

All production animations are rendered in full high-definition (1080p at 60 fps) and organized in [`linear_algebra/animations/`](linear_algebra/animations/):

| Sprint | Scene Name | Resolution / FPS | Animation File Link |
|---|---|---|---|
| **01** | `VectorSpanScene` | 1080p60 (30.0s) | [`linear_algebra/animations/sprint_01_VectorSpanScene_1080p60.mp4`](linear_algebra/animations/sprint_01_VectorSpanScene_1080p60.mp4) |
| **02** | `LinearMappingsScene` | 1080p60 (38.5s) | [`linear_algebra/animations/sprint_02_LinearMappingsScene_1080p60.mp4`](linear_algebra/animations/sprint_02_LinearMappingsScene_1080p60.mp4) |
| **03** | `ColumnNullSpaceScene` | 1080p60 (48.0s) | [`linear_algebra/animations/sprint_03_ColumnNullSpaceScene_1080p60.mp4`](linear_algebra/animations/sprint_03_ColumnNullSpaceScene_1080p60.mp4) |
| **04** | `ChangeOfBasisScene` | 1080p60 (44.5s) | [`linear_algebra/animations/sprint_04_ChangeOfBasisScene_1080p60.mp4`](linear_algebra/animations/sprint_04_ChangeOfBasisScene_1080p60.mp4) |
| **05** | `NormsInnerProductsScene` | 1080p60 (45.0s) | [`linear_algebra/animations/sprint_05_NormsInnerProductsScene_1080p60.mp4`](linear_algebra/animations/sprint_05_NormsInnerProductsScene_1080p60.mp4) |
| **06** | `OrthogonalProjectionsScene` | 1080p60 (49.0s) | [`linear_algebra/animations/sprint_06_OrthogonalProjectionsScene_1080p60.mp4`](linear_algebra/animations/sprint_06_OrthogonalProjectionsScene_1080p60.mp4) |
| **07** | `GramSchmidtQRScene` | 1080p60 (52.0s) | [`linear_algebra/animations/sprint_07_GramSchmidtQRScene_1080p60.mp4`](linear_algebra/animations/sprint_07_GramSchmidtQRScene_1080p60.mp4) |
| **08** | `OrthogonalRotationsScene` | 1080p60 (46.0s) | [`linear_algebra/animations/sprint_08_OrthogonalRotationsScene_1080p60.mp4`](linear_algebra/animations/sprint_08_OrthogonalRotationsScene_1080p60.mp4) |

---

### Render & Verification Quickstart

```bash
# Activate project virtual environment
source .venv/bin/activate

# Run full 587-test invariant & adversarial suite across Sprints 1–8
pytest tests/ -v

# Fast Preview Renders (480p15) — Master Combined Scenes
manim -pql linear_algebra/sprint_01_vectors_span/scene_vectors_span.py VectorSpanScene
manim -pql linear_algebra/sprint_02_linear_mappings/scene_linear_mappings.py LinearMappingsScene
manim -pql linear_algebra/sprint_03_column_null_space/scene_column_null_space.py ColumnNullSpaceScene
manim -pql linear_algebra/sprint_04_change_of_basis/scene_change_of_basis.py ChangeOfBasisScene
manim -pql linear_algebra/sprint_05_norms_inner_products/scene_norms_inner_products.py NormsInnerProductsScene
manim -pql linear_algebra/sprint_06_orthogonal_projections/scene_orthogonal_projections.py OrthogonalProjectionsScene
manim -pql linear_algebra/sprint_07_gram_schmidt_qr/scene_gram_schmidt_qr.py GramSchmidtQRScene
manim -pql linear_algebra/sprint_08_orthogonal_matrices_rotations/scene_orthogonal_matrices_rotations.py OrthogonalRotationsScene

# Production High-Definition Renders (1080p60 Headless)
manim -qh linear_algebra/sprint_01_vectors_span/scene_vectors_span.py VectorSpanScene
manim -qh linear_algebra/sprint_02_linear_mappings/scene_linear_mappings.py LinearMappingsScene
manim -qh linear_algebra/sprint_03_column_null_space/scene_column_null_space.py ColumnNullSpaceScene
manim -qh linear_algebra/sprint_04_change_of_basis/scene_change_of_basis.py ChangeOfBasisScene
manim -qh linear_algebra/sprint_05_norms_inner_products/scene_norms_inner_products.py NormsInnerProductsScene
manim -qh linear_algebra/sprint_06_orthogonal_projections/scene_orthogonal_projections.py OrthogonalProjectionsScene
manim -qh linear_algebra/sprint_07_gram_schmidt_qr/scene_gram_schmidt_qr.py GramSchmidtQRScene
manim -qh linear_algebra/sprint_08_orthogonal_matrices_rotations/scene_orthogonal_matrices_rotations.py OrthogonalRotationsScene
```

---

### Curriculum Roadmap

| Sprint | Topic | Geometric Focus | References | Status |
|---|---|---|---|---|
| **01** | **Vectors, Linear Combinations & Span** | Oblique grid tiling, displacement concatenation, collinear collapse | MML §2.4, VMLS Ch. 1, 5, ILA Ch. 1–2 | **Completed** |
| **02** | **Linear Mappings & Matrix Composition** | Parallel grid invariance, basis columns, $BA \neq AB$ via rotation + shear | MML §2.5, §2.7, VMLS Ch. 6–7, 10, ILA Ch. 6 | **Completed** |
| **03** | **Column Space, Null Space & Rank-Nullity** | 3D rank-2 collapse onto 2D plane, 1D null line to $\mathbf{0}$, $\mathbf{x}_p + \mathcal{N}(A)$ | MML §2.5–2.7, Kreyszig §7.3–7.5, Grewal Ch. 2 | **Completed** |
| **04** | **Change of Basis & Similarity ($P^{-1}AP$)** | Standard vs. eigen-grid overlay, $P^{-1} \to D \to P$, defective Jordan collapse | MML §2.7.2, ILA Ch. 6, Kreyszig Ch. 8 | **Completed** |
| **05** | **Norms ($L_1, L_2, L_\infty$) & Cauchy-Schwarz** | Unit ball morphing ($L_1 \to L_2 \to L_\infty$), ruler projection, SPD ellipsoid warping | MML §3.1–3.3, ILA Ch. 3, VMLS Ch. 3–4 | **Completed** |
| **06** | **Orthogonal Projections onto Lines & Subspaces** | Distance bowl minimization, $P = A(A^\top A)^{-1}A^\top$, Pythagoras split | MML §3.8, ILA Ch. 3, 6, VMLS Ch. 11–12 | **Completed** |
| **07** | **Gram-Schmidt Orthonormalization & $QR$** | Sequential shadow stripping in 3D, $A = QR$, volume conservation $\prod R_{ii}$ | MML §3.8.3, VMLS Ch. 5.4, 10.4, Kreyszig §20.9 | **Completed** |
| **08** | **Orthogonal Matrices & Rotations ($SO(n)$)** | Rigid isometries, proper vs improper reflections, Cartan–Dieudonné, RoPE | MML §3.9, ILA Ch. 6, Kreyszig §8.3 | **Completed** |

---

## 2. Master Study Guide: How to Study Sprints 1–8

To achieve 90%+ geometric and analytical mastery across GATE CS/DA and Frontier ML, study each sprint using this **5-Phase Active Loop** rather than passively reading textbooks or watching videos:

### Phase 1: Anchor the Spatial Intuition First (15 mins)
1. **Read `brief.md` before opening any textbook**:
   - Internalize Lens 1 (*Geometric Action*) and Lens 2 (*Geometric Invariant*). Ask yourself: *"What moves, and what refuses to change?"*
2. **Render & Watch the 4-Act Animation (`manim -pql`)**:
   - Watch Act 1 & Act 2 **with your eyes on the ghost grid and basis vectors** ($\hat{\imath}$ green, $\hat{\jmath}$ red, $\hat{k}$ blue).
   - Pause at Act 3 when the HUD equation appears (`storyboard.md` gives exact timestamps) and map every colored symbol in the `MathTex` card to the corresponding arrow or subspace on screen.

### Phase 2: Targeted Multi-Book Triangulation (35 mins)
Use the **Canonical Textbook Cross-Reference Matrix** in each sprint's `README.md` to read the exact sections in a deliberate sequence:
1. **Immersive Linear Algebra (ILA — Interactive Browser)**:
   - Open `https://immersivemath.com/ila/tableofcontents.html` first. Drag the interactive vector tips in the browser figures to feel the spatial sensitivity before reading formal proofs (Ch. 1–2 for Sprint 1, Ch. 6 for Sprints 2, 4, 8; Ch. 3 for Sprints 5–6; Ch. 7 for Sprint 6–7).
2. **Boyd & Vandenberghe (`VMLS.pdf`)**:
   - Read for **applied intuition, block-matrix mechanics, and least-squares optimization**:
     - Ch. 1 & 5: Vectors, linear combinations, span (Sprint 1)
     - Ch. 6–7, 10: Linear functions, matrix multiplication, composition (Sprint 2)
     - Ch. 10: Column and null space, inverse systems (Sprint 3)
     - Ch. 6, 10: Coordinate basis changes (Sprint 4)
     - Ch. 3–4: Norms, distances, inner products, angles, Cauchy-Schwarz (Sprint 5)
     - Ch. 11–12: Least squares, orthogonal projections, normal equations (Sprint 6)
     - Ch. 5.4, 10.4, 11.4: Gram-Schmidt orthonormalization, QR factorization (Sprint 7)
     - Ch. 6: Orthogonal transformations, rotations, reflections (Sprint 8)
3. **Deisenroth, Faisal & Ong (`Mathematics for Machine Learning.pdf`)**:
   - Read for **rigorous definitions, vector space geometry, and Frontier ML connections**:
     - §2.4: Vector spaces, linear independence, basis (Sprint 1)
     - §2.5–2.7: Linear mappings, matrix representations (Sprint 2)
     - §2.5–2.7: Image, kernel, rank-nullity (Sprint 3)
     - §2.7.2: Basis change commutative diagrams (Sprint 4)
     - §3.1–3.3: Norms, inner products, lengths, angles, Cauchy-Schwarz (Sprint 5)
     - §3.8: Orthogonal projections onto lines and general subspaces (Sprint 6)
     - §3.8.3: Gram-Schmidt orthonormalization (Sprint 7)
     - §3.9: Orthogonal matrices, rotations, transformations (Sprint 8)
4. **Kreyszig (10th Ed. Ch. 7–8, 20.9) & B.S. Grewal (Ch. 2)**:
   - Read for **theorems, rank-nullity proofs, QR numerical methods, and GATE CS/DA PYQ archetypes**.

### Phase 3: Break the Code via `tweaks_prompts.md` (25 mins)
Never treat `scene_*.py` as read-only. Open `tweaks_prompts.md` in the sprint folder:
1. **Execute the Parameter Tweaks**:
   - Change the matrices/vectors at the top of `scene_*.py` as instructed (e.g., orientation-reversing negative determinants in Sprint 2, rank-1 collapse in Sprint 3, complex rotation eigenvalues in Sprint 4, $p < 1$ quasi-norm stars in Sprint 5, ill-conditioned Lauchli matrices in Sprint 6 & 7, or 3D non-commutative axes in Sprint 8).
   - **Predict on paper** what the grid, basis vectors, and HUD invariants will look like *before* running `manim -pql`.
2. **Solve the 4 Intuition Stress-Test Prompts**:
   - Write down your geometric + algebraic answer first, then run the embedded NumPy verification snippet in `tweaks_prompts.md` to check your derivation against exact floating-point output.

### Phase 4: Read the Test Suite as Executable Theorems (15 mins)
Open `tests/test_sprint0X_invariants.py` and `tests/test_adversarial_sprintX.py`:
- Every `test_*` function is a **machine-verified mathematical theorem** (e.g., verifying $P^2 = P$ and $P^\top = P$ across thousands of random projections, $A^\top A = R^\top R$, or $Q^\top Q = I$).
- Run `pytest tests/test_sprint0X_invariants.py -v` and read how mathematical invariants translate into `np.allclose` assertions.

### Phase 5: Re-Synthesize into your Obsidian Brain (15 mins)
- Port the **Geometric Action**, **Geometric Invariants**, and **Dual Bridge (Frontier ML + GATE Shortcut)** directly into your personal Obsidian notes (`~/brain`).
- Link theorems back to the repository's rendered 1080p60 animations in `linear_algebra/animations/`.

---

## 3. Scratchpad & Prototyping

Isolated verification scripts for fast algebraic and numerical checks prior to animation implementation:
- [matrix_expansion.py](scratchpad_codes/matrix_expansion.py): Verifies quadratic forms $(\mathbf{a} - \mathbf{b})^\top \mathbf{M} (\mathbf{a} - \mathbf{b})$ under symmetric and non-symmetric matrices $\mathbf{M}$.

---

## 4. Foundational Philosophy & Blueprint

To anchor everything into a single, permanent architecture, we synthesize the core philosophy, the geometric translations, the contrarian focus strategies, and the two-year execution plan into one unified blueprint.

Mathematics and computer science fuse into a single, intuitive language that operates through spatial and mechanical reasoning.

---

### Part 1: The Core Philosophy (The Unified Field)

The foundational mistake of modern technical education is treating math as abstract symbol manipulation and computer science as writing lines of text code. To achieve muscle memory, you must adopt a single, elegant worldview:

* **Mathematics** is the study of pure structure, relationship, and transformation. It is the abstract blueprint of how things can move, warp, scale, and balance across arbitrary dimensions.
* **Computer Science** is the physical manifestation of that blueprint. It is the art of using physical matter (silicon, electricity) and time to build, automate, and navigate those mathematical structures.

When you look at software code, you are not looking at text; you are looking at a machine that constructs a geometric landscape in memory. When you look at an equation, you are looking at a description of how that landscape bends or flows.

---

### Part 2: The Geometric Translation of Technical Landscapes

To stop pushing your brain to calculate, offload symbolic logic onto your visual and spatial systems:

#### 1. Formal Logic & Discrete Math (Spatial Boundaries)
* **The Symbols**: $\land$ (AND), $\lor$ (OR), $\neg$ (NOT), $\implies$ (Implication).
* **The Geometry**: Logic is the science of spatial containment and boundaries. A proposition or data state is a designated region inside a universal coordinate space.
* **The Intuition**: The logical statement $A \implies B$ ("If A, then B") simply means the region A is entirely enclosed inside the larger boundary of region B. Modus Ponens (knowing that if you are in A, you must be in B) is just realizing that if you are physically standing inside the inner circle, you are automatically standing inside the outer circle.

#### 2. Linear Algebra (Warping Grid Lines)
* **The Symbols**: Matrices, Vectors, Determinants, Eigenvalues.
* **The Geometry**: Linear algebra is the visual art of stretching, shearing, compressing, and rotating space while ensuring the origin stays fixed and parallel grid lines remain straight.
* **The Intuition**: A matrix is not a grid of numbers; it is a recipe for a dynamic space warp. Multiplying a vector by a matrix means watching the entire coordinate system deform, pulling your vector along with it. The determinant measures how much an area or volume scales up or shrinks down during that warp. Eigenvectors are the special, unyielding pathways in space that do not change their physical direction when the universe warps around them—they simply stretch or shrink along their original axes.

#### 3. Calculus & Machine Learning Optimization (Mountain Terrains)
* **The Symbols**: Gradients, Loss Functions, Backpropagation, Derivatives.
* **The Geometry**: Machine learning is the exploration of high-dimensional physical topography.
* **The Intuition**: A machine learning model's loss function creates a literal landscape of mountains, valleys, and deep ravines. The gradient vector acts like a physical compass needle pointing directly up the steepest cliff face. Training a neural network using gradient descent is not an abstract arithmetic loop; it is the physical sensation of an object rolling down a rugged mountain path, navigating momentum, trying to locate the lowest possible valley floor (global minimum) without getting caught in a shallow pothole (local minimum).

#### 4. Probability & Causal AI (Hyper-Dimensional Volumes)
* **The Symbols**: Joint Probability $P(X, Y)$, Conditional Probability $P(X \mid Y)$, Marginalization.
* **The Geometry**: Probability is the geometry of slicing, dicing, and projecting smoke clouds.
* **The Intuition**: A complex joint probability distribution is like a high-dimensional cloud of smoke suspended inside a glass box. Conditioning on an event ($P(X \mid Y)$) means taking a flat, razor-sharp geometric slice right through that smoke at a specific coordinate. Marginalization (summing out a variable) is the act of turning off a dimension and looking at the flat shadow that the high-dimensional smoke cloud casts onto a lower-dimensional wall.

#### 5. Systems Architecture & Version Control (Flows, Allocations, and Trees)
* **Git**: Git is not a checklist of file backups; it is a Directed Acyclic Graph (DAG) growing across time. Branches are literal forks in a physical road. Running a git rebase means picking up a detached branch structure, lifting it by its base node, and physically sliding it over to paste it onto the tip of another node.
* **Operating Systems (Virtual Memory)**: The kernel acts like a magical optical illusionist. It convinces every single running process (tenant) that it owns a flat, continuous, unbroken infinity of space (linear memory address space). Behind the scenes, the kernel is secretly chopping their belongings into uniform physical chunks (pages) and scattering them across completely non-contiguous rooms inside the physical hardware chips (RAM).

---

### Part 3: The Focus Matrix (Prioritizing via Cognitive Constraints)

Because human energy, time, and focus are finite, prioritize deep mechanical anchors over superficial signals:

| What to Focus on MORE (Deep Anchors) | What to Focus on LESS (Superficial Signals) |
|---|---|
| **First-Principles Mimicry**: Building micro-versions of core engines (custom malloc, minimal Git, primitive compiler shell) from scratch. | **LeetCode Puzzle Grinding**: Memorizing isolated template tricks to solve hundreds of isolated competitive programming puzzles out of context. |
| **Structural & Paradigm Mastery**: Understanding the deep mathematical architecture behind a tool (e.g., how an asynchronous event loop works conceptually). | **Framework & API Chasing**: Memorizing specific, rapidly changing syntax quirks of trending libraries, wrappers, and high-level tools. |
| **Visual/Dynamic Prototyping**: Forcing yourself to animate, slide, and sketch mathematical parameters using interactive engines. | **Monolithic Note Encyclopedias**: Transcribing vast textbook chapters into exhaustive, dead walls of text inside your Obsidian vault. |

---

### Part 4: The 2-Year Execution Pipeline

A four-semester sequence taking you from raw logical gates to high-dimensional neural network optimization:

#### Year 1: Constructing the Spatial Infrastructure
* **Semester 1 (Months 1–6): The Digital Spine**
  * *Focus*: Demystify the leap from formal logic symbols to electrical signals and physical hardware memory.
  * *Canon*: *Book of Proof* (Hammack), *Code* (Petzold), *Computer Systems: A Programmer's Perspective* (Bryant & O'Hallaron).
  * *Projects*: Build an 8-bit Arithmetic Logic Unit (ALU) out of virtual NAND gates. Write a custom memory manager (`malloc` and `free`) in raw C99 to manipulate hardware byte arrays directly.
* **Semester 2 (Months 7–12): The Warp & The Topography**
  * *Focus*: Shift from sequential data layouts to spatial transformations (Linear Algebra) and fluid, continuous change (Calculus).
  * *Canon*: *Linear Algebra Done Right* (Axler), *Introduction to Algorithms* (CLRS).
  * *Tools*: GeoGebra for visual slider experimentation, Manim (Python) for programmatic animations.
  * *Projects*: Build a 3D wireframe graphics rasterizer from scratch in Python or C without libraries (custom projection and rotation matrices). Build a binary graph/network database engine with direct disk persistence.

#### Year 2: Achieving the Deep Synthesis
* **Semester 3 (Months 13–18): Randomness & Time Realities**
  * *Focus*: Master the geometry of high-dimensional probability and kernel orchestration of multi-processing space and time.
  * *Canon*: *Introduction to Probability* (Blitzstein & Hwang), *Operating Systems: Three Easy Pieces* (OSTEP).
  * *Projects*: Write a functional command-line Linux shell in C handling process forks (`fork()`), execution maps, and pipes. Build a mini-Git version control tool from scratch using SHA-1 content-addressable DAG nodes.
* **Semester 4 (Months 19–24): The Intelligence Continuum**
  * *Focus*: Fuse systems, calculus, linear algebra, and data networks to build machine learning intelligence from first principles.
  * *Canon*: *Mathematics for Machine Learning* (Deisenroth), *Deep Learning* (Goodfellow).
  * *Projects*: Write a scalar-valued automatic differentiation engine (`micrograd` style) from scratch in pure Python. Upgrade it to a Tensor/Matrix framework and write a visual tracer mapping gradient descent across a multi-dimensional loss surface.

---

### Part 5: The System Integration (Your Obsidian Engine)

Convert your notes into a **Triple-Anchor Intuition Matrix**. Every note for a core mathematical theorem or computer science concept contains three linked elements:

1. **The Human Narrative**: The historical frustration or bottleneck that forced the invention of the concept.
2. **The Geometric Blueprint**: An embedded visual asset (GeoGebra link, Manim animation snippet, or canvas sketch) showing how the concept deforms or behaves spatially.
3. **The Code Manifestation**: A minimal, clean, first-principles code snippet (in C, Rust, or Python) demonstrating the backbone execution of that structure.
