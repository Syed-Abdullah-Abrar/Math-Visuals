# Linear Algebra Curriculum

A visual, first-principles curriculum in Linear Algebra grounded in spatial transformations, geometric invariants, and coordinate geometry, modeled after Grant Sanderson (3Blue1Brown).

---

## Canonical References

- **Mathematics for Machine Learning (MML)**: Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong (Chapters 2–4, 10) — `/home/uni-kernel/brain/03_RESOURCES/Books/AI-ML/Mathematics/Mathematics for Machine Learning.pdf`
- **Introduction to Applied Linear Algebra (VMLS)**: Stephen Boyd, Lieven Vandenberghe (Chapters 1, 3–7, 10–12, 18) — `/home/uni-kernel/brain/03_RESOURCES/Books/Applied mathematics/Algebra/VMLS.pdf`
- **Immersive Linear Algebra (ILA)**: J. Ström, K. Åström, T. Akenine-Möller (Chapters 1–8) — `https://immersivemath.com/ila/tableofcontents.html`
- **Advanced Engineering Mathematics (Kreyszig, 10th Ed.)** & **Higher Engineering Mathematics (B.S. Grewal)**: Part B Chapters 7 & 8; Grewal Chapter 2 + GATE CS/DA PYQs — `/home/uni-kernel/brain/03_RESOURCES/Books/`

---

## Sprint Modules

| Sprint | Directory | Topic | Primary Text References | Core Deliverables | Status |
|---|---|---|---|---|---|
| **01** | [`sprint_01_vectors_span/`](sprint_01_vectors_span/) | Vectors, Linear Combinations & Span | MML §2.4 \| ILA Ch. 1–2 \| VMLS Ch. 1, 5 | Brief, Storyboard, `scene_vectors_span.py`, Tweaks & Prompts, Tests | **Completed** |
| **02** | [`sprint_02_linear_mappings/`](sprint_02_linear_mappings/) | Linear Mappings & Matrix Composition | MML §2.7 \| ILA Ch. 6 \| VMLS Ch. 6–7, 10 | Brief, Storyboard, `scene_linear_mappings.py`, Tweaks & Prompts, Tests | **Completed** |
| **03** | [`sprint_03_column_null_space/`](sprint_03_column_null_space/) | Column Space, Null Space & Rank-Nullity (3D) | MML §2.5–2.7 \| Kreyszig §7.3–7.5 \| Grewal Ch. 2 | Brief, Storyboard, `scene_column_null_space.py` (`ThreeDScene`), Tweaks & Prompts, Tests | **Completed** |
| **04** | [`sprint_04_change_of_basis/`](sprint_04_change_of_basis/) | Change of Basis & Similarity ($P^{-1}AP$) | MML §2.7.2 \| ILA Ch. 6 | Brief, Storyboard, `scene_change_of_basis.py`, Tweaks & Prompts, Tests | **Completed** |
| **05** | [`sprint_05_norms_inner_products/`](sprint_05_norms_inner_products/) | Norms ($L_1, L_2, L_\infty$), Inner Products & Cauchy-Schwarz | MML §3.1–3.3 \| ILA Ch. 3 \| VMLS Ch. 3–4 | Brief, Storyboard, `scene_norms_inner_products.py`, Tweaks & Prompts, Tests | **Completed** |
| **06** | [`sprint_06_orthogonal_projections/`](sprint_06_orthogonal_projections/) | Orthogonal Projections onto Lines & Subspaces | MML §3.8 \| ILA Ch. 3, 6 \| VMLS Ch. 11–12 | Brief, Storyboard, `scene_orthogonal_projections.py` (`ThreeDScene`), Tweaks & Prompts, Tests | **Completed** |
| **07** | [`sprint_07_gram_schmidt_qr/`](sprint_07_gram_schmidt_qr/) | Gram-Schmidt Orthonormalization & $QR$ Decomposition | MML §3.8.3 \| VMLS Ch. 5.4, 10.4, 11.4 \| Kreyszig §20.9 | Brief, Storyboard, `scene_gram_schmidt_qr.py` (`ThreeDScene`), Tweaks & Prompts, Tests | **Completed** |
| **08** | [`sprint_08_orthogonal_matrices_rotations/`](sprint_08_orthogonal_matrices_rotations/) | Orthogonal Matrices, Rotations ($SO(n)$) & Givens/Householder | MML §3.9 \| ILA Ch. 6 \| Kreyszig §8.3 | Brief, Storyboard, `scene_orthogonal_matrices_rotations.py`, Tweaks & Prompts, Tests | **Completed** |

---

## Curated 1080p60 Production Animations

All production animations are rendered in full high-definition (1080p at 60 fps) and organized in [`linear_algebra/animations/`](animations/):

| Sprint | Scene Name | Resolution / FPS | Animation File Link |
|---|---|---|---|
| **01** | `VectorSpanScene` | 1080p60 (30.0s) | [`animations/sprint_01_VectorSpanScene_1080p60.mp4`](animations/sprint_01_VectorSpanScene_1080p60.mp4) |
| **02** | `LinearMappingsScene` | 1080p60 (38.5s) | [`animations/sprint_02_LinearMappingsScene_1080p60.mp4`](animations/sprint_02_LinearMappingsScene_1080p60.mp4) |
| **03** | `ColumnNullSpaceScene` | 1080p60 (48.0s) | [`animations/sprint_03_ColumnNullSpaceScene_1080p60.mp4`](animations/sprint_03_ColumnNullSpaceScene_1080p60.mp4) |
| **04** | `ChangeOfBasisScene` | 1080p60 (44.5s) | [`animations/sprint_04_ChangeOfBasisScene_1080p60.mp4`](animations/sprint_04_ChangeOfBasisScene_1080p60.mp4) |
| **05** | `NormsInnerProductsScene` | 1080p60 (45.0s) | [`animations/sprint_05_NormsInnerProductsScene_1080p60.mp4`](animations/sprint_05_NormsInnerProductsScene_1080p60.mp4) |
| **06** | `OrthogonalProjectionsScene` | 1080p60 (49.0s) | [`animations/sprint_06_OrthogonalProjectionsScene_1080p60.mp4`](animations/sprint_06_OrthogonalProjectionsScene_1080p60.mp4) |
| **07** | `GramSchmidtQRScene` | 1080p60 (52.0s) | [`animations/sprint_07_GramSchmidtQRScene_1080p60.mp4`](animations/sprint_07_GramSchmidtQRScene_1080p60.mp4) |
| **08** | `OrthogonalRotationsScene` | 1080p60 (46.0s) | [`animations/sprint_08_OrthogonalRotationsScene_1080p60.mp4`](animations/sprint_08_OrthogonalRotationsScene_1080p60.mp4) |

---

## Key Concepts & Geometric Insights Across Sprints 5–8

### Sprint 5: Norms ($L_1, L_2, L_\infty$), Inner Products & Cauchy-Schwarz
- **Continuous Unit Ball Morphing ($L_1 \to L_2 \to L_\infty$)**: The unit ball $B_p = \{\mathbf{x} : \|\mathbf{x}\|_p \le 1\}$ morphs from a sharp rotated diamond ($p=1$) to a Euclidean circle ($p=2$) to an axis-aligned square ($p=\infty$). When $p < 1$, the shape pinches inward into a non-convex star, breaking the triangle inequality ($\|\mathbf{e}_1 + \mathbf{e}_2\|_{0.5} = 4 > 2$).
- **Dot Product as Ruler Projection**: The inner product $\mathbf{x}^\top\mathbf{y}$ drops an orthogonal shadow of $\mathbf{x}$ onto $\mathbf{y}$'s unit ruler and scales by $\|\mathbf{y}\|_2$. Cauchy-Schwarz $|\mathbf{x}^\top\mathbf{y}| \le \|\mathbf{x}\|_2\|\mathbf{y}\|_2$ geometrically means a shadow length cannot exceed the hypotenuse radius, saturating only when vectors are collinear.
- **SPD Inner Products Warp Space**: Any symmetric positive-definite matrix $A \succ 0$ defines $\langle\mathbf{x},\mathbf{y}\rangle_A = \mathbf{x}^\top A\mathbf{y}$, which warps the circular unit ball into an $A$-unit ellipsoid via Cholesky factor $L$ ($A = L L^\top$). Vectors obtuse in standard coordinates become orthogonal under the whitened geometry.
- **Frontier ML & GATE Insights**:
  - Attention scaling ($1/\sqrt{d_k}$) prevents cosine similarity $\mathbf{q}^\top\mathbf{k}$ from saturating softmax gradients.
  - $L_1$ regularization produces exact sparse weights because linear objective contours touch the sharp diamond corners first; $L_2$ shrinks coordinates uniformly.
  - RMSNorm projects activation vectors onto the sphere of radius $\sqrt{d}$ without mean subtraction.
  - Sylvester's criterion tests positive definiteness in seconds via strictly positive leading principal minors ($\Delta_k > 0$).

---

### Sprint 6: Orthogonal Projections onto 1D Lines & $k$-D Subspaces
- **Minimum Distance Geometry**: For any target $\mathbf{y}$ and subspace $U = \mathcal{C}(A)$, the distance $\|\mathbf{y} - A\mathbf{x}\|_2$ forms a strictly convex quadratic bowl. Its unique global minimum occurs at the foot of the perpendicular, where the error vector $\mathbf{r} = \mathbf{y} - A\hat{\mathbf{x}}$ is orthogonal to every column of $A$:
  $$A^\top(\mathbf{y} - A\hat{\mathbf{x}}) = \mathbf{0} \implies A^\top A\hat{\mathbf{x}} = A^\top\mathbf{y} \implies P = A(A^\top A)^{-1}A^\top$$
- **Projector Invariants**:
  - Idempotence: $P^2 = P$ (projecting a shadow does nothing new).
  - Symmetry: $P^\top = P \iff$ the projection is orthogonal ($90^\circ$ drop).
  - Trace equals rank: $\operatorname{Tr}(P) = \operatorname{rank}(P) = k = \dim(U)$, with eigenvalues strictly in $\{0, 1\}$.
  - Pythagorean decomposition: $\mathbf{y} = P\mathbf{y} + (I - P)\mathbf{y}$, where $\|\mathbf{y}\|_2^2 = \|P\mathbf{y}\|_2^2 + \|(I - P)\mathbf{y}\|_2^2$.
- **Frontier ML & GATE Insights**:
  - Ordinary Least Squares (OLS) is orthogonal projection onto the column space via hat matrix $H = A(A^\top A)^{-1}A^\top$.
  - Ridge regression adds $\lambda I$ to solve ill-conditioned inverses ($(A^\top A + \lambda I)^{-1}$), applying SVD shrinkage factors $\frac{\sigma_i^2}{\sigma_i^2 + \lambda}$.
  - Rank-1 projector shortcut: for any nonzero vector $\mathbf{u}$, $P = \frac{\mathbf{u}\mathbf{u}^\top}{\|\mathbf{u}\|_2^2}$ instantly gives $\operatorname{rank}=1, \operatorname{Tr}=1, \det=0$.

---

### Sprint 7: Gram-Schmidt Orthonormalization & $QR$ Decomposition
- **Sequential Orthogonal Stripping**: Starting from a skewed basis $\{\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3\}$, each vector is stripped of its parallel shadows onto the preceding orthonormal vectors:
  $$\mathbf{u}_1 = \mathbf{a}_1, \quad \mathbf{q}_1 = \frac{\mathbf{u}_1}{\|\mathbf{u}_1\|_2}$$
  $$\mathbf{u}_2 = \mathbf{a}_2 - (\mathbf{q}_1^\top\mathbf{a}_2)\mathbf{q}_1, \quad \mathbf{q}_2 = \frac{\mathbf{u}_2}{\|\mathbf{u}_2\|_2}$$
  $$\mathbf{u}_3 = \mathbf{a}_3 - (\mathbf{q}_1^\top\mathbf{a}_3)\mathbf{q}_1 - (\mathbf{q}_2^\top\mathbf{a}_3)\mathbf{q}_2, \quad \mathbf{q}_3 = \frac{\mathbf{u}_3}{\|\mathbf{u}_3\|_2}$$
- **The Geometry of $A = QR$**: $Q \in O(n)$ is a rigid orthonormal frame lying on the unit sphere $S^{n-1}$ ($Q^\top Q = I$), while $R$ is upper-triangular, recording the forward-causal shear and scaling coefficients ($R_{ii} = \|\mathbf{u}_i\|_2 > 0, R_{ij} = \mathbf{q}_i^\top\mathbf{a}_j$).
- **Volume Conservation**: The volume of the oblique parallelepiped spanned by $A$'s columns equals the product of the orthogonal altitudes: $|\det(A)| = \prod_{i=1}^n R_{ii}$.
- **Frontier ML & Numerical Conditioning**:
  - Normal equations square the condition number: $\kappa_2(A^\top A) = \kappa_2(A)^2$. On ill-conditioned inputs like the Lauchli matrix ($\kappa \approx 10^8$), $A^\top A$ becomes singular in IEEE-754 `float64`, whereas $QR$ solves $R\hat{\mathbf{x}} = Q^\top\mathbf{b}$ at $\kappa_2(R) = \kappa_2(A)$, preserving numerical precision.
  - The Muon optimizer uses polar-factor orthogonalization ($UV^\top$) via Newton-Schulz iterations to stabilize momentum update tensors.

---

### Sprint 8: Orthogonal Matrices, Rotations ($SO(n)$) & Givens/Householder
- **Rigid Isometries**: Orthogonal matrices $Q \in O(n)$ preserve lengths ($\|Q\mathbf{x}\|_2 = \|\mathbf{x}\|_2$), pairwise distances, and inner products/angles ($Q^\top Q = I \iff Q^{-1} = Q^\top$).
- **Topology of $O(n)$**:
  - $SO(n)$ ($\det Q = +1$): Proper rotations connected to $I_n$, preserving orientation/handedness.
  - $O(n) \setminus SO(n)$ ($\det Q = -1$): Reflections reversing orientation, separated from $SO(n)$ by the singular wall $\det = 0$.
- **Cartan–Dieudonné Theorem**: Every orthogonal transformation in $\mathbb{R}^n$ decomposes into at most $n$ reflections. Composing two reflections across hyperplanes separated by angle $\Delta\alpha$ yields a pure rotation through $2\Delta\alpha$.
- **Non-Commutativity in 3D**: While 2D rotations commute ($R(\alpha)R(\beta) = R(\beta)R(\alpha)$), 3D rotations in $SO(3)$ do **not** commute ($R_x R_y \ne R_y R_x$), as shown by 3D cube reorientations.
- **Givens vs Householder**:
  - Givens rotation $G(i, j, \theta) \in SO(n)$ performs a surgical 2D plane rotation to zero a single target entry.
  - Householder reflector $H = I - 2\frac{\mathbf{v}\mathbf{v}^\top}{\|\mathbf{v}\|_2^2} \in O(n)$ reflects across $\mathbf{v}^\perp$, zeroing an entire subvector at once. Using the sign rule $\mathbf{v} = \mathbf{x} + \operatorname{sgn}(x_1)\|\mathbf{x}\|_2\mathbf{e}_1$ prevents catastrophic floating-point cancellation.
- **Frontier ML**:
  - Rotary Position Embeddings (RoPE) apply $SO(2)$ block rotations to query/key vectors, guaranteeing attention scores depend only on relative token displacement $(n - m)$.
  - Isometric orthogonal weight initialization preserves signal variance across 100+ deep linear layers ($\|W_L \cdots W_1 \mathbf{x}\| = \|\mathbf{x}\|$).

---

## Standard 4-Part Protocol

Every sprint directory adheres to a strict 4-part pedagogical and engineering structure:
1. `brief.md` — **Geometric Consensus Brief** (~350 words across 4 lenses: Geometric Action, Geometric Invariant, Algebraic-to-Visual Dictionary, and Dual Bridge to Frontier ML + GATE CS/DA).
2. `storyboard.md` — **4-Act Visual Storyboard** (Act 1 Concrete Spatial Puzzle, Act 2 Dynamic Deformation & Ghost Space, Act 3 Color-Coded Algebraic Morph, Act 4 Degenerate/Edge-Case Stress Test).
3. `scene_*.py` — **Production ManimCE v0.18+ Python Script** adhering to the 8 ManimCE rules (3b1b color palette, zero text overlap via `make_hud_panel`, `ValueTracker` updaters, exact `numpy.linalg` calculations).
4. `tweaks_prompts.md` — **Interactive Parameter Tweaks & Self-Test Prompts** with executable NumPy verification snippets.
5. `README.md` — **Master Companion Guide** integrating textbook cross-references, scene architecture, and CLI commands.

---

## Verification & Test Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all 587 automated invariant and adversarial tests across Sprints 1–8
pytest tests/ -v

# Render Master Scenes (Fast 480p15 Preview)
manim -pql linear_algebra/sprint_01_vectors_span/scene_vectors_span.py VectorSpanScene
manim -pql linear_algebra/sprint_02_linear_mappings/scene_linear_mappings.py LinearMappingsScene
manim -pql linear_algebra/sprint_03_column_null_space/scene_column_null_space.py ColumnNullSpaceScene
manim -pql linear_algebra/sprint_04_change_of_basis/scene_change_of_basis.py ChangeOfBasisScene
manim -pql linear_algebra/sprint_05_norms_inner_products/scene_norms_inner_products.py NormsInnerProductsScene
manim -pql linear_algebra/sprint_06_orthogonal_projections/scene_orthogonal_projections.py OrthogonalProjectionsScene
manim -pql linear_algebra/sprint_07_gram_schmidt_qr/scene_gram_schmidt_qr.py GramSchmidtQRScene
manim -pql linear_algebra/sprint_08_orthogonal_matrices_rotations/scene_orthogonal_matrices_rotations.py OrthogonalRotationsScene

# Render Production 1080p60 Scenes (Headless)
manim -qh linear_algebra/sprint_05_norms_inner_products/scene_norms_inner_products.py NormsInnerProductsScene
manim -qh linear_algebra/sprint_06_orthogonal_projections/scene_orthogonal_projections.py OrthogonalProjectionsScene
manim -qh linear_algebra/sprint_07_gram_schmidt_qr/scene_gram_schmidt_qr.py GramSchmidtQRScene
manim -qh linear_algebra/sprint_08_orthogonal_matrices_rotations/scene_orthogonal_matrices_rotations.py OrthogonalRotationsScene
```
