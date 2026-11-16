# Sprint 4: Change of Basis & Similarity Transformations ($P^{-1}AP$)

## Overview & Curriculum Context

Sprint 4 delivers the definitive visual module for Change of Basis and Similarity Transformations following the 3Blue1Brown pedagogical standard. Rather than treating similarity conjugation $A = PDP^{-1}$ as an abstract algebraic trick, this module illuminates its geometric core: a linear operator is an intrinsic spatial action that exists independently of coordinates. In standard Cartesian coordinates, a symmetric matrix $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ couples stretching and shearing, rotating generic vectors. However, when viewed through the lens of its eigen-basis $P = [\mathbf{v}_1 \mid \mathbf{v}_2]$, the exact same transformation uncouples into pure, independent 1D scalings along orthogonal axes.

Furthermore, this sprint visually decomposes the similarity cycle into three intuitive geometric steps ($P^{-1} \to D \to P$):
1. **$P^{-1}$**: Standard-to-eigen change of basis, reprojecting standard coordinates into coordinates relative to the eigen-basis.
2. **$D$**: Decoupled diagonal scaling, stretching eigen-axes independently by eigenvalues $\lambda_1 = 3$ and $\lambda_2 = 1$ without cross-talk or rotation.
3. **$P$**: Eigen-to-standard restoration, translating scaled components back into standard Euclidean coordinates to reproduce $A\mathbf{x}$.

Finally, the module stress-tests the limits of diagonalizability with defective Jordan matrices $M_{\text{def}} = \begin{bmatrix} 2 & 1 \\ 0 & 2 \end{bmatrix}$, showing why an eigen-grid cannot be formed when geometric multiplicity is strictly less than algebraic multiplicity.

This module grounds core concepts across canonical literature:
- **Mathematics for Machine Learning (MML)**: Basis change, coordinates, transformation matrices, similarity, and diagonalizability (§2.7).
- **Introduction to Applied Linear Algebra (VMLS - Boyd & Vandenberghe)**: Matrix operations, change of coordinates, and diagonal representations (Ch. 6 & 10).
- **Immersive Linear Algebra (ILA - Ström, Åström et al.)**: Geometric interpretations of eigenvectors, eigenvalues, and basis changes (Ch. 6).
- **Advanced Engineering Mathematics (Kreyszig 10th Ed.)**: Matrix eigenvalue problems, similarity transformations, diagonalization, invariant trace and determinant (Part B: Ch. 8).

---

## The 4-Part Pedagogical Protocol

Sprint 4 is organized around the rigorous 4-part protocol ensuring pedagogical clarity, mathematical precision, and automated testability:

```
linear_algebra/sprint_04_change_of_basis/
├── README.md                   # Master companion guide (this document)
├── brief.md                    # Part 1: Geometric Consensus Brief (~350 words, 4 lenses)
├── storyboard.md               # Part 2: 4-Act Visual Storyboard & Choreography
├── scene_change_of_basis.py    # Part 3: Production ManimCE v0.18+ Animation Script
└── tweaks_prompts.md           # Part 4: Interactive Tweaks & Intuition Stress-Test Prompts
```

| Part | Document | Primary Role |
|---|---|---|
| **Part 1** | [`brief.md`](./brief.md) | **Geometric Consensus Brief**: Synthesizes Geometric Action, Geometric Invariants, Algebraic-to-Visual Dictionary, and Dual Bridge (Frontier ML + GATE CS/DA shortcuts). |
| **Part 2** | [`storyboard.md`](./storyboard.md) | **4-Act Visual Storyboard**: Frame-by-frame spatial choreography detailing timing (60–120s), camera motions, color palettes, and pinned HUD states. |
| **Part 3** | [`scene_change_of_basis.py`](./scene_change_of_basis.py) | **Production ManimCE Code**: Executable animation script adhering to the 8 ManimCE coding rules (zero text overlap, ValueTracker, exact NumPy math). |
| **Part 4** | [`tweaks_prompts.md`](./tweaks_prompts.md) | **Interactive Tweaks & Intuition Prompts**: Parameter modification matrix and 4 rigorous conceptual stress-test problems with executable NumPy proofs. |

---

## Canonical Textbook Cross-Reference Matrix

| Topic / Invariant | MML Reference | VMLS Reference | Kreyszig Reference | Visual / Algorithmic Manifestation |
|---|---|---|---|---|
| **Coordinate Transition** | §2.7.2 (Def 2.18–2.19) | §6.1 (pp. 107–110) | §8.4 (pp. 340–345) | $[\mathbf{x}]_B = P^{-1}\mathbf{x}$; reprojecting vector onto eigen-grid scaffolding |
| **Basis Change Matrix $P$** | §2.7.2 (Theorem 2.20) | §6.1 & §10.1 | §8.4 | $P = [\mathbf{v}_1 \mid \mathbf{v}_2]$; columns are orthogonal eigenvectors |
| **Similarity Conjugation** | §2.7.2 (Def 2.22) | §10.1 (pp. 175–178) | §8.4 (Theorem 1) | $A = P D P^{-1} \iff D = P^{-1} A P$; similarity equivalence |
| **Trace Invariance** | §2.7.2 (p. 56) | §10.2 | §8.4 (Theorem 2) | $\text{Tr}(P^{-1}AP) = \text{Tr}(A) = \lambda_1 + \lambda_2 = 4$; first-order dilation |
| **Determinant Invariance** | §2.7.2 (p. 56) | §10.2 | §8.4 (Theorem 2) | $\det(P^{-1}AP) = \det(A) = \lambda_1 \lambda_2 = 3$; global area scaling |
| **Spectrum Invariance** | §2.7.2 | §10.2 | §8.4 (Theorem 2) | Characteristic polynomial $\lambda^2 - 4\lambda + 3 = 0$ conserved |
| **Matrix Powers $A^k$** | §2.7.2 | §10.2 | §8.4 (p. 343) | $A^k = P D^k P^{-1}$; fast $\mathcal{O}(1)$ closed-form exponentiation |
| **Defective Non-Diagonalizability** | §2.7.2 | §6.1 | §8.4 (Def 2) | $M_{\text{def}} = \begin{bmatrix} 2 & 1 \\ 0 & 2 \end{bmatrix}$; geometric mult $1 < 2$; 1D eigenspace collapse |
| **ML: PCA Covariance Decorrelation** | §10.3 (pp. 317–330) | §10.3 | §8.5 | $\Sigma = Q \Lambda Q^\top$; projection $Z = X Q$ completely diagonalizes covariance |
| **GATE: Cayley-Hamilton Inverse** | §2.7.2 | §10.2 | §8.4 (p. 344) | $p(A) = A^2 - 4A + 3I = 0 \implies A^{-1} = \frac{1}{3}(4I - A)$ |

---

## Scene Manifest & Architecture

The production script `scene_change_of_basis.py` provides six modular `Scene` classes designed for independent rendering or cohesive end-to-end playback:

```
scene_change_of_basis.py
├── BaseChangeOfBasisScene (Shared utilities, HUD panel factory, color palettes, coordinate planes)
├── Act1SpatialPuzzle (~18s)             [Scene]
├── Act2DynamicDeformation (~24s)        [Scene]
├── Act3AlgebraicMorph (~22s)            [Scene]
├── Act4DegenerateStressTest (~24s)      [Scene]
└── ChangeOfBasisScene (~88s)            [Master Combined Scene]
```

### Scene Descriptions
1. **`Act1SpatialPuzzle`**:
   - Poses the fundamental reachability dilemma: In standard Cartesian coordinates, $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ couples stretching and shearing, rotating $\mathbf{x} = [2, 0]^\top$ into $[4, 2]^\top$.
   - Highlights that off-diagonal terms entangle coordinates. Can we find a coordinate frame with zero rotation?
2. **`Act2DynamicDeformation`**:
   - Overlays standard Cartesian grid with oblique eigen-grid $P = [\mathbf{v}_1 \mid \mathbf{v}_2]$ where $\mathbf{v}_1 = [1, 1]^\top$ and $\mathbf{v}_2 = [-1, 1]^\top$.
   - Uses `ValueTracker` to animate continuous homotopy $M(t) = (1-t)I + tA$, demonstrating that while the standard grid shears, the eigen-grid lines slide along themselves without tilting. $\mathbf{v}_1$ stretches $3\times$ and $\mathbf{v}_2$ scales $1\times$ with zero rotation.
3. **`Act3AlgebraicMorph`**:
   - Animates the 3-step similarity sequence on $\mathbf{x} = [2, 0]^\top$:
     - Step 1: $P^{-1}$ standard-to-eigen change of basis ($[2, 0]^\top \to [1, -1]_B$).
     - Step 2: $D$ decoupled diagonal scaling ($[1, -1]_B \to [3, -1]_B$).
     - Step 3: $P$ restoration to standard frame ($3\mathbf{v}_1 - 1\mathbf{v}_2 = [4, 2]^\top = A\mathbf{x}$).
   - Highlights conserved similarity invariants: $\text{Tr}(P^{-1}AP) = \text{Tr}(A) = 4$, $\det(P^{-1}AP) = \det(A) = 3$.
4. **`Act4DegenerateStressTest`**:
   - Geometrically investigates defective shear matrix $M_{\text{def}} = \begin{bmatrix} 2 & 1 \\ 0 & 2 \end{bmatrix}$.
   - Demonstrates that although algebraic multiplicity of $\lambda = 2$ is 2, the eigenspace $\text{nullity}(M_{\text{def}} - 2I)$ is only 1D ($\text{span}([1, 0]^\top)$).
   - Proves visually that without two linearly independent eigenvectors, the eigenvector matrix $P$ is singular ($\det(P) = 0$), preventing similarity diagonalization and necessitating Jordan Canonical Form.
5. **`ChangeOfBasisScene`**:
   - The master comprehensive narrative stringing Acts 1 through 4 together with seamless camera and HUD transitions.

---

## CLI Render Commands

Run all rendering commands from the project root using the dedicated Python virtual environment:

```bash
# 1. Fast Low-Quality Preview (480p15) — Master Scene
.venv/bin/manim -ql linear_algebra/sprint_04_change_of_basis/scene_change_of_basis.py ChangeOfBasisScene

# 2. Render Individual Modular Acts
.venv/bin/manim -ql linear_algebra/sprint_04_change_of_basis/scene_change_of_basis.py Act1SpatialPuzzle
.venv/bin/manim -ql linear_algebra/sprint_04_change_of_basis/scene_change_of_basis.py Act2DynamicDeformation
.venv/bin/manim -ql linear_algebra/sprint_04_change_of_basis/scene_change_of_basis.py Act3AlgebraicMorph
.venv/bin/manim -ql linear_algebra/sprint_04_change_of_basis/scene_change_of_basis.py Act4DegenerateStressTest

# 3. High-Quality Production Render (1080p60)
.venv/bin/manim -qh linear_algebra/sprint_04_change_of_basis/scene_change_of_basis.py ChangeOfBasisScene
```

---

## Programmatic Verification

Sprint 4 is covered by automated unit and adversarial test suites in `tests/`:

```bash
# Run full 4-tier invariant test suite (65 tests, including static AST checks)
.venv/bin/pytest tests/test_sprint04_invariants.py -v

# Run 8-tier adversarial stress suite (100k random vectors, ill-conditioned matrices, Jordan blocks)
.venv/bin/python tests/test_adversarial_sprint4.py
```
