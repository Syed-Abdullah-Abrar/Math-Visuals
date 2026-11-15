# Sprint 2: Linear Mappings & Matrix Multiplication as Composition

## Overview & Curriculum Context

Sprint 2 delivers the geometric consensus module explaining linear transformations and matrix multiplication following the 3Blue1Brown visual standard. Rather than treating matrices as static number arrays, this module visualizes matrices as dynamic spatial operators that continuously deform space while strictly preserving parallel, evenly spaced coordinate grid lines and fixing the origin in place.

Furthermore, this sprint demystifies matrix multiplication by grounding it directly in sequential geometric function composition: multiplying $BA$ corresponds to applying transformation $A$ first, followed by transformation $B$ on the already-deformed coordinate grid. By contrasting sequential operations involving shear ($S$) and rotation ($R$), the module provides an unshakeable geometric proof for matrix non-commutativity ($SR \neq RS$).

This module grounds core concepts across canonical literature:
- **Mathematics for Machine Learning (MML)**: Linear mappings, coordinates, transformation matrices, dimension isomorphism, and composition (§2.5).
- **Introduction to Applied Linear Algebra (VMLS - Boyd & Vandenberghe)**: Matrix operations, linear functions, superposition, matrix multiplication as composition, and non-commutativity (Ch. 6, 7 & 10).
- **Immersive Linear Algebra (ILA - Ström, Åström et al.)**: Geometric interpretations of linear mappings, basis deformations, and 2D matrix transformations (Ch. 6).
- **Advanced Engineering Mathematics (Kreyszig 10th Ed.)**: Linear systems, matrix algebra, transformations, determinant product theorems, and non-commutativity (Part B: Ch. 7).

---

## The 4-Part Pedagogical Protocol

Sprint 2 is organized around the rigorous 4-part protocol ensuring pedagogical depth, production reproducibility, and mathematical precision:

```
linear_algebra/sprint_02_linear_mappings/
├── README.md                 # Master companion guide (this document)
├── brief.md                  # Part 1: Geometric Consensus Brief (~350 words)
├── storyboard.md             # Part 2: 4-Act Visual Storyboard & Choreography
├── scene_linear_mappings.py  # Part 3: Production ManimCE v0.18+ Animation Script
└── tweaks_prompts.md         # Part 4: Interactive Tweaks & Intuition Stress-Test Prompts
```

| Part | Document | Primary Role |
|---|---|---|
| **Part 1** | [`brief.md`](./brief.md) | **Geometric Consensus Brief**: Synthesizes the Geometric Action, Geometric Invariants, Algebraic-to-Visual Dictionary, and Dual Bridge (Frontier ML + GATE CS/DA shortcuts). |
| **Part 2** | [`storyboard.md`](./storyboard.md) | **4-Act Visual Storyboard**: Frame-by-frame spatial choreography detailing timing (60–120s), camera motions, color palettes, and pinned HUD states. |
| **Part 3** | [`scene_linear_mappings.py`](./scene_linear_mappings.py) | **Production ManimCE Code**: Executable animation script adhering to the 8 ManimCE coding rules (zero text overlap, ValueTracker, exact NumPy math). |
| **Part 4** | [`tweaks_prompts.md`](./tweaks_prompts.md) | **Interactive Tweaks & Intuition Prompts**: Parameter modification matrix and 4 rigorous conceptual stress-test problems with executable NumPy proofs. |

---

## Canonical Textbook Cross-Reference Matrix

| Topic / Invariant | MML Reference | VMLS Reference | ILA Reference | Visual / Algorithmic Manifestation |
|---|---|---|---|---|
| **Linearity Axioms** | §2.5 (Def 2.15) | §7.1 (pp. 127–130) | §6.1 (Def 6.1) | Superposition $T(c\mathbf{u} + d\mathbf{v}) = cT(\mathbf{u}) + dT(\mathbf{v})$ |
| **Origin Anchor Invariant** | §2.5 (p. 48) | §7.1 (p. 128) | §6.1 | $A\mathbf{0} = \mathbf{0}$; white origin dot remains immovable |
| **Parallel Grid Invariance** | §2.5 (Theorem 2.17) | §7.1 (pp. 131–134) | §6.2 | Tangent direction $A\mathbf{v}$ identical for parallel lines; grid lines stay parallel |
| **Uniform Spacing Preservation** | §2.5 | §7.1 | §6.2 | Consecutive interval steps $\Delta t (A\mathbf{v})$ remain strictly constant |
| **Basis Destination Tracking** | §2.5 (Def 2.18–2.19) | §6.1 & §7.2 | §6.2 (Thm 6.2) | Matrix columns are transformed unit basis vectors: $A = [\mathbf{a}_1 \mid \mathbf{a}_2]$ |
| **Sequential Composition** | §2.5 (p. 52) | §10.1 (pp. 175–178) | §6.3 | Matrix product $BA$ evaluates $A$ first, then $B$ on deformed grid |
| **Non-Commutativity ($BA \ne AB$)**| §2.5 (p. 53) | §10.2 (pp. 178–180) | §6.3 | Spatial divergence $\|\Delta\| = \sqrt{13}$ under rotated shear vs sheared rotation |
| **Determinant Multiplicativity** | §2.5 (Theorem 2.20) | §10.2 | §6.4 | Unit parallelogram area scaling commutes: $\det(BA) = \det(B)\det(A)$ |
| **Cyclic Trace Invariance** | §2.5 | §10.2 | §6.4 | $\text{Tr}([S, R]) = \text{Tr}(SR - RS) = 0$; trace invariant under cyclic permutation |

---

## Scene Manifest & Architecture

The production script `scene_linear_mappings.py` provides five modular `Scene` classes designed for independent rendering or cohesive end-to-end playback:

```
scene_linear_mappings.py
├── BaseLinearMappingsScene (Shared utilities, HUD panel factory, color palettes, coordinate planes)
├── Act1SpatialPuzzle (~18s)              [Scene]
├── Act2DynamicDeformation (~24s)        [Scene]
├── Act3AlgebraicMorph (~20s)            [Scene]
├── Act4NonCommutativeStressTest (~24s)  [Scene]
└── LinearMappingsScene (~86s)           [Master Combined Scene]
```

### Scene Descriptions
1. **`Act1SpatialPuzzle`**:
   - Poses the fundamental reachability challenge: How does knowing the destination of only two basis vectors $\hat{\imath} \to \mathbf{a}_1 = [2, 1]^\top$ and $\hat{\jmath} \to \mathbf{a}_2 = [0, 2]^\top$ determine the mapping of arbitrary vector $\mathbf{x} = [1.5, 1.0]^\top$?
   - Demonstrates linear combination $1.5\mathbf{a}_1 + 1.0\mathbf{a}_2$ converging on target $A\mathbf{x} = [3.0, 3.5]^\top$.
2. **`Act2DynamicDeformation`**:
   - Uses `ValueTracker` instances to continuously sweep matrix homotopy parameter $t \in [0, 1]$ in $M(t) = (1-t)I + tA$.
   - Directly contrasts dynamic deforming grid against static ghost Cartesian background grid.
   - Live dynamic readout of unit parallelogram area scaling from $1.00$ to $\det(A) = 4.00$.
3. **`Act3AlgebraicMorph`**:
   - Connects matrix multiplication directly to sequential function composition: $(B \circ A)(\mathbf{x}) = B(A\mathbf{x}) = (BA)\mathbf{x}$.
   - Applies $90^\circ$ rotation $B$ on the already-deformed grid $A$, transforming basis columns to $B\mathbf{a}_1 = [-1, 2]^\top$ and $B\mathbf{a}_2 = [-2, 0]^\top$.
   - Highlights determinant area product $\det(BA) = \det(B)\det(A) = 1 \times 4 = 4$.
4. **`Act4NonCommutativeStressTest`**:
   - Geometrically proves non-commutativity using horizontal shear $S = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$ and rotation $R = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$.
   - Tests probe vector $\mathbf{x} = [2, 3]^\top$, showing that $(SR)\mathbf{x} = [-1, 2]^\top$ while $(RS)\mathbf{x} = [-3, 5]^\top$.
   - Illustrates the spatial divergence vector of length $\|\Delta\| = \sqrt{13} \approx 3.61$ and the commutator $[S, R] = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \ne \mathbf{0}$ with $\text{Tr}([S, R]) = 0$.
5. **`LinearMappingsScene`**:
   - The master comprehensive narrative stringing Acts 1 through 4 together with seamless camera and HUD transitions.

---

## CLI Render Commands

Run all rendering commands from the project root using the dedicated Python virtual environment:

```bash
# 1. Fast Low-Quality Preview (480p15) — Recommended for iteration
.venv/bin/manim -ql linear_algebra/sprint_02_linear_mappings/scene_linear_mappings.py LinearMappingsScene

# 2. Render Individual Acts
.venv/bin/manim -ql linear_algebra/sprint_02_linear_mappings/scene_linear_mappings.py Act1SpatialPuzzle
.venv/bin/manim -ql linear_algebra/sprint_02_linear_mappings/scene_linear_mappings.py Act2DynamicDeformation
.venv/bin/manim -ql linear_algebra/sprint_02_linear_mappings/scene_linear_mappings.py Act3AlgebraicMorph
.venv/bin/manim -ql linear_algebra/sprint_02_linear_mappings/scene_linear_mappings.py Act4NonCommutativeStressTest

# 3. High-Definition Production Render (1080p60)
.venv/bin/manim -qh linear_algebra/sprint_02_linear_mappings/scene_linear_mappings.py LinearMappingsScene

# 4. Save Final Keyframe Image (PNG)
.venv/bin/manim -ql -s linear_algebra/sprint_02_linear_mappings/scene_linear_mappings.py LinearMappingsScene
```

---

## Mathematical Invariants & Verification Checklist

All code and companion texts conform to automated testing standards in `tests/test_sprint02_invariants.py` and `tests/test_adversarial_sprint2.py`:

- [x] **Linearity Axioms**: Additivity $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$ and scalar homogeneity $T(c\mathbf{u}) = cT(\mathbf{u})$ verified across 100,000 vectors.
- [x] **Origin Fixed Point**: $A\mathbf{0} = \mathbf{0}$ verified to machine precision ($< 10^{-15}$).
- [x] **Basis Invertibility**: $\det(A) = 4.0 \ne 0 \implies \text{Full Rank } 2$.
- [x] **Grid Parallelism Invariant**: Tangent vectors $A\mathbf{v}$ invariant across all parallel horizontal and vertical lines.
- [x] **Determinant Multiplicativity**: $\det(AB) = \det(A)\det(B)$ verified across 10,000 random matrix pairs.
- [x] **Geometric Non-Commutativity**: Spatial discrepancy $\|\Delta\| = \sqrt{13}$ confirmed for probe $[2, 3]^\top$ under $SR$ vs $RS$.
- [x] **Commutator Lie Algebra Invariant**: $\text{Tr}([S, R]) = 0.0$ confirmed across 252 $(k, \theta)$ configurations.
- [x] **ManimCE AST Cleanliness**: Zero deprecated symbols (`CONFIG`, `ShowCreation`, `TexMobject`, `TextMobject`); `make_hud_panel` z-index 10.

To execute the test suites:
```bash
# 1. Determinant and invariant pytest suite
.venv/bin/pytest tests/test_sprint02_invariants.py -v

# 2. Adversarial numerical stress suite
.venv/bin/python tests/test_adversarial_sprint2.py
```
