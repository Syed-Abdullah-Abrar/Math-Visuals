# Linear Algebra Curriculum

A visual, first-principles curriculum in Linear Algebra grounded in spatial transformations, geometric invariants, and coordinate geometry, modeled after Grant Sanderson (3Blue1Brown).

---

## Canonical References

- **Mathematics for Machine Learning (MML)**: Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong (Chapters 2–4, 10) — `/home/uni-kernel/brain/03_RESOURCES/Books/AI-ML/Mathematics/Mathematics for Machine Learning.pdf`
- **Introduction to Applied Linear Algebra (VMLS)**: Stephen Boyd, Lieven Vandenberghe (Chapters 1, 5, 6, 7, 8, 10, 12, 18) — `/home/uni-kernel/brain/03_RESOURCES/Books/Applied mathematics/Algebra/VMLS.pdf`
- **Immersive Linear Algebra (ILA)**: J. Ström, K. Åström, T. Akenine-Möller (Chapters 1–8) — `https://immersivemath.com/ila/tableofcontents.html`
- **Advanced Engineering Mathematics (Kreyszig, 10th Ed.)** & **Higher Engineering Mathematics (B.S. Grewal)**: Part B Chapters 7 & 8; Grewal Chapter 2 + GATE CS/DA PYQs — `/home/uni-kernel/brain/03_RESOURCES/Books/`

---

## Sprint Modules

| Sprint | Directory | Topic | Core Deliverables | Status |
|---|---|---|---|---|
| **01** | [`sprint_01_vectors_span/`](sprint_01_vectors_span/) | Vectors, Linear Combinations & Span | Brief, Storyboard, `scene_vectors_span.py`, Tweaks & Prompts, Tests (`test_sprint01_invariants.py`, `test_adversarial_sprint1.py`) | **Completed** |
| **02** | [`sprint_02_linear_mappings/`](sprint_02_linear_mappings/) | Linear Mappings & Matrix Multiplication as Composition | Brief, Storyboard, `scene_linear_mappings.py`, Tweaks & Prompts, Tests (`test_sprint02_invariants.py`, `test_adversarial_sprint2.py`) | **Completed** |
| **03** | [`sprint_03_column_null_space/`](sprint_03_column_null_space/) | Column Space, Null Space & Rank-Nullity Theorem (3D) | Brief, Storyboard, `scene_column_null_space.py` (`ThreeDScene`), Tweaks & Prompts, Tests (`test_sprint03_invariants.py`, `test_adversarial_sprint3.py`) | **Completed** |
| **04** | [`sprint_04_change_of_basis/`](sprint_04_change_of_basis/) | Change of Basis & Similarity Transformations ($P^{-1}AP$) | Brief, Storyboard, `scene_change_of_basis.py`, Tweaks & Prompts, Tests (`test_sprint04_invariants.py`, `test_adversarial_sprint4.py`) | **Completed** |
| **05** | `sprint_05_orthogonality_projections/` | Orthogonal Projections & Least Squares Normal Equations | Orthogonal complements, Gram-Schmidt, $P = A(A^\top A)^{-1}A^\top$ | Planned |
| **06** | `sprint_06_determinants_orientation/` | Determinants, Signed Volume & Orientation | Signed area/volume scaling, orientation reversal, multilinear alternating forms | Planned |
| **07** | `sprint_07_spectral_theory/` | Spectral Theory & Symmetric Matrices | Orthogonal eigenvectors $Q\Lambda Q^\top$, Rayleigh quotient, quadratic forms | Planned |
| **08** | `sprint_08_svd_pca/` | Singular Value Decomposition (SVD) & PCA | Rotation-stretch-rotation $U\Sigma V^\top$, Eckart-Young low-rank theorem, LoRA | Planned |

---

## Standard 4-Part Protocol

Every sprint directory adheres to a strict 4-part pedagogical and engineering structure:
1. `brief.md` — **Geometric Consensus Brief** (~350 words across 4 lenses: Geometric Action, Geometric Invariant, Algebraic-to-Visual Dictionary, and Dual Bridge to Frontier ML + GATE CS/DA).
2. `storyboard.md` — **4-Act Visual Storyboard** (Act 1 Concrete Spatial Puzzle, Act 2 Dynamic Deformation & Ghost Space, Act 3 Color-Coded Algebraic Morph, Act 4 Degenerate/Edge-Case Stress Test).
3. `scene_*.py` — **Production ManimCE v0.18+ Python Script** adhering to the 8 ManimCE rules (3b1b color palette, zero text overlap via `make_hud_panel`, `ValueTracker` updaters, exact `numpy.linalg` calculations).
4. `tweaks_prompts.md` — **Interactive Parameter Tweaks & Self-Test Prompts** with executable NumPy verification snippets.
5. `README.md` — **Master Companion Guide** integrating textbook cross-references, scene architecture, and CLI commands.

---

## Quickstart Commands

```bash
# Activate virtual environment from repository root
source .venv/bin/activate

# Run all 290 automated invariant and adversarial tests across Sprints 1–4
pytest tests/ -v

# Render Master Scenes (Fast 480p15 Preview)
manim -pql linear_algebra/sprint_01_vectors_span/scene_vectors_span.py VectorSpanScene
manim -pql linear_algebra/sprint_02_linear_mappings/scene_linear_mappings.py LinearMappingsScene
manim -pql linear_algebra/sprint_03_column_null_space/scene_column_null_space.py ColumnNullSpaceScene
manim -pql linear_algebra/sprint_04_change_of_basis/scene_change_of_basis.py ChangeOfBasisScene
```
