# Linear Algebra Curriculum

A visual, first-principles curriculum in Linear Algebra grounded in spatial transformations and coordinate geometry.

---

## Canonical References

- **Mathematics for Machine Learning (MML)**: Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong (Chapters 2–4).
- **Introduction to Applied Linear Algebra (VMLS)**: Stephen Boyd, Lieven Vandenberghe (Chapters 1, 5, 7, 8, 12, 18).
- **Immersive Linear Algebra (ILA)**: J. Ström, K. Åström, T. Akenine-Möller (Chapters 1–8).

---

## Sprint Modules

| Sprint | Directory | Topic | Deliverables | Status |
|---|---|---|---|---|
| **01** | [`sprint_01_vectors_span/`](sprint_01_vectors_span/) | Vectors, Linear Combinations & Span | Brief, Storyboard, ManimCE script (`scene_vectors_span.py`), Tweaks & Prompts | **Active** |
| **02** | `sprint_02_linear_transformations/` | Linear Transformations & Matrix Action | Coordinate warping, basis vectors as column destinations | Planned |
| **03** | `sprint_03_determinants/` | Determinants & Orientation Scaling | Signed area/volume scaling, orientation reversal, null-space collapse | Planned |
| **04** | `sprint_04_linear_systems/` | Systems of Linear Equations | Hyperplane intersections, column span consistency, Rouché-Capelli | Planned |
| **05** | `sprint_05_subspaces_ftla/` | Vector Subspaces & FTLA | 4 fundamental subspaces, rank-nullity theorem | Planned |
| **06** | `sprint_06_orthogonality/` | Orthogonality & Projections | Orthogonal complements, Gram-Schmidt, least squares | Planned |
| **07** | `sprint_07_spectral_theory/` | Spectral Theory | Invariant axes, eigenvalues, diagonalization | Planned |
| **08** | `sprint_08_svd_pca/` | SVD & PCA | Rotation-stretch-rotation decomposition, low-rank approximation | Planned |

---

## Standard 4-Part Protocol

Every sprint directory adheres to a 4-part structure:
1. `brief.md` — Geometric Consensus Brief (~350 words: Geometric Action, Geometric Invariant, Algebraic-to-Visual Dictionary, Dual Bridge).
2. `storyboard.md` — 4-Act Visual Storyboard (Spatial puzzle, dynamic deformation, algebraic morph, degenerate stress test).
3. `scene_*.py` — Production ManimCE Python animation script adhering to the 8 design rules.
4. `tweaks_prompts.md` — Interactive parameter modifications and conceptual intuition stress-test prompts.
5. `README.md` — Master companion guide integrating all 4 parts.

---

## Quickstart

```bash
# Render Sprint 1 Master Scene
manim -ql sprint_01_vectors_span/scene_vectors_span.py VectorSpanScene

# Test numerical invariants
python -m pytest ../tests/test_sprint01_invariants.py
```
