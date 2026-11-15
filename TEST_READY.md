# Test Suite Ready: Linear Algebra Visual Curriculum (Sprints 1–4)

**Status**: READY FOR MILESTONE IMPLEMENTATION  
**Date**: 2026-11-15  
**Test Suite Pass Rate**: 100% (273 Passed, 15 Skipped pending scene script creation, 0 Failed)  
**Execution Time**: 0.58s via `.venv/bin/pytest tests/`  

---

## 1. Executive Summary

Milestone 1 (Curriculum Test Infrastructure & Automated Invariant Suites) is complete. Comprehensive mathematical invariant suites and adversarial stress suites have been constructed and verified for Sprints 01 through 04.

All test suites strictly adhere to:
- **Progressive Testability**: Mathematical invariants execute deterministically against textbook ground truth (MML, VMLS, Kreyszig). Static AST scene integrity tests gracefully skip (`self.skipTest` / `[SKIPPED]`) until scene scripts are created in Milestones 2, 3, and 4.
- **Strict ManimCE v0.18+ Rules**: Verification of zero deprecated symbols (`CONFIG`, `ShowCreation`, `TexMobject`, `TextMobject`), `ThreeDScene` base inheritance for 3D camera sweeps, and `add_fixed_in_frame_mobjects` for pinned viewport HUD cards.
- **Zero Flakiness / Determinism**: Explicit RNG seeds across all randomized fixtures, relative precision error bounds, and avoidance of floating-point equality comparisons.

---

## 2. Test File Inventory

| Test File | Sprint / Domain | Tier Architecture | Items | Pass / Skip | Execution |
|---|---|---|---|---|---|
| `tests/test_sprint01_invariants.py` | Sprint 1: Vectors & Span | Tiers 1–4 Unittest | 61 | 61 Passed | pytest / unittest |
| `tests/test_adversarial_sprint1.py` | Sprint 1: Vectors & Span | Adversarial Stress | 8 | 8 Passed | pytest / python |
| `tests/test_sprint02_invariants.py` | Sprint 2: Linear Mappings | Tiers 1–5 Unittest | 65 | 60 Passed, 5 Skipped | pytest / unittest |
| `tests/test_adversarial_sprint2.py` | Sprint 2: Linear Mappings | Adversarial Stress | 8 | 8 Passed | pytest / python |
| `tests/test_sprint03_invariants.py` | Sprint 3: Column & Null Space | Tiers 1–5 Unittest | 65 | 60 Passed, 5 Skipped | pytest / unittest |
| `tests/test_adversarial_sprint3.py` | Sprint 3: Column & Null Space | Adversarial Stress | 8 | 8 Passed | pytest / python |
| `tests/test_sprint04_invariants.py` | Sprint 4: Change of Basis | Tiers 1–5 Unittest | 65 | 60 Passed, 5 Skipped | pytest / unittest |
| `tests/test_adversarial_sprint4.py` | Sprint 4: Change of Basis | Adversarial Stress | 8 | 8 Passed | pytest / python |
| **Total Test Suite** | **Sprints 1–4 Full Curriculum** | **Comprehensive** | **288** | **273 Passed, 15 Skipped** | **pytest tests/** |

---

## 3. Verified Mathematical Invariants by Sprint

### Sprint 2: Linear Mappings & Matrix Composition ($BA \neq AB$)
- **Linearity Superposition**: $T(c\mathbf{u} + d\mathbf{v}) = cT(\mathbf{u}) + dT(\mathbf{v})$ and $T(\mathbf{0}) = \mathbf{0}$.
- **Parallel Grid Preservation**: Direction vectors remain collinear; grid spacing remains equidistant; unit squares map to closed parallelograms.
- **Basis Tracking**: $A\hat{\imath} = \mathbf{a}_1, A\hat{\jmath} = \mathbf{a}_2$; arbitrary vector synthesis $A\mathbf{x} = x\mathbf{a}_1 + y\mathbf{a}_2$; unit square deformed area equals $|\det(A)| = 4.0$.
- **Composition & Associativity**: $(BA)\mathbf{x} = B(A\mathbf{x})$; $((CB)A)\mathbf{x} = (C(BA))\mathbf{x}$; $(AB)^{-1} = B^{-1}A^{-1}$.
- **Non-Commutativity**: $SR \neq RS$ for shear $S = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$ and $90^\circ$ rotation $R = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$; spatial divergence on test points ($[1, 0]^\top \to [1, 1]^\top$ vs $[0, 1]^\top$); Lie algebra commutator trace $\text{Tr}([S, R]) = 0$.
- **Dual Bridges**: ML multi-layer linear collapse $W_2 W_1 \mathbf{x}$, bilinear attention projections $QK^\top = X(W_Q W_K^\top)X^\top$, BatchNorm weight folding; GATE matrix chain multiplication dynamic programming dimensions.

### Sprint 3: Column Space, Null Space & Rank-Nullity Theorem ($2 + 1 = 3$)
- **Rank-2 $3 \times 3$ Architecture**: $A = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix}$, $\text{rank}(A) = 2$, $\det(A) = 0.0$, column sum dependency $\mathbf{a}_3 = \mathbf{a}_1 + \mathbf{a}_2$.
- **Column Space Plane**: $\mathcal{C}(A)$ 2D plane $x + y - z = 0$ with normal $\mathbf{n} = [1, 1, -1]^\top$ orthogonal to all columns.
- **Null Space Line**: $\mathcal{N}(A)$ 1D line $t[1, 1, -1]^\top$ collapsing to $\mathbf{0}$.
- **Rank-Nullity Theorem**: $\text{rank}(A) + \text{nullity}(A) = 2 + 1 = 3 = n$.
- **Affine Solution Subspace**: $\mathbf{x}_p = [1, 2, 0]^\top$ maps to target $\mathbf{b} = [1, 2, 3]^\top$; affine line $\mathbf{x}_p + t\mathbf{v}_{\text{null}}$ maps uniformly to $\mathbf{b}$; minimum-norm solution $\mathbf{x}_{LN} = [0, 1, 1]^\top$ satisfies $\mathbf{x}_{LN} \perp \mathcal{N}(A)$ with norm $\sqrt{2}$.
- **Dual Bridges**: ML overparameterized model null spaces with zero loss gradient; LoRA low-rank adaptation decomposition $\Delta W = BA$ with bounded rank $r \ll d$; GATE CSE 3D Rouché-Capelli classification.

### Sprint 4: Change of Basis & Similarity Transformations ($A = PDP^{-1}$)
- **Change of Basis Matrix**: $P = [\mathbf{v}_1 \mid \mathbf{v}_2] = \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}$, $\det(P) = 2.0$, $\text{rank}(P) = 2$; coordinate change $[\mathbf{x}]_B = P^{-1}\mathbf{x}$ and recovery $\mathbf{x} = P[\mathbf{x}]_B$.
- **Similarity Transformation**: $A = PDP^{-1}$ for symmetric $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ with $D = \text{diag}(3, 1)$; decoupled axis stretching without cross-axis shearing.
- **3-Step Sequence**: $[2, 0]^\top \xrightarrow{P^{-1}} [1, -1]_B \xrightarrow{D} [3, -1]_B \xrightarrow{P} [4, 2]^\top == A[2, 0]^\top$.
- **Spectral Invariants**: Trace $\text{Tr}(P^{-1}AP) = \text{Tr}(A) = 4.0$; Determinant $\det(P^{-1}AP) = \det(A) = 3.0$; Characteristic polynomial $\det(\lambda I - P^{-1}AP) = \det(\lambda I - A)$; Fast exponentiation $A^k = PD^kP^{-1}$.
- **Defective Matrix**: $M_{\text{def}} = \begin{bmatrix} 2 & 1 \\ 0 & 2 \end{bmatrix}$ having algebraic multiplicity 2 but geometric multiplicity 1, proving non-diagonalizability.
- **Dual Bridges**: ML PCA covariance diagonalization $\Sigma = Q\Lambda Q^\top$ decorrelating features; SVD Gram matrix relation $X^\top X = V\Sigma^2 V^\top$; GATE Cayley-Hamilton theorem $p(A) = A^2 - 4A + 3I = 0$ and $A^{-1} = \frac{1}{3}(4I - A)$.

---

## 4. How to Run the Test Suite

```bash
# 1. Full Pytest Suite (All Sprints)
.venv/bin/pytest tests/

# 2. Verbose Pytest Run
.venv/bin/pytest -v tests/

# 3. Individual Sprint Invariant Suites
.venv/bin/pytest tests/test_sprint02_invariants.py
.venv/bin/pytest tests/test_sprint03_invariants.py
.venv/bin/pytest tests/test_sprint04_invariants.py

# 4. Standalone Adversarial Stress Suites
.venv/bin/python tests/test_adversarial_sprint1.py
.venv/bin/python tests/test_adversarial_sprint2.py
.venv/bin/python tests/test_adversarial_sprint3.py
.venv/bin/python tests/test_adversarial_sprint4.py
```

---

## 5. Verification Sign-Off

The test infrastructure is fully operational and verified. When implementing Milestone 2, Milestone 3, and Milestone 4 deliverables, all code changes will immediately be validated by these suites.
