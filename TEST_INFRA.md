# Test Infrastructure: Linear Algebra & Mathematical Animation Suite

## 1. Overview & Architecture

The testing infrastructure for `math_cs` validates the mathematical correctness, numerical stability, rendering integrity, and pedagogical invariants across the entire 4-sprint linear algebra animation curriculum. Testing operates across two distinct, complementary layers:

1. **Automated Mathematical & Invariant Verification (`tests/`)**:
   - **Sprint 1**: `tests/test_sprint01_invariants.py` (61 tests) and `tests/test_adversarial_sprint1.py` (8 stress tests).
   - **Sprint 2**: `tests/test_sprint02_invariants.py` (65 tests) and `tests/test_adversarial_sprint2.py` (8 stress tests).
   - **Sprint 3**: `tests/test_sprint03_invariants.py` (65 tests) and `tests/test_adversarial_sprint3.py` (8 stress tests).
   - **Sprint 4**: `tests/test_sprint04_invariants.py` (65 tests) and `tests/test_adversarial_sprint4.py` (8 stress tests).
   - Total Suite: **288 test items** running in `<0.6s` with zero video rendering overhead.
2. **Visual & Rendering Integration (`media/videos/`)**:
   - Manim Community Edition (ManimCE v0.21.0) CLI verification ensuring zero runtime exceptions, valid LaTeX rendering via `dvisvgm`, and zero layout/HUD text occlusions.

---

## 2. Environment & Dependencies

| Component | Specification | Purpose |
|---|---|---|
| **Python** | 3.12.13 | Core runtime environment (`.venv`) |
| **NumPy** | 2.5.3 | High-performance vectorized numerical linear algebra |
| **SciPy** | 1.18.1 | Convex hull calculations, spatial analysis, SVD factorizations |
| **ManimCE** | 0.21.0 | 3b1b standard geometric animation engine |
| **FFmpeg** | 8.0.1 | Video encoding (H.264 / AAC) at 480p15 and 1080p60 |
| **TeX Live** | 2025 (`dvisvgm 3.6`) | Mathematical formula typesetting |

---

## 3. Curriculum Test Hierarchy & Invariant Framework

### 3.1 Suite Inventory & Mapping
```
tests/
├── test_sprint01_invariants.py     (61 tests: Vectors, Linear Combinations & Span)
├── test_adversarial_sprint1.py     (8 stress tests: Condition numbers, Convex hull, Attention)
├── test_sprint02_invariants.py     (65 tests: Linear Mappings, Basis Tracking, Composition)
├── test_adversarial_sprint2.py     (8 stress tests: 100k vectors, Commutators, Det product)
├── test_sprint03_invariants.py     (65 tests: Column Space, Null Space, Rank-Nullity, Affine line)
├── test_adversarial_sprint3.py     (8 stress tests: 100k points, 3D Direct sum, Least-norm)
├── test_sprint04_invariants.py     (65 tests: Change of Basis, Similarity Conjugation, Defective)
└── test_adversarial_sprint4.py     (8 stress tests: 100k vectors, Powers up to A^100, PCA)
```

---

### 3.2 Sprint 1: Vectors, Linear Combinations & Span
- **Tier 1 (Feature Coverage, 27 tests)**: Vector addition axioms (commutativity, associativity, identity, inverse, parallelogram law, triangle inequality); scalar scaling homogeneity and distributivity; linear combinations and convex bounds; 2D basis span and bijectivity; origin anchor invariant.
- **Tier 2 (Boundaries, 25 tests)**: Collinear rank-1 collapse ($\det=0$); 0D zero-vector trivial subspace; negative scalar direction inversion; orthogonal basis decoupling ($\kappa=1.0$); large coordinates ($10^{12}$) and microscopic coordinates ($10^{-12}$).
- **Tier 3 (Cross-Feature, 5 tests)**: Orthogonality with distributivity; SO(2) rotation preserving span; rank invariance under non-zero scaling; collinear collapse under linear maps; affine shift invariance.
- **Tier 4 (Real-World, 5 tests)**: Transformer attention convex combinations and temperature limits ($T \to 0, T \to \infty$); Word2Vec parallelogram semantic offsets ($\text{king} - \text{man} + \text{woman} \approx \text{queen}$); GATE CSE Rouché-Capelli classification and Rank-Nullity theorem.
- **Adversarial Suite (8 tests)**: 100k target bijectivity, collinear reachability boundaries, stable softmax attention convex bounds, intuition prompts 1–4.

---

### 3.3 Sprint 2: Linear Mappings & Matrix Multiplication as Composition
- **Tier 1 (Feature Coverage, 25 tests)**:
  - Linearity definition: $T(c\mathbf{u} + d\mathbf{v}) = cT(\mathbf{u}) + dT(\mathbf{v})$ and $T(\mathbf{0}) = \mathbf{0}$.
  - Parallel grid preservation: straight lines map to straight lines, parallel lines stay parallel, equidistant grid spacing is preserved, unit square maps to closed parallelogram.
  - Basis tracking: $\hat{\imath} \to \mathbf{a}_1, \hat{\jmath} \to \mathbf{a}_2$; arbitrary vector synthesis $\mathbf{x} = x\mathbf{a}_1 + y\mathbf{a}_2$; unit square area equals $|\det(A)| = 4.0$; orientation preservation ($\det(A) > 0$).
  - Sequential composition: $(BA)\mathbf{x} = B(A\mathbf{x})$; associativity $((CB)A)\mathbf{x} = (C(BA))\mathbf{x}$; distributivity $C(A+B) = CA + CB$; identity neutrality $AI = IA = A$.
  - Non-commutativity: $SR \neq RS$ for shear $S = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$ and $90^\circ$ rotation $R = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$; point destination divergence ($\mathbf{x} = [1, 0]^\top \to [1, 1]^\top$ under $SR$ vs $[0, 1]^\top$ under $RS$); Lie algebra commutator trace $\text{Tr}([S, R]) = 0$; shoe-and-sock inverse rule $(SR)^{-1} = R^{-1}S^{-1} \neq S^{-1}R^{-1}$.
- **Tier 2 (Boundaries, 25 tests)**:
  - Singular rank-1 matrix collapse to 1D line $y = 0.5x$, non-trivial kernel $[-2, 1]^\top$, LinAlgError on inversion, unreachable target detection.
  - Zero matrix rank 0 annihilation $0\mathbf{x} = \mathbf{0}$, full space kernel $\mathcal{N}(0) = \mathbb{R}^2$, composition nilpotence.
  - Identity matrix isometry $||\mathbf{x}|| = ||I\mathbf{x}||$, $\det=1$, $\text{rank}=2$, all vectors as eigenvectors ($\lambda=1$).
  - $180^\circ$ rotation inversion $R(\pi) = -I$, reflection parity reversal ($\det=-1$), 4-quadrant Cartesian coverage, involution $M^2 = I$.
  - Ill-conditioned mappings with condition number $\kappa(A) \ge 10^8$, almost-collinear columns, inversion sensitivity, singular value ratio $\sigma_1/\sigma_2 \ge 10^8$.
- **Tier 3 (Cross-Feature, 5 tests)**: Determinant multiplication rule $\det(AB) = \det(A)\det(B)$; associativity with scalar scaling; inverse composition rule; trace cyclic invariance $\text{Tr}(ABC) = \text{Tr}(BCA) = \text{Tr}(CAB)$; Sylvester's rank composition inequality.
- **Tier 4 (Real-World, 5 tests)**: ML Feedforward multi-layer linear collapse $W_2 W_1 \mathbf{x} = W_{\text{collapsed}} \mathbf{x}$ (necessity of activations); Attention query-key composition $Q K^\top = X(W_Q W_K^\top)X^\top$; BatchNorm inference folding $W_{\text{fused}} = \text{diag}(\gamma/\sigma)W$; GATE Matrix chain multiplication dynamic programming cost optimization ($7,500$ vs $75,000$ ops); GATE Rouché-Capelli consistency.
- **Tier 5 (AST & Syntax, 5 tests)**: Static AST verification of `scene_linear_mappings.py` (zero deprecated symbols `CONFIG`, `ShowCreation`, `TexMobject`, `TextMobject`, `LinearMappingsScene` class definition, `make_hud_panel` usage, security check against `eval`/`exec`). Gracefully skips if scene file is pending Milestone 2 implementation.
- **Adversarial Suite (8 tests)**: 100k vector composition linearity stress; condition number sweep up to $\kappa = 10^{14}$; commutator $[S, R]$ landscape across 252 $(k, \theta)$ points; determinant multiplicativity over 10,000 pairs; static AST check; prompts 1–4 verification.

---

### 3.4 Sprint 3: Column Space, Null Space & Rank-Nullity Theorem
- **Tier 1 (Feature Coverage, 25 tests)**:
  - Canonical Rank-2 $3 \times 3$ architecture: $A = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix}$, $\text{rank}(A) = 2$, $\det(A) = 0.0$, column dependency $\mathbf{a}_3 = \mathbf{a}_1 + \mathbf{a}_2$, SVD spectrum with two non-zero singular values.
  - Column space plane: $\mathcal{C}(A)$ plane equation $x + y - z = 0$ with normal $\mathbf{n} = [1, 1, -1]^\top$; orthogonality to all columns; origin containment; target $\mathbf{b} = [1, 2, 3]^\top \in \mathcal{C}(A)$.
  - Null space line: $\mathcal{N}(A)$ 1D line $t[1, 1, -1]^\top$; annihilation $A\mathbf{v}_{\text{null}} = \mathbf{0}$; origin containment; kernel uniqueness up to scalar.
  - Rank-Nullity Theorem: $\text{rank}(A) + \text{nullity}(A) = 2 + 1 = 3 = n$; RREF 2 pivots and 1 free variable; dimension conservation across 20 randomized matrices.
  - Affine solution line: Particular solution $\mathbf{x}_p = [1, 2, 0]^\top$ maps to $\mathbf{b} = [1, 2, 3]^\top$; affine line $\mathbf{x}_p + t\mathbf{v}_{\text{null}}$ maps uniformly to $\mathbf{b}$; affine line strictly parallel to null line; minimum-norm solution $\mathbf{x}_{LN} = [0, 1, 1]^\top$ has norm $\sqrt{2}$ and is orthogonal to $\mathbf{v}_{\text{null}}$ ($\mathbf{x}_{LN} \perp \mathcal{N}(A)$).
- **Tier 2 (Boundaries, 25 tests)**:
  - Inconsistent system $\mathbf{b}_{\text{unreach}} = [1, 2, 0]^\top$ detected via $\text{rank}([A \mid \mathbf{b}]) = 3 > 2$; geometric distance to plane $\sqrt{3}$ equals least-squares residual; normal equations consistency.
  - Rank-1 collapse ($3 \times 3$ matrix with rank 1, nullity 2 / 2D null plane).
  - Full-rank-3 trivial null space ($\text{nullity}=0$, universal reachability, unique solutions).
  - Zero matrix 3D rank 0 collapse ($\text{nullity}=3$, solution space $\mathbb{R}^3$).
  - Large coordinate precision ($10^{10}$) and microscopic coordinates ($10^{-10}$); near-rank-deficiency thresholding ($\sigma_3 = 10^{-14}$ vs $\sigma_3 = 0$).
- **Tier 3 (Cross-Feature, 5 tests)**: Orthogonal complement $\mathcal{N}(A) = (\mathcal{C}(A^\top))^\perp$; projection matrix $P_C = A(A^\top A)^+ A^\top$ symmetry ($P^\top = P$) and idempotence ($P^2 = P$); direct sum decomposition $\mathbb{R}^3 = \mathcal{C}(A^\top) \oplus \mathcal{N}(A)$; affine projection invariance onto row space collapsing to $\mathbf{x}_{LN}$; Moore-Penrose pseudoinverse 4 Penrose axioms.
- **Tier 4 (Real-World, 5 tests)**: ML Overparameterized Jacobian null space (dimension $\ge P - N$, loss flatness along null directions); LoRA low-rank adaptation decomposition $\Delta W = BA$ with rank $r \ll d$; Linear autoencoder $3 \to 2 \to 3$ bottleneck; GATE CSE 3D Rouché-Capelli classification; Abstract vector space dimension theorem.
- **Tier 5 (AST & Syntax, 5 tests)**: Static AST verification of `scene_column_null_space.py` (`ThreeDScene` inheritance, `add_fixed_in_frame_mobjects` HUD pinning, zero deprecated symbols, `ColumnNullSpaceScene` class definition, `make_hud_panel` usage). Gracefully skips if scene file is pending Milestone 3 implementation.
- **Adversarial Suite (8 tests)**: 100,000 random 3D points decomposed into $\mathcal{C}(A^\top) \oplus \mathcal{N}(A)$ with zero relative cross-correlation; projection matrix idempotence over 100,000 points; Rouché-Capelli classification across 10,000 synthesized systems; affine distance minimizer verification over 10,001 points; AST check; prompts 1–4 verification.

---

### 3.5 Sprint 4: Change of Basis & Similarity Transformations
- **Tier 1 (Feature Coverage, 25 tests)**:
  - Basis matrix $P = [\mathbf{v}_1 \mid \mathbf{v}_2] = \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}$, $\det(P) = 2.0$, $\text{rank}(P) = 2$; coordinate mapping $[\mathbf{x}]_B = P^{-1}\mathbf{x}$ and reconstruction $\mathbf{x} = P[\mathbf{x}]_B$; canonical basis representation $[1, 0]^\top, [0, 1]^\top$; round-trip bijectivity.
  - Coordinate transition calculus: standard basis mapping to columns of $P^{-1}$; barycentric reconstruction; coordinate map as linear isomorphism; Gram matrix inner product $\mathbf{u}^\top \mathbf{v} = [\mathbf{u}]_B^\top (P^\top P) [\mathbf{v}]_B$.
  - Similarity transformation conjugation: $A = PDP^{-1}$ for symmetric $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$; extraction $D = P^{-1}AP = \text{diag}(3, 1)$; eigenvector action $A\mathbf{v}_1 = 3\mathbf{v}_1, A\mathbf{v}_2 = 1\mathbf{v}_2$; identity similarity $I^{-1}AI = A$; transitivity of similarity.
  - Diagonal scaling action: decoupled coordinate scaling without cross-axis coupling; first eigenaxis threefold stretch; second eigenaxis invariance; diagonal area scaling $\det(D) = 3.0$; diagonal powers $D^k = \text{diag}(3^k, 1^k)$.
  - 3-step geometric sequence: Canonical point $[2, 0]^\top \xrightarrow{P^{-1}} [1, -1]_B \xrightarrow{D} [3, -1]_B \xrightarrow{P} [4, 2]^\top == A[2, 0]^\top$; batch verification over 100 random vectors.
- **Tier 2 (Boundaries, 25 tests)**:
  - Defective matrix $M_{\text{def}} = \begin{bmatrix} 2 & 1 \\ 0 & 2 \end{bmatrix}$: algebraic multiplicity 2 vs geometric multiplicity 1, single 1D horizontal eigenspace along $[1, 0]^\top$, non-diagonalizability failure, Jordan block power growth $J^k = \begin{bmatrix} 2^k & k 2^{k-1} \\ 0 & 2^k \end{bmatrix}$.
  - Repeated eigenvalues with full eigenspace ($A = cI$, algebraic mult 2 = geometric mult 2, isotropic scaling, similarity invariance $P^{-1}(cI)P = cI$).
  - Symmetric matrix orthogonal eigenvectors ($\mathbf{v}_1 \cdot \mathbf{v}_2 = 0$, orthonormal $Q^\top Q = I$, spectral decomposition $A = \lambda_1 \mathbf{q}_1 \mathbf{q}_1^\top + \lambda_2 \mathbf{q}_2 \mathbf{q}_2^\top$, strictly real eigenvalues).
  - Zero eigenvalue / singular matrix similarity ($A_{\text{sing}} = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}$, $\lambda = 2, 0$, diagonal form $\text{diag}(2, 0)$, null space alignment with zero-eigenvalue eigenvector).
  - Negative eigenvalues / hyperbolic saddle reflection ($D = \text{diag}(2, -1)$, $\det = -2$, orientation reversal, parity flip in odd powers).
- **Tier 3 (Cross-Feature, 5 tests)**: Similarity invariants across 50 random change-of-basis matrices: determinant invariance $\det(P^{-1}AP) = \det(A) = 3.0$; trace invariance $\text{Tr}(P^{-1}AP) = \text{Tr}(A) = 4.0$; characteristic polynomial invariance $\det(\lambda I - P^{-1}AP) = \det(\lambda I - A)$; eigenvalue spectrum invariance; matrix powers $A^k = PD^kP^{-1}$.
- **Tier 4 (Real-World, 5 tests)**: ML PCA covariance diagonalization $\Sigma = Q\Lambda Q^\top$ decorrelating features; SVD relation to Gram matrix $X^\top X = V \Sigma^2 V^\top$; Graph Laplacian $L = D - A = U\Lambda U^\top$ spectral decoupling; GATE Cayley-Hamilton theorem $p(A) = A^2 - 4A + 3I = 0$ and inverse formula $A^{-1} = \frac{1}{3}(4I - A)$; Complete diagonalizability criteria.
- **Tier 5 (AST & Syntax, 5 tests)**: Static AST verification of `scene_change_of_basis.py` (zero deprecated symbols, `ChangeOfBasisScene` class definition, `make_hud_panel` usage, eigenvector configuration constants, zero `eval`/`exec`). Gracefully skips if scene file is pending Milestone 4 implementation.
- **Adversarial Suite (8 tests)**: 100,000 random vectors under similarity cycle with relative precision loss $< 10^{-14}$; matrix powers up to $A^{100}$ verifying asymptotic eigenspace domination; Jordan block power growth; ill-conditioned similarity matrices $\kappa(P) \to 10^8$; AST check; prompts 1–4 verification.

---

## 4. Execution Commands

### Run Complete Curriculum Test Suite (All Sprints)
```bash
/home/uni-kernel/dev/math_cs/.venv/bin/pytest tests/
```

### Run Specific Sprint Invariant Suites (Unittest / Pytest)
```bash
# Sprint 1
/home/uni-kernel/dev/math_cs/.venv/bin/pytest tests/test_sprint01_invariants.py

# Sprint 2
/home/uni-kernel/dev/math_cs/.venv/bin/pytest tests/test_sprint02_invariants.py

# Sprint 3
/home/uni-kernel/dev/math_cs/.venv/bin/pytest tests/test_sprint03_invariants.py

# Sprint 4
/home/uni-kernel/dev/math_cs/.venv/bin/pytest tests/test_sprint04_invariants.py
```

### Run Adversarial Stress Test Suites (Standalone Python Execution)
```bash
/home/uni-kernel/dev/math_cs/.venv/bin/python tests/test_adversarial_sprint1.py
/home/uni-kernel/dev/math_cs/.venv/bin/python tests/test_adversarial_sprint2.py
/home/uni-kernel/dev/math_cs/.venv/bin/python tests/test_adversarial_sprint3.py
/home/uni-kernel/dev/math_cs/.venv/bin/python tests/test_adversarial_sprint4.py
```

### Fast Scene Render Verification (`manim -ql`)
```bash
# Sprint 1
/home/uni-kernel/dev/math_cs/.venv/bin/manim -ql linear_algebra/sprint_01_vectors_span/scene_vectors_span.py VectorSpanScene

# Sprint 2 (Milestone 2)
/home/uni-kernel/dev/math_cs/.venv/bin/manim -ql linear_algebra/sprint_02_linear_mappings/scene_linear_mappings.py LinearMappingsScene

# Sprint 3 (Milestone 3)
/home/uni-kernel/dev/math_cs/.venv/bin/manim -ql linear_algebra/sprint_03_column_null_space/scene_column_null_space.py ColumnNullSpaceScene

# Sprint 4 (Milestone 4)
/home/uni-kernel/dev/math_cs/.venv/bin/manim -ql linear_algebra/sprint_04_change_of_basis/scene_change_of_basis.py ChangeOfBasisScene
```

---

## 5. Continuous Quality & Defect Prevention

- **Deterministic Seeds**: All randomized tests initialize explicit RNG instances via `np.random.default_rng(seed)` (seeds 201–208 for Sprint 2, 301–306 for Sprint 3, 401–405 for Sprint 4).
- **Floating-Point Stability**: Relative error metrics are utilized for astronomical/microscopic coordinate regimes ($10^{12}$ to $10^{-12}$), with adaptive tolerance thresholds (`rtol=1e-10`, `atol=1e-12`).
- **Progressive Testability**: Static AST checks gracefully skip with `self.skipTest` if scene scripts are not yet implemented, allowing the full mathematical test infrastructure to run independently during milestone development.
- **HUD Pinning Invariant**: For 3D scenes (Sprint 3), `add_fixed_in_frame_mobjects` is strictly enforced via AST traversal to prevent 2D typography from being occluded or rotated by 3D camera sweeps.
