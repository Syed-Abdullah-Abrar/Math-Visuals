# Test Infrastructure: Linear Algebra & Mathematical Animation Suite

## 1. Overview & Architecture

The testing infrastructure for `math_cs` validates the mathematical correctness, numerical stability, rendering integrity, and pedagogical invariants of the linear algebra animation curriculum. Testing operates across two distinct but complementary layers:

1. **Automated Mathematical & Invariant Verification (`tests/`)**:
   - `tests/test_sprint01_invariants.py`: Requirement-driven, 4-tier opaque-box test suite implementing 61 deterministic unit tests with `unittest` and `numpy`.
   - `tests/test_adversarial_sprint1.py`: Adversarial numerical stress testing suite probing extreme condition numbers, randomized large-scale targets ($N=100,000$), convex hulls, and GATE/ML dual bridges.
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

## 3. Test Hierarchy: The 4-Tier Test Framework

The primary test suite in `tests/test_sprint01_invariants.py` adheres to a strict 4-tier testing hierarchy:

```
tests/test_sprint01_invariants.py
├── Tier 1: Feature Coverage (27 tests)
│   ├── TestTier1VectorAddition (6 tests)
│   ├── TestTier1ScalarScaling (5 tests)
│   ├── TestTier1LinearCombinations (5 tests)
│   ├── TestTier1BasisSpan2D (5 tests)
│   └── TestTier1OriginAnchorInvariant (5 tests)
├── Tier 2: Boundary & Corner Cases (25 tests)
│   ├── TestTier2CollinearCollapse (5 tests)
│   ├── TestTier2ZeroVectorSubspace (5 tests)
│   ├── TestTier2NegativeScalars (5 tests)
│   ├── TestTier2OrthogonalBasis (5 tests)
│   └── TestTier2LargeCoordinatesPrecision (5 tests)
├── Tier 3: Cross-Feature Combinations (5 tests)
│   └── TestTier3CrossFeatureCombinations (5 tests)
└── Tier 4: Real-World Scenarios (5 tests)
    └── TestTier4RealWorldScenarios (5 tests)
```

### Tier 1: Feature Coverage
Validates the fundamental vector space axioms and geometric definitions:
- **Vector Addition**: Commutativity ($u + v = v + u$), Associativity, Additive Identity ($\mathbf{0}$), Additive Inverse ($-\mathbf{v}$), Parallelogram Law ($||u+v||^2 + ||u-v||^2 = 2(||u||^2 + ||v||^2)$), and Triangle Inequality.
- **Scalar Scaling**: Distributivity over scalars and vectors, scalar associativity, identity, and norm homogeneity ($||c\mathbf{v}|| = |c|\cdot||\mathbf{v}||$).
- **Linear Combinations**: Exact solution verification ($2\mathbf{v}_1 + 1\mathbf{v}_2 = \mathbf{b}$), shortfall analysis, matrix-vector product duality, convex combination boundedness, and conic combinations.
- **2D Basis Span**: Full rank ($\text{rank}=2$), non-zero determinant ($\det=5.0$), linear independence ($\mathbf{c}=\mathbf{0}$ kernel), universal reachability across $\mathbb{R}^2$, and bijectivity.
- **Origin Anchor Invariant**: Subspace zero-closure ($\mathbf{0} \in \text{span}(S)$), orthogonal projection preservation, and norm minimizer properties.

### Tier 2: Boundary & Corner Cases
Stress-tests degenerate boundaries and pathological inputs:
- **Collinear Collapse**: Rank deficiency ($\text{rank}=1, \det=0$), 1D line constraint ($y = 0.5x$), unreachable target detection ($\mathbf{b}_{\text{unreachable}} = [1, 4]^\top$), Rouché-Capelli rank failure, and infinite solutions on subspace line.
- **0D Zero Vector Subspace**: Rank 0 matrix, trivial subspace $\text{span}(\{\mathbf{0}\}) = \{\mathbf{0}\}$, non-zero target unreachability, and dimension 0 verification.
- **Negative Scalars**: Direction inversion by $\pi$ radians, Quadrant III reachability, 4-quadrant Cartesian coverage, and continuity through the origin.
- **Orthogonal Basis**: Inner product decoupling ($\mathbf{v}_1 \cdot \mathbf{v}_2 = 0$), condition number $\kappa(A) = 1.0$, Pythagorean decoupling, and projection coordinate recovery.
- **Large Coordinates & Precision**: Astronomical coordinates ($10^{12}$), microscopic coordinates ($10^{-12}$), ill-conditioned almost-collinear bases, and high dynamic range superposition.

### Tier 3: Cross-Feature Combinations
Tests pairwise interactions across multiple features:
- Distributivity under orthogonality.
- Collinear collapse invariance under linear transformations.
- Origin anchor invariance under affine shifts.
- SO(2) rotation preserving span and 4-quadrant reachability.
- Rank invariance under arbitrary non-zero scalar rescaling.

### Tier 4: Real-World Scenarios
Grounds geometric concepts in applied AI and theoretical computer science:
- **Transformer Attention**: Proves attention output $\text{Attention}(Q, K, V) = \text{softmax}(QK^\top / \sqrt{d_k})V$ is strictly inside the convex hull of Value vectors, with partition of unity ($\sum \alpha_i = 1, \alpha_i \ge 0$), and verifies temperature limits ($T \to \infty$ centroid, $T \to 0$ argmax).
- **Word2Vec Linear Analogy**: Verifies semantic vector offset $\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}$ using parallelogram vector arithmetic and cosine similarity.
- **GATE CSE Linear Systems Consistency**: Comprehensive Rouché-Capelli classification (unique, infinite, inconsistent) and Rank-Nullity Theorem verification ($\text{nullity}(A) + \text{rank}(A) = n$).

---

## 4. Execution Commands

### Run Full E2E Invariant Test Suite
```bash
/home/uni-kernel/dev/math_cs/.venv/bin/python -m unittest -v tests/test_sprint01_invariants.py
```

### Run Adversarial Stress Tests
```bash
/home/uni-kernel/dev/math_cs/.venv/bin/python tests/test_adversarial_sprint1.py
```

### Render Manim Animations (Low-Resolution Preview)
```bash
/home/uni-kernel/dev/math_cs/.venv/bin/manim -ql linear_algebra/sprint_01_vectors_span/scene_vectors_span.py \
    Act1SpatialPuzzle Act2DynamicDeformation Act3AlgebraicMorph Act4DegenerateStressTest VectorSpanScene
```

### Render Master Scene (Production High-Resolution 1080p60)
```bash
/home/uni-kernel/dev/math_cs/.venv/bin/manim -qh linear_algebra/sprint_01_vectors_span/scene_vectors_span.py VectorSpanScene
```

---

## 5. Continuous Quality & Defect Prevention

- **Deterministic Seeds**: All randomized tests initialize explicit RNG instances via `np.random.default_rng(seed)`.
- **Floating-Point Stability**: Invertibility and orthogonality checks use adaptive tolerance thresholds (`rtol=1e-10`, `atol=1e-12`), avoiding raw equality comparisons on floating-point scalars.
- **HUD Layout Constraints**: Manim scenes strictly observe viewport boundaries ($X \in [-7.11, 7.11]$, $Y \in [-4.0, 4.0]$) and utilize container z-indexing (`z_index >= 10`) to eliminate text-geometry overlap.
