# Part 4: Interactive Parameter Tweaks & Intuition Stress-Test Prompts

## 1. Interactive Parameter Tweaks Guide

The ManimCE animation script (`scene_change_of_basis.py`) defines key transformation matrices, eigenvector bases, coordinate vectors, and visual styling parameters as module-level constants. Students and researchers can modify these values to explore alternate symmetric operators, oblique non-orthogonal bases, distinct eigenvalue spectra, and defective shear matrices.

### Parameter Configuration Matrix

| Parameter Name | Default Value | Recommended Test Values | Pedagogical Objective |
|---|---|---|---|
| `A_MAT` / `A_MATRIX` | `[[2.0, 1.0], [1.0, 2.0]]` | `[[3.0, 1.0], [1.0, 3.0]]`, `[[4.0, 2.0], [2.0, 1.0]]` | Observe change in eigenvalue spread and principal dilation axes for symmetric operators. |
| `V1` / `EIGEN_V1` | `np.array([1.0, 1.0])` | `[1.0, 2.0]`, `[2.0, 1.0]` | Alter primary eigen-axis orientation ($\theta = 45^\circ \to \theta \approx 63.4^\circ$). |
| `V2` / `EIGEN_V2` | `np.array([-1.0, 1.0])` | `[-2.0, 1.0]`, `[-1.0, 2.0]` | Verify that orthogonal eigenvectors satisfy $\mathbf{v}_1 \cdot \mathbf{v}_2 = 0$. |
| `D_MAT` / `D_MATRIX` | `np.diag([3.0, 1.0])` | `np.diag([4.0, 2.0])`, `np.diag([2.0, -1.0])` | Explore isotropic scaling, hyperbolic saddle stretching, and reflection ($\lambda < 0$). |
| `TEST_X` | `np.array([2.0, 0.0])` | `[1.0, 1.0]`, `[0.0, 2.0]`, `[3.0, -1.0]` | Track 3-step sequence $P^{-1} \to D \to P$ on vectors in different quadrants and on eigen-axes. |
| `M_DEF` | `[[2.0, 1.0], [0.0, 2.0]]` | `[[3.0, 1.0], [0.0, 3.0]]`, `[[1.0, 2.0], [0.0, 1.0]]` | Probe defective Jordan blocks; observe geometric multiplicity deficiency $\text{nullity}(M - \lambda I) = 1 < 2$. |
| `EIGEN_GRID_COLOR`| `"#E0AF68"` (Amber) | `"#58C4DD"` (Blue), `"#9A72AC"` (Purple) | Custom visual contrast for oblique eigen-grid coordinate scaffolding. |

### How to Modify and Re-Render

1. **Edit constants in `scene_change_of_basis.py`**:
   ```python
   # Configuration Constants
   A_MATRIX = np.array([[2.0, 1.0], [1.0, 2.0]])
   V1 = np.array([1.0, 1.0])
   V2 = np.array([-1.0, 1.0])
   D_MATRIX = np.diag([3.0, 1.0])
   TEST_X = np.array([2.0, 0.0])
   M_DEF = np.array([[2.0, 1.0], [0.0, 2.0]])
   ```
2. **Execute fast low-quality preview render**:
   ```bash
   .venv/bin/manim -ql linear_algebra/sprint_04_change_of_basis/scene_change_of_basis.py ChangeOfBasisScene
   ```
3. **Execute single act preview**:
   ```bash
   .venv/bin/manim -ql linear_algebra/sprint_04_change_of_basis/scene_change_of_basis.py Act1SpatialPuzzle
   .venv/bin/manim -ql linear_algebra/sprint_04_change_of_basis/scene_change_of_basis.py Act2DynamicDeformation
   .venv/bin/manim -ql linear_algebra/sprint_04_change_of_basis/scene_change_of_basis.py Act3AlgebraicMorph
   .venv/bin/manim -ql linear_algebra/sprint_04_change_of_basis/scene_change_of_basis.py Act4DegenerateStressTest
   ```

---

## 2. Intuition Stress-Test Prompts

### Prompt 1: Eigen-Grid Invariant Directions & Zero Rotation vs Generic Vector Rotation
- **Question**:  
  Why does an eigenvector $\mathbf{v}$ of matrix $A$ experience zero spatial rotation ($A\mathbf{v} \parallel \mathbf{v}$), while almost every generic vector $\mathbf{x} \in \mathbb{R}^2$ undergoes both stretching and angular rotation? How does the 2D cross product mathematically detect this rotational divergence?
- **Intuitive & Mathematical Proof**:  
  By definition, an eigenvector satisfies $A\mathbf{v} = \lambda \mathbf{v}$ for scalar $\lambda \in \mathbb{R}$. Geometrically, the matrix $A$ acts on the 1D subspace $\text{span}(\mathbf{v})$ as pure scalar dilation by $\lambda$, keeping all transformed vectors strictly collinear with the original line of action.  
  
  In $\mathbb{R}^2$, the angular rotation between any non-zero vector $\mathbf{u}$ and its image $A\mathbf{u}$ is captured by the 2D cross product:
  $$\mathbf{u} \times A\mathbf{u} = u_1 (A\mathbf{u})_2 - u_2 (A\mathbf{u})_1 = \|\mathbf{u}\| \|A\mathbf{u}\| \sin(\Delta \theta)$$
  - For eigenvectors $\mathbf{v}_1 = [1, 1]^\top$ and $\mathbf{v}_2 = [-1, 1]^\top$:
    $$A\mathbf{v}_1 = 3\mathbf{v}_1 \implies \mathbf{v}_1 \times (3\mathbf{v}_1) = 3(\mathbf{v}_1 \times \mathbf{v}_1) = 0 \implies \Delta \theta = 0^\circ$$
    $$A\mathbf{v}_2 = 1\mathbf{v}_2 \implies \mathbf{v}_2 \times (1\mathbf{v}_2) = 1(\mathbf{v}_2 \times \mathbf{v}_2) = 0 \implies \Delta \theta = 0^\circ$$
  - For a generic non-eigenvector $\mathbf{x} = [1, 0]^\top$:
    $$A\mathbf{x} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}\begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$$
    $$\mathbf{x} \times A\mathbf{x} = 1(1) - 0(2) = 1 \ne 0 \implies \sin(\Delta \theta) = \frac{1}{\sqrt{5}} \ne 0 \implies \Delta \theta \approx 26.57^\circ$$
  Therefore, the eigen-grid represents the unique orthogonal coordinate frame where spatial shearing is completely eliminated.
- **NumPy Verification**:
  ```python
  import numpy as np

  A = np.array([[2.0, 1.0], [1.0, 2.0]])
  v1 = np.array([1.0, 1.0])
  v2 = np.array([-1.0, 1.0])

  # 1. Invariant directions: cross product is strictly 0
  out1 = A @ v1
  out2 = A @ v2
  cross1 = v1[0] * out1[1] - v1[1] * out1[0]
  cross2 = v2[0] * out2[1] - v2[1] * out2[0]
  assert np.isclose(cross1, 0.0, atol=1e-12)
  assert np.isclose(cross2, 0.0, atol=1e-12)
  assert np.allclose(out1, 3.0 * v1, atol=1e-12)
  assert np.allclose(out2, 1.0 * v2, atol=1e-12)

  # 2. Generic vector rotates: cross product is non-zero
  v_generic = np.array([1.0, 0.0])
  out_gen = A @ v_generic
  cross_gen = v_generic[0] * out_gen[1] - v_generic[1] * out_gen[0]
  assert not np.isclose(cross_gen, 0.0)
  assert np.isclose(cross_gen, 1.0)
  print("Prompt 1 Verification PASSED: Zero rotation on eigen-axes confirmed.")
  ```
- **Textbook Grounding**: MML §2.7.2 (Basis Change); VMLS §6.1 & §10.1; Kreyszig §8.1 (Eigenvalue Problems).

---

### Prompt 2: The 3-Step Geometric Similarity Cycle ($P^{-1} \to D \to P$)
- **Question**:  
  How does the similarity factorization $A = P D P^{-1}$ decompose a coupled 2D spatial transformation into three elementary geometric operations? Trace the coordinates of $\mathbf{x} = [2, 0]^\top$ through each stage of the cycle.
- **Intuitive & Mathematical Proof**:  
  Let $P = [\mathbf{v}_1 \mid \mathbf{v}_2] = \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}$ be the eigen-basis matrix, with inverse $P^{-1} = \frac{1}{2}\begin{bmatrix} 1 & 1 \\ -1 & 1 \end{bmatrix}$ and diagonal eigenvalue matrix $D = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}$.  
  Direct transformation yields $A\mathbf{x} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}\begin{bmatrix} 2 \\ 0 \end{bmatrix} = \begin{bmatrix} 4 \\ 2 \end{bmatrix}$.  
  The similarity sequence reproduces this result via three coordinate transitions:
  1. **Stage 1 ($P^{-1}$)**: Standard-to-eigen change of basis:
     $$[\mathbf{x}]_B = P^{-1}\mathbf{x} = \frac{1}{2}\begin{bmatrix} 1 & 1 \\ -1 & 1 \end{bmatrix}\begin{bmatrix} 2 \\ 0 \end{bmatrix} = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$$
     *Check*: $1\mathbf{v}_1 - 1\mathbf{v}_2 = \begin{bmatrix} 1 \\ 1 \end{bmatrix} - \begin{bmatrix} -1 \\ 1 \end{bmatrix} = \begin{bmatrix} 2 \\ 0 \end{bmatrix} = \mathbf{x}$.
  2. **Stage 2 ($D$)**: Decoupled diagonal scaling along eigen-axes:
     $$[\mathbf{y}]_B = D [\mathbf{x}]_B = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}\begin{bmatrix} 1 \\ -1 \end{bmatrix} = \begin{bmatrix} 3(1) \\ 1(-1) \end{bmatrix} = \begin{bmatrix} 3 \\ -1 \end{bmatrix}$$
     The first coordinate scales by $\lambda_1 = 3$; the second coordinate scales by $\lambda_2 = 1$.
  3. **Stage 3 ($P$)**: Eigen-to-standard restoration:
     $$\mathbf{y} = P [\mathbf{y}]_B = \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}\begin{bmatrix} 3 \\ -1 \end{bmatrix} = 3\begin{bmatrix} 1 \\ 1 \end{bmatrix} - 1\begin{bmatrix} -1 \\ 1 \end{bmatrix} = \begin{bmatrix} 4 \\ 2 \end{bmatrix} = A\mathbf{x}$$
  The composition $P \circ D \circ P^{-1}$ is identical to $A$, proving that $A$ is simply decoupled diagonal scaling viewed from a rotated coordinate frame.
- **NumPy Verification**:
  ```python
  import numpy as np

  A = np.array([[2.0, 1.0], [1.0, 2.0]])
  v1 = np.array([1.0, 1.0])
  v2 = np.array([-1.0, 1.0])
  P = np.column_stack([v1, v2])
  P_inv = np.linalg.inv(P)
  D = np.diag([3.0, 1.0])
  x = np.array([2.0, 0.0])

  # Step 1: P^-1
  c1 = P_inv @ x
  assert np.allclose(c1, [1.0, -1.0])

  # Step 2: D
  c2 = D @ c1
  assert np.allclose(c2, [3.0, -1.0])

  # Step 3: P
  x_final = P @ c2
  assert np.allclose(x_final, [4.0, 2.0])
  assert np.allclose(A @ x, x_final)
  print("Prompt 2 Verification PASSED: 3-step sequence P^-1 -> D -> P confirmed.")
  ```
- **Textbook Grounding**: MML §2.7.2 (Similarity Transformations); Kreyszig §8.4 (Diagonalization).

---

### Prompt 3: ML Bridge: Covariance Diagonalization & PCA Decorrelation
- **Question**:  
  In Principal Component Analysis (PCA), how does the similarity transformation of a sample covariance matrix $\Sigma = \frac{1}{N-1} X^\top X$ into diagonal form $\Lambda = Q^\top \Sigma Q$ eliminate cross-feature correlations?
- **Intuitive & Mathematical Proof**:  
  Let $X \in \mathbb{R}^{N \times d}$ denote mean-centered dataset observations. The sample covariance matrix is:
  $$\Sigma = \frac{1}{N-1} X^\top X \in \mathbb{R}^{d \times d}$$
  Because $\Sigma$ is real symmetric and positive semi-definite ($\mathbf{z}^\top \Sigma \mathbf{z} = \frac{1}{N-1}\|X\mathbf{z}\|^2 \ge 0$), the Spectral Theorem guarantees the existence of an orthonormal basis of eigenvectors $Q = [\mathbf{q}_1 \mid \dots \mid \mathbf{q}_d]$ such that $Q^\top Q = I$ and:
  $$\Sigma = Q \Lambda Q^\top \iff \Lambda = Q^\top \Sigma Q = \text{diag}(\lambda_1, \dots, \lambda_d)$$
  When we project data onto the principal axes via $Z = X Q$, the sample covariance matrix of the transformed features $Z$ becomes:
  $$\text{Cov}(Z) = \frac{1}{N-1} Z^\top Z = \frac{1}{N-1} (XQ)^\top (XQ) = Q^\top \left(\frac{1}{N-1} X^\top X\right) Q = Q^\top \Sigma Q = \Lambda$$
  Because $\Lambda$ is strictly diagonal, the covariance between distinct features $z_i$ and $z_j$ ($i \ne j$) is zero: $\text{Cov}(z_i, z_j) = 0$. The similarity transformation has completely decorrelated the data dimensions.
- **NumPy Verification**:
  ```python
  import numpy as np

  A = np.array([[2.0, 1.0], [1.0, 2.0]])
  rng = np.random.default_rng(4108)

  # Generate 1000 correlated 2D samples with covariance A
  L = np.linalg.cholesky(A)
  X = rng.standard_normal((1000, 2)) @ L.T
  X_cent = X - np.mean(X, axis=0)
  cov_X = (X_cent.T @ X_cent) / 999

  # Eigendecomposition of covariance
  eigvals, Q = np.linalg.eigh(cov_X)
  # Project data onto principal eigen-basis
  Z = X_cent @ Q
  cov_Z = (Z.T @ Z) / 999

  # Off-diagonal covariance is decorrelated (< 0.05 for finite sample)
  assert abs(cov_Z[0, 1]) < 0.05
  print(f"Prompt 3 Verification PASSED: Off-diagonal cov reduced from {cov_X[0, 1]:.2f} to {cov_Z[0, 1]:.4f}.")
  ```
- **Textbook Grounding**: MML §10.3 (Principal Component Analysis); VMLS §10.3.

---

### Prompt 4: GATE Bridge: Cayley-Hamilton Theorem & Inverse Computation
- **Question**:  
  State the Cayley-Hamilton Theorem and apply it to matrix $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ to evaluate $p(A)$ and derive an instant closed-form expression for $A^{-1}$ without performing row operations or computing adjugates.
- **Intuitive & Mathematical Proof**:  
  **Cayley-Hamilton Theorem**: Every square matrix $A \in \mathbb{R}^{n \times n}$ satisfies its own characteristic polynomial:
  $$p(\lambda) = \det(\lambda I - A) = 0 \implies p(A) = \mathbf{0}$$
  For our canonical $2 \times 2$ symmetric matrix $A$:
  $$p(\lambda) = \det\begin{bmatrix} \lambda - 2 & -1 \\ -1 & \lambda - 2 \end{bmatrix} = (\lambda - 2)^2 - 1 = \lambda^2 - 4\lambda + 3$$
  Applying the theorem:
  $$p(A) = A^2 - 4A + 3I = \mathbf{0}$$
  To find $A^{-1}$, multiply both sides of the identity by $A^{-1}$:
  $$A^{-1}(A^2 - 4A + 3I) = A - 4I + 3A^{-1} = \mathbf{0}$$
  Rearranging for $A^{-1}$:
  $$3A^{-1} = 4I - A \implies A^{-1} = \frac{1}{3}(4I - A)$$
  Substituting numerical values:
  $$A^{-1} = \frac{1}{3}\left( \begin{bmatrix} 4 & 0 \\ 0 & 4 \end{bmatrix} - \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \right) = \frac{1}{3}\begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix} = \begin{bmatrix} 2/3 & -1/3 \\ -1/3 & 2/3 \end{bmatrix}$$
  *Verification*:
  $$A A^{-1} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} 2/3 & -1/3 \\ -1/3 & 2/3 \end{bmatrix} = \begin{bmatrix} 4/3 - 1/3 & -2/3 + 2/3 \\ 2/3 - 2/3 & -1/3 + 4/3 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = I$$
  This allows calculating high powers and inverses in competitive exams in seconds.
- **NumPy Verification**:
  ```python
  import numpy as np

  A = np.array([[2.0, 1.0], [1.0, 2.0]])
  I = np.eye(2)

  # Cayley-Hamilton: A^2 - 4A + 3I == 0
  p_A = A @ A - 4.0 * A + 3.0 * I
  assert np.allclose(p_A, np.zeros((2, 2)), atol=1e-12)

  # Fast inverse formula: A^-1 = (1/3)*(4I - A)
  A_inv_ch = (1.0 / 3.0) * (4.0 * I - A)
  A_inv_true = np.linalg.inv(A)
  assert np.allclose(A_inv_ch, A_inv_true, atol=1e-12)
  print("Prompt 4 Verification PASSED: Cayley-Hamilton theorem and inverse verified.")
  ```
- **Textbook Grounding**: Kreyszig §8.4 (Eigenvalue Problems); B.S. Grewal §2.15 (Cayley-Hamilton Theorem).
