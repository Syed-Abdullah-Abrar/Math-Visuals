# Part 4: Interactive Parameter Tweaks & Intuition Stress-Test Prompts

## 1. Interactive Parameter Tweaks Guide

The ManimCE animation script (`scene_linear_mappings.py`) defines key transformation matrices, vector coordinates, and visual styling parameters as module-level constants. Students and researchers can modify these values to explore alternate linear operators, shears, rotations, reflections, and composition orders.

### Parameter Configuration Matrix

| Parameter Name | Default Value | Recommended Test Values | Pedagogical Objective |
|---|---|---|---|
| `A_MAT` | `[[2.0, 0.0], [1.0, 2.0]]` | `[[3.0, 0.0], [0.0, 2.0]]`, `[[1.0, 2.0], [2.0, 1.0]]` | Observe diagonal scaling vs non-orthogonal shearing and area magnification $\det(A)$. |
| `PUZZLE_X` | `np.array([1.5, 1.0])` | `[2.0, -1.0]`, `[-1.0, 1.5]` | Test basis expansion across different quadrants and sign combinations. |
| `S_MAT` | `[[1.0, 1.0], [0.0, 1.0]]` | `[[1.0, -2.0], [0.0, 1.0]]`, `[[1.0, 0.5], [0.0, 1.0]]` | Vary horizontal shear intensity $k$; observe rhomboidal skewing without vertical displacement. |
| `R_MAT` | `[[0.0, -1.0], [1.0, 0.0]]` | `[[0.0, 1.0], [-1.0, 0.0]]` ($-90^\circ$), `[[-1.0, 0.0], [0.0, -1.0]]` ($180^\circ$) | Explore clockwise rotation and point reflection through origin. |
| `STRESS_X` | `np.array([2.0, 3.0])` | `[1.0, 1.0]`, `[3.0, -2.0]` | Measure spatial divergence between $(SR)\mathbf{x}$ and $(RS)\mathbf{x}$ across distinct test probes. |
| `SUBSPACE_COLOR`| `"#9A72AC"` (Purple) | `"#58C4DD"` (Blue), `"#2A9D8F"` (Teal) | Alter unit parallelogram fill color for customized contrast. |

### How to Modify and Re-Render

1. **Edit constants in `scene_linear_mappings.py`**:
   ```python
   # Configuration Constants
   A_MAT = np.array([[2.0, 0.0], [1.0, 2.0]])
   PUZZLE_X = np.array([1.5, 1.0])
   S_MAT = np.array([[1.0, 1.0], [0.0, 1.0]])
   R_MAT = np.array([[0.0, -1.0], [1.0, 0.0]])
   STRESS_X = np.array([2.0, 3.0])
   ```
2. **Execute fast low-quality preview render**:
   ```bash
   .venv/bin/manim -ql scene_linear_mappings.py LinearMappingsScene
   ```
3. **Execute single act preview**:
   ```bash
   .venv/bin/manim -ql scene_linear_mappings.py Act1SpatialPuzzle
   ```

---

## 2. Intuition Stress-Test Prompts

### Prompt 1: Parallel Grid Line Preservation & Area Scaling Factor $\det(A)$
- **Question**:  
  Why does an arbitrary invertible matrix $A \in \mathbb{R}^{2 \times 2}$ guarantee that parallel lines remain parallel and that grid lines remain evenly spaced? Prove that the area scaling factor of any enclosed shape is identically $|\det(A)|$.
- **Intuitive & Mathematical Proof**:  
  Any straight line in $\mathbb{R}^2$ can be parameterized as $L(t) = \mathbf{p} + t \mathbf{v}$, where $\mathbf{p}$ is an initial anchor point, $\mathbf{v}$ is the tangent direction vector, and $t \in \mathbb{R}$. Applying a linear mapping $A$:
  $$A(L(t)) = A(\mathbf{p} + t \mathbf{v}) = A\mathbf{p} + t (A\mathbf{v})$$
  Because $A$ is linear, the image is another straight line anchored at $A\mathbf{p}$ with direction vector $A\mathbf{v}$. If two lines $L_1$ and $L_2$ are parallel, they share direction vector $\mathbf{v}$. Their transformed images share direction vector $A\mathbf{v}$, guaranteeing that $A(L_1) \parallel A(L_2)$.  
  
  Equispaced points $\mathbf{x}_k = \mathbf{p} + k \Delta t \mathbf{v}$ transform to $A\mathbf{x}_k = A\mathbf{p} + k \Delta t (A\mathbf{v})$. The step vector between consecutive points is $\Delta t (A\mathbf{v})$, which is constant and independent of $k$, preserving uniform grid spacing.
  
  For area scaling, the unit square spanned by $\hat{\imath} = [1, 0]^\top$ and $\hat{\jmath} = [0, 1]^\top$ has area $1.0$. Under $A$, it maps to the parallelogram spanned by $\mathbf{a}_1 = A\hat{\imath}$ and $\mathbf{a}_2 = A\hat{\jmath}$. The signed area of this parallelogram is given by the 2D determinant (cross product):
  $$\text{Area} = |a_{11} a_{22} - a_{12} a_{21}| = |\det(A)|$$
  By Green's theorem and Riemann summation, any arbitrary 2D shape can be partitioned into infinitesimal squares of area $dA$; each transforms into a parallelogram of area $|\det(A)| dA$, yielding a global area scaling factor of $|\det(A)|$.
- **NumPy Verification**:
  ```python
  import numpy as np

  # Canonical Sprint 2 matrix: det = 4
  A = np.array([[2.0, 0.0], [1.0, 2.0]])

  # 1. Verify parallel tangent preservation across horizontal and vertical grid lines
  x_coords = np.linspace(-4, 4, 9)
  y_coords = np.linspace(-4, 4, 9)

  for x in x_coords:
      line_pts = np.column_stack([np.full(50, x), np.linspace(-4, 4, 50)])
      mapped_line = line_pts @ A.T
      tangent = mapped_line[1] - mapped_line[0]
      expected_tangent = (line_pts[1] - line_pts[0]) @ A.T
      assert np.allclose(tangent, expected_tangent, atol=1e-12)

  for y in y_coords:
      line_pts = np.column_stack([np.linspace(-4, 4, 50), np.full(50, y)])
      mapped_line = line_pts @ A.T
      tangent = mapped_line[1] - mapped_line[0]
      expected_tangent = (line_pts[1] - line_pts[0]) @ A.T
      assert np.allclose(tangent, expected_tangent, atol=1e-12)

  # 2. Verify area scaling factor det(A) == 4.0
  det_A = np.linalg.det(A)
  assert np.isclose(det_A, 4.0, atol=1e-12)
  print("Prompt 1 Verification PASSED: Parallelism and area scaling = 4.0 confirmed.")
  ```
- **Textbook Grounding**: MML §2.5 (Linear Mappings, Def 2.15 & 2.19); VMLS §7.1 (Linear Functions); Kreyszig §7.4.

---

### Prompt 2: Basis Tracking & Orientation Preservation vs Inversion
- **Question**:  
  Why are the columns of a transformation matrix $A = [\mathbf{a}_1 \mid \mathbf{a}_2]$ sufficient to determine the transformed position of every vector in the space? How does the sign of $\det(A)$ determine whether the spatial orientation is preserved or inverted (flipped)?
- **Intuitive & Mathematical Proof**:  
  Any vector $\mathbf{x} = \begin{bmatrix} x \\ y \end{bmatrix}$ is expressed uniquely in the standard basis as $\mathbf{x} = x \hat{\imath} + y \hat{\jmath}$. By linearity:
  $$A\mathbf{x} = A(x\hat{\imath} + y\hat{\jmath}) = x(A\hat{\imath}) + y(A\hat{\jmath}) = x\mathbf{a}_1 + y\mathbf{a}_2$$
  The columns $\mathbf{a}_1$ and $\mathbf{a}_2$ represent the destinations of $\hat{\imath}$ and $\hat{\jmath}$. The entire space is merely a continuous extension of this two-vector coordinate framework.
  
  In $\mathbb{R}^2$, the orientation of the ordered basis $\{\mathbf{a}_1, \mathbf{a}_2\}$ is measured by the 2D cross product:
  $$\mathbf{a}_1 \times \mathbf{a}_2 = a_{11} a_{22} - a_{12} a_{21} = \det(A)$$
  - If $\det(A) > 0$: $\mathbf{a}_2$ remains counter-clockwise from $\mathbf{a}_1$. Orientation is preserved (right-handed system remains right-handed).
  - If $\det(A) < 0$: $\mathbf{a}_2$ is clockwise from $\mathbf{a}_1$. Space has been flipped over (reflected across a hyperplane), inverting orientation.
- **NumPy Verification**:
  ```python
  import numpy as np

  # Preserving matrix (det = +4) vs Inverting matrix (det = -4)
  A_pos = np.array([[2.0, 0.0], [1.0, 2.0]])
  A_neg = np.array([[2.0, 0.0], [1.0, -2.0]])

  e1 = np.array([1.0, 0.0])
  e2 = np.array([0.0, 1.0])

  # Basis destinations
  assert np.allclose(A_pos @ e1, [2.0, 1.0])
  assert np.allclose(A_pos @ e2, [0.0, 2.0])

  # 2D orientation cross product (u0*v1 - u1*v0)
  u_pos = A_pos @ e1
  v_pos = A_pos @ e2
  cross_pos = u_pos[0] * v_pos[1] - u_pos[1] * v_pos[0]

  u_neg = A_neg @ e1
  v_neg = A_neg @ e2
  cross_neg = u_neg[0] * v_neg[1] - u_neg[1] * v_neg[0]

  assert cross_pos > 0.0, "Positive determinant must preserve counter-clockwise orientation"
  assert cross_neg < 0.0, "Negative determinant must invert to clockwise orientation"
  assert np.isclose(cross_pos, np.linalg.det(A_pos))
  assert np.isclose(cross_neg, np.linalg.det(A_neg))
  print("Prompt 2 Verification PASSED: Basis tracking and orientation flip confirmed.")
  ```
- **Textbook Grounding**: MML §2.5 (Theorem 2.17); VMLS §6.1; ILA §6.2.

---

### Prompt 3: Geometric Non-Commutativity ($SR \neq RS$) & Spatial Divergence
- **Question**:  
  Using a $90^\circ$ rotation $R = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$ and a horizontal shear $S = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$, geometrically prove why matrix multiplication is non-commutative ($SR \neq RS$). What is the exact spatial distance between $(SR)\mathbf{x}$ and $(RS)\mathbf{x}$ for probe point $\mathbf{x} = [2, 3]^\top$?
- **Intuitive & Mathematical Proof**:  
  Matrix multiplication represents sequential composition from right to left.
  - In $SR$: Rotation $R$ occurs first, swinging the axes by $90^\circ$ counter-clockwise. Then shear $S$ shifts horizontal coordinates proportional to their new vertical coordinates.
    $$SR = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 1 & -1 \\ 1 & 0 \end{bmatrix}$$
  - In $RS$: Shear $S$ occurs first, slanting the vertical grid lines to the right. Then rotation $R$ turns the already-sheared grid by $90^\circ$.
    $$RS = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 0 & -1 \\ 1 & 1 \end{bmatrix}$$
  Because each operation deforms the coordinate scaffolding before the subsequent operation takes place, the final outcomes diverge:
  $$(SR)\mathbf{x} = \begin{bmatrix} 1 & -1 \\ 1 & 0 \end{bmatrix}\begin{bmatrix} 2 \\ 3 \end{bmatrix} = \begin{bmatrix} -1 \\ 2 \end{bmatrix}$$
  $$(RS)\mathbf{x} = \begin{bmatrix} 0 & -1 \\ 1 & 1 \end{bmatrix}\begin{bmatrix} 2 \\ 3 \end{bmatrix} = \begin{bmatrix} -3 \\ 5 \end{bmatrix}$$
  The spatial discrepancy vector is:
  $$\Delta = (SR)\mathbf{x} - (RS)\mathbf{x} = \begin{bmatrix} -1 - (-3) \\ 2 - 5 \end{bmatrix} = \begin{bmatrix} 2 \\ -3 \end{bmatrix}$$
  The Euclidean distance between the two endpoints is:
  $$\|\Delta\| = \sqrt{2^2 + (-3)^2} = \sqrt{13} \approx 3.60555$$
  The commutator is $[S, R] = SR - RS = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \neq \mathbf{0}$, yet $\text{Tr}([S, R]) = 1 + (-1) = 0$ and $\det(SR) = \det(RS) = 1.0$.
- **NumPy Verification**:
  ```python
  import numpy as np
  import math

  S = np.array([[1.0, 1.0], [0.0, 1.0]])
  R = np.array([[0.0, -1.0], [1.0, 0.0]])
  SR = S @ R
  RS = R @ S

  x = np.array([2.0, 3.0])
  dest_SR = SR @ x
  dest_RS = RS @ x

  assert np.allclose(dest_SR, [-1.0, 2.0])
  assert np.allclose(dest_RS, [-3.0, 5.0])

  discrepancy = np.linalg.norm(dest_SR - dest_RS)
  assert np.isclose(discrepancy, math.sqrt(13.0))

  commutator = SR - RS
  assert not np.allclose(commutator, np.zeros((2, 2)))
  assert np.isclose(np.trace(commutator), 0.0)
  assert np.isclose(np.linalg.det(SR), 1.0)
  assert np.isclose(np.linalg.det(RS), 1.0)
  print(f"Prompt 3 Verification PASSED: Non-commutativity discrepancy = {discrepancy:.4f} == sqrt(13).")
  ```
- **Textbook Grounding**: VMLS §10.1–10.2; MML §2.5; Kreyszig §7.4.

---

### Prompt 4: Frontier ML Attention Projections as Fused Bilinear Operators
- **Question**:  
  In Transformer Multi-Head Self-Attention, input token representations $X \in \mathbb{R}^{T \times d_{\text{model}}}$ are projected via weight matrices $W_Q, W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ to form Query $Q = X W_Q$ and Key $K = X W_K$. Prove algebraically and geometrically that computing the unnormalized attention matrix $A = Q K^\top / \sqrt{d_k}$ is identical to applying a single composed bilinear operator $M_{\text{fused}} = (W_Q W_K^\top) / \sqrt{d_k} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ directly between token vectors.
- **Intuitive & Mathematical Proof**:  
  Using the associativity of matrix multiplication:
  $$Q K^\top = (X W_Q)(X W_K)^\top = (X W_Q)(W_K^\top X^\top) = X (W_Q W_K^\top) X^\top$$
  Dividing by the temperature scale factor $\sqrt{d_k}$:
  $$\frac{Q K^\top}{\sqrt{d_k}} = X \left( \frac{W_Q W_K^\top}{\sqrt{d_k}} \right) X^\top = X M_{\text{fused}} X^\top$$
  Geometrically, rather than projecting high-dimensional token vectors down into lower-dimensional query and key subspaces and evaluating their Euclidean dot product, attention computes the pairwise bilinear metric induced by the low-rank kernel matrix $M_{\text{fused}}$. The rank of $M_{\text{fused}}$ is bounded by $\min(d_{\text{model}}, d_k) = d_k$, demonstrating that multi-head attention acts as a low-rank directional similarity lens on the latent representation space.
- **NumPy Verification**:
  ```python
  import numpy as np

  rng = np.random.default_rng(2108)
  d_model = 64
  d_k = 16
  T = 10

  W_Q = rng.standard_normal((d_model, d_k))
  W_K = rng.standard_normal((d_model, d_k))
  X = rng.standard_normal((T, d_model))

  # 1. Direct separate projections
  Q = X @ W_Q
  K = X @ W_K
  A_direct = (Q @ K.T) / np.sqrt(d_k)

  # 2. Composed fused bilinear operator
  M_fused = (W_Q @ W_K.T) / np.sqrt(d_k)
  A_fused = X @ M_fused @ X.T

  # Verify equivalence with strict numerical precision
  np.testing.assert_allclose(A_direct, A_fused, atol=1e-12)
  assert np.linalg.matrix_rank(M_fused) <= d_k
  print("Prompt 4 Verification PASSED: Multi-head attention bilinear composition matches exactly.")
  ```
- **Textbook Grounding**: Vaswani et al. (2017) "Attention Is All You Need"; MML §2.5 & §3.2 (Bilinear Forms).
