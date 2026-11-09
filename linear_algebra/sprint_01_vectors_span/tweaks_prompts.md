# Part 4: Interactive Parameter Tweaks & Intuition Stress-Test Prompts

## 1. Interactive Parameter Tweaks Guide

The ManimCE animation script (`scene_vectors_span.py`) defines key geometric, algebraic, and aesthetic parameters at the top of the file as global constants. Users and students can alter these parameters to experiment with coordinate geometries, basis orientations, and edge cases.

### Parameter Configuration Matrix

| Parameter Name | Default Value | Recommended Test Values | Pedagogical Objective |
|---|---|---|---|
| `V1_COORDS` | `np.array([2.0, 1.0])` | `[1.0, 0.0]`, `[3.0, -1.0]` | Alter primary basis direction and test non-orthogonal coordinate frames. |
| `V2_COORDS` | `np.array([-1.0, 2.0])` | `[0.0, 1.0]`, `[1.0, 2.0]` | Test orthogonal vs non-orthogonal basis vectors in Act 1 & 2. |
| `COLLINEAR_FACTOR` | `2.0` | `-1.5`, `0.5`, `3.0` | Test anti-parallel ($k < 0$) and fractional ($0 < k < 1$) collinear collapses in Act 4. |
| `TARGET_COORDS` | `np.array([3.0, 4.0])` | `[-3.0, 2.0]`, `[0.0, 5.0]` | Shift reachable puzzle target into different quadrants; test negative coefficients. |
| `UNREACHABLE_TARGET` | `np.array([1.0, 4.0])` | `[2.0, -3.0]`, `[0.0, 2.0]` | Test off-line points when collinear line is $y = 0.5x$ ($y \ne 0.5x$). |
| `SWEEP_RANGE` | `(-2.5, 2.5)` | `(-4.0, 4.0)`, `(-1.0, 1.0)` | Expand or restrict slider sweep range to test viewport boundaries. |
| `GRID_OPACITY` | `0.25` | `0.10` to `0.50` | Adjust contrast between coordinate plane and vector arrows. |
| `SUBSPACE_OPACITY` | `0.25` | `0.15` to `0.45` | Control transparency of the 2D span wash in Act 3. |

### How to Modify and Re-render

1. **Locate configuration block in `scene_vectors_span.py`**:
   ```python
   # Configuration Constants
   V1_COORDS = np.array([2.0, 1.0])
   V2_COORDS = np.array([-1.0, 2.0])
   TARGET_COORDS = np.array([3.0, 4.0])
   COLLINEAR_FACTOR = 2.0
   ```
2. **Execute fast low-quality preview render**:
   ```bash
   python -m manim -ql scene_vectors_span.py VectorSpanScene
   ```
3. **Execute single act preview**:
   ```bash
   python -m manim -ql scene_vectors_span.py Act1SpatialPuzzle
   ```

---

## 2. Intuition Stress-Test Prompts

### Prompt 1: Subspace Closure & The Origin Anchor
- **Question**:  
  Consider the line $L = \left\{ \begin{bmatrix} x \\ y \end{bmatrix} \in \mathbb{R}^2 \;\middle|\; y = 2x + 1 \right\}$. Is $L$ a linear subspace of $\mathbb{R}^2$? Use the parallelogram law of vector addition to explain why any affine flat not passing through the origin fails subspace closure.
- **Intuitive & Mathematical Proof**:  
  By Definition 2.10 in MML, every linear subspace must contain the zero vector $\mathbf{0} = [0, 0]^\top$. In $L$, substituting $x=0$ yields $y = 1 \ne 0$, so $\mathbf{0} \notin L$.  
  Furthermore, vector addition violates closure. Select two points on $L$:
  $$\mathbf{u} = \begin{bmatrix} 0 \\ 1 \end{bmatrix} \in L \quad (1 = 2(0) + 1), \qquad \mathbf{v} = \begin{bmatrix} 1 \\ 3 \end{bmatrix} \in L \quad (3 = 2(1) + 1)$$
  Their vector sum is:
  $$\mathbf{u} + \mathbf{v} = \begin{bmatrix} 1 \\ 4 \end{bmatrix}$$
  Testing membership in $L$: $2(1) + 1 = 3 \ne 4$. The sum escapes the line. Geometrically, the parallelogram formed by $\mathbf{u}$ and $\mathbf{v}$ has one vertex at the origin $\mathbf{0}$. Because the origin is not on $L$, the opposite diagonal vertex $\mathbf{u} + \mathbf{v}$ is offset by twice the constant shift ($+2$), landing outside $L$.
- **NumPy Verification**:
  ```python
  import numpy as np

  # Define line y = 2x + 1
  u = np.array([0.0, 1.0])
  v = np.array([1.0, 3.0])
  w = u + v  # [1.0, 4.0]

  # Subspace closure check
  is_w_on_line = np.isclose(w[1], 2 * w[0] + 1)
  assert not is_w_on_line, "Sum must escape non-origin line!"
  print("Prompt 1 Verification Passed: Offset lines violate closure under addition.")
  ```
- **Textbook Grounding**: MML §2.4.3 (Definition 2.10); ILA §8.1 (Theorem 8.1).

---

### Prompt 2: Independence-Dimension Inequality in $\mathbb{R}^n$
- **Question**:  
  Suppose you are given three non-zero vectors $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3 \in \mathbb{R}^2$. Can they ever be linearly independent? What is the maximum number of linearly independent vectors in $\mathbb{R}^n$, and what geometric invariant guarantees this bound?
- **Intuitive & Mathematical Proof**:  
  According to the Independence-Dimension Inequality (VMLS §5.1, p. 91), any collection of $k > n$ vectors in $\mathbb{R}^n$ is guaranteed to be linearly dependent. In $\mathbb{R}^2$, ambient space has only 2 orthogonal directions of motion. If $\mathbf{v}_1$ and $\mathbf{v}_2$ are linearly independent, they form a basis and span all of $\mathbb{R}^2$. Consequently, the third vector $\mathbf{v}_3$ is already reachable:
  $$\mathbf{v}_3 = c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 \implies c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 - 1 \mathbf{v}_3 = \mathbf{0}$$
  Because the coefficient of $\mathbf{v}_3$ is $-1 \ne 0$, this provides a non-trivial linear combination equaling $\mathbf{0}$, proving linear dependence. Geometrically, a third vector cannot introduce a third independent direction in a 2-dimensional plane.
- **NumPy Verification**:
  ```python
  import numpy as np

  # Three arbitrary non-zero vectors in R^2
  v1 = np.array([2.0, 1.0])
  v2 = np.array([-1.0, 2.0])
  v3 = np.array([5.0, -3.0])

  A = np.column_stack([v1, v2, v3])  # 2x3 matrix
  rank = np.linalg.matrix_rank(A)
  assert rank <= 2, "Rank in R^2 cannot exceed ambient dimension 2"

  # Find non-trivial null vector: A @ c = 0
  U, S, Vt = np.linalg.svd(A)
  null_c = Vt[-1]  # right singular vector corresponding to smallest singular value
  assert not np.allclose(null_c, 0.0)
  assert np.allclose(A @ null_c, np.zeros(2))
  print("Prompt 2 Verification Passed: 3 vectors in R^2 are strictly linearly dependent.")
  ```
- **Textbook Grounding**: VMLS §5.1 (Independence-Dimension Inequality); MML §2.5 (Definition 2.12).

---

### Prompt 3: GATE CS/DA Consistency & The Rouché-Capelli Theorem
- **Question**:  
  Consider the linear system $A \mathbf{x} = \mathbf{b}$ where $A = \begin{bmatrix} 2 & 4 \\ 1 & 2 \end{bmatrix}$ and $\mathbf{b} = \begin{bmatrix} 1 \\ 4 \end{bmatrix}$. How does Gaussian elimination mechanically expose that $\mathbf{b} \notin \text{span}(\text{columns of } A)$? State the exact Rouché-Capelli rank condition for zero solutions.
- **Intuitive & Mathematical Proof**:  
  The columns of $A$ are $\mathbf{a}_1 = [2, 1]^\top$ and $\mathbf{a}_2 = [4, 2]^\top = 2 \mathbf{a}_1$. They are collinear, so $\text{Col}(A)$ is the 1D line $y = 0.5x$.  
  The target vector $\mathbf{b} = [1, 4]^\top$ has $y = 4 \ne 0.5(1) = 0.5$, so $\mathbf{b} \notin \text{Col}(A)$.  
  Setting up the augmented matrix $[A \mid \mathbf{b}]$:
  $$\begin{bmatrix} 2 & 4 & 1 \\ 1 & 2 & 4 \end{bmatrix} \xrightarrow{R_2 \leftarrow R_2 - 0.5 R_1} \begin{bmatrix} 2 & 4 & 1 \\ 0 & 0 & 3.5 \end{bmatrix}$$
  The second row represents the impossible equation $0 x_1 + 0 x_2 = 3.5$.  
  The Rouché-Capelli Theorem dictates:
  $$\text{rank}(A) = 1 < \text{rank}([A \mid \mathbf{b}]) = 2 \implies \text{Zero Solutions (Inconsistent System)}$$
  The augmented column $\mathbf{b}$ introduces a new pivot, meaning it adds a new dimension not covered by the columns of $A$.
- **NumPy Verification**:
  ```python
  import numpy as np

  A = np.array([[2.0, 4.0], [1.0, 2.0]])
  b_unreachable = np.array([1.0, 4.0])
  b_reachable = np.array([6.0, 3.0])  # 3 * a1

  rank_A = np.linalg.matrix_rank(A)
  rank_aug_unreachable = np.linalg.matrix_rank(np.column_stack([A, b_unreachable]))
  rank_aug_reachable = np.linalg.matrix_rank(np.column_stack([A, b_reachable]))

  assert rank_A == 1
  assert rank_aug_unreachable == 2  # Inconsistent: rank(A) < rank([A|b])
  assert rank_aug_reachable == 1    # Consistent: rank(A) == rank([A|b])
  print("Prompt 3 Verification Passed: Inconsistency verified via rank discrepancy.")
  ```
- **Textbook Grounding**: MML §2.3.1; GATE CSE / DA Linear Algebra Syllabus.

---

### Prompt 4: Machine Learning Attention as a Constrained Convex Subspace
- **Question**:  
  In Transformer Self-Attention, context representations are generated by:
  $$\mathbf{y} = \sum_{j=1}^N \alpha_j \mathbf{v}_j, \quad \alpha = \text{softmax}\left(\frac{\mathbf{q}^\top K}{\sqrt{d_k}}\right)$$
  Why is $\mathbf{y}$ geometrically constrained to the convex hull of $\{\mathbf{v}_1, \dots, \mathbf{v}_N\}$ rather than the full linear span? What happens to $\mathbf{y}$ in the two temperature limits $\tau \to 0$ and $\tau \to \infty$?
- **Intuitive & Mathematical Proof**:  
  By definition of the softmax function:
  $$\alpha_j = \frac{\exp(z_j / \tau)}{\sum_{m=1}^N \exp(z_m / \tau)} > 0 \quad \text{and} \quad \sum_{j=1}^N \alpha_j = 1$$
  A linear combination with non-negative coefficients that sum to 1 is strictly a **convex combination** (VMLS §1.3, p. 18). While an unconstrained linear combination can reach any point in the infinite linear span $\mathbb{R}^d$, a convex combination is geometrically confined to the convex hull (the polygon, simplex, or polytope defined by the endpoints of the value vectors).
  - **Limit $\tau \to 0$ (Freezing / Hard Attention)**: Softmax converges to one-hot $\text{argmax}$. The output collapses to a single vertex of the convex hull: $\mathbf{y} \to \mathbf{v}_{j^*}$.
  - **Limit $\tau \to \infty$ (Infinite Temperature / Uniform Attention)**: Softmax weights become uniformly equal: $\alpha_j \to 1/N$. The output collapses to the geometric centroid: $\mathbf{y} \to \frac{1}{N} \sum_{j=1}^N \mathbf{v}_j$.
- **NumPy Verification**:
  ```python
  import numpy as np

  def softmax(logits, tau=1.0):
      scaled = (logits - np.max(logits)) / tau
      exp_z = np.exp(scaled)
      return exp_z / np.sum(exp_z)

  # Value vectors in R^2
  V = np.array([[2.0, 1.0], [-1.0, 2.0], [3.0, 4.0]])
  logits = np.array([1.5, 3.2, 0.8])  # argmax is index 1

  # Normal temperature
  alpha_normal = softmax(logits, tau=1.0)
  assert np.isclose(np.sum(alpha_normal), 1.0)
  assert np.all(alpha_normal >= 0.0)

  # Low temperature (tau -> 0)
  alpha_cold = softmax(logits, tau=1e-5)
  assert np.isclose(alpha_cold[1], 1.0)
  assert np.allclose(alpha_cold @ V, V[1])

  # High temperature (tau -> inf)
  alpha_hot = softmax(logits, tau=1e7)
  assert np.allclose(alpha_hot, 1.0 / 3.0)
  assert np.allclose(alpha_hot @ V, np.mean(V, axis=0))
  print("Prompt 4 Verification Passed: Attention bounded in convex hull; limit states confirmed.")
  ```
- **Textbook Grounding**: VMLS §1.3 (Convex Combinations & Mixtures); Vaswani et al. (2017).
