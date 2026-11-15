# Part 3: Interactive Parameter Tweaks Guide & Intuition Stress-Test Prompts

## Section 1: Interactive Parameter Tweaks Guide

The modular script `scene_column_null_space.py` exposes canonical matrix and geometric constants at lines 47–58. Modifying these parameters alters the transformation geometry, planar equation, and subspace dimensions, allowing students and researchers to explore arbitrary $3 \times 3$ linear systems.

### Parameter Configuration Matrix
| Parameter Name | Default Value | Recommended Test Values | Pedagogical Objective |
|---|---|---|---|
| `A_MAT` | `[[1, 0, 1], [0, 1, 1], [1, 1, 2]]` | `[[1, 0, 0], [0, 1, 0], [0, 0, 0]]` | Test rank-2 coordinate plane projection ($xy$-plane, $z=0$, null vector along $\hat{k}$). |
| `A_MAT` (Rank 1) | `[[1, 0, 1], [0, 1, 1], [1, 1, 2]]` | `[[1, 2, 3], [2, 4, 6], [-1, -2, -3]]` | Observe catastrophic rank-1 collapse: 3D space collapses to a 1D line, null space becomes a 2D plane ($1 + 2 = 3$). |
| `A_MAT` (Rank 3) | `[[1, 0, 1], [0, 1, 1], [1, 1, 2]]` | `[[2, 1, 0], [0, 3, 1], [1, 0, 4]]` | Observe full rank 3: trivial null space $\{\mathbf{0}\}$, universal reachability, $\det(A) = 25 \ne 0$ ($3 + 0 = 3$). |
| `TARGET_B` | `[1.0, 2.0, 3.0]` | `[3.0, 1.0, 4.0]` | Test alternative reachable targets on plane $x + y - z = 0$ ($3 + 1 - 4 = 0$). |
| `B_UNREACH` | `[1.0, 0.0, 0.0]` | `[0.0, 0.0, 2.0]` | Test off-plane targets with varying orthogonal distance $|1(0) + 1(0) - 1(2)|/\sqrt{3} = 2/\sqrt{3}$. |
| `X_P` | `[1.0, 2.0, 0.0]` | `[0.0, 1.0, 1.0]` | Set particular solution directly to the least-norm solution $\mathbf{x}_{\text{LN}} = A^+ \mathbf{b}$, verifying $\mathbf{x}_{\text{LN}} \perp \mathbf{v}_{\text{null}}$. |
| `camera_rate` | `0.08` | `0.04` to `0.15` | Tune ambient orbital rotation velocity for optimal 3D perspective and depth perception. |

### CLI Render Commands
```bash
# Fast low-quality preview render of complete 4-Act master scene (480p15)
.venv/bin/manim -ql linear_algebra/sprint_03_column_null_space/scene_column_null_space.py ColumnNullSpaceScene

# High-quality production release render (1080p60)
.venv/bin/manim -qh linear_algebra/sprint_03_column_null_space/scene_column_null_space.py ColumnNullSpaceScene

# Modular single-act preview renders for rapid inspection
.venv/bin/manim -ql linear_algebra/sprint_03_column_null_space/scene_column_null_space.py Act1SpatialPuzzle
.venv/bin/manim -ql linear_algebra/sprint_03_column_null_space/scene_column_null_space.py Act2DynamicDeformation
.venv/bin/manim -ql linear_algebra/sprint_03_column_null_space/scene_column_null_space.py Act3AlgebraicMorph
.venv/bin/manim -ql linear_algebra/sprint_03_column_null_space/scene_column_null_space.py Act4DegenerateStressTest
```

---

## Section 2: Intuition Stress-Test Prompts

### Prompt 1: Why Does the Column Space Form a 2D Plane $x + y - z = 0$?
**Question**:  
Given matrix $A = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix}$, why is the set of all reachable vectors $A\mathbf{x}$ strictly constrained to a 2D plane in $\mathbb{R}^3$? How do the columns $\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3$ uniquely determine the plane normal $\mathbf{n} = [1, 1, -1]^\top$ and Cartesian equation $x + y - z = 0$?

**Intuitive & Mathematical Proof**:  
1. *Column Interpretation*: Any output vector $\mathbf{y} = A\mathbf{x}$ is a linear combination of the columns of $A$:
   $$\mathbf{y} = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + x_3 \mathbf{a}_3 = x_1 \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix} + x_2 \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix} + x_3 \begin{bmatrix} 1 \\ 1 \\ 2 \end{bmatrix}$$
2. *Linear Dependency*: Notice that column 3 is precisely the sum of columns 1 and 2:
   $$\mathbf{a}_3 = \mathbf{a}_1 + \mathbf{a}_2 = \begin{bmatrix} 1+0 \\ 0+1 \\ 1+1 \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \\ 2 \end{bmatrix}$$
   Therefore, any output can be rewritten purely in terms of $\mathbf{a}_1$ and $\mathbf{a}_2$:
   $$\mathbf{y} = (x_1 + x_3)\mathbf{a}_1 + (x_2 + x_3)\mathbf{a}_2$$
   Because $\mathbf{a}_1$ and $\mathbf{a}_2$ are linearly independent, their span forms a two-dimensional subspace (plane passing through the origin).
3. *Plane Normal Vector*: The normal vector $\mathbf{n}$ is orthogonal to both spanning columns:
   $$\mathbf{n} = \mathbf{a}_1 \times \mathbf{a}_2 = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix} \times \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix} = \begin{bmatrix} (0)(1) - (1)(1) \\ (1)(0) - (1)(1) \\ (1)(1) - (0)(0) \end{bmatrix} = \begin{bmatrix} -1 \\ -1 \\ 1 \end{bmatrix} \sim \begin{bmatrix} 1 \\ 1 \\ -1 \end{bmatrix}$$
4. *Cartesian Equation*: Since the plane passes through the origin $(0, 0, 0)$:
   $$\mathbf{n} \cdot \mathbf{y} = 0 \iff 1(x) + 1(y) - 1(z) = 0 \iff x + y - z = 0$$
   Every reachable vector must have its $z$-coordinate equal to the sum of its $x$- and $y$-coordinates ($z = x + y$).

**Executable NumPy Verification**:
```python
import numpy as np

A = np.array([
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 2.0]
])
a1, a2, a3 = A[:, 0], A[:, 1], A[:, 2]

# 1. Dependency check
assert np.allclose(a3, a1 + a2)
assert np.linalg.matrix_rank(A) == 2

# 2. Normal vector via cross product
n = np.cross(a1, a2)  # [-1, -1, 1]
normal_canonical = np.array([1.0, 1.0, -1.0])
assert np.allclose(n, -normal_canonical)

# 3. Orthogonality to all columns
for col in [a1, a2, a3]:
    assert np.isclose(np.dot(normal_canonical, col), 0.0)

# 4. 1,000 random inputs produce outputs on plane x + y - z = 0
rng = np.random.default_rng(42)
X = rng.uniform(-100, 100, size=(1000, 3))
Y = X @ A.T
plane_residuals = Y @ normal_canonical
assert np.allclose(plane_residuals, 0.0, atol=1e-12)
print("Prompt 1: Column Space Plane x + y - z = 0 Verified!")
```

**Textbook Grounding**:  
- Mathematics for Machine Learning (MML) §2.6 (Image / Column Space, Definition 2.23)
- Introduction to Applied Linear Algebra (VMLS) §5.1 (Range / Column Space, pp. 83–94)
- Erwin Kreyszig — Advanced Engineering Mathematics 10th Ed. §7.4 (Linear Independence and Vector Space, pp. 272–280)

---

### Prompt 2: How Does the 1D Null Line March into the Origin?
**Question**:  
What algebraic mechanism forces the one-dimensional line $t [1, 1, -1]^\top$ ($t \in \mathbb{R}$) to collapse completely to the zero vector $\mathbf{0}$? How does the Rank-Nullity Theorem guarantee that exactly one dimension of information is annihilated?

**Intuitive & Mathematical Proof**:  
1. *Kernel Definition*: The null space $\mathcal{N}(A)$ consists of all vectors $\mathbf{x} \in \mathbb{R}^3$ satisfying $A\mathbf{x} = \mathbf{0}$.
   $$A\mathbf{x} = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + x_3 \mathbf{a}_3 = \mathbf{0}$$
2. *Linear Combination Cancellation*: Since $\mathbf{a}_3 = \mathbf{a}_1 + \mathbf{a}_2$, setting $x_1 = 1, x_2 = 1, x_3 = -1$ yields:
   $$1\mathbf{a}_1 + 1\mathbf{a}_2 - 1\mathbf{a}_3 = \mathbf{a}_1 + \mathbf{a}_2 - (\mathbf{a}_1 + \mathbf{a}_2) = \mathbf{0}$$
   Therefore, $\mathbf{v}_{\text{null}} = \begin{bmatrix} 1 \\ 1 \\ -1 \end{bmatrix}$ is an element of $\mathcal{N}(A)$.
3. *Homogeneity of the Line*: For any scalar multiple $t \in \mathbb{R}$:
   $$A(t \mathbf{v}_{\text{null}}) = t (A \mathbf{v}_{\text{null}}) = t \mathbf{0} = \mathbf{0}$$
   The entire continuous infinite line $L = \{ t \mathbf{v}_{\text{null}} \mid t \in \mathbb{R} \}$ maps to the single point $(0, 0, 0)$.
4. *Rank-Nullity Conservation*: By the Fundamental Theorem of Linear Maps:
   $$\text{rank}(A) + \text{nullity}(A) = n = 3$$
   Since $\text{rank}(A) = 2$, $\text{nullity}(A) = 3 - 2 = 1$. The null space is strictly one-dimensional. Exactly one spatial degree of freedom is destroyed by the mapping.

**Executable NumPy Verification**:
```python
import numpy as np

A = np.array([
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 2.0]
])
v_null = np.array([1.0, 1.0, -1.0])

# 1. Null vector annihilation
assert np.allclose(A @ v_null, np.zeros(3))

# 2. Entire line collapses across 1,000 t values
t_vals = np.linspace(-1e5, 1e5, 1000)
line_pts = np.outer(t_vals, v_null)
collapsed = line_pts @ A.T
assert np.allclose(collapsed, 0.0, atol=1e-10)

# 3. SVD confirms exactly one zero singular value
s = np.linalg.svd(A, compute_uv=False)
rank = int(np.sum(s > 1e-12))
nullity = int(np.sum(s <= 1e-12))
assert rank == 2 and nullity == 1
assert rank + nullity == 3
print("Prompt 2: Null Space 1D Line Annihilation & Rank-Nullity Verified!")
```

**Textbook Grounding**:  
- Mathematics for Machine Learning (MML) §2.6 (Theorem 2.24: Rank-Nullity Theorem)
- Erwin Kreyszig — Advanced Engineering Mathematics 10th Ed. §7.4 (Fundamental Theorem of Linear Systems, pp. 280–287)
- B.S. Grewal — Higher Engineering Mathematics Ch. 2.7 (Rank of a Matrix and Consistency)

---

### Prompt 3: Why Do Solutions to $A\mathbf{x} = \mathbf{b}$ Form the Affine Subspace $\mathbf{x}_p + \mathcal{N}(A)$?
**Question**:  
When solving $A\mathbf{x} = \mathbf{b}$ for target $\mathbf{b} = [1, 2, 3]^\top$, why is the solution set an infinite line shifted away from the origin? Why does the least-norm solution $\mathbf{x}_{\text{LN}} = [0, 1, 1]^\top$ have minimum Euclidean length $\sqrt{2}$ and lie perpendicular to the null space?

**Intuitive & Mathematical Proof**:  
1. *Particular Solution*: Let $\mathbf{x}_p = [1, 2, 0]^\top$. Evaluating $A\mathbf{x}_p$:
   $$A\mathbf{x}_p = 1\mathbf{a}_1 + 2\mathbf{a}_2 + 0\mathbf{a}_3 = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix} + \begin{bmatrix} 0 \\ 2 \\ 2 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} = \mathbf{b}$$
2. *Superposition of General Solutions*: Let $\mathbf{x}$ be any solution satisfying $A\mathbf{x} = \mathbf{b}$. Then:
   $$A(\mathbf{x} - \mathbf{x}_p) = A\mathbf{x} - A\mathbf{x}_p = \mathbf{b} - \mathbf{b} = \mathbf{0} \implies \mathbf{x} - \mathbf{x}_p \in \mathcal{N}(A)$$
   Hence $\mathbf{x} = \mathbf{x}_p + \mathbf{x}_n$ where $\mathbf{x}_n \in \mathcal{N}(A) = \{ t \mathbf{v}_{\text{null}} \mid t \in \mathbb{R} \}$. The complete solution set is the affine line $\mathbf{x}_p + \mathcal{N}(A)$.
3. *Least-Norm Solution via Row Space*: By the Fundamental Theorem of Linear Algebra, $\mathbb{R}^3 = \mathcal{C}(A^\top) \oplus \mathcal{N}(A)$. Any vector $\mathbf{x}(t) = \mathbf{x}_p + t \mathbf{v}_{\text{null}}$ decomposes into row space and null space components:
   $$\|\mathbf{x}(t)\|^2 = \|\mathbf{x}_{\text{row}}\|^2 + \|\mathbf{x}_{\text{null}}(t)\|^2 \ge \|\mathbf{x}_{\text{row}}\|^2$$
   The minimum norm occurs when the null component is zero ($\mathbf{x} \in \mathcal{C}(A^\top)$), yielding the Moore-Penrose solution $\mathbf{x}_{\text{LN}} = A^+ \mathbf{b} = [0, 1, 1]^\top$.
4. *Orthogonality & Minimal Length*:
   $$\mathbf{x}_{\text{LN}} \cdot \mathbf{v}_{\text{null}} = 0(1) + 1(1) + 1(-1) = 0$$
   $$\|\mathbf{x}_{\text{LN}}\| = \sqrt{0^2 + 1^2 + 1^2} = \sqrt{2} \approx 1.4142$$
   For any other solution ($t \ne 0$), $\|\mathbf{x}_{\text{LN}} + t \mathbf{v}_{\text{null}}\|^2 = 2 + 3t^2 > 2$.

**Executable NumPy Verification**:
```python
import numpy as np

A = np.array([
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 2.0]
])
b = np.array([1.0, 2.0, 3.0])
x_p = np.array([1.0, 2.0, 0.0])
v_null = np.array([1.0, 1.0, -1.0])

# 1. Particular solution matches target
assert np.allclose(A @ x_p, b)

# 2. Entire affine line maps to b
for t in np.linspace(-50, 50, 21):
    x_t = x_p + t * v_null
    assert np.allclose(A @ x_t, b, atol=1e-12)

# 3. Least-norm solution via pseudoinverse
x_LN = np.linalg.pinv(A) @ b
assert np.allclose(x_LN, np.array([0.0, 1.0, 1.0]))
assert np.isclose(np.dot(x_LN, v_null), 0.0)
assert np.isclose(np.linalg.norm(x_LN), np.sqrt(2.0))

# 4. Strict minimizer across 1,000 points on line
t_sweep = np.linspace(-10, 10, 1001)
norms = [np.linalg.norm(x_LN + t * v_null) for t in t_sweep]
assert np.isclose(min(norms), np.sqrt(2.0), atol=1e-12)
print("Prompt 3: Affine Solution Line & Least-Norm Optimizer Verified!")
```

**Textbook Grounding**:  
- Mathematics for Machine Learning (MML) §2.6 (Affine Subspaces, Definition 2.25)
- Introduction to Applied Linear Algebra (VMLS) §10.2 (Least-Squares and Least-Norm Solutions)
- Erwin Kreyszig — Advanced Engineering Mathematics 10th Ed. §7.5 (Solutions of Linear Systems)

---

### Prompt 4: How Does Rank-Nullity Explain Overparameterized Loss Valleys & LoRA?
**Question**:  
Why do modern deep learning models ($P \gg N$ parameters) have connected valleys of zero loss, and how does Low-Rank Adaptation (LoRA) use column space and null space geometry to fine-tune billion-parameter LLMs efficiently?

**Intuitive & Mathematical Proof**:  
1. *Overparameterization & Wide Jacobians*: Consider a dataset of $N$ training examples and a neural network with $P$ parameters. The output Jacobian $J = \frac{\partial \hat{\mathbf{y}}}{\partial \mathbf{w}} \in \mathbb{R}^{N \times P}$ maps parameter perturbations to output changes: $\Delta \hat{\mathbf{y}} \approx J \Delta \mathbf{w}$.
2. *Massive Null Space*: Applying the Rank-Nullity Theorem to $J$:
   $$\dim(\mathcal{C}(J)) + \dim(\mathcal{N}(J)) = P \implies \text{nullity}(J) = P - \text{rank}(J) \ge P - N$$
   When $P = 7 \times 10^9$ (e.g., Llama-7B) and $N = 10^5$, $\text{nullity}(J) \approx 7 \times 10^9$. The null space represents millions of orthogonal parameter directions along which predictions and loss remain completely invariant ($J \Delta \mathbf{w} = \mathbf{0}$).
3. *LoRA (Low-Rank Adaptation)*: Instead of updating full weight matrix $W \in \mathbb{R}^{d \times k}$ (with $d \cdot k$ degrees of freedom), LoRA factorizes the update:
   $$\Delta W = B A, \qquad B \in \mathbb{R}^{d \times r}, \quad A \in \mathbb{R}^{r \times k}, \quad r \ll \min(d, k)$$
   - *Column Space Boundedness*: The column space $\mathcal{C}(\Delta W) \subseteq \mathcal{C}(B)$ has dimension at most $r$. The adaptation is confined to an $r$-dimensional manifold.
   - *Null Space Conservation*: The null space $\mathcal{N}(\Delta W)$ has dimension $k - r$. For inputs orthogonal to $A^\top$, $\Delta W \mathbf{x} = \mathbf{0}$. The adapted model leaves vast regions of pre-trained feature space completely undisturbed, preventing catastrophic forgetting.

**Executable NumPy Verification**:
```python
import numpy as np

# 1. Overparameterized Jacobian null space demonstration
rng = np.random.default_rng(2026)
N_samples = 8
P_params = 32
J = rng.standard_normal((N_samples, P_params))
rank_J = np.linalg.matrix_rank(J)
nullity_J = P_params - rank_J
assert rank_J == N_samples
assert nullity_J == P_params - N_samples  # 24 null directions

# Parameter perturbation in null space produces ZERO output change
_, _, Vt = np.linalg.svd(J)
v_null = Vt[rank_J]  # First right singular vector in kernel
output_delta = J @ v_null
assert np.allclose(output_delta, 0.0, atol=1e-12)

# 2. LoRA Low-Rank Decomposition
d, k, r = 128, 128, 4
B = rng.standard_normal((d, r))
A_lora = rng.standard_normal((r, k))
Delta_W = B @ A_lora

rank_lora = np.linalg.matrix_rank(Delta_W)
assert rank_lora == r  # Rank is bounded by r=4 << 128
nullity_lora = k - rank_lora
assert nullity_lora == 124  # 124 dimensions preserved untouched!
print("Prompt 4: Overparameterized Null Space & LoRA Geometry Verified!")
```

**Textbook Grounding**:  
- Mathematics for Machine Learning (MML) §2.6 & §10.3 (Rank, Nullity, and Low-Rank Approximation)
- Edward J. Hu et al. (2021) — *LoRA: Low-Rank Adaptation of Large Language Models*
- Erwin Kreyszig — Advanced Engineering Mathematics 10th Ed. §7.4 (Fundamental Theorem of Linear Systems)
