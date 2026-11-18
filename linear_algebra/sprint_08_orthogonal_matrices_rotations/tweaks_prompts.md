# Part 4: Interactive Parameter Tweaks & Intuition Stress-Test Prompts

## 1. Interactive Parameter Tweaks Guide

The ManimCE animation script (`scene_orthogonal_matrices_rotations.py`) exposes orthogonal rotation matrices, shear stress matrices, reflection angles, Givens/Householder test vectors, and numerical drift parameters as module-level NumPy constants. Students and engineers can modify these constants to probe alternate Pythagorean triples, higher-angle shears, different mirror pairs, and floating-point conditioning thresholds.

### Parameter Configuration Matrix

| Parameter Name | Default Value | Recommended Test Values | Pedagogical Objective |
|---|---|---|---|
| `Q_PYTH` | `[[0.6, -0.8], [0.8, 0.6]]` | `[[5/13, -12/13], [12/13, 5/13]]`, `R(pi/3)` | Test exact rational Pythagorean triples in $SO(2)$ preserving $S^1$ and $90^\circ$ angles. |
| `M_SHEAR` | `[[1.0, 1.2], [0.0, 1.0]]` | `[[1.0, 2.0], [0.0, 1.0]]`, `np.diag([2.0, 0.5])` | Probe non-orthogonal $\det = 1$ area-preserving operators that warp $S^1$ into ellipses. |
| `ALPHA_1`, `ALPHA_2` | `math.pi / 12.0`, `math.pi / 4.0` | `0.0, math.pi / 6.0`, `math.pi / 8.0, math.pi / 2.0` | Observe Cartan–Dieudonné angle doubling $H(\alpha_2)H(\alpha_1) = R(2(\alpha_2 - \alpha_1))$. |
| `TEST_W` | `np.array([0.9, 0.8])` | `[1.2, -0.5]`, `[0.0, 1.5]`, `[-1.0, 0.8]` | Compare norm and angle distortion under $M_{\text{shear}}$ vs rigid isometry under $Q_{\text{pyth}}$. |
| `X_GIVENS` | `np.array([3.0, 2.0, 4.0])` | `[0.0, 2.0, 0.0]`, `[5.0, -1.0, 12.0]` | Trigger Givens $(0, 2)$ coordinate zeroing and test the $(0, 0)$ zero-pair `hypot` guard. |
| `X_HH` / `X_CANCEL` | `[1.0, 2.0, 2.0]` / `[1.0, 1e-9]` | `[1.0, 1e-12]`, `[-1.0, 1e-15]`, `[0.0, 0.0]` | Stress-test Householder $+\operatorname{sgn}(x_1)$ cancellation avoidance and $\mathbf{x} = \mathbf{0}$ guard. |
| `Q_DRIFT_N` | `1000` | `500`, `2000`, `10000` | Measure accumulated $\mathcal{O}(n\epsilon_{\text{mach}})$ orthogonality drift vs exponential blowup. |

### How to Modify and Re-Render

1. **Edit constants in `scene_orthogonal_matrices_rotations.py`**:
   ```python
   # Configuration Constants
   Q_PYTH = np.array([[0.6, -0.8], [0.8, 0.6]])
   M_SHEAR = np.array([[1.0, 1.2], [0.0, 1.0]])
   ALPHA_1 = math.pi / 12.0
   ALPHA_2 = math.pi / 4.0
   X_GIVENS = np.array([3.0, 2.0, 4.0])
   X_HH = np.array([1.0, 2.0, 2.0])
   X_CANCEL = np.array([1.0, 1e-9])
   Q_DRIFT_N = 1000
   ```
2. **Execute fast low-quality preview render**:
   ```bash
   .venv/bin/manim -pql linear_algebra/sprint_08_orthogonal_matrices_rotations/scene_orthogonal_matrices_rotations.py OrthogonalRotationsScene
   ```
3. **Execute single act preview**:
   ```bash
   .venv/bin/manim -ql linear_algebra/sprint_08_orthogonal_matrices_rotations/scene_orthogonal_matrices_rotations.py Act1SpatialPuzzle
   .venv/bin/manim -ql linear_algebra/sprint_08_orthogonal_matrices_rotations/scene_orthogonal_matrices_rotations.py Act2DynamicDeformation
   .venv/bin/manim -ql linear_algebra/sprint_08_orthogonal_matrices_rotations/scene_orthogonal_matrices_rotations.py Act3AlgebraicMorph
   .venv/bin/manim -ql linear_algebra/sprint_08_orthogonal_matrices_rotations/scene_orthogonal_matrices_rotations.py Act4DegenerateStressTest
   ```

---

## 2. Intuition Stress-Test Prompts

### Prompt 1: Rigid Isometry vs Area-Preserving Shear ($\det = 1$ Is Necessary, Not Sufficient)
- **Question**:  
  Why does $M_{\text{shear}} = \begin{bmatrix} 1 & 1.2 \\ 0 & 1 \end{bmatrix}$ satisfy $\det(M_{\text{shear}}) = 1$ (preserving signed 2D area) while failing to preserve the unit circle, vector lengths, and pairwise angles—and how do $Q^\top Q = I$ and singular values $\sigma_1 = \sigma_2 = 1$ distinguish true rigid motions like $Q_{\text{pyth}} = \begin{bmatrix} 0.6 & -0.8 \\ 0.8 & 0.6 \end{bmatrix}$?
- **Intuitive & Mathematical Proof**:  
  A matrix $Q \in \mathbb{R}^{n \times n}$ preserves all Euclidean lengths and angles if and only if $Q^\top Q = I_n$, because:
  $$\|Q\mathbf{x}\|_2^2 = \mathbf{x}^\top (Q^\top Q)\mathbf{x}, \qquad (Q\mathbf{x})^\top(Q\mathbf{y}) = \mathbf{x}^\top (Q^\top Q)\mathbf{y}$$
  Taking determinants gives $1 = \det(Q^\top Q) = (\det Q)^2 \implies \det(Q) = \pm 1$. However, the converse is false: $\det(A) = 1$ constrains only the **product** of singular values $\sigma_1 \sigma_2 = |\det A| = 1$, leaving one axis free to stretch ($\sigma_1 > 1$) while the orthogonal axis compresses ($\sigma_2 = 1/\sigma_1 < 1$).
  - For $M_{\text{shear}} = \begin{bmatrix} 1 & 1.2 \\ 0 & 1 \end{bmatrix}$: $\det(M_{\text{shear}}) = 1$, yet $M_{\text{shear}}\mathbf{e}_1 = [1, 0]^\top$ (norm $1$) and $M_{\text{shear}}\mathbf{e}_2 = [1.2, 1]^\top$ (norm $\sqrt{2.44} \approx 1.5620$). The angle between them collapses from $90^\circ$ to:
    $$\omega = \arccos\left(\frac{1.2}{\sqrt{2.44}}\right) \approx 39.81^\circ$$
    and its singular values split to $\sigma_1 \approx 1.7662$ and $\sigma_2 \approx 0.5662$, warping $S^1$ into a tilted ellipse.
  - For $Q_{\text{pyth}} = \begin{bmatrix} 0.6 & -0.8 \\ 0.8 & 0.6 \end{bmatrix}$: $Q_{\text{pyth}}^\top Q_{\text{pyth}} = I_2 \implies \sigma_1 = \sigma_2 = 1$ and $\kappa_2(Q_{\text{pyth}}) = 1$, locking $S^1$ and every pairwise angle rigidly.
- **NumPy Verification**:
  ```python
  import math
  import numpy as np

  Q = np.array([[0.6, -0.8], [0.8, 0.6]])
  M = np.array([[1.0, 1.2], [0.0, 1.0]])

  assert np.isclose(np.linalg.det(Q), 1.0)
  assert np.isclose(np.linalg.det(M), 1.0)
  assert np.allclose(Q.T @ Q, np.eye(2), atol=1e-15)
  assert not np.allclose(M.T @ M, np.eye(2), atol=1e-6)
  assert np.allclose(np.linalg.svd(Q, compute_uv=False), [1.0, 1.0], atol=1e-15)

  s_M = np.linalg.svd(M, compute_uv=False)
  assert s_M[0] > 1.7 and s_M[1] < 0.6
  e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
  cos_M = np.dot(M @ e1, M @ e2) / (np.linalg.norm(M @ e1) * np.linalg.norm(M @ e2))
  angle_deg = math.degrees(math.acos(cos_M))
  assert math.isclose(angle_deg, 39.8055710922652, rel_tol=1e-10)
  print(f"Prompt 1 Verification PASSED: M_shear det=1 yet angle={angle_deg:.2f} deg, svd={s_M}.")
  ```
- **Textbook Grounding**: MML §3.4 (Def. 3.8, eqs. 3.29–3.32); VMLS §10.4; Kreyszig §8.3 (Thms 2–4).

---

### Prompt 2: Cartan–Dieudonné Composition ($H_2 H_1 = R(2\Delta\alpha)$) & 3D $SO(3)$ Non-Commutativity
- **Question**:  
  Why do two consecutive 2D reflections $H(\alpha_2)H(\alpha_1)$ always yield a proper rotation $R(2(\alpha_2 - \alpha_1)) \in SO(2)$, why does swapping their order reverse the rotation angle, and why do 3D rotations $R_1(\alpha), R_2(\beta) \in SO(3)$ fail to commute even about the same origin?
- **Intuitive & Mathematical Proof**:  
  A reflection across a line through the origin at angle $\alpha$ is $H(\alpha) = \begin{bmatrix} \cos 2\alpha & \sin 2\alpha \\ \sin 2\alpha & -\cos 2\alpha \end{bmatrix}$ with $\det H(\alpha) = -1$. Multiplying two reflections yields $\det(H_2 H_1) = (-1)(-1) = +1$ and:
  $$H(\alpha_2)H(\alpha_1) = \begin{bmatrix} \cos 2\alpha_2\cos 2\alpha_1 + \sin 2\alpha_2\sin 2\alpha_1 & \cos 2\alpha_2\sin 2\alpha_1 - \sin 2\alpha_2\cos 2\alpha_1 \\ \sin 2\alpha_2\cos 2\alpha_1 - \cos 2\alpha_2\sin 2\alpha_1 & \sin 2\alpha_2\sin 2\alpha_1 + \cos 2\alpha_2\cos 2\alpha_1 \end{bmatrix} = \begin{bmatrix} \cos 2(\alpha_2-\alpha_1) & -\sin 2(\alpha_2-\alpha_1) \\ \sin 2(\alpha_2-\alpha_1) & \cos 2(\alpha_2-\alpha_1) \end{bmatrix} = R(2\Delta\alpha)$$
  Reversing the reflection order replaces $\Delta\alpha = \alpha_2 - \alpha_1$ with $-\Delta\alpha$, producing $R(-2\Delta\alpha) = R(2\Delta\alpha)^\top$ (non-commutative unless $\Delta\alpha \in \frac{\pi}{2}\mathbb{Z}$).
  
  In $\mathbb{R}^2$, all proper rotations share the single normal axis orthogonal to the plane, so $R(\theta_1)R(\theta_2) = R(\theta_1 + \theta_2) = R(\theta_2)R(\theta_1)$ (Abelian). In $\mathbb{R}^3$, distinct rotation axes transform under earlier rotations. Taking $R_1(\pi/2)$ (around $x$) and $R_2(\pi/2)$ (around $y$) acting on $\mathbf{e}_1 = [1, 0, 0]^\top$:
  - $R_1(\pi/2)R_2(\pi/2)\mathbf{e}_1 = R_1(\pi/2)[0, 0, -1]^\top = [0, 1, 0]^\top$
  - $R_2(\pi/2)R_1(\pi/2)\mathbf{e}_1 = R_2(\pi/2)[1, 0, 0]^\top = [0, 0, -1]^\top$
  Their outputs differ by $\|[0, 1, 0]^\top - [0, 0, -1]^\top\|_2 = \sqrt{2}$.
- **NumPy Verification**:
  ```python
  import math
  import numpy as np

  def H2d(alpha: float) -> np.ndarray:
      return np.array([
          [math.cos(2.0 * alpha), math.sin(2.0 * alpha)],
          [math.sin(2.0 * alpha), -math.cos(2.0 * alpha)],
      ])

  a1, a2 = math.pi / 12.0, math.pi / 4.0  # 15 deg, 45 deg -> 2*(45-15) = 60 deg
  da = 2.0 * (a2 - a1)
  R_exp = np.array([[math.cos(da), -math.sin(da)], [math.sin(da), math.cos(da)]])
  assert np.allclose(H2d(a2) @ H2d(a1), R_exp, atol=1e-15)
  assert np.allclose(H2d(a1) @ H2d(a2), R_exp.T, atol=1e-15)

  R1 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
  R2 = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
  e1 = np.array([1.0, 0.0, 0.0])
  assert np.allclose(R1 @ R2 @ e1, [0.0, 1.0, 0.0], atol=1e-15)
  assert np.allclose(R2 @ R1 @ e1, [0.0, 0.0, -1.0], atol=1e-15)
  assert np.isclose(np.linalg.norm((R1 @ R2 - R2 @ R1) @ e1), math.sqrt(2.0), atol=1e-15)
  print("Prompt 2 Verification PASSED: H2@H1 == R(2*Delta_alpha) and SO(3) non-commutativity confirmed.")
  ```
- **Textbook Grounding**: MML §3.9.1–3.9.4 (eqs. 3.76–3.79); VMLS §7.1; Kreyszig §8.3.

---

### Prompt 3: Givens vs. Householder Elimination & Catastrophic Cancellation at $\mathbf{x} = [1, 10^{-9}]^\top$
- **Question**:  
  How do Givens rotations $G(i, j, \theta) \in SO(n)$ and Householder reflectors $H \in O(n)\setminus SO(n)$ trade off sparsity vs. bulk subcolumn elimination, why is a `hypot` guard required at $x_i = x_j = 0$, and why does naive $\mathbf{v} = \mathbf{x} - \|\mathbf{x}\|_2\mathbf{e}_1$ fail catastrophically at $\mathbf{x} = [1, 10^{-9}]^\top$ while $\mathbf{v} = \mathbf{x} + \operatorname{sgn}(x_1)\|\mathbf{x}\|_2\mathbf{e}_1$ preserves full precision?
- **Intuitive & Mathematical Proof**:  
  - **Givens $G(i, j, \theta)$** embeds a $2\times 2$ rotation in rows/columns $(i, j)$ with $r = \operatorname{hypot}(x_i, x_j)$, $c = x_i / r$, $s = -x_j / r$. It modifies only 2 rows ($\mathcal{O}(n)$ work per zero), preserving sparsity in banded/sparse matrices; checking $r = 0 \implies G = I_n$ prevents $0/0$ `NaN`, and `math.hypot` prevents overflow at $10^{200}$.
  - **Householder $H = I - 2\mathbf{v}\mathbf{v}^\top / \|\mathbf{v}\|_2^2$** zeros all $n - 1$ entries below entry 1 in a single rank-1 update ($2\times$ fewer FLOPs than Givens for dense $QR$).
  - **Cancellation at $\mathbf{x} = [1.0, 10^{-9}]^\top$**: In IEEE-754 `float64` ($\epsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$), $1 + 10^{-18}$ rounds to `1.0`, so $\|\mathbf{x}\|_2 = 1.0$. The naive difference $\mathbf{v}_{\text{bad}} = \mathbf{x} - \|\mathbf{x}\|_2\mathbf{e}_1 = [0.0, 10^{-9}]^\top$ loses all significant bits of $v_1$ and reflects vertically along $\mathbf{e}_2$, yielding $H_{\text{bad}}\mathbf{x} = [1.0, -10^{-9}]^\top$ (failing to zero $x_2$). Using $\mathbf{v}_{\text{good}} = \mathbf{x} + \operatorname{sgn}(x_1)\|\mathbf{x}\|_2\mathbf{e}_1 = [2.0, 10^{-9}]^\top$ adds same-sign magnitudes so $\|\mathbf{v}_{\text{good}}\|_2 \ge \|\mathbf{x}\|_2$, giving $H_{\text{good}}\mathbf{x} = [-1.0, 0.0]^\top$ with $|(H_{\text{good}}\mathbf{x})_2| < 10^{-15}$.
- **NumPy Verification**:
  ```python
  import math
  import numpy as np

  def givens_2d_safe(xi: float, xj: float) -> np.ndarray:
      r = math.hypot(xi, xj)
      if r == 0.0:
          return np.eye(2)
      c, s = xi / r, -xj / r
      return np.array([[c, -s], [s, c]])

  assert np.allclose(givens_2d_safe(0.0, 0.0), np.eye(2))
  assert np.allclose(givens_2d_safe(3.0, 4.0) @ np.array([3.0, 4.0]), [5.0, 0.0], atol=1e-14)

  x = np.array([1.0, 1e-9])
  nx = np.linalg.norm(x)
  v_bad = x - np.array([nx, 0.0])
  H_bad = np.eye(2) - 2.0 * np.outer(v_bad, v_bad) / np.dot(v_bad, v_bad)
  assert abs((H_bad @ x)[1]) > 1e-10  # Naive fails completely

  sigma = 1.0 if x[0] >= 0.0 else -1.0
  v_good = x + sigma * nx * np.array([1.0, 0.0])
  H_good = np.eye(2) - 2.0 * np.outer(v_good, v_good) / np.dot(v_good, v_good)
  assert abs((H_good @ x)[1]) < 1e-15  # Sign-aware succeeds
  print("Prompt 3 Verification PASSED: Givens zero-pair guard and Householder sign-aware stability verified.")
  ```
- **Textbook Grounding**: MML §3.9.3 (Def. 3.11, eqs. 3.80–3.81); Kreyszig §20.9 (eqs. 2, 4, 7–11).

---

### Prompt 4: ML & GATE Bridge — Spectral Unit Circle ($|\lambda_i| = 1$), $Q^{1000}$ Drift, RoPE & Euler Axis
- **Question**:  
  Why do all eigenvalues of $Q \in O(n)$ lie on the unit circle ($|\lambda_i| = 1$) and why does this bound $Q^{1000}$ drift and preserve gradient norms across $100$-layer networks, power Rotary Position Embeddings (RoPE), and yield GATE $SO(3)$ trace/eigenvalue shortcuts?
- **Intuitive & Mathematical Proof**:  
  If $Q\mathbf{v} = \lambda\mathbf{v}$ ($\mathbf{v} \ne \mathbf{0}$, possibly complex), then $\|\mathbf{v}\|_2^2 = \mathbf{v}^* Q^\top Q \mathbf{v} = \|Q\mathbf{v}\|_2^2 = |\lambda|^2 \|\mathbf{v}\|_2^2 \implies |\lambda| = 1$. Since all singular values also equal $1$, iterating $Q^{1000}$ introduces zero exponential growth or decay—only linear $\mathcal{O}(n\epsilon_{\text{mach}}) \approx 10^{-13}$ rounding drift in `float64`.
  1. **ML — Orthogonal Init & RoPE**: In a 100-layer linear stack $\mathbf{h}_{100} = W_{100}\cdots W_1\mathbf{x}$ with $W_\ell \in O(d)$, $\|\mathbf{h}_{100}\|_2 = \|\mathbf{x}\|_2$. In Transformer RoPE, $(R(m\theta)\mathbf{q})^\top(R(n\theta)\mathbf{k}) = \mathbf{q}^\top R(m\theta)^\top R(n\theta)\mathbf{k} = \mathbf{q}^\top R((n-m)\theta)\mathbf{k}$.
  2. **GATE Shortcut — $SO(3)$ Euler Axis & Trace**: For any $R \in SO(3)$, $\det(R) = +1$ and complex eigenvalues appear in conjugate pairs $e^{\pm i\theta}$, forcing the third real eigenvalue to be $\lambda_1 = +1$ (Euler's rotation axis), so $\operatorname{Tr}(R) = 1 + e^{i\theta} + e^{-i\theta} = 1 + 2\cos\theta$.
- **NumPy Verification**:
  ```python
  import math
  import numpy as np

  Q = np.array([[math.cos(0.37), -math.sin(0.37)], [math.sin(0.37), math.cos(0.37)]])
  P = np.eye(2)
  for _ in range(1000):
      P = P @ Q
  assert np.linalg.norm(P.T @ P - np.eye(2), "fro") < 1e-12

  def R2(a: float) -> np.ndarray:
      return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])

  q, k = np.array([1.2, -0.7]), np.array([-0.5, 2.1])
  assert math.isclose((R2(17 * 0.25) @ q) @ (R2(42 * 0.25) @ k), q @ (R2((42 - 17) * 0.25) @ k), rel_tol=1e-13)

  theta_3d = 0.85
  R3 = np.array([
      [1.0, 0.0, 0.0],
      [0.0, math.cos(theta_3d), -math.sin(theta_3d)],
      [0.0, math.sin(theta_3d), math.cos(theta_3d)],
  ])
  eigs = np.linalg.eigvals(R3)
  assert np.allclose(np.abs(eigs), 1.0, atol=1e-14)
  assert math.isclose(np.trace(R3), 1.0 + 2.0 * math.cos(theta_3d), rel_tol=1e-14)
  print("Prompt 4 Verification PASSED: Q^1000 drift, RoPE relative invariance, and SO(3) trace verified.")
  ```
- **Textbook Grounding**: MML §3.4 & §3.9; VMLS §10.4; Kreyszig §8.3 (Thm 4, Thm 5, Ex. 5, Prob. 12).
