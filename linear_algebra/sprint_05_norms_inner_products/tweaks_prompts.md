# Part 4: Interactive Parameter Tweaks & Intuition Stress-Test Prompts

## 1. Interactive Parameter Tweaks Guide

All canonical vectors, norms, and inner-product matrices are declared at the top of `linear_algebra/sprint_05_norms_inner_products/scene_norms_inner_products.py` and verified by Rule 6 NumPy pre-flight assertions.

### Parameter Configuration Matrix

| Constant | Default Value | Visual Role | Safe Exploration Range |
|---|---|---|---|
| `X_PUZZLE` | `np.array([3.0, 4.0])` | Act 1 primary 3-4-5 vector ($L_1=7, L_2=5, L_\infty=4$) | Any 2D vector with $\|x\|_2 \le 5$ inside `NumberPlane` |
| `Y_PUZZLE` | `np.array([2.0, 1.0])` | Act 1 ruler vector ($\mathbf{x}^\top\mathbf{y}=10$, $\operatorname{proj}=[4,2]$) | Any nonzero vector inside `[-5, 5] x [-3, 3]` |
| `X_VEC`, `Y_VEC` | `[1.0, 2.0]`, `[2.0, 1.0]` | Canonical Cauchy-Schwarz pair ($\mathbf{x}^\top\mathbf{y}=4 < 5$, $\cos\omega=0.8$) | Any nonzero 2D pair (update Rule 6 asserts if changed) |
| `Y_RULER` | `np.array([1.6, 1.2])` | Act 2B/3A tilted ruler ($\|\mathbf{y}\|_2 = 2.0$, $36.87^\circ$) | Any nonzero 2D vector |
| `X_SWEEP_NORM` | `2.5` | Radius of sweeping vector $\mathbf{x}(\omega)$ in Acts 2B–3A | `1.5` to `3.2` (keeps tip inside HUD-safe zone) |
| `A_SPD` | `np.array([[2.0, 1.0], [1.0, 2.0]])` | Act 3B SPD matrix defining $\langle\mathbf{x},\mathbf{y}\rangle_A$ ellipse | Any $2 \times 2$ matrix with $A = A^\top$ and $\lambda_{\min}(A) > 0$ |
| `U_A_ORTH`, `V_A_ORTH` | `[1.0, 0.0]`, `[-1.0, 2.0]` | $A$-orthogonal pair ($\mathbf{u}^\top A\mathbf{v}=0$, $\mathbf{u}^\top\mathbf{v}=-1$) | Any pair satisfying $\mathbf{u}^\top A\mathbf{v}=0$ |
| `QUASI_P` | `0.5` | Act 4 non-convex quasi-norm exponent | `0.25` to `0.85` (all strictly $< 1$) |
| `A_INDEF`, `X_INDEF` | `diag(1.0, -1.0)`, `[1.0, 2.0]` | Act 4 indefinite bilinear form ($\mathbf{x}^\top A\mathbf{x} = -3 < 0$) | Any symmetric matrix with $\det(A) < 0$ |

### How to Modify and Re-Render
1. Edit the target constant at the top of `linear_algebra/sprint_05_norms_inner_products/scene_norms_inner_products.py` (updating matching Rule 6 assertions if you alter canonical numbers).
2. Run a fast low-resolution preview (`480p15`) for rapid iteration:
   ```bash
   .venv/bin/manim -pql linear_algebra/sprint_05_norms_inner_products/scene_norms_inner_products.py Act2DynamicDeformation
   ```
3. Render the full composite scene at production quality (`1080p60`, headless `-qh` or preview `-pqh`):
   ```bash
   .venv/bin/manim -qh linear_algebra/sprint_05_norms_inner_products/scene_norms_inner_products.py NormsInnerProductsScene
   ```

---

## 2. Intuition Stress-Test Prompts

### Prompt 1: Nested $L_p$ Unit Balls, Norm Equivalence & Non-Convexity at $p=0.5$
- **Question**: Why are the $L_1, L_2, L_\infty$ unit balls strictly nested ($B_1 \subseteq B_2 \subseteq B_\infty$) while touching at the axis points $(\pm 1, 0), (0, \pm 1)$, and why does dropping to $p=0.5$ break the triangle inequality?
- **Intuitive & Mathematical Proof**: For any $\mathbf{x} \in \mathbb{R}^n$, $|x_i| \le \|\mathbf{x}\|_\infty$ and $\|\mathbf{x}\|_p$ is monotonically non-increasing in $p$ because raising ratios $|x_i|/\|\mathbf{x}\|_\infty \le 1$ to a larger power $p$ shrinks every term. Hence $\|\mathbf{x}\|_\infty \le \|\mathbf{x}\|_2 \le \|\mathbf{x}\|_1 \le \sqrt{n}\|\mathbf{x}\|_2 \le n\|\mathbf{x}\|_\infty$, which reverses containment on unit balls $\{\mathbf{x}:\|\mathbf{x}\|_p \le 1\}$: $B_1 \subseteq B_2 \subseteq B_\infty$. Along the diagonal $y=x$, the boundary point is $(2^{-1/p}, 2^{-1/p})$, which expands monotonically from $(0.5, 0.5)$ at $p=1$ to $(1/\sqrt{2}, 1/\sqrt{2})$ at $p=2$ to $(1, 1)$ at $p=\infty$, while $1$-sparse axis points $\pm\mathbf{e}_i$ have $\|\pm\mathbf{e}_i\|_p = 1$ for all $p$. For $p=0.5 < 1$, the midpoint $\frac{1}{2}(\mathbf{e}_1 + \mathbf{e}_2) = [0.5, 0.5]^\top$ has $\|\frac{1}{2}(\mathbf{e}_1+\mathbf{e}_2)\|_{0.5} = (2\cdot 0.5^{0.5})^2 = 2 > 1$, so the unit ball is concave and $\|\mathbf{e}_1+\mathbf{e}_2\|_{0.5} = 4 > 1 + 1$ violates the triangle inequality.
- **NumPy Verification**:
  ```python
  import numpy as np

  rng = np.random.default_rng(501)
  X = rng.normal(size=(1000, 6))
  n = X.shape[1]
  l_inf = np.linalg.norm(X, ord=np.inf, axis=1)
  l_2 = np.linalg.norm(X, ord=2, axis=1)
  l_1 = np.linalg.norm(X, ord=1, axis=1)
  assert np.all(l_inf <= l_2 + 1e-12)
  assert np.all(l_2 <= l_1 + 1e-12)
  assert np.all(l_1 <= np.sqrt(n) * l_2 + 1e-12)
  assert np.all(l_2 <= np.sqrt(n) * l_inf + 1e-12)

  # Diagonal boundary points in R^2: (2^(-1/p), 2^(-1/p))
  for p in [1.0, 2.0, 10.0, 100.0]:
      diag_pt = np.array([2.0 ** (-1.0 / p), 2.0 ** (-1.0 / p)])
      assert np.isclose(np.linalg.norm(diag_pt, ord=p), 1.0)

  # Non-convexity / triangle inequality failure at p = 0.5
  u, v = np.array([1.0, 0.0]), np.array([0.0, 1.0])
  norm_half = lambda z: float(np.sum(np.abs(z) ** 0.5) ** 2.0)
  assert np.isclose(norm_half(u + v), 4.0) and norm_half(u + v) > norm_half(u) + norm_half(v)
  ```
- **Textbook Grounding**: MML §3.1 (Def. 3.1, Ex. 3.1–3.2, Fig. 3.3); VMLS Ex. 3.5.

---

### Prompt 2: Cauchy-Schwarz Proof, Collinear Saturation & Parallelogram Law Test
- **Question**: How does the Cauchy-Schwarz inequality guarantee that angles $\omega \in [0, \pi]$ are well-defined, when does equality hold, and why can $L_1$ and $L_\infty$ never come from an inner product?
- **Intuitive & Mathematical Proof**: For nonzero $\mathbf{x}, \mathbf{y}$ in any inner-product space, set $\alpha = \|\mathbf{x}\|, \beta = \|\mathbf{y}\|$ and expand $0 \le \|\beta\mathbf{x} \mp \alpha\mathbf{y}\|^2 = 2\alpha^2\beta^2 \mp 2\alpha\beta\langle\mathbf{x},\mathbf{y}\rangle$, yielding $|\langle\mathbf{x},\mathbf{y}\rangle| \le \|\mathbf{x}\|\|\mathbf{y}\|$ with equality iff $\beta\mathbf{x} = \pm\alpha\mathbf{y}$ (collinear). Thus $\cos\omega = \langle\mathbf{x},\mathbf{y}\rangle / (\|\mathbf{x}\|\|\mathbf{y}\|) \in [-1, 1]$ always yields a unique angle $\omega \in [0, \pi]$, and $\mathbf{x}^\top\mathbf{y}$ measures the signed length of $\operatorname{proj}_{\mathbf{y}}(\mathbf{x})$ times $\|\mathbf{y}\|_2$. Moreover, any norm induced by an inner product obeys the Jordan–von Neumann parallelogram law $\|\mathbf{x}+\mathbf{y}\|^2 + \|\mathbf{x}-\mathbf{y}\|^2 = 2(\|\mathbf{x}\|^2 + \|\mathbf{y}\|^2)$; testing $\mathbf{e}_1, \mathbf{e}_2$ gives $8 \ne 4$ for $L_1$ and $2 \ne 4$ for $L_\infty$, disqualifying both.
- **NumPy Verification**:
  ```python
  import numpy as np

  x = np.array([1.0, 2.0])
  y = np.array([2.0, 1.0])
  proj = (np.dot(x, y) / np.dot(y, y)) * y
  resid = x - proj
  assert np.allclose(proj, np.array([1.6, 0.8]))
  assert np.allclose(resid, np.array([-0.6, 1.2]))
  assert np.isclose(np.dot(resid, y), 0.0)
  assert abs(np.dot(x, y)) == 4.0 < np.linalg.norm(x) * np.linalg.norm(y)  # 4 < 5

  # Collinear saturation
  assert np.isclose(np.dot(x, 3.0 * x), np.linalg.norm(x) * np.linalg.norm(3.0 * x))
  assert np.isclose(np.dot(x, -3.0 * x), -np.linalg.norm(x) * np.linalg.norm(-3.0 * x))

  # Parallelogram law: L2 passes, L1 and L_inf fail
  e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
  assert np.isclose(np.linalg.norm(e1 + e2, 2)**2 + np.linalg.norm(e1 - e2, 2)**2,
                    2 * (np.linalg.norm(e1, 2)**2 + np.linalg.norm(e2, 2)**2))
  assert not np.isclose(np.linalg.norm(e1 + e2, 1)**2 + np.linalg.norm(e1 - e2, 1)**2,
                        2 * (np.linalg.norm(e1, 1)**2 + np.linalg.norm(e2, 1)**2))
  assert not np.isclose(np.linalg.norm(e1 + e2, np.inf)**2 + np.linalg.norm(e1 - e2, np.inf)**2,
                        2 * (np.linalg.norm(e1, np.inf)**2 + np.linalg.norm(e2, np.inf)**2))
  ```
- **Textbook Grounding**: MML §3.3 (Eq. 3.17, p. 75), §3.4 (Eqs. 3.24–3.25), §3.8.1 (Eq. 3.44); VMLS §3.4 & Ex. 3.4b.

---

### Prompt 3 (ML Bridge): RMS Norm, Z-Score Standardization, Chebyshev Bound & $L_1$/$L_2$ Geometry
- **Question**: How do RMS norms, z-score standardization, Chebyshev's inequality, and cosine similarity tie into neural-network normalization and Lasso vs. Ridge regularization?
- **Intuitive & Mathematical Proof**: $\operatorname{rms}(\mathbf{x}) = \|\mathbf{x}\|_2/\sqrt{n}$ satisfies $\operatorname{rms}(\mathbf{x})^2 = \operatorname{avg}(\mathbf{x})^2 + \operatorname{std}(\mathbf{x})^2$. Z-score standardization $\mathbf{z} = (\mathbf{x}-\operatorname{avg}(\mathbf{x})\mathbf{1})/\operatorname{std}(\mathbf{x})$ centers and scales every non-constant vector onto the sphere $\|\mathbf{z}\|_2 = \sqrt{n}$ ($\operatorname{rms}(\mathbf{z}) = 1$), while **RMSNorm** skips centering and projects directly to $\|\mathbf{x}/\operatorname{rms}(\mathbf{x})\|_2 = \sqrt{n}$. Counting entries with $|x_i| \ge a$ obeys $k a^2 \le \sum_{i=1}^n x_i^2 = n\,\operatorname{rms}(\mathbf{x})^2 \implies k/n \le (\operatorname{rms}(\mathbf{x})/a)^2$. Finally, on $L_2$-normalized embeddings $\hat{\mathbf{u}}, \hat{\mathbf{v}}$, $\|\hat{\mathbf{u}}-\hat{\mathbf{v}}\|_2^2 = 2(1 - \cos\omega)$, whereas constraining $\boldsymbol{\beta}$ to the sharp $L_1$ diamond snaps solutions onto sparse coordinate vertices ($\beta_2 = 0$).
- **NumPy Verification**:
  ```python
  import numpy as np

  x = np.array([1.0, -2.0, 3.0, 2.0])
  avg_x = np.mean(x)
  std_x = np.linalg.norm(x - avg_x) / np.sqrt(len(x))
  assert np.isclose(std_x, np.sqrt(3.5))  # exact sqrt(3.5) ~ 1.870829 (VMLS p.53 prints 1.872 typo)
  assert np.isclose(np.linalg.norm(x)**2 / len(x), avg_x**2 + std_x**2)

  z = (x - avg_x) / std_x
  assert np.isclose(np.mean(z), 0.0)
  assert np.isclose(np.linalg.norm(z) / np.sqrt(len(z)), 1.0)
  assert np.isclose(np.linalg.norm(z), np.sqrt(len(z)))

  # Chebyshev bound: fraction with |x_i| >= a is <= (rms(x)/a)^2
  a = 2.5
  rms_x = np.linalg.norm(x) / np.sqrt(len(x))
  frac = np.mean(np.abs(x) >= a)
  assert frac <= (rms_x / a) ** 2 + 1e-12
  ```
- **Textbook Grounding**: VMLS §3.1 (Eq. 3.2), §3.3 (Eqs. 3.4–3.5, Fig. 3.5), Ex. 3.24b; MML §8.2.3 (Ex. 8.3).

---

### Prompt 4 (GATE Bridge): Sylvester's Criterion for SPD Inner Products, $A$-Orthogonality & C-S Extrema
- **Question**: How can we verify in seconds whether a symmetric matrix $A$ defines a valid inner product, test $A$-orthogonality, and compute $\max_{\mathbf{x}^\top A\mathbf{x} \le 1} \mathbf{c}^\top \mathbf{x}$ without Lagrange multipliers?
- **Intuitive & Mathematical Proof**: By MML Theorem 3.5 and Sylvester's criterion, symmetric $A$ defines a valid inner product $\langle\mathbf{x},\mathbf{y}\rangle_A = \mathbf{x}^\top A\mathbf{y}$ iff all leading principal minors $\Delta_1, \dots, \Delta_n > 0$. Cholesky factorization $A = L L^\top$ whitens coordinates $\tilde{\mathbf{x}} = L^\top\mathbf{x}$ so $\langle\mathbf{x},\mathbf{y}\rangle_A = \tilde{\mathbf{x}}^\top \tilde{\mathbf{y}}$; vectors like $\mathbf{u}=[1,0]^\top, \mathbf{v}=[-1,2]^\top$ can have $\mathbf{u}^\top\mathbf{v} = -1 \ne 0$ yet $\mathbf{u}^\top A\mathbf{v} = 0$. Applying Cauchy-Schwarz to $\langle A^{1/2}\mathbf{x}, A^{-1/2}\mathbf{c}\rangle$ directly gives $\max_{\mathbf{x}^\top A\mathbf{x} \le 1} \mathbf{c}^\top \mathbf{x} = \sqrt{\mathbf{c}^\top A^{-1}\mathbf{c}}$.
- **NumPy Verification**:
  ```python
  import numpy as np

  A_spd = np.array([[2.0, 1.0], [1.0, 2.0]])
  A_indef = np.array([[1.0, 0.0], [0.0, -1.0]])
  assert A_spd[0, 0] > 0 and np.linalg.det(A_spd) > 0  # Sylvester minors: 2, 3 > 0
  assert np.linalg.det(A_indef) < 0  # fails Sylvester: -1 < 0

  u = np.array([1.0, 0.0])
  v = np.array([-1.0, 2.0])
  assert np.isclose(u @ v, -1.0)
  assert np.isclose(u @ A_spd @ v, 0.0)

  # GATE shortcut: max_{x^T A x <= 1} c^T x = sqrt(c^T A^{-1} c)
  c = np.array([3.0, -1.0])
  A_inv_c = np.linalg.solve(A_spd, c)
  bound = float(np.sqrt(c @ A_inv_c))
  x_star = A_inv_c / bound
  assert np.isclose(x_star @ A_spd @ x_star, 1.0)
  assert np.isclose(c @ x_star, bound)
  ```
- **Textbook Grounding**: MML §3.2.2–3.2.3 (Thm. 3.5, Exs. 3.4, 3.7), §4.3 (Thm. 4.18), Ex. 3.9; Kreyszig Ch. 8.
