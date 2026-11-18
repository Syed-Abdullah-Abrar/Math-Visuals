# Part 4: Interactive Parameter Tweaks & Intuition Stress-Test Prompts

## 1. Interactive Parameter Tweaks Guide

The ManimCE animation script (`scene_gram_schmidt_qr.py`) defines the canonical input matrix $A$, orthonormal factor $Q$, upper-triangular factor $R$, target vector $\mathbf{b}$, degenerate column $\mathbf{a}_3^{\text{deg}}$, and visual styling constants at module level. Learners and engineers can modify these parameters to explore arbitrary skewed bases, sign flips, rank-1 collapses, or alternate 3D camera trajectories.

### Parameter Configuration Matrix

| Parameter Name | Default Value | Recommended Test Values | Pedagogical Objective |
|---|---|---|---|
| `A_MAT` | `[[2, 0, -1], [1, 2, -1], [2, 2, 3]]` | `[[3, 3, -2], [4, 4, 4], [0, 5, 3]]`, `[[1, 1, 1], [0, 1, 1], [0, 0, 1]]` | Compare Pythagorean-quadruple integer $QR$ against reflection-containing $Q$ ($\det Q = -1$) or unit-shear bases. |
| `B_TARGET` | `np.array([1.0, -2.0, 3.0])` | `[3.0, 3.0, 3.0]`, `[2.0, 1.0, 2.0]` | Observe how $Q^\top \mathbf{b}$ gives exact coordinates in the $\{\mathbf{q}_1, \mathbf{q}_2, \mathbf{q}_3\}$ frame and drives $R\hat{\mathbf{x}} = Q^\top\mathbf{b}$. |
| `A3_DEG` | `np.array([-2.0, 1.0, 0.0])` | `2*A1_COORDS - A2_COORDS`, `np.zeros(3)` | Test coplanar collapse onto $\operatorname{span}(\mathbf{q}_1, \mathbf{q}_2)$ vs. immediate zero-vector termination. |
| `PHI_DEFAULT` / `THETA_DEFAULT` | `70 * DEGREES`, `-45 * DEGREES` | `65 * DEGREES, -30 * DEGREES`, `80 * DEGREES, -60 * DEGREES` | Inspect perpendicularity of $\mathbf{u}_2, \mathbf{u}_3$ and the plane $\operatorname{span}(\mathbf{q}_1, \mathbf{q}_2)$ from different 3D elevations. |
| `SUBSPACE_COLOR` | `"#2A9D8F"` (Teal) | `"#7AA2F7"`, `"#9ECE6A"` | Adjust contrast of $\operatorname{span}(\mathbf{q}_1, \mathbf{q}_2)$ and 3D right-angle elbow markers. |
| `AMBIENT_ROTATION_RATE` | `0.06` rad/s | `0.00` (static), `0.08` (faster parallax) | Trade off active 3D parallax sweeps against 1080p60 rendering speed. |

### How to Modify and Re-Render

1. **Edit constants in `scene_gram_schmidt_qr.py`** (ensuring pre-flight `assert` statements match any new matrix constants, or deriving `Q_MAT, R_MAT` via Gram-Schmidt).
2. **Execute fast low-quality preview render (`-ql` / `-pql`)**:
   ```bash
   .venv/bin/manim -ql linear_algebra/sprint_07_gram_schmidt_qr/scene_gram_schmidt_qr.py GramSchmidtQRScene
   .venv/bin/manim -pql linear_algebra/sprint_07_gram_schmidt_qr/scene_gram_schmidt_qr.py GramSchmidtQRScene
   ```
3. **Render individual modular acts**:
   ```bash
   .venv/bin/manim -ql linear_algebra/sprint_07_gram_schmidt_qr/scene_gram_schmidt_qr.py Act1SpatialPuzzle
   .venv/bin/manim -ql linear_algebra/sprint_07_gram_schmidt_qr/scene_gram_schmidt_qr.py Act2DynamicDeformation
   .venv/bin/manim -ql linear_algebra/sprint_07_gram_schmidt_qr/scene_gram_schmidt_qr.py Act3AlgebraicMorph
   .venv/bin/manim -ql linear_algebra/sprint_07_gram_schmidt_qr/scene_gram_schmidt_qr.py Act4DegenerateStressTest
   ```
4. **Execute high-quality 1080p60 production render (`-qh` / `-pqh`)**:
   ```bash
   .venv/bin/manim -qh linear_algebra/sprint_07_gram_schmidt_qr/scene_gram_schmidt_qr.py GramSchmidtQRScene
   .venv/bin/manim -pqh linear_algebra/sprint_07_gram_schmidt_qr/scene_gram_schmidt_qr.py GramSchmidtQRScene
   ```

---

## 2. Intuition Stress-Test Prompts

### Prompt 1: 3D Orthogonal Stripping & Nested Span Invariance
- **Question**:  
  During 3D Gram-Schmidt on $\{\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3\}$, why does subtracting the projections onto earlier $\mathbf{q}_i$ leave every prefix span invariant ($\operatorname{span}(\mathbf{a}_1, \dots, \mathbf{a}_k) = \operatorname{span}(\mathbf{q}_1, \dots, \mathbf{q}_k)$ for $k=1,2,3$), and how does the squared norm $\|\mathbf{a}_k\|_2^2$ decompose across column $k$ of $R$?
- **Intuitive & Mathematical Proof**:  
  By induction (VMLS §5.4), at step $k$ we define:
  $$\mathbf{u}_k = \mathbf{a}_k - \sum_{i=1}^{k-1} (\mathbf{q}_i^\top \mathbf{a}_k)\mathbf{q}_i, \qquad \mathbf{q}_k = \frac{\mathbf{u}_k}{\|\mathbf{u}_k\|_2}$$
  1. **Mutual Inclusion of Prefix Spans**: Since $\mathbf{q}_1, \dots, \mathbf{q}_{k-1} \in \operatorname{span}(\mathbf{a}_1, \dots, \mathbf{a}_{k-1})$ by the inductive hypothesis, $\mathbf{u}_k$ and $\mathbf{q}_k$ are linear combinations of $\{\mathbf{a}_1, \dots, \mathbf{a}_k\}$, giving $\operatorname{span}(\mathbf{q}_1, \dots, \mathbf{q}_k) \subseteq \operatorname{span}(\mathbf{a}_1, \dots, \mathbf{a}_k)$. Conversely, rearranging gives $\mathbf{a}_k = \sum_{i=1}^{k-1} R_{ik}\mathbf{q}_i + R_{kk}\mathbf{q}_k$ with $R_{kk} = \|\mathbf{u}_k\|_2 > 0$, so $\mathbf{a}_k \in \operatorname{span}(\mathbf{q}_1, \dots, \mathbf{q}_k)$. Hence the spans are identical for every $k$, and their orthogonal projectors coincide:
     $$P_k = A_k(A_k^\top A_k)^{-1} A_k^\top = Q_k Q_k^\top$$
  2. **Pythagorean Energy Decomposition**: Because $\{\mathbf{q}_1, \dots, \mathbf{q}_k\}$ are mutually orthonormal ($\mathbf{q}_i^\top \mathbf{q}_j = \delta_{ij}$), expanding $\|\mathbf{a}_k\|_2^2 = \mathbf{a}_k^\top \mathbf{a}_k$ eliminates all cross-terms:
     $$\|\mathbf{a}_k\|_2^2 = \sum_{i=1}^{k} R_{ik}^2$$
     For our canonical $\mathbf{a}_3 = [-1, -1, 3]^\top$, $\|\mathbf{a}_3\|_2^2 = 1 + 1 + 9 = 11 = R_{13}^2 + R_{23}^2 + R_{33}^2 = 1^2 + 1^2 + 3^2$.
- **NumPy Verification**:
  ```python
  import numpy as np

  A = np.array([[2.0, 0.0, -1.0], [1.0, 2.0, -1.0], [2.0, 2.0, 3.0]])
  Q = np.zeros((3, 3))
  R = np.zeros((3, 3))

  for j in range(3):
      v = A[:, j].copy()
      for i in range(j):
          R[i, j] = np.dot(Q[:, i], A[:, j])
          v = v - R[i, j] * Q[:, i]
      R[j, j] = np.linalg.norm(v)
      Q[:, j] = v / R[j, j]

  assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-12)
  assert np.allclose(Q @ R, A, atol=1e-12)

  # Nested prefix span equality via orthogonal projector equivalence
  for k in (1, 2, 3):
      A_k = A[:, :k]
      Q_k = Q[:, :k]
      P_A = A_k @ np.linalg.inv(A_k.T @ A_k) @ A_k.T
      P_Q = Q_k @ Q_k.T
      assert np.allclose(P_A, P_Q, atol=1e-12)
      assert np.isclose(np.sum(A[:, k - 1] ** 2), np.sum(R[:k, k - 1] ** 2), atol=1e-12)

  print("Prompt 1 Verification PASSED: Nested span invariance & Pythagorean split confirmed.")
  ```
- **Textbook Grounding**: MML §3.8.3 (Eq. 3.67–3.68); VMLS §5.4 (Algorithm 5.1, pp. 99–100) & Ex. 10.40.

---

### Prompt 2: Upper-Triangular $R$, QR Uniqueness vs. Sign Diagonal $D = \operatorname{diag}(\pm 1)$, and $A^\top A = R^\top R$
- **Question**:  
  Why is $R$ strictly upper triangular ($R_{ij} = \mathbf{q}_i^\top \mathbf{a}_j = 0$ for $i > j$), why is thin $A = QR$ unique once $R_{ii} > 0$ is enforced (versus the $2^k$ ambiguity $D = \operatorname{diag}(\pm 1)$), and what geometric meaning is encoded by $A^\top A = R^\top R$?
- **Intuitive & Mathematical Proof**:  
  1. **Upper Triangularity**: By Prompt 1, $\mathbf{a}_j \in \operatorname{span}(\mathbf{q}_1, \dots, \mathbf{q}_j)$. By construction, $\mathbf{q}_i \perp \operatorname{span}(\mathbf{q}_1, \dots, \mathbf{q}_{i-1})$ for every $i > j$. Therefore $R_{ij} = \mathbf{q}_i^\top \mathbf{a}_j = 0$ whenever $i > j$, making $R$ upper triangular.
  2. **Uniqueness up to $D = \operatorname{diag}(\pm 1)$**: Suppose full-column-rank $A \in \mathbb{R}^{m\times k}$ has two thin factorizations $A = Q_1 R_1 = Q_2 R_2$. Since $R_1, R_2$ are invertible upper triangular, define $D := R_2 R_1^{-1}$ (which is also upper triangular). Then $Q_1 = Q_2 D$, and orthonormality gives:
     $$I_k = Q_1^\top Q_1 = D^\top (Q_2^\top Q_2) D = D^\top D \implies D^{-1} = D^\top$$
     Because $D$ is upper triangular while $D^\top$ is lower triangular, $D$ must be **diagonal** with $D_{ii}^2 = 1$, i.e., $D = \operatorname{diag}(\pm 1, \dots, \pm 1)$ ($2^k$ sign choices). Enforcing $R_{ii} > 0$ forces $D_{ii} > 0 \implies D = I_k$, making $(Q, R)$ strictly unique. (Note: LAPACK/Householder `np.linalg.qr` can return negative $R_{ii}$; canonicalize via $D = \operatorname{diag}(\operatorname{sgn}(R_{ii}))$, $Q \leftarrow QD, R \leftarrow DR$.)
  3. **Gram Isometry ($A^\top A = R^\top R$)**: Substituting $A = QR$ yields:
     $$A^\top A = (QR)^\top(QR) = R^\top (Q^\top Q) R = R^\top R$$
     Thus the columns of $R$ in $\mathbb{R}^k$ preserve every column norm and pairwise inner product (and hence every angle) of the original columns of $A$ in $\mathbb{R}^m$, while $R$ is the Cholesky factor ($R = L^\top$) of the Gram matrix $A^\top A$.
- **NumPy Verification**:
  ```python
  import numpy as np

  A = np.array([[2.0, 0.0, -1.0], [1.0, 2.0, -1.0], [2.0, 2.0, 3.0]])
  Q_canon = np.array([[2.0, -2.0, -1.0], [1.0, 2.0, -2.0], [2.0, 1.0, 2.0]]) / 3.0
  R_canon = np.array([[3.0, 2.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 3.0]])

  # Upper-triangular and positive diagonal check
  assert np.allclose(np.tril(R_canon, -1), 0.0)
  assert np.all(np.diag(R_canon) > 0.0)
  assert np.allclose(A.T @ A, R_canon.T @ R_canon, atol=1e-12)

  # Canonicalize Householder QR output to match positive-diagonal GS uniquely
  Q_hh, R_hh = np.linalg.qr(A)
  signs = np.where(np.diag(R_hh) >= 0.0, 1.0, -1.0)
  D_fix = np.diag(signs)
  Q_hh_pos = Q_hh @ D_fix
  R_hh_pos = D_fix @ R_hh
  assert np.allclose(Q_hh_pos, Q_canon, atol=1e-12)
  assert np.allclose(R_hh_pos, R_canon, atol=1e-12)

  # Arbitrary sign diagonal D preserves factorization A = (Q D)(D R)
  D_flip = np.diag([1.0, -1.0, -1.0])
  assert np.allclose((Q_canon @ D_flip) @ (D_flip @ R_canon), A, atol=1e-12)
  print("Prompt 2 Verification PASSED: Upper-triangular R, uniqueness, and A^T A = R^T R confirmed.")
  ```
- **Textbook Grounding**: VMLS §10.4 (pp. 189–191), Ex. 10.39 (p. 197).

---

### Prompt 3: Frontier ML Bridge — Orthogonalization in Muon / Orthogonal Weight Init & Lauchli QR Conditioning
- **Question**:  
  Why do **Muon** (orthogonalized momentum updates) and **Orthogonal Weight Initialization** (`nn.init.orthogonal_`) rely on orthonormalization to prevent spectral collapse across depth, and why does QR least squares survive the $\epsilon = 10^{-8}$ Lauchli matrix where Normal Equations collapse?
- **Intuitive & Mathematical Proof**:  
  1. **Spectral Preservation in Deep Layers & Muon**: Any linear map $W$ has singular values $\sigma_1 \ge \dots \ge \sigma_d$; iterating $W^L$ amplifies ratios $(\sigma_1/\sigma_d)^L$, causing exploding/vanishing gradients. Sampling $W \sim \mathcal{N}(0, 1)^{d\times d}$ and taking $Q = Q_{\text{qr}}\operatorname{diag}(\operatorname{sgn}(R_{ii}))$ yields a Haar-uniform orthogonal matrix where all $\sigma_i(Q) = 1$ ($\kappa_2(Q) = 1$, $\|Q^L\mathbf{x}\|_2 = \|\mathbf{x}\|_2$). Likewise, **Muon** replaces a momentum update $G$ with its nearest semi-orthogonal polar factor $UV^\top$ (approximated efficiently via odd-polynomial Newton-Schulz iterations $X_{t+1} = aX_t + b(X_t X_t^\top)X_t + c(X_t X_t^\top)^2 X_t$), equalizing singular scales across update directions.
  2. **Conditioning ($\kappa$ vs. $\kappa^2$) on Lauchli $\epsilon = 10^{-8}$**: Consider the $4\times 3$ Lauchli matrix $A(\epsilon) = \begin{bmatrix} 1 & 1 & 1 \\ \epsilon & 0 & 0 \\ 0 & \epsilon & 0 \\ 0 & 0 & \epsilon \end{bmatrix}^\top$ transposed properly as $\begin{bmatrix} 1 & 1 & 1 \\ \epsilon & 0 & 0 \\ 0 & \epsilon & 0 \\ 0 & 0 & \epsilon \end{bmatrix}$ ($\kappa_2(A) \approx \sqrt{3}/\epsilon \approx 1.73\times 10^8$ at $\epsilon = 10^{-8}$). Forming Normal Equations $A^\top A$ requires computing $1 + \epsilon^2 = 1 + 10^{-16} = 1.0$ in IEEE-754 `float64` ($\epsilon_{\text{mach}} \approx 2.22\times 10^{-16}$), collapsing $A^\top A$ to the rank-1 matrix of all ones (`LinAlgError`). Meanwhile, Classical Gram-Schmidt loses orthogonality at $\mathcal{O}(\epsilon_{\text{mach}}\kappa^2) \sim 10^0$, Modified Gram-Schmidt drifts at $\mathcal{O}(\epsilon_{\text{mach}}\kappa) \sim 10^{-8}$, and Householder QR maintains $\mathcal{O}(\epsilon_{\text{mach}}) \sim 10^{-16}$ orthogonality and solves $R\hat{\mathbf{x}} = Q^\top\mathbf{b}$ cleanly.
- **NumPy Verification**:
  ```python
  import numpy as np

  def cgs_qr(A):
      m, n = A.shape
      Q, R = np.zeros((m, n)), np.zeros((n, n))
      for j in range(n):
          v = A[:, j].copy()
          for i in range(j):
              R[i, j] = np.dot(Q[:, i], A[:, j])
              v -= R[i, j] * Q[:, i]
          R[j, j] = np.linalg.norm(v)
          Q[:, j] = v / R[j, j]
      return Q, R

  def mgs_qr(A):
      m, n = A.shape
      Q, R = np.zeros((m, n)), np.zeros((n, n))
      V = A.astype(float).copy()
      for i in range(n):
          R[i, i] = np.linalg.norm(V[:, i])
          Q[:, i] = V[:, i] / R[i, i]
          for j in range(i + 1, n):
              R[i, j] = np.dot(Q[:, i], V[:, j])
              V[:, j] -= R[i, j] * Q[:, i]
      return Q, R

  # 1. Orthogonal Weight Init norm preservation across 20 layers
  rng = np.random.default_rng(703)
  W = rng.standard_normal((32, 32))
  Q_raw, R_raw = np.linalg.qr(W)
  Q_haar = Q_raw @ np.diag(np.where(np.diag(R_raw) >= 0, 1.0, -1.0))
  x0 = rng.standard_normal(32)
  x = x0.copy()
  for _ in range(20):
      x = Q_haar @ x
  assert np.isclose(np.linalg.norm(x), np.linalg.norm(x0), atol=1e-11)

  # 2. Lauchli epsilon = 1e-8 stability comparison
  eps = 1e-8
  A_lauchli = np.array([[1.0, 1.0, 1.0], [eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]])
  Q_cgs, _ = cgs_qr(A_lauchli)
  Q_mgs, _ = mgs_qr(A_lauchli)
  Q_hh, R_hh = np.linalg.qr(A_lauchli)

  err_cgs = np.linalg.norm(Q_cgs.T @ Q_cgs - np.eye(3), "fro")
  err_mgs = np.linalg.norm(Q_mgs.T @ Q_mgs - np.eye(3), "fro")
  err_hh = np.linalg.norm(Q_hh.T @ Q_hh - np.eye(3), "fro")
  assert err_cgs > 0.1 and err_mgs < 1e-7 and err_hh < 1e-14

  # Normal equations collapse vs QR solve
  x_true = np.array([1.0, -2.0, 1.0])
  b = A_lauchli @ x_true
  normal_failed = False
  try:
      x_ne = np.linalg.solve(A_lauchli.T @ A_lauchli, A_lauchli.T @ b)
      if np.linalg.norm(x_ne - x_true) > 1e-2:
          normal_failed = True
  except np.linalg.LinAlgError:
      normal_failed = True
  assert normal_failed, "Normal equations should fail or lose precision at eps=1e-8"

  x_qr = np.linalg.solve(R_hh, Q_hh.T @ b)
  assert np.linalg.norm(x_qr - x_true) / np.linalg.norm(x_true) < 1e-7
  print(f"Prompt 3 Verification PASSED: CGS={err_cgs:.2e}, MGS={err_mgs:.2e}, HH={err_hh:.2e}.")
  ```
- **Textbook Grounding**: VMLS §10.4 & §12.3 (Algorithm 12.1); Saxe et al. (2013) orthogonal init; Mezzadri (2007) Haar QR sign correction; Jordan et al. (2024) Muon optimizer.

---

### Prompt 4: GATE CS/DA Shortcut — Back-Substitution $R\hat{\mathbf{x}} = Q^\top\mathbf{b}$, $|\det A| = \prod R_{ii}$ & Rank from $R_{ii}=0$
- **Question**:  
  Given $A = QR$ ($R_{ii} \ge 0$), how do you rapidly compute (a) the exact or least-squares solution $\hat{\mathbf{x}}$ via back-substitution $R\hat{\mathbf{x}} = Q^\top\mathbf{b}$, (b) $|\det(A)| = \prod R_{ii}$ (noting the $\det(Q) = \pm 1$ sign distinction), and (c) rank / linear dependence from $R_{ii} = 0$?
- **Intuitive & Mathematical Proof**:  
  1. **$\mathcal{O}(n^2)$ Back-Substitution ($R\hat{\mathbf{x}} = Q^\top\mathbf{b}$)**: For canonical $A = \begin{bmatrix} 2 & 0 & -1 \\ 1 & 2 & -1 \\ 2 & 2 & 3 \end{bmatrix}$ and $\mathbf{b} = [1, -2, 3]^\top$, we first compute $\mathbf{y} = Q^\top\mathbf{b} = \frac{1}{3}\begin{bmatrix} 2 & 1 & 2 \\ -2 & 2 & 1 \\ -1 & -2 & 2 \end{bmatrix}\begin{bmatrix} 1 \\ -2 \\ 3 \end{bmatrix} = \begin{bmatrix} 2 \\ -1 \\ 3 \end{bmatrix}$. Solving $R\hat{\mathbf{x}} = \mathbf{y}$ bottom-up:
     - $3x_3 = 3 \implies x_3 = 1$
     - $2x_2 + 1(1) = -1 \implies x_2 = -1$
     - $3x_1 + 2(-1) + 1(1) = 2 \implies x_1 = 1 \implies \hat{\mathbf{x}} = [1, -1, 1]^\top$
     This requires only $1 + 3 + \dots + (2n-1) = n^2$ flops after $QR$ factoring (VMLS Alg 11.1–11.2).
  2. **Determinant Magnitude vs. $\det(Q) = \pm 1$**: Since $\det(Q) \in \{+1, -1\}$ and $\det(R) = \prod_{i=1}^n R_{ii}$, we always have $|\det(A)| = \prod_{i=1}^n R_{ii}$. For our canonical matrix, $\det(Q) = +1$ so $\det(A) = 3 \cdot 2 \cdot 3 = +18$. Caution for GATE MCQs: if $Q$ contains an odd number of reflections ($\det(Q) = -1$, e.g., $A_{\text{refl}} = Q_{\text{refl}} R$), then $\prod R_{ii} = |\det(A)|$ while $\det(A) = -\prod R_{ii}$!
  3. **Early Termination / Rank Detection ($R_{ii} = 0$)**: Replacing $\mathbf{a}_3$ with $\mathbf{a}_3^{\text{deg}} = \mathbf{a}_2 - \mathbf{a}_1 = [-2, 1, 0]^\top = -1\mathbf{q}_1 + 2\mathbf{q}_2$ produces $\mathbf{u}_3 = \mathbf{0}$ and $R_{33} = 0$. Gram-Schmidt halts early (VMLS Alg 5.1 Step 2), certifying $\operatorname{rank}(A_{\text{deg}}) = 2 < 3$ and singular $R$.
- **NumPy Verification**:
  ```python
  import numpy as np

  A = np.array([[2.0, 0.0, -1.0], [1.0, 2.0, -1.0], [2.0, 2.0, 3.0]])
  Q = np.array([[2.0, -2.0, -1.0], [1.0, 2.0, -2.0], [2.0, 1.0, 2.0]]) / 3.0
  R = np.array([[3.0, 2.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 3.0]])
  b = np.array([1.0, -2.0, 3.0])

  # 1. Back-substitution R x = Q^T b
  y = Q.T @ b
  assert np.allclose(y, [2.0, -1.0, 3.0], atol=1e-12)
  x = np.zeros(3)
  for i in range(2, -1, -1):
      x[i] = (y[i] - np.dot(R[i, i + 1 :], x[i + 1 :])) / R[i, i]
  assert np.allclose(x, [1.0, -1.0, 1.0], atol=1e-12)
  assert np.allclose(A @ x, b, atol=1e-12)

  # 2. |det(A)| == prod(diag(R)), contrast det(Q) = +1 vs -1
  assert np.isclose(abs(np.linalg.det(A)), np.prod(np.diag(R)), atol=1e-12)
  Q_refl = Q @ np.diag([1.0, 1.0, -1.0])
  A_refl = Q_refl @ R
  assert np.isclose(np.linalg.det(Q_refl), -1.0, atol=1e-12)
  assert np.isclose(np.linalg.det(A_refl), -np.prod(np.diag(R)), atol=1e-12)
  assert np.isclose(abs(np.linalg.det(A_refl)), np.prod(np.diag(R)), atol=1e-12)

  # 3. Rank detection when R_33 == 0
  a3_deg = A[:, 1] - A[:, 0]
  u3_deg = a3_deg - np.dot(Q[:, 0], a3_deg) * Q[:, 0] - np.dot(Q[:, 1], a3_deg) * Q[:, 1]
  r33_deg = np.linalg.norm(u3_deg)
  assert np.isclose(r33_deg, 0.0, atol=1e-12)
  assert np.linalg.matrix_rank(np.column_stack([A[:, :2], a3_deg])) == 2
  print("Prompt 4 Verification PASSED: Back-sub, |det A| = prod R_ii, and R_33=0 rank check confirmed.")
  ```
- **Textbook Grounding**: VMLS §5.4 (Alg 5.1 early termination), §11.3 (Alg 11.1–11.2, $n^2$ back-substitution); Kreyszig §20.9; GATE CS/DA Linear Algebra syllabus.
