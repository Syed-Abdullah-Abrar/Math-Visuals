# Part 4: Interactive Parameter Tweaks & Intuition Stress-Test Prompts

## Section 1: Interactive Parameter Tweaks Guide

The ManimCE animation script (`scene_orthogonal_projections.py`) defines its canonical 1D direction, 2D subspace basis matrix, off-plane targets, affine offset, rank-deficient matrix, and oblique projector as module-level constants. Modifying these constants allows students and researchers to explore arbitrary projections onto lines, planes, affine manifolds, and ill-conditioned or skewed projection regimes.

### Parameter Configuration Matrix
| Parameter Name | Default Value | Recommended Test Values | Pedagogical Objective |
|---|---|---|---|
| `B_1D` | `np.array([1.0, 2.0, 2.0])` | `[2.0, 1.0, 0.0]`, `[1.0, 1.0, 1.0]` | Alter the 1D target line direction; observe how $\lambda = \mathbf{b}^\top\mathbf{x}/\|\mathbf{b}\|^2$ scales while $P_1 = \mathbf{b}\mathbf{b}^\top/\|\mathbf{b}\|^2$ remains invariant under $\mathbf{b} \mapsto c\mathbf{b}$. |
| `X_1D` | `np.array([3.0, 3.0, 0.0])` | `[1.0, 1.0, 1.0]`, `[2.0, 4.0, 4.0]` | Probe off-line vs. already-on-line vectors ($\mathbf{x} \in U \implies \pi_U(\mathbf{x}) = \mathbf{x}$) and orthogonal vectors ($\mathbf{x} \perp \mathbf{b} \implies \pi_U(\mathbf{x}) = \mathbf{0}$). |
| `A_MAT` | `[[1, 0], [1, 1], [1, 2]]` | `[[1, 0], [0, 1], [0.5, 0.5]]`, `[[1, 0], [0, 1], [0, 0]]` | Test tilted vs. coordinate $xy$-plane projections; verify $P = A(A^\top A)^{-1}A^\top$ is invariant under any invertible column reparameterization $A \mapsto AM$. |
| `Y_TARGET` | `np.array([3.0, -1.0, 1.0])` | `[1.0, 2.0, 1.0]`, `[2.0, 0.0, -2.0]` | Move the gold off-plane target to test minimal residual $\|\mathbf{y} - P\mathbf{y}\|_2$ and Pythagorean decomposition $\|\mathbf{y}\|_2^2 = \|P\mathbf{y}\|_2^2 + \|(I-P)\mathbf{y}\|_2^2$. |
| `X0_AFF` | `np.array([0.0, 0.0, 1.0])` | `[0.0, 0.0, 1.5]`, `[1.0, -2.0, 1.0]` | Shift the affine flat $L = \mathbf{x}_0 + U$; verify $\pi_L(\mathbf{x}) = \mathbf{x}_0 + P(\mathbf{x} - \mathbf{x}_0)$ stays unchanged if $\mathbf{x}_0$ shifts by any vector inside $U$. |
| `A_DEG` | `[[1, 2], [1, 2], [1, 2]]` | `[[1, -1], [2, -2], [1, -1]]` | Trigger rank-deficient collapse ($\det(A^\top A) = 0$) and compare Moore-Penrose $A^+$ against Ridge regularization $(A^\top A + \lambda I)^{-1}A^\top$. |
| `P_OB` | `[[1, 0, 0], [0, 1, 0], [-1, 2, 0]]` | `[[1, 0, 0], [0, 1, 0], [-0.5, 1.5, 0]]` | Contrast oblique idempotents ($P_{\text{ob}}^2 = P_{\text{ob}}$, $P_{\text{ob}}^\top \ne P_{\text{ob}}$, $\|P_{\text{ob}}\|_2 > 1$) against symmetric orthogonal projection. |
| `camera_rate` | `0.08` | `0.04` to `0.14` | Tune ambient orbital rotation speed for 3D depth clarity while HUDs remain fixed in frame. |

### CLI Render Commands
```bash
# Fast low-quality preview render of complete 4-Act master scene (480p15)
.venv/bin/manim -pql linear_algebra/sprint_06_orthogonal_projections/scene_orthogonal_projections.py OrthogonalProjectionsScene
.venv/bin/manim -ql linear_algebra/sprint_06_orthogonal_projections/scene_orthogonal_projections.py OrthogonalProjectionsScene

# High-quality production release render (1080p60)
.venv/bin/manim -pqh linear_algebra/sprint_06_orthogonal_projections/scene_orthogonal_projections.py OrthogonalProjectionsScene
.venv/bin/manim -qh linear_algebra/sprint_06_orthogonal_projections/scene_orthogonal_projections.py OrthogonalProjectionsScene

# Modular single-act preview renders for rapid inspection
.venv/bin/manim -ql linear_algebra/sprint_06_orthogonal_projections/scene_orthogonal_projections.py Act1SpatialPuzzle
.venv/bin/manim -ql linear_algebra/sprint_06_orthogonal_projections/scene_orthogonal_projections.py Act2DynamicDeformation
.venv/bin/manim -ql linear_algebra/sprint_06_orthogonal_projections/scene_orthogonal_projections.py Act3AlgebraicMorph
.venv/bin/manim -ql linear_algebra/sprint_06_orthogonal_projections/scene_orthogonal_projections.py Act4DegenerateStressTest
```

---

## Section 2: Intuition Stress-Test Prompts

### Prompt 1: Minimum-Distance Optimality & The Orthogonality Principle
**Question**:  
Why does the closest point $\hat{\mathbf{y}} = A\hat{\mathbf{x}}$ in $\mathcal{C}(A)$ to $\mathbf{y}$ occur if and only if the residual $\mathbf{r} = \mathbf{y} - \hat{\mathbf{y}}$ is orthogonal to every column of $A$? Derive the normal equations $A^\top A\hat{\mathbf{x}} = A^\top \mathbf{y}$ and prove global optimality without calculus.

**Intuitive & Mathematical Proof**:  
1. *Orthogonal Splitting of Squared Error*: For any candidate vector $\mathbf{x} \in \mathbb{R}^k$, add and subtract the candidate $\hat{\mathbf{y}} = A\hat{\mathbf{x}}$ inside the residual:
   $$\|\mathbf{y} - A\mathbf{x}\|_2^2 = \|(\mathbf{y} - A\hat{\mathbf{x}}) + A(\hat{\mathbf{x}} - \mathbf{x})\|_2^2 = \|\mathbf{y} - A\hat{\mathbf{x}}\|_2^2 + \|A(\hat{\mathbf{x}} - \mathbf{x})\|_2^2 + 2(\hat{\mathbf{x}} - \mathbf{x})^\top A^\top (\mathbf{y} - A\hat{\mathbf{x}})$$
2. *Vanishing Cross-Term via Normal Equations*: The cross-term vanishes for all $\mathbf{x} \in \mathbb{R}^k$ if and only if:
   $$A^\top(\mathbf{y} - A\hat{\mathbf{x}}) = \mathbf{0} \iff A^\top A\hat{\mathbf{x}} = A^\top \mathbf{y}$$
   Geometrically, this states that $\mathbf{r} = \mathbf{y} - A\hat{\mathbf{x}}$ is perpendicular to every vector $A\mathbf{z} \in \mathcal{C}(A)$ (the Orthogonality Principle).
3. *Strict Global Minimum*: Once $A^\top(\mathbf{y} - A\hat{\mathbf{x}}) = \mathbf{0}$ holds, every deviation $\mathbf{x} \ne \hat{\mathbf{x}}$ yields:
   $$\|\mathbf{y} - A\mathbf{x}\|_2^2 = \|\mathbf{y} - A\hat{\mathbf{x}}\|_2^2 + \|A(\mathbf{x} - \hat{\mathbf{x}})\|_2^2$$
   When $A$ has full column rank, $A(\mathbf{x} - \hat{\mathbf{x}}) \ne \mathbf{0}$ for $\mathbf{x} \ne \hat{\mathbf{x}}$, so $\|A(\mathbf{x} - \hat{\mathbf{x}})\|_2^2 > 0$ and $\hat{\mathbf{x}}$ is the unique strict global minimizer. For our canonical trajectory $\mathbf{x}(s) = [2+s, -1-s]^\top$, $A(\mathbf{x}(s)-\hat{\mathbf{x}}) = [s, 0, -s]^\top$ has squared norm $2s^2$, giving exact distance $d(s) = \sqrt{6 + 2s^2}$.

**Executable NumPy Verification**:
```python
import numpy as np

A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
y = np.array([3.0, -1.0, 1.0])

x_hat = np.linalg.solve(A.T @ A, A.T @ y)
y_hat = A @ x_hat
r_hat = y - y_hat

assert np.allclose(x_hat, [2.0, -1.0])
assert np.allclose(y_hat, [2.0, 1.0, 0.0])
assert np.allclose(A.T @ r_hat, 0.0, atol=1e-12)

rng = np.random.default_rng(601)
X_cand = rng.uniform(-10.0, 10.0, size=(1000, 2))
for x in X_cand:
    lhs = np.linalg.norm(y - A @ x) ** 2
    rhs = np.linalg.norm(r_hat) ** 2 + np.linalg.norm(A @ (x - x_hat)) ** 2
    assert np.isclose(lhs, rhs, atol=1e-10)
    assert lhs > np.linalg.norm(r_hat) ** 2

# Verify exact Act 2 sliding curve d(s) = sqrt(6 + 2*s^2)
for s in np.linspace(-2.0, 1.5, 71):
    x_s = np.array([2.0 + s, -1.0 - s])
    assert np.isclose(np.linalg.norm(y - A @ x_s), np.sqrt(6.0 + 2.0 * s**2), atol=1e-12)
print("Prompt 1: Minimum-Distance Optimality & Orthogonality Principle Verified!")
```

**Textbook Grounding**:  
- Introduction to Applied Linear Algebra (VMLS) §12.1–12.2 Eqs. (12.4), (12.7), (12.9), Fig. 12.2 (pp. 225–231)  
- Mathematics for Machine Learning (MML) §3.8.2 Eqs. (3.51)–(3.59), Example 3.11 (pp. 85–88)

---

### Prompt 2: Projector Invariants ($\operatorname{Tr}(P)=\operatorname{rank}(P)$, $P^2=P$, $P^\top=P$) & The Oblique Contrast
**Question**:  
Why are the eigenvalues of any projection matrix restricted to $\{0, 1\}$, why does $\operatorname{Tr}(P) = \operatorname{rank}(P) = \dim(U)$, and what geometrically breaks when idempotence holds ($P_{\text{ob}}^2 = P_{\text{ob}}$) but symmetry fails ($P_{\text{ob}}^\top \ne P_{\text{ob}}$)?

**Intuitive & Mathematical Proof**:  
1. *Eigenvalues in $\{0, 1\}$*: If $P\mathbf{v} = \lambda \mathbf{v}$ ($\mathbf{v} \ne \mathbf{0}$), applying $P^2 = P$ gives $\lambda^2 \mathbf{v} = P^2\mathbf{v} = P\mathbf{v} = \lambda \mathbf{v}$, so $\lambda(\lambda - 1) = 0 \implies \lambda \in \{0, 1\}$. Because the minimal polynomial divides $\lambda(\lambda - 1)$ (distinct linear factors), every projector $P$ is diagonalizable with $k = \operatorname{rank}(P)$ eigenvalues equal to $1$ and $n - k$ eigenvalues equal to $0$.
2. *Trace Equals Rank*: Summing the eigenvalues yields $\operatorname{Tr}(P) = \sum_{i=1}^n \lambda_i = \operatorname{rank}(P) = \dim(U)$.
3. *Why Symmetry Forces Orthogonality*: When $P^\top = P$, the $1$-eigenspace $U$ and $0$-eigenspace $\mathcal{N}(P)$ are orthogonal ($U^\perp = \mathcal{N}(P)$). Hence $\|P\|_2 = 1$ and every projection minimizes Euclidean distance. When symmetry fails ($P_{\text{ob}}^\top \ne P_{\text{ob}}$), rays strike $U$ at an oblique angle $\theta_{\min} < 90^\circ$, stretching singular values to $\|P_{\text{ob}}\|_2 = \frac{1}{\sin\theta_{\min}} > 1$ (here $\sqrt{6}$) and breaking the Pythagorean theorem.

**Executable NumPy Verification**:
```python
import numpy as np

A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
P = A @ np.linalg.inv(A.T @ A) @ A.T
P_ob = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 2.0, 0.0]])
y_test = np.array([1.0, 1.0, 3.0])

# Orthogonal projector checks
eigs_P = np.sort(np.linalg.eigvalsh(P))
assert np.allclose(eigs_P, [0.0, 1.0, 1.0], atol=1e-12)
assert np.isclose(np.trace(P), np.linalg.matrix_rank(P))
assert np.isclose(np.linalg.norm(P, 2), 1.0)

# Oblique projector checks: still idempotent, Tr == rank == 2, eigenvalues {0, 1, 1}, but asymmetric & expansive!
assert np.allclose(P_ob @ P_ob, P_ob)
assert not np.allclose(P_ob.T, P_ob)
eigs_ob = np.sort(np.real(np.linalg.eigvals(P_ob)))
assert np.allclose(eigs_ob, [0.0, 1.0, 1.0], atol=1e-12)
assert np.isclose(np.trace(P_ob), 2.0) and np.linalg.matrix_rank(P_ob) == 2

_, s_ob, Vt_ob = np.linalg.svd(P_ob)
assert np.isclose(s_ob[0], np.sqrt(6.0))
assert np.isclose(np.linalg.norm(P_ob @ Vt_ob[0]), np.sqrt(6.0))
assert np.linalg.norm(y_test - P_ob @ y_test) > np.linalg.norm(y_test - P @ y_test)
print("Prompt 2: Projector Invariants & Oblique Amplification Verified!")
```

**Textbook Grounding**:  
- Mathematics for Machine Learning (MML) §3.8 Definition 3.10, pp. 82–88  
- Introduction to Applied Linear Algebra (VMLS) §12.2

---

### Prompt 3: Frontier ML Bridge — Ridge Shrinkage, PCA Reconstruction & Attention Subspace Projections
**Question**:  
How do orthogonal projections unify Ridge regression ($H_\lambda = A(A^\top A + \lambda I)^{-1}A^\top$), PCA ($P_M = BB^\top$), and rank-deficient normal equations? Specifically, why does Ridge break idempotence ($H_\lambda^2 \ne H_\lambda$) while fixing singularity, and why does PCA achieve exact minimum reconstruction error?

**Intuitive & Mathematical Proof**:  
1. *SVD Filter View of Ridge*: Let $A = U\Sigma V^\top$ be the SVD of $A \in \mathbb{R}^{n \times k}$ with singular values $\sigma_1, \dots, \sigma_r > 0$. Unregularized orthogonal projection onto $\mathcal{C}(A)$ applies step weights $1$ on $\mathcal{C}(A)$ and $0$ on $\mathcal{C}(A)^\perp$: $P = U_r U_r^\top$. Ridge regularization replaces the inverse $\sigma_i^{-1}$ with damped factors $\frac{\sigma_i}{\sigma_i^2 + \lambda}$, giving hat matrix:
   $$H_\lambda = A(A^\top A + \lambda I)^{-1}A^\top = \sum_{i=1}^r \frac{\sigma_i^2}{\sigma_i^2 + \lambda}\mathbf{u}_i \mathbf{u}_i^\top$$
   For every $\lambda > 0$, $A^\top A + \lambda I \succ 0$ is strictly positive definite even when $\operatorname{rank}(A) < k$. However, because $\frac{\sigma_i^2}{\sigma_i^2 + \lambda} \in (0, 1)$, squaring $H_\lambda$ squares those weights, so $H_\lambda^2 \ne H_\lambda$. As $\lambda \to 0^+$, the Ridge solution converges smoothly to the minimum-norm pseudoinverse solution $A^+\mathbf{y}$.
2. *PCA Reconstruction Optimality*: For centered data $\{\mathbf{x}_n\}_{n=1}^N \subset \mathbb{R}^D$, projecting onto any $M$-dimensional orthonormal basis $B$ ($B^\top B = I_M$) gives orthogonal error $\|\mathbf{x}_n - BB^\top \mathbf{x}_n\|_2^2$. By the Eckart-Young / PCA theorem, choosing $B$ as the top-$M$ eigenvectors of covariance $S$ minimizes average reconstruction error to $\sum_{j=M+1}^D \lambda_j$.
3. *Attention Subspaces*: In Transformer multi-head attention, $W_Q, W_K, W_V$ project embeddings into $d_k$-D subspaces where scaled inner products $QK^\top$ evaluate alignment before $W_O$ maps back into the residual stream.

**Executable NumPy Verification**:
```python
import numpy as np

# 1. Rank-deficient Ridge vs Moore-Penrose pseudoinverse
A_deg = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
y = np.array([3.0, -1.0, 1.0])
s_deg = np.linalg.svd(A_deg, compute_uv=False)

lam = 0.5
H_lam = A_deg @ np.linalg.inv(A_deg.T @ A_deg + lam * np.eye(2)) @ A_deg.T
assert np.allclose(H_lam.T, H_lam)
assert not np.allclose(H_lam @ H_lam, H_lam)  # Shrinkage breaks idempotence
eff_df = sum((si**2) / (si**2 + lam) for si in s_deg if si > 1e-12)
assert np.isclose(np.trace(H_lam), eff_df, atol=1e-12)

x_rls_tiny = np.linalg.solve(A_deg.T @ A_deg + 1e-10 * np.eye(2), A_deg.T @ y)
x_pinv = np.linalg.pinv(A_deg) @ y
assert np.allclose(x_rls_tiny, x_pinv, atol=1e-6)

# 2. PCA reconstruction error == discarded eigenvalues
rng = np.random.default_rng(603)
X = rng.standard_normal((200, 3))
X_c = X - X.mean(axis=0)
S = (X_c.T @ X_c) / len(X_c)
eigvals, eigvecs = np.linalg.eigh(S)
idx = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
B_M = eigvecs[:, :2]
P_M = B_M @ B_M.T
recon_err = np.mean(np.sum((X_c - X_c @ P_M.T) ** 2, axis=1))
assert np.isclose(recon_err, eigvals[2], atol=1e-12)
print("Prompt 3: Ridge Shrinkage, Pseudoinverse & PCA Projection Verified!")
```

**Textbook Grounding**:  
- Mathematics for Machine Learning (MML) §9.2.3 Eq. (9.34), §10.3 Eqs. (10.26)–(10.29)  
- Introduction to Applied Linear Algebra (VMLS) §11.5 (p. 215), §15.1–15.4

---

### Prompt 4: GATE CS/DA Shortcut — Rank-1 Projector Spectrum & Instant Pythagorean Split
**Question**:  
Given any nonzero $\mathbf{u} \in \mathbb{R}^n$ and $P = \frac{\mathbf{u}\mathbf{u}^\top}{\|\mathbf{u}\|_2^2}$, how can you read off $\operatorname{rank}(P)$, $\det(P)$ ($n \ge 2$), $\operatorname{Tr}(P)$, all eigenvalues, $(I - P)^k$, and the lengths $\|P\mathbf{y}\|_2$, $\|(I - P)\mathbf{y}\|_2$ in under 15 seconds without forming $P$?

**Intuitive & Mathematical Proof**:  
1. *Instant Eigenspectrum*: $P\mathbf{u} = \frac{\mathbf{u}(\mathbf{u}^\top\mathbf{u})}{\|\mathbf{u}\|_2^2} = \mathbf{u}$ gives one eigenvalue $\lambda_1 = 1$, while for every $\mathbf{w} \in \mathbf{u}^\perp$ ($n - 1$ linearly independent directions), $P\mathbf{w} = \mathbf{0}$ gives $n - 1$ eigenvalues equal to $0$.
2. *Rank, Trace, Determinant & Powers*: Immediately $\operatorname{rank}(P) = 1$, $\operatorname{Tr}(P) = 1$, and for $n \ge 2$, $\det(P) = 0$ (zero eigenvalues exist). Since $I - P$ is an orthogonal projector onto $\mathbf{u}^\perp$, it is idempotent: $(I - P)^k = I - P$ for every integer $k \ge 1$.
3. *Vector Lengths Without Matrix Multiplication*:
   $$\|P\mathbf{y}\|_2 = \frac{|\mathbf{u}^\top \mathbf{y}|}{\|\mathbf{u}\|_2}, \qquad \|(I - P)\mathbf{y}\|_2 = \sqrt{\|\mathbf{y}\|_2^2 - \|P\mathbf{y}\|_2^2}$$

**Executable NumPy Verification**:
```python
import numpy as np

u = np.array([1.0, 2.0, 2.0])
y = np.array([3.0, 3.0, 0.0])
P = np.outer(u, u) / np.dot(u, u)
I_minus_P = np.eye(3) - P

assert np.linalg.matrix_rank(P) == 1
assert np.isclose(np.trace(P), 1.0)
assert np.isclose(np.linalg.det(P), 0.0)
assert np.allclose(np.sort(np.linalg.eigvalsh(P)), [0.0, 0.0, 1.0])
assert np.allclose(np.linalg.matrix_power(I_minus_P, 7), I_minus_P)

proj_len = abs(np.dot(u, y)) / np.linalg.norm(u)
res_len = np.sqrt(np.linalg.norm(y)**2 - proj_len**2)
assert np.isclose(proj_len, np.linalg.norm(P @ y)) and np.isclose(proj_len, 3.0)
assert np.isclose(res_len, np.linalg.norm(I_minus_P @ y)) and np.isclose(res_len, 3.0)
print("Prompt 4: GATE Rank-1 Projector & Instant Pythagorean Split Verified!")
```

**Textbook Grounding**:  
- Mathematics for Machine Learning (MML) §3.8.1 Eqs. (3.41)–(3.46), Example 3.10 (pp. 82–85)  
- GATE CS/DA Linear Algebra: Rank-1 Outer Product Spectrum & Orthogonal Projection Shortcuts
