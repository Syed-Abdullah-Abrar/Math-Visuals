# Part 2: 4-Act Visual Storyboard — Orthogonal Matrices, Rotations $SO(n)$ & Givens/Householder

## Scene Specifications & Global Canvas Layout
- **Frame Resolution**: `1920x1080` (16:9 widescreen, world bounds `[-7.11, 7.11] x [-4.0, 4.0]`)
- **Background Color**: Dark slate (`#1C1C24`)
- **Coordinate Grids (Strict Isotropic 1:1 Scaling)**:
  - **Split Dual Planes (Act 1)**: `NumberPlane` with `x_range=[-2.5, 2.5, 1]`, `y_range=[-2.0, 2.0, 1]`, `x_length=4.5, y_length=3.6` (exact isotropic `0.9` scale) centered at `LEFT * 3.1 + DOWN * 0.1` and `RIGHT * 3.1 + DOWN * 0.1`.
  - **Central Plane (Acts 2 & 3)**: `NumberPlane` with `x_range=[-3, 3, 1]`, `y_range=[-2.2, 2.2, 1]`, `x_length=6.6, y_length=4.84` (exact isotropic `1.1` scale) shifted `DOWN * 0.05`.
  - **Isometric 3D Projection (Act 2 Beat 4)**: Side-by-side NumPy-projected 3D wireframe cubes + RGB triads $(\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3)$ driven by exact $3 \times 3$ matrices $R_1(90^\circ), R_2(90^\circ) \in SO(3)$ so 2D circular precision in Acts 1/3 is never distorted by perspective foreshortening.
- **Origin Anchor**: Permanent high-contrast white dot at `(0, 0)` labeled $\mathbf{0}$ (`z_index=8`) plus ghost unit circle $S^1$ (`GHOST_COLOR = "#414868"`).
- **HUD Panel Configuration (`make_hud_panel`, `z_index=10`, corner `buff=0.25`)**:
  - `UL` (Upper-Left): Act title & pedagogical subtitle (`font_size=18..19` / `14`)
  - `UR` (Upper-Right): Matrix formula & color-coded algebraic readout (`font_size=16..18`)
  - `DL` (Lower-Left): Pre-sized live `DecimalNumber` telemetry (`det`, `||Qx||`, angle, $x_2$, or $n$)
  - `DR` (Lower-Right): Invariant badge / stability contrast readout

---

## Act 1: Concrete Spatial Puzzle (Isometry vs. Area-Preserving Shear Distortion)
- **Duration**: 0:00 – 0:18 (18.0 s)
- **Pedagogical Objective**: Refute the misconception that $\det(A) = 1$ implies rigid rotation. Contrast area-preserving shear $M_{\text{shear}} = \begin{bmatrix} 1 & 1.2 \\ 0 & 1 \end{bmatrix}$ ($\det = 1$, yet warps $S^1$ into a tilted ellipse and collapses $90^\circ \to 39.81^\circ$) against Pythagorean orthogonal $Q_{\text{pyth}} = \begin{bmatrix} 0.6 & -0.8 \\ 0.8 & 0.6 \end{bmatrix} \in SO(2)$ ($\det = 1$, $S^1$ locked, $\|\mathbf{w}\|_2$ and $90^\circ$ preserved).

### Visual Assets & Setup
- Left Split Plane (`"Area-Preserving Shear M_shear"` in `ERROR_COLOR` `#F7768E`): ghost circle $S^1$, live deformed circle `ParametricFunction`, basis $\mathbf{e}_1, \mathbf{e}_2$, test vector $\mathbf{w} = [0.9, 0.8]^\top$ (`VEC_COLOR` `#F7D96F`), `safe_angle_arc`.
- Right Split Plane (`"Orthogonal Q in SO(2)"` in `I_HAT_COLOR` `#83C167`): ghost circle $S^1$, live circle, rotated orthonormal basis $\mathbf{q}_1, \mathbf{q}_2$, test vector $\mathbf{w} = [0.9, 0.8]^\top$, `safe_right_angle` elbow marker.

### Choreography & Timeline
| Time (s) | Phase | Camera / Canvas Action | HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| **0:00–0:04** | Dual Frame Setup | Left & Right isotropic split planes fade in with ghost unit circles, unit circles, $\hat{\imath}$ (green), $\hat{\jmath}$ (red), right-angle elbows, and yellow vector $\mathbf{w} = [0.9, 0.8]^\top$. | UL: "Act 1: Isometry vs. Shear Distortion" <br> UR: $M_{\text{shear}} = \begin{bmatrix} 1 & 1.2 \\ 0 & 1 \end{bmatrix}$ vs $Q = \begin{bmatrix} 0.6 & -0.8 \\ 0.8 & 0.6 \end{bmatrix}$ | Initial unit frames: $\det = 1.00$, $\angle = 90.0^\circ$ |
| **0:04–0:11** | Continuous Homotopy vs Geodesic Rotation | `t_tracker: 0 -> 1` (`run_time=3.4`, `smooth`). Left linearly interpolates $M(t) = (1-t)I + tM_{\text{shear}}$: unit circle squashes into tilted ellipse ($\sigma_1 \approx 1.77, \sigma_2 \approx 0.57$), $\mathbf{w}$ stretches from $1.20$ to $2.02$, and angle shrinks $90.0^\circ \to 39.81^\circ$. Right sweeps geodesic $R(t \cdot 53.13^\circ)$: live circle stays locked on ghost $S^1$, `RightAngle` remains $90.0^\circ$, $\|\mathbf{w}\|_2 = 1.20$ unchanged. | DL: `\|\|M w\|\| = 1.20 -> 2.02`, `\angle = 90.0° -> 39.81°` (`ERROR_COLOR`) <br> DR: `\|\|Q w\|\| = 1.20`, `\angle = 90.0°` (`I_HAT_COLOR`) | $\det(M_{\text{shear}}) = 1$ preserves area ONLY; $Q^\top Q = I \implies \|Q\mathbf{x}\|_2 = \|\mathbf{x}\|_2$ |
| **0:11–0:18** | Invariant Verdict | Pulse ghost vs deformed ellipse on Left and rigid `RightAngle` + $S^1$ lock on Right. Clear updaters and fade out Act 1 cleanly. | DR: "det=1 is NOT enough: need Q^T Q = I" | Singular values: $\sigma(M) = \{1.77, 0.57\}$ vs $\sigma(Q) = \{1, 1\}$ |

---

## Act 2: Dynamic Deformation & Ghost Space ($SO(2)$, Reflections, Cartan–Dieudonné & $SO(3)$ Non-Commutativity)
- **Duration**: 0:18 – 0:46 (28.0 s)
- **Pedagogical Objective**: Explore the two disconnected halves of $O(2)$ ($\det = +1$ vs $\det = -1$), prove the 2D Cartan–Dieudonné reflection composition $H(\alpha_2)H(\alpha_1) = R(2\Delta\alpha)$, and contrast 2D Abelian $SO(2)$ against 3D non-commutative $SO(3)$.

### Visual Assets & Setup
- Central isotropic plane (`scale=1.1`), ghost circle, live circle, orthonormal frame $(\mathbf{q}_1, \mathbf{q}_2)$, `safe_right_angle`, and chiral orientation triangle + directed arc (CCW green `#83C167` for $\det = +1$, CW coral `#F7768E` for $\det = -1$).
- Beat 4: Side-by-side 3D isometric wireframe cubes + RGB triads at `LEFT * 3.1 + DOWN * 0.15` and `RIGHT * 3.1 + DOWN * 0.15`.

### Choreography & Timeline
| Time (s) | Phase | Camera / Canvas Action | HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| **0:18–0:24** | Beat 1: Continuous $SO(2)$ Sweep | `theta_tracker: 0 -> 2*pi/3 -> 0` (`run_time=3.5`). Frame glides rigidly around $S^1$; `RightAngle` and CCW chiral loop remain invariant. | UL: "Act 2: SO(2), Reflections & SO(3)" <br> DL: `det(R(θ)) = +1.00`, `CCW (+1)` | Proper rotations form connected manifold $SO(2)$ |
| **0:24–0:31** | Beat 2: Improper Reflection ($\det = -1$) | Mirror line $\ell_1$ at $\alpha_1 = 15^\circ$ draws. `s_perp: +1 -> -1` (`run_time=2.5`) passes through singular line collapse ($\det = 0$) and flips frame across $\ell_1$. Orientation loop turns CW (`ERROR_COLOR`), while `RightAngle` holds $90^\circ$. | UR: $H(\alpha_1) = \begin{bmatrix} \cos 2\alpha_1 & \sin 2\alpha_1 \\ \sin 2\alpha_1 & -\cos 2\alpha_1 \end{bmatrix}$ <br> DL: `det(H_1) = -1.00`, `CW (-1)` | Topological wall $\det = 0$ separates $SO(2)$ from $O(2)\setminus SO(2)$ |
| **0:31–0:38** | Beat 3: Cartan–Dieudonné $H_2 H_1 = R(2\Delta\alpha)$ | Add second mirror $\ell_2$ at $\alpha_2 = 45^\circ$ ($\Delta\alpha = 30^\circ$). Vector $\mathbf{x} = [1.5, 0.2]^\top$ bounces across $\ell_1$ (`H_1 x`) then across $\ell_2$ (`H_2 H_1 x = R(60°) x`). Arc sweeps $2\Delta\alpha = 60^\circ$; CCW orientation restored! | UR: $H_2 H_1 = R(2\Delta\alpha) = R(60^\circ)$ <br> DR: `det = (-1)(-1) = +1` | Product of 2 reflections in $\mathbb{R}^2$ is always in $SO(2)$ |
| **0:38–0:46** | Beat 4: 2D Commutative vs 3D $SO(3)$ Non-Commutative | Central plane fades out; side-by-side 3D cubes appear. Left applies $R_1(90^\circ)$ then $R_2(90^\circ)$ ($\mathbf{e}_1 \to [0,0,-1]^\top$, down); Right applies $R_2(90^\circ)$ then $R_1(90^\circ)$ ($\mathbf{e}_1 \to [0,1,0]^\top$, right). | UR: `2D: R(α)R(β)=R(β)R(α)` vs `3D: R_1 R_2 ≠ R_2 R_1` <br> DL: `(R_2 R_1)e_1 = [0,0,-1]^T` <br> DR: `(R_1 R_2)e_1 = [0,1,0]^T` | $SO(3)$ is non-Abelian: $\|R_1 R_2 - R_2 R_1\|_F = 2.0$ |

---

## Act 3: Color-Coded Algebraic Morph ($Q^\top Q = I$, Givens Rotation & Householder Reflection)
- **Duration**: 0:46 – 1:12 (26.0 s)
- **Pedagogical Objective**: Derive $Q^{-1} = Q^\top$ visually, then contrast the two canonical orthogonal zeroing primitives on $\mathbf{x} = [1.2, 1.6]^\top$ ($r = 2.0$, the $(0,2)$ planar slice of $\mathbf{x}_{\text{giv}} = [3, 2, 4]^\top$ scaled by $0.4$): Givens circular glide vs. Householder hyperplane bounce.

### Visual Assets & Setup
- Central isotropic plane (`scale=1.1`), unit circle $S^1$ (Beat 1) and radius-$2.0$ circle (Beats 2 & 3).
- Color-coded `MathTex` equations in `UR`: `Q_PYTH` (`#83C167`, `#FC6255`), Givens `G` (`#58C4DD`), Householder `H` (`#9A72AC`, `#F7D96F`, `#F9E498`).
- Two `safe_right_angle` markers in Beat 3: at origin ($\hat{\mathbf{v}} \perp \mathbf{v}^\perp$) and chord midpoint $[-0.4, 0.8]^\top$.

### Choreography & Timeline
| Time (s) | Phase | Camera / Canvas Action | HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| **0:46–0:53** | Beat 1: Algebraic Morph $Q^\top Q = I$ | Orthonormal columns $\mathbf{q}_1 = [0.6, 0.8]^\top, \mathbf{q}_2 = [-0.8, 0.6]^\top$ display with `RightAngle`. Test vector $\mathbf{x} = [1.4, 0.4]^\top$ rotates via $Q$ to $Q\mathbf{x}$, then snaps cleanly back to $\mathbf{x}$ under $Q^\top$. | UR: $Q^\top Q = I_2 \implies Q^{-1} = Q^\top$ <br> DL: $\det(Q)^2 = 1 \implies \det(Q) = \pm 1$ | Orthonormal columns invert by transposition |
| **0:53–1:02** | Beat 2: Givens Rotation $G(0, 2, \theta)$ | Radius-$2.0$ circle with $\mathbf{x} = [1.2, 1.6]^\top$ (`c=0.6, s=-0.8`). `phi_tracker: 0 -> -53.13°` (`run_time=2.5`): arrow glides along circle to $[2.0, 0.0]^\top$ while dashed $x_j$ drop collapses to zero. | UR: $G = \begin{bmatrix} 0.6 & 0.8 \\ -0.8 & 0.6 \end{bmatrix} \in SO(2)$ <br> DL: `x_j: 1.60 -> 0.00`, `\|\|x\|\| = 2.00` | Planar rotation zeros single entry $x_j$ ($n-2$ coords untouched) |
| **1:02–1:12** | Beat 3: Householder Reflector $H$ | Reset $\mathbf{x} = [1.2, 1.6]^\top$. Draw mirror line $\mathbf{v}^\perp$, unit normal $\hat{\mathbf{v}}$ ($\mathbf{v} = [3.2, 1.6]^\top$), dashed chord $\mathbf{x} \leftrightarrow [-2, 0]^\top$, and two `RightAngle`s. `s_hh: 0 -> 2` bounces $\mathbf{x}$ straight across $\mathbf{v}^\perp$ onto $[-2.0, 0.0]^\top$. | UR: $H = I - 2\frac{\mathbf{v}\mathbf{v}^\top}{\|\mathbf{v}\|^2},\; H^2 = I,\; \det(H) = -1$ <br> DR: `x -> -sgn(x_1)\|\|x\|\| e_1` | Rank-1 reflection zeros entire subcolumn in 1 step |

---

## Act 4: Degenerate & Edge-Case Stress Test (Cancellation, $Q^{1000}$ Drift & Sparsity Tradeoff)
- **Duration**: 1:12 – 1:36 (24.0 s)
- **Pedagogical Objective**: Expose floating-point failure modes and numerical tradeoffs: (1) Householder catastrophic cancellation at $\mathbf{x} = [1.0, 10^{-9}]^\top$ + Givens $x_i = x_j = 0$ guard, (2) $Q^{1000}$ drift stability vs non-orthogonal blowup/collapse, and (3) Givens sparse vs. Householder dense $4 \times 4$ elimination.

### Visual Assets & Setup
- Beat 1: Side-by-side comparison cards (`ERROR_COLOR` vs `I_HAT_COLOR`) illustrating IEEE-754 mantissa loss for $\mathbf{v}_{\text{bad}} = \mathbf{x} - \|\mathbf{x}\|\mathbf{e}_1$ vs constructive $\mathbf{v}_{\text{good}} = \mathbf{x} + \operatorname{sgn}(x_1)\|\mathbf{x}\|\mathbf{e}_1$, plus Givens $(0,0) \implies G = I$ badge.
- Beat 2: Left isotropic orbit plane (`x_length=4.0, y_length=4.0`) + Right norm-vs-$n$ `Axes(x_range=[0, 1000, 250], y_range=[0, 5, 1])`.
- Beat 3: Side-by-side $4 \times 4$ cell matrices comparing Givens surgical 2-row update vs Householder bulk column zeroing + Master Summary Card.

### Choreography & Timeline
| Time (s) | Phase | Camera / Canvas Action | HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| **1:12–1:20** | Beat 1: Cancellation & Zero-Pair Guard | Left card shows $\mathbf{x} = [1, 10^{-9}]^\top \implies \mathbf{v}_{\text{bad}} = [0, 10^{-9}]^\top$, failing with $H_{\text{bad}}\mathbf{x} = [1, -10^{-9}]^\top$. Right card shows $\mathbf{v}_{\text{good}} = [2, 10^{-9}]^\top \implies H_{\text{good}}\mathbf{x} = [-1, 0]^\top$. Bottom badge shows Givens `hypot(0,0)=0 -> G=I`. | UL: "Act 4: Stability & Edge Cases" <br> DR: `\|(H_good x)_2\| < 1e-15` | $+\operatorname{sgn}(x_1)$ guarantees $\|\mathbf{v}\|_2 \ge \|\mathbf{x}\|_2$ |
| **1:20–1:28** | Beat 2: $Q^{1000}$ Repeated Drift | `n_tracker: 0 -> 1000` (`run_time=3.0`, `linear`). Exact $Q_{\text{drift}}^n$ stays locked on radius-1 circle (`drift ~ 1.47e-13`), while $1.0015 R$ spirals out to `4.48` and $0.998 R$ collapses to `0.14`. | DL: `n = 0..1000`, `\|\|Q^n x\|\| = 1.00` <br> DR: `\|\|1.0015^n\|\| -> 4.48`, `\|\|0.998^n\|\| -> 0.14` | Unit eigenvalues $|\lambda_i| = 1$ bound $Q^{1000}$ error to $\mathcal{O}(n\epsilon_{\text{mach}})$ |
| **1:28–1:36** | Beat 3: Sparsity Grid & Master Summary | $4 \times 4$ grids animate Givens surgical zero vs Householder bulk column zeroing, followed by centered Master Summary Card (`wait(2.5)`). | Center HUD Card: $Q^\top Q=I$, $\det=\pm 1$, $SO(n)$ vs $O(n)\setminus SO(n)$, Givens vs Householder | Givens: sparse/banded $\mathcal{O}(n)$; Householder: dense $QR$ ($2\times$ fewer FLOPs) |
