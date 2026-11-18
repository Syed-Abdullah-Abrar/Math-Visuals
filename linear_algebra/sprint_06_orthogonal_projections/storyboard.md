# Part 2: 4-Act Visual Storyboard — Orthogonal Projections onto 1D Lines & $k$-D Subspaces

## Canvas & 3D Camera Configuration
- **Resolution**: $1920 \times 1080$ (16:9 widescreen, 60 fps production / 15 fps preview)
- **Background Color**: `#1C1C24` (Deep obsidian slate)
- **3D Coordinate Engine**: ManimCE v0.21.0 `ThreeDScene` with `ThreeDAxes(x_range=[-4, 4, 1], y_range=[-4, 4, 1], z_range=[-4, 4, 1], x_length=7.0, y_length=7.0, z_length=5.0)`
- **Initial Camera Orientation**: $\phi = 70^\circ$, $\theta = -45^\circ$
- **Dynamic Camera Action**: `self.begin_ambient_camera_rotation(rate=0.08)` for continuous 3D depth perception
- **HUD Pinning Rule**: All 2D text, equations, live `DecimalNumber` counters, and the corner error graph are viewport-pinned via `self.add_fixed_in_frame_mobjects(...)` inside dark backing `RoundedRectangle` cards (`make_hud_panel`, `fill_opacity=0.92`, `stroke_width=1.5`, `z_index=10`). Dynamic `DecimalNumber` objects re-register via `self.camera.add_fixed_in_frame_mobjects(mob)` after `set_value(...)`. Clean teardown uses `FadeOut` followed by `remove_fixed_in_frame_mobjects` and `self.remove`.
- **Semantic Color Palette**:
  - Basis Column 1 ($\mathbf{a}_1$): Green `#83C167`
  - Basis Column 2 ($\mathbf{a}_2$): Red `#FC6255`
  - Input / Candidate Projection ($\hat{\mathbf{y}}$, $\pi_U(\mathbf{x})$): Blue `#58C4DD`
  - Target Subspace ($U = \operatorname{span}(\mathbf{b})$ or $\mathcal{C}(A)$): Teal `#2A9D8F` (surface opacity 0.35)
  - Off-Plane Target ($\mathbf{x}_{1D}, \mathbf{y}$): Gold `#F9E498`
  - Orthogonal Residual / Error ($\mathbf{r} = \mathbf{y} - \hat{\mathbf{y}}$): Coral Red `#F7768E`
  - Orthogonal Elbow / Right-Angle Marker: Yellow `#F7D96F`
  - Affine Flat / Ridge Rescue: Yellow `#F7D96F` / Orange `#FF9E64`
  - Origin Anchor $\mathbf{0}$: White `#FFFFFF`

---

## Act 1: Concrete Spatial Puzzle — 1D Line Projection & Idempotence (0:00 – 0:15)
**Pedagogical Objective**: Establish the 1D line projection via MML Example 3.10 ($\mathbf{b} = [1, 2, 2]^\top$, $\mathbf{x}_{1D} = [3, 3, 0]^\top$), verify idempotence $\pi^2 = \pi$, and pose the 2D subspace minimum-distance puzzle.

### Act 1 Choreography Table
| Time (s) | Phase | Camera / 3D Canvas Action | Pinned HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| 0:00–0:03 | 1D Line & Target Setup | `ThreeDAxes` and white origin dot fade in ($\phi=70^\circ, \theta=-45^\circ$, ambient rotation `0.08`). Teal line $\operatorname{span}(\mathbf{b})$ along $\mathbf{b} = [1, 2, 2]^\top$ and gold target $\mathbf{x}_{1D} = [3, 3, 0]^\top$ appear. | `UL`: `"Sprint 6: Orthogonal Projections"` / `"Act 1: 1D Line Projection"`. `UR`: $\mathbf{b}=[1,2,2]^\top$, $\mathbf{x}=[3,3,0]^\top$. | 1D subspace $U = \operatorname{span}(\mathbf{b}) \subset \mathbb{R}^3$, $\|\mathbf{b}\|_2 = 3$. |
| 0:03–0:07 | Sliding Ruler Projection | Blue candidate dot $\lambda\mathbf{b}$ slides from $\lambda = 0$ to $\lambda = 1$ as coral residual tether $\mathbf{x} - \lambda\mathbf{b}$ snaps perpendicular to $\mathbf{b}$. | `UR` updates: $\lambda = \frac{\mathbf{b}^\top\mathbf{x}}{\|\mathbf{b}\|^2} = 1$, $P_1 = \frac{\mathbf{b}\mathbf{b}^\top}{\|\mathbf{b}\|^2}$. | $\langle \mathbf{x} - \lambda\mathbf{b}, \mathbf{b} \rangle = 0 \implies \hat{\mathbf{y}}_{1D} = [1, 2, 2]^\top$. |
| 0:07–0:10 | 3D Right-Angle & Idempotence | Yellow 3D right-angle elbow (`VMobject`) illuminates at $[1, 2, 2]^\top$ between $\mathbf{b}$ and $\mathbf{r}_{1D} = [2, 1, -2]^\top$. Pulse on $\hat{\mathbf{y}}_{1D}$ illustrates $\pi(\pi(\mathbf{x})) = \pi(\mathbf{x})$. | `DL`: Idempotence badge $\pi^2 = \pi$ ($P_1^2 = P_1$, $\operatorname{Tr}(P_1) = 1$). | $3^2 + 3^2 = (3\sqrt{2})^2$ and $P_1 \hat{\mathbf{y}}_{1D} = \hat{\mathbf{y}}_{1D}$. |
| 0:10–0:15 | 2D Plane Puzzle Handoff | 1D objects hold for puzzle preview posing the 2D subspace minimum-distance question, then Act 1 unpins cleanly. | `DR`: `"Next: 2D Subspace U = C(A)"` / `"Where on U is ||y - y_hat||_2 minimized?"` | Transition from 1D line to $k=2$ subspace $U \subset \mathbb{R}^3$. |

---

## Act 2: Dynamic Deformation — Sliding $\hat{\mathbf{y}} = A\mathbf{x}(s)$ & Pinned Corner Error Graph (0:15 – 0:36)
**Pedagogical Objective**: Sweep a live candidate $\hat{\mathbf{y}}(s) = A\mathbf{x}(s)$ across the 2D subspace plane $U = \mathcal{C}(A)$ while plotting $d(s) = \|\mathbf{y} - \hat{\mathbf{y}}(s)\|_2 = \sqrt{6 + 2s^2}$ in a pinned 2D corner graph, proving the global minimum occurs precisely when $\mathbf{y} - \hat{\mathbf{y}} \perp \mathcal{C}(A)$.

### Act 2 Choreography Table
| Time (s) | Phase | Camera / 3D Canvas Action | Pinned HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| 0:15–0:17 | Plane & Target Lock | Translucent teal plane $U = \mathcal{C}(A)$ (`x - 2y + z = 0`) expands with green $\mathbf{a}_1 = [1,1,1]^\top$, red $\mathbf{a}_2 = [0,1,2]^\top$, and gold target $\mathbf{y} = [3, -1, 1]^\top$. | `UL`: `"Act 2: Minimum Distance on k-D Subspace"`. `DR`: Pinned 2D `Axes` error graph ($s \in [-2, 1.5]$, $d \in [0, 4]$). | $U = \mathcal{C}(A)$, $\dim(U) = 2$, normal $\mathbf{n} = [1, -2, 1]^\top$. |
| 0:17–0:23 | Live Trajectory Sweep (`s: -2 -> 0`) | `ValueTracker(s)` sweeps from $-2.0$ to $0.0$. Blue dot $\hat{\mathbf{y}}(s) = [2+s, 1, -s]^\top$ glides across $U$ with dynamic blue segment and coral error tether $\mathbf{y} - \hat{\mathbf{y}}(s)$. Pinned corner curve & dot descend the parabola. | `DL`: Live `DecimalNumber` counters `s` and `d(s) = ||y - Ax(s)||_2`. `DR`: Error curve traces $d(s) = \sqrt{6 + 2s^2}$ down to $\sqrt{6} \approx 2.449$. | Monotone descent to unique minimizer $\hat{\mathbf{x}} = [2, -1]^\top$ at $s = 0$. |
| 0:23–0:27 | Orthogonality Illumination at $s=0$ | Sweep pauses at $s = 0$ ($\hat{\mathbf{y}}^* = [2, 1, 0]^\top$, $\mathbf{r}^* = [1, -2, 1]^\top$). Dual yellow 3D right-angle elbows light up at $\hat{\mathbf{y}}^*$ against $\mathbf{a}_1$ and $-\hat{\mathbf{y}}^*$. | `UR` highlights below: `"Orthogonality Principle: A^T(y - Ax_hat) = 0!"`. | $A^\top \mathbf{r}^* = \mathbf{0}$, global min $d(0) = \sqrt{6} \approx 2.4495$. |
| 0:27–0:36 | Overshoot (`s: 0 -> 1.5`) & Return | `s` continues to $+1.5$ (error climbs back up to $\sqrt{10.5} \approx 3.240$), then snaps smoothly back to $s = 0$ to cement strict convexity; updaters cleared and HUDs unpinned. | `DL` & `DR`: Live readouts climb past $\sqrt{6}$ then lock at $s = 0.00$, $d = 2.449$. | $\forall \mathbf{x} \ne \hat{\mathbf{x}}: \|\mathbf{y} - A\mathbf{x}\|_2 > \|\mathbf{y} - A\hat{\mathbf{x}}\|_2$. |

---

## Act 3: Color-Coded Algebraic Morph — Normal Equations, $P$, $I-P$ & Pythagoras (0:36 – 0:52)
**Pedagogical Objective**: Derive $P = A(A^\top A)^{-1}A^\top$ from $A^\top(\mathbf{y} - A\hat{\mathbf{x}}) = \mathbf{0}$, reveal the orthogonal complement $I - P$ along $U^\perp$, and verify the Pythagorean Theorem and spectral projector invariants.

### Act 3 Choreography Table
| Time (s) | Phase | Camera / 3D Canvas Action | Pinned HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| 0:36–0:40 | Orthogonal Split Triangle | 3D triangle forms from origin: $\mathbf{y} = [3,-1,1]^\top$ (Gold), $\hat{\mathbf{y}}^* = P\mathbf{y} = [2,1,0]^\top$ (Blue) on teal plane $U$, and $\mathbf{r}^* = (I-P)\mathbf{y} = [1,-2,1]^\top$ (Coral) along normal line $U^\perp$. | `UL`: `"Act 3: Normal Equations & Projector P"`. `UR`: $A^\top(\mathbf{y} - A\hat{\mathbf{x}}) = \mathbf{0} \implies A^\top A\hat{\mathbf{x}} = A^\top\mathbf{y}$. | Direct sum $\mathbb{R}^3 = U \oplus U^\perp$, $U^\perp = \mathcal{N}(A^\top)$. |
| 0:40–0:44 | Algebraic Morph to $P$ | `UR` panel transitions to closed-form $P = A(A^\top A)^{-1}A^\top$ and $I - P$ while normal line $U^\perp$ and $\mathbf{r}^*$ draw from origin. | `UR`: Color-coded $P = \frac{1}{6}\begin{bmatrix} 5 & 2 & -1 \\ 2 & 2 & 2 \\ -1 & 2 & 5 \end{bmatrix}$ and $I - P$. | $A^\top A = \begin{bmatrix} 3 & 3 \\ 3 & 5 \end{bmatrix}$ ($\det = 6 \ne 0$). |
| 0:44–0:47 | Pythagorean Check | Highlight lengths along triangle edges: $\|\mathbf{y}\|^2 = 11$, $\|P\mathbf{y}\|^2 = 5$, $\|(I-P)\mathbf{y}\|^2 = 6$, with 3D right-angle elbow at $\hat{\mathbf{y}}^*$. | `DL`: $\|\mathbf{y}\|_2^2 = \|P\mathbf{y}\|_2^2 + \|(I-P)\mathbf{y}\|_2^2 \iff 11 = 5 + 6$. | $P(I - P) = \mathbf{0} \implies \langle P\mathbf{y}, (I-P)\mathbf{y} \rangle = 0$. |
| 0:47–0:52 | Projector Invariants Card | Complement ray $U^\perp$ and plane $U$ rotate while invariants summary card settles in `DR`, then Act 3 cleans up. | `DR`: $P^2 = P$, $P^\top = P$, $\operatorname{Tr}(P) = \operatorname{rank}(P) = 2$, $\lambda(P) = \{0, 1\}$. | Complete spectral & algebraic signature of orthogonal projectors. |

---

## Act 4: Degenerate Stress Test — Affine Flats, Rank Deficiency, Lauchli & Oblique Projectors (0:52 – 1:10)
**Pedagogical Objective**: Contrast orthogonal linear projections against shifted affine subspaces, rank-deficient normal equations (pseudoinverse $AA^+$ and Ridge), Lauchli $\kappa^2$ ill-conditioning, and asymmetric oblique projectors ($P_{\text{ob}}^\top \ne P_{\text{ob}}$).

### Act 4 Choreography Table
| Time (s) | Phase | Camera / 3D Canvas Action | Pinned HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| 0:52–0:57 | Affine Subspace Shift | Translucent yellow plane $L = \mathbf{x}_0 + U$ ($\mathbf{x}_0 = [0, 0, 1]^\top$) appears shifted above teal $U$. $\mathbf{y}_{\text{aff}} = [3,-1,2]^\top$ drops orthogonally to $\pi_L(\mathbf{y}_{\text{aff}}) = [2,1,1]^\top$. | `UL`: `"Act 4: Affine, Rank-Deficient & Oblique"`. `UR`: $\pi_L(\mathbf{x}) = \mathbf{x}_0 + P(\mathbf{x} - \mathbf{x}_0)$. | Distance preserved ($d = \sqrt{6}$), yet $\pi_L(\mathbf{0}) \ne \mathbf{0}$ (not linear). |
| 0:57–1:03 | Rank Deficiency & Lauchli | $L$ fades; degenerate columns $A_{\text{deg}} = [\mathbf{a}_1 \mid 2\mathbf{a}_1]$ collapse onto 1D line $\operatorname{span}([1,1,1]^\top)$ ($\det(A^\top A) = 0$). Residual vector drops via $AA^+$ / Ridge rescue. | `UR` updates: $\det(A^\top A) = 0 \implies A^+$, Ridge $(A^\top A + \lambda I)^{-1}A^\top$. `DL`: Lauchli warning $\kappa(A_\varepsilon^\top A_\varepsilon) = \kappa(A_\varepsilon)^2 \approx 2\times 10^{16}$ (use QR/SVD!). | Pseudoinverse $AA^+$ & Ridge ($\lambda > 0$) restore unique stable projection. |
| 1:03–1:10 | Oblique vs Orthogonal Contrast | Restore teal plane $U$ and target $\mathbf{y}_{\text{test}} = [1, 1, 3]^\top$. Show vertical skewed projection $P_{\text{ob}}\mathbf{y}_{\text{test}} = [1,1,1]^\top$ (coral ray, error $2.0$) vs shortest perpendicular drop $P\mathbf{y}_{\text{test}} = \frac{1}{3}[2,5,8]^\top$ (blue, error $\sqrt{6}/3 \approx 0.816$). | `DR`: Oblique Verdict Card: $P_{\text{ob}}^2 = P_{\text{ob}}$, $P_{\text{ob}}^\top \ne P_{\text{ob}}$, $\|P_{\text{ob}}\|_2 = \sqrt{6} > 1$, error $2.0 > 0.816$. | Symmetry $P^\top = P$ is mandatory for contractiveness ($\|P\|_2 = 1$) & minimum distance. |
