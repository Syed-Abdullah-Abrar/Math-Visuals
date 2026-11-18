# Part 2: 4-Act Visual Storyboard — Gram-Schmidt Orthonormalization & QR Decomposition

## Canvas & 3D Camera Configuration
- **Resolution**: $1920 \times 1080$ (16:9 widescreen, 60 fps production `-qh` / 15 fps preview `-ql`)
- **Background Color**: `#1C1C24` (Deep obsidian slate)
- **3D Coordinate Engine**: ManimCE `ThreeDScene` with **isotropic** `ThreeDAxes(x_range=[-4, 4, 1], y_range=[-4, 4, 1], z_range=[-4, 4, 1], x_length=5.6, y_length=5.6, z_length=5.6)` (`0.7` Manim units per data unit along all three axes so the unit sphere $S^2$ and $90^\circ$ angles have zero $z$-compression distortion)
- **Initial Camera Orientation**: $\phi = 70^\circ$, $\theta = -45^\circ$
- **Dynamic Camera Action**: Targeted `move_camera` / brief orbital `begin_ambient_camera_rotation(rate=0.06)` sweeps during active 3D beats, pausing ambient rotation (`stop_ambient_camera_rotation()`) during static holds and algebraic HUD beats so frame caching stays responsive at 1080p60
- **HUD Pinning Rule**: All 2D text, formulas, `DecimalNumber(..., mob_class=Text)` readouts, and matrix cards are pinned to the viewport via `self.add_fixed_in_frame_mobjects(panel)` (and re-pinned on dynamic updater ticks) inside dark `RoundedRectangle` panels (`PANEL_BG="#16161E"`, `fill_opacity=0.92`, `stroke_color="#414868"`, `stroke_width=1.5`, `z_index=10`). Calling `unpin_hud` removes panels from both `fixed_in_frame_mobjects` and `self.mobjects`
- **3D Right-Angle Rule**: All 3D perpendicularity markers use a custom `Polygon` / `VMobject` elbow helper (`create_right_angle_3d`) in 3D space (never `manim.RightAngle`, which requires $z=0$)
- **Semantic Color Palette**:
  - Column 1 / Basis 1 ($\mathbf{a}_1, \mathbf{q}_1$): Green `#83C167` (`I_HAT_COLOR`)
  - Column 2 / Basis 2 ($\mathbf{a}_2, \mathbf{q}_2$): Red `#FC6255` (`J_HAT_COLOR`)
  - Column 3 / Basis 3 ($\mathbf{a}_3, \mathbf{q}_3$): Blue `#58C4DD` (`K_HAT_COLOR`)
  - Intermediate Plane $\operatorname{span}(\mathbf{q}_1, \mathbf{q}_2)$ & Right-Angle Markers: Teal `#2A9D8F` (`SUBSPACE_COLOR`)
  - Projection Shadows ($\mathbf{p}_{21}, \mathbf{p}_{12,3}$) & Stripped Vectors ($\mathbf{u}_k$): Yellow `#F7D96F` (`VEC_COLOR`)
  - Target Vector $\mathbf{b} = [1, -2, 3]^\top$: Gold `#F9E498` (`TARGET_COLOR`)
  - Degenerate / Error / CGS Drift Alert: Coral Red `#F7768E` (`ERROR_COLOR`)
  - Origin $\mathbf{0}$: White `#FFFFFF` (`ORIGIN_COLOR`)

---

## Act 1: Concrete Spatial Puzzle — Skewed 3D Frame (0:00 – 0:18)
**Pedagogical Objective**: Establish the isotropic 3D axes, present the skewed input vectors $\{\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3\}$ forming an oblique parallelepiped alongside target $\mathbf{b}$, and expose why non-zero pairwise dot products couple coordinates.

### Act 1 Choreography Table
| Time (s) | Phase | Camera / 3D Canvas Action | Pinned HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| 0:00–0:04 | Isotropic 3D Setup | `ThreeDAxes` (`[-4,4]^3`, 5.6 units) and white origin dot $\mathbf{0}$ fade in at $(\phi=70^\circ, \theta=-45^\circ)$. | `UL` Title card: `"Sprint 7: Gram-Schmidt & QR"`, `"Act 1: Skewed 3D Basis Puzzle"`. | Origin anchor $\mathbf{0} \in \mathbb{R}^3$. |
| 0:04–0:10 | Skewed Column Frame | Three 3D arrows sprout from $\mathbf{0}$: $\mathbf{a}_1=[2,1,2]^\top$ (Green), $\mathbf{a}_2=[0,2,2]^\top$ (Red), $\mathbf{a}_3=[-1,-1,3]^\top$ (Blue) + faint parallelepiped wireframe. | `UR` Matrix card: $A = [\mathbf{a}_1 \mid \mathbf{a}_2 \mid \mathbf{a}_3] = \begin{bmatrix}2&0&-1\\1&2&-1\\2&2&3\end{bmatrix}$. | Full rank $\det(A) = 18 \ne 0$, skewed frame. |
| 0:10–0:14 | Target & Coupling Probe | Gold target arrow $\mathbf{b} = [1, -2, 3]^\top$ and tip `Dot3D` appear. Brief 3s orbital sweep highlights oblique angles. | `DL` Probe card: $\mathbf{a}_1^\top\mathbf{a}_2 = 6$, $\mathbf{a}_1^\top\mathbf{a}_3 = 3$, $\mathbf{a}_2^\top\mathbf{a}_3 = 4 \ne 0$. | Non-orthogonal basis: $A^\top A$ has non-zero off-diagonals. |
| 0:14–0:18 | Orthonormalization Question | Ambient orbit pauses cleanly; parallelepiped holds in frame. | `DR` Question badge: `"How do we straighten A into an orthonormal frame Q without losing span?"` | Goal: $Q^\top Q = I_3$ with nested span equality. |

---

## Act 2: Dynamic Deformation — Sequential 3D Stripping onto $S^2$ (0:18 – 0:48)
**Pedagogical Objective**: Animate step-by-step Gram-Schmidt orthogonal stripping in 3D: normalizing $\mathbf{a}_1 \to \mathbf{q}_1$, subtracting $\operatorname{proj}_{\mathbf{q}_1}(\mathbf{a}_2)$ via `ValueTracker` with live dot-product readout $\to 0.00$ to form $\mathbf{q}_2$, and subtracting $\operatorname{proj}_{\operatorname{span}(\mathbf{q}_1,\mathbf{q}_2)}(\mathbf{a}_3)$ to form $\mathbf{q}_3$ on the unit sphere $S^2$.

### Act 2 Choreography Table
| Time (s) | Phase | Camera / 3D Canvas Action | Pinned HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| 0:18–0:23 | Unit Sphere & Step 1 ($\mathbf{q}_1$) | 3 orthogonal great-circle rings of $S^2$ + brief translucent sphere surface fade in (`Surface` fades out after reveal). $\mathbf{a}_1$ ($\|\mathbf{a}_1\|=3$) shrinks along its ray to unit $\mathbf{q}_1 = \frac{1}{3}[2,1,2]^\top$ on $S^2$. | `UL` `"Act 2: Sequential Orthogonal Stripping"`, `UR` Step 1: $\mathbf{u}_1 = \mathbf{a}_1$, $R_{11}=3$, $\mathbf{q}_1 = \mathbf{u}_1 / 3$. | $\|\mathbf{q}_1\|_2 = 1$, $\operatorname{span}(\mathbf{q}_1) = \operatorname{span}(\mathbf{a}_1)$. |
| 0:23–0:33 | Step 2: Stripping $\mathbf{a}_2 \to \mathbf{u}_2 \to \mathbf{q}_2$ | Golden shadow arrow $2\mathbf{q}_1$ and dashed drop-line appear from $\mathbf{a}_2$ to $2\mathbf{q}_1$. `ValueTracker` $t: 0 \to 1$ slides $\mathbf{v}_2(t) = \mathbf{a}_2 - 2t\mathbf{q}_1$ into perpendicular $\mathbf{u}_2 = \frac{2}{3}[-2,2,1]^\top$; 3D teal `Polygon` right-angle elbow snaps at $\mathbf{0}$; $\mathbf{u}_2$ ($\|\mathbf{u}_2\|=2$) scales onto $S^2$ as $\mathbf{q}_2$. | `DR` Live `DecimalNumber` HUD: $\mathbf{q}_1^\top \mathbf{v}_2(t): 2.00 \to 0.00$; `UR` updates $R_{12}=2, R_{22}=2$. | $\mathbf{u}_2 = \mathbf{a}_2 - (\mathbf{q}_1^\top\mathbf{a}_2)\mathbf{q}_1 \perp \mathbf{q}_1$, $\operatorname{span}(\mathbf{q}_1,\mathbf{q}_2) = \operatorname{span}(\mathbf{a}_1,\mathbf{a}_2)$. |
| 0:33–0:44 | Step 3: Stripping $\mathbf{a}_3 \to \mathbf{u}_3 \to \mathbf{q}_3$ | Translucent teal plane $\mathcal{S}_2 = \operatorname{span}(\mathbf{q}_1,\mathbf{q}_2)$ appears. Planar shadow $\mathbf{p}_{12,3} = [0,1,1]^\top = \mathbf{q}_1+\mathbf{q}_2$ and drop-line render. `ValueTracker` $t: 0\to 1$ strips $\mathbf{v}_3(t) = \mathbf{a}_3 - t\mathbf{p}_{12,3}$ into altitude $\mathbf{u}_3 = [-1,-2,2]^\top$; scales ($\|\mathbf{u}_3\|=3$) to $\mathbf{q}_3$ on $S^2$; all three 3D right-angle elbows snap at $\mathbf{0}$. | `DR` Live `DecimalNumber` HUD: $(\mathbf{q}_1^\top\mathbf{v}_3, \mathbf{q}_2^\top\mathbf{v}_3) \to (0.00, 0.00)$; `UR` updates $R_{13}=1, R_{23}=1, R_{33}=3$. | $\mathbf{u}_3 \perp \mathcal{S}_2$, $Q^\top Q = I_3$, $\det(Q) = +1$. |
| 0:44–0:48 | Orthonormal Frame Lock | Brief 3s camera orbit reveals symmetric orthogonal tripod $\{\mathbf{q}_1, \mathbf{q}_2, \mathbf{q}_3\}$ inscribed in $S^2$, then stops rotation. | `DL` Invariant badge: `"Orthonormal Frame Q: Q^T Q = I_3, ||q_i|| = 1"`. | Rigid orthonormal basis spanning $\mathbb{R}^3$. |

---

## Act 3: Algebraic Morph — Upper-Triangular $R$, Expansion & Back-Substitution (0:48 – 1:10)
**Pedagogical Objective**: Visually connect tip-to-tail reconstruction $\mathbf{a}_j = \sum_{i=1}^j R_{ij}\mathbf{q}_i$ to upper-triangular $R$ and sign uniqueness ($R_{ii}>0$), then demonstrate inversion-free expansion $\mathbf{b}=Q(Q^\top\mathbf{b})$, $R\hat{\mathbf{x}}=Q^\top\mathbf{b}$, and $|\det A|=\prod R_{ii}=18$.

### Act 3 Choreography Table
| Time (s) | Phase | Camera / 3D Canvas Action | Pinned HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| 0:48–0:56 | Tip-to-Tail $A = QR$ | Orthonormal frame $\{\mathbf{q}_1,\mathbf{q}_2,\mathbf{q}_3\}$ stays on canvas; scaled components $1\mathbf{q}_1 + 1\mathbf{q}_2 + 3\mathbf{q}_3$ assemble tip-to-tail to reconstruct $\mathbf{a}_3 = [-1,-1,3]^\top$. | `UR` Color-coded $A = Q R$ matrix card highlighting zero sub-diagonal ($R_{ij}=0, i>j$) and positive pivots $R_{ii} \in \{3,2,3\}$. | $R_{ij} = \mathbf{q}_i^\top \mathbf{a}_j$, unique thin QR via $R_{ii} > 0$. |
| 0:56–1:03 | Orthonormal Expansion $\mathbf{b}$ | Gold target $\mathbf{b}=[1,-2,3]^\top$ appears alongside its orthogonal coordinates $2\mathbf{q}_1 - 1\mathbf{q}_2 + 3\mathbf{q}_3$. | `DL` Expansion card: $Q^\top\mathbf{b} = [2, -1, 3]^\top$, $\mathbf{b} = Q(Q^\top\mathbf{b})$ (zero inversion!). | Coordinates via inner products: $c_i = \mathbf{q}_i^\top\mathbf{b}$. |
| 1:03–1:10 | Back-Sub & Volume | Static camera hold during algebraic solve reveal (`stop_ambient_camera_rotation` active for fast caching). | `DR` Solve & Volume card: $R\hat{\mathbf{x}} = Q^\top\mathbf{b} \implies \hat{\mathbf{x}}=[1,-1,1]^\top$; $|\det A| = R_{11}R_{22}R_{33} = 3\cdot 2\cdot 3 = 18$. | $\mathcal{O}(n^2)$ back-substitution & Gram identity $A^\top A = R^\top R$. |

---

## Act 4: Degenerate Stress Test — Linear Dependence & CGS vs. MGS Drift (1:10 – 1:30)
**Pedagogical Objective**: Stress-test Gram-Schmidt when $\mathbf{a}_3^{\text{deg}} \in \operatorname{span}(\mathbf{q}_1, \mathbf{q}_2)$ causes total collapse $\mathbf{u}_3 \to \mathbf{0}$ ($R_{33}=0$, early termination), and compare floating-point orthogonality drift between Classical GS, Modified GS, and Householder QR.

### Act 4 Choreography Table
| Time (s) | Phase | Camera / 3D Canvas Action | Pinned HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| 1:10–1:16 | Coplanar Degenerate Probe | Plane $\mathcal{S}_2 = \operatorname{span}(\mathbf{q}_1,\mathbf{q}_2)$ reappears with $\mathbf{q}_1, \mathbf{q}_2$. Coral arrow $\mathbf{a}_3^{\text{deg}} = \mathbf{a}_2 - \mathbf{a}_1 = [-2,1,0]^\top$ lies flush inside $\mathcal{S}_2$. | `UL` `"Act 4: Linear Dependence & Numerical Drift"`, `UR` $\mathbf{a}_3^{\text{deg}} = -1\mathbf{q}_1 + 2\mathbf{q}_2 \in \mathcal{S}_2$. | Exact linear dependence: $\det(A_{\text{deg}}) = 0$. |
| 1:16–1:22 | Zero-Residual Collapse | `ValueTracker` $t: 0\to 1$ strips shadow $(-1\mathbf{q}_1 + 2\mathbf{q}_2)$ from $\mathbf{a}_3^{\text{deg}}$; tip contracts completely into white origin dot $\mathbf{0}$. | `DL` Alert badge: $\mathbf{u}_3 = \mathbf{0} \implies R_{33} = 0$, Early Termination ($\operatorname{rank} = 2$). | VMLS Alg 5.1 Step 2 halts before $0/0$. |
| 1:22–1:30 | CGS vs. MGS vs. Householder | Static camera hold displaying crisp floating-point verdict card comparing Lauchli ($\epsilon=10^{-8}, \kappa \approx 1.73\times 10^8$) orthogonality drift. | `DR` Stability Verdict Card: `CGS: 7.1e-1 (O(eps*k^2))` vs `MGS: 1.2e-8 (O(eps*k))` vs `Householder: 2.7e-16 (O(eps))` + Normal Eq `LinAlgError`. | MGS/Householder preserve orthogonality & avoid $\kappa(A)^2$ collapse. |
