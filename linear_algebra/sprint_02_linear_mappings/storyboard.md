# Part 2: 4-Act Visual Storyboard — Linear Mappings & Matrix Composition

## Scene Specifications & Global Canvas Layout
- **Frame Resolution**: 1920x1080 (16:9 widescreen)
- **Background Color**: Dark slate (`#1C1C24`)
- **Coordinate Grid**: `NumberPlane` with x-range $[-6, 6]$, y-range $[-4, 4]$, clamped scale `x_length=9.6, y_length=6.4` (0.8 scale factor protecting viewport $[-7.11, 7.11] \times [-4.0, 4.0]$)
- **Origin Anchor**: Permanent high-contrast white dot at $(0, 0)$ labeled $\mathbf{0}$
- **HUD Panel Configuration**:
  - `UL` (Upper-Left): Module & Act title badges (`make_hud_panel`)
  - `UR` (Upper-Right): Dynamic Matrix readout $A(t)$ & Basis destination coordinates
  - `LL` (Lower-Left): Geometric metric badge ($\det(A)$, Area Scale Factor, Orientation)
  - `LR` (Lower-Right): Pedagogical cross-reference / Dual Bridge note

---

## Act 1: Concrete Spatial Puzzle (Tracking the Basis Arrow Destinations)
- **Duration**: 0:00 – 0:18 (18.0 s)
- **Pedagogical Objective**: Pose the fundamental question: "How can knowing the destinations of only two unit vectors determine the destination of every point in the entire infinite 2D plane?" Track $\hat{\imath} \to \mathbf{a}_1 = [2, 1]^\top$ and $\hat{\jmath} \to \mathbf{a}_2 = [0, 2]^\top$.

### Visual Assets & Setup
- Unit basis vectors: $\hat{\imath} = [1, 0]^\top$ in `I_HAT_COLOR` (`#83C167`, Green), $\hat{\jmath} = [0, 1]^\top$ in `J_HAT_COLOR` (`#FC6255`, Red).
- Transformed columns: $\mathbf{a}_1 = [2, 1]^\top$, $\mathbf{a}_2 = [0, 2]^\top$.
- Arbitrary input vector: $\mathbf{x} = 1.5\hat{\imath} + 1.0\hat{\jmath} = [1.5, 1.0]^\top$ in `VEC_COLOR` (`#F7D96F`, Yellow).
- Destination point: $A\mathbf{x} = 1.5\mathbf{a}_1 + 1.0\mathbf{a}_2 = [3.0, 3.5]^\top$ in `TARGET_COLOR` (`#F9E498`, Gold).

### Choreography & Timeline
| Time (s) | Phase | Camera / Canvas Action | HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| **0:00–0:04** | Frame Setup & Basis Intro | NumberPlane fades in; origin dot $\mathbf{0}$ pulses. Unit arrows $\hat{\imath}$ (green) and $\hat{\jmath}$ (red) grow from $(0, 0)$ via `GrowArrow`. | UL: "Act 1: Spatial Puzzle" <br> UR: Basis vectors $\hat{\imath}, \hat{\jmath}$ | Standard orthonormal basis $\{\hat{\imath}, \hat{\jmath}\}$ spans $\mathbb{R}^2$ |
| **0:04–0:09** | Vector Decomposition & Mystery Target | Yellow input vector $\mathbf{x} = [1.5, 1.0]^\top$ appears. Dashed projection lines show $1.5\hat{\imath} + 1.0\hat{\jmath}$. Gold target dot appears at mystery location $[3.0, 3.5]^\top$. | UL: "Where does $\mathbf{x} = 1.5\hat{\imath} + 1.0\hat{\jmath}$ map under $T$?" | Linearity requires $T(\mathbf{x}) = 1.5 T(\hat{\imath}) + 1.0 T(\hat{\jmath})$ |
| **0:09–0:14** | Basis Transformation | Green arrow $\hat{\imath}$ rotates and stretches to $\mathbf{a}_1 = [2, 1]^\top$; red arrow $\hat{\jmath}$ stretches to $\mathbf{a}_2 = [0, 2]^\top$. Matrix columns form: $A = [\mathbf{a}_1 \mid \mathbf{a}_2] = \begin{bmatrix} 2 & 0 \\ 1 & 2 \end{bmatrix}$. | UR: Matrix $A = \begin{bmatrix} 2 & 0 \\ 1 & 2 \end{bmatrix}$ | Matrix columns record basis destinations |
| **0:14–0:18** | Linear Combination Convergence | Scaled transformed vectors $1.5\mathbf{a}_1$ and $1.0\mathbf{a}_2$ chain tip-to-tail. Their composite tip hits the mystery target $[3.0, 3.5]^\top$ with pinpoint precision. Target pulses. | UL: "$T(\mathbf{x}) = 1.5\mathbf{a}_1 + 1.0\mathbf{a}_2 = A\mathbf{x}$" <br> LL: "Full Rank: $\det(A) = 4$" | Matrix-vector product is a linear combination of columns |

---

## Act 2: Dynamic Deformation & Ghost Space (Continuous Matrix Interpolation)
- **Duration**: 0:18 – 0:42 (24.0 s)
- **Pedagogical Objective**: Demonstrate that a linear transformation is a global geometric deformation of coordinate space; show that straight grid lines remain straight, parallel, and evenly spaced while the origin remains pinned. Contrast dynamic deformed grid with static ghost Cartesian grid.

### Visual Assets & Setup
- `ValueTracker` $t \in [0, 1]$ parameterizing homotopy $M(t) = (1-t)I + tA = \begin{bmatrix} 1+t & 0 \\ t & 1+t \end{bmatrix}$.
- Static ghost Cartesian grid rendered with faint dashed lines (`TEAL_E`, `stroke_opacity=0.20`, `stroke_width=0.8`).
- Dynamic deforming grid lines updated via `always_redraw` (`stroke_color=LINE_COLOR`, `stroke_width=1.5`).
- Unit square morphing into parallelogram of area $\det(M(t)) = (1+t)^2$, filled with `SUBSPACE_COLOR` (`#9A72AC`, Purple, `fill_opacity=0.25`).
- Live HUD readouts for matrix entries and area scale factor.

### Choreography & Timeline
| Time (s) | Phase | Camera / Canvas Action | HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| **0:18–0:24** | Ghost Grid & Homotopy Setup | Static ghost grid locks in place. Translucent purple unit square $[0, 1] \times [0, 1]$ highlights between $\hat{\imath}$ and $\hat{\jmath}$. Slider $t$ initializes at $0.0$. | UL: "Act 2: Dynamic Deformation" <br> UR: $M(t) = (1-t)I + tA$ | At $t=0$, $M(0) = I$; identity mapping leaves space intact |
| **0:24–0:34** | Continuous Space Deformation | ValueTracker sweeps $t: 0 \to 1$. Coordinate grid lines smoothly stretch and tilt. Horizontal lines remain parallel to each other; vertical lines remain parallel to each other. Spacing between adjacent lines remains uniform. Origin anchor dot $\mathbf{0}$ stays immovable. | UR: Dynamic matrix entries <br> LL: $\text{Area} = \det(M(t)) = (1+t)^2$ <br> Readout sweeps $1.0 \to 4.0$ | Straight lines map to straight lines; parallel lines remain parallel; origin invariant $M(t)\mathbf{0} = \mathbf{0}$ |
| **0:34–0:42** | Grid Invariance & Determinant Zoom | Deformed grid stabilizes at $t=1$. Purple unit square is now a sheared parallelogram spanned by $\mathbf{a}_1 = [2, 1]^\top$ and $\mathbf{a}_2 = [0, 2]^\top$. Area readout highlights $\det(A) = 4.0$. Parallelogram edges match basis arrows. | UL: "Parallel & Evenly Spaced Grid Lines Preserved" <br> LL: "$\det(A) = 4.0$ (Area quadrupled)" | $\det(A)$ represents the uniform geometric area scaling factor |

---

## Act 3: Color-Coded Algebraic Morph (Matrix Multiplication as Composition)
- **Duration**: 0:42 – 1:02 (20.0 s)
- **Pedagogical Objective**: Connect matrix multiplication directly to sequential function composition; show that $BA$ means applying $A$ first, then applying $B$ to the already-deformed grid.

### Visual Assets & Setup
- Initial transformation $A = \begin{bmatrix} 2 & 0 \\ 1 & 2 \end{bmatrix}$ (stretch & shear).
- Second transformation $B$: $90^\circ$ rotation matrix $R = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$.
- Composed transformation $BA = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}\begin{bmatrix} 2 & 0 \\ 1 & 2 \end{bmatrix} = \begin{bmatrix} -1 & -2 \\ 2 & 0 \end{bmatrix}$.
- Two-stage continuous transition with intermediate basis arrows.

### Choreography & Timeline
| Time (s) | Phase | Camera / Canvas Action | HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| **0:42–0:47** | Composition Architecture | Transformed grid under $A$ is shown. Formula appears in center HUD: $(B \circ A)(\mathbf{x}) = B(A\mathbf{x}) = (BA)\mathbf{x}$. Color-coded labels emphasize right-to-left evaluation order ($A$ first, then $B$). | UL: "Act 3: Sequential Composition" <br> UR: $A$ applied first, $B$ applied second | Function composition associativity: $(B \circ A)(\mathbf{x}) = B(T_A(\mathbf{x}))$ |
| **0:47–0:54** | Stage 2 Transformation $B$ | Transformation $B$ ($90^\circ$ rotation) executes on the deformed grid. The basis columns $\mathbf{a}_1 = [2, 1]^\top$ and $\mathbf{a}_2 = [0, 2]^\top$ rotate counter-clockwise by $90^\circ$ into $B\mathbf{a}_1 = [-1, 2]^\top$ and $B\mathbf{a}_2 = [-2, 0]^\top$. | UR: $BA = \begin{bmatrix} -1 & -2 \\ 2 & 0 \end{bmatrix}$ | Columns of $BA$ are $B$ applied to columns of $A$: $[B\mathbf{a}_1 \mid B\mathbf{a}_2]$ |
| **0:54–1:02** | Determinant Product Invariant | Unit area parallelogram rotates with space, maintaining area 4. Area badge indicates $\det(BA) = \det(B) \times \det(A) = 1 \times 4 = 4$. Faded flash highlights equivalence of single composite matrix $C = BA$. | LL: "$\det(BA) = \det(B)\det(A) = 1 \times 4 = 4$" <br> LR: "MML §2.5 Theorem 2.20" | Determinant multiplication theorem $\det(BA) = \det(B)\det(A)$ |

---

## Act 4: Degenerate & Edge-Case Stress Test (Geometric Non-Commutativity $BA \neq AB$)
- **Duration**: 1:02 – 1:26 (24.0 s)
- **Pedagogical Objective**: Geometrically prove why matrix multiplication is non-commutative ($SR \neq RS$). Apply shear $S = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$ and $90^\circ$ rotation $R = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$ in opposite orders, highlighting the divergence of basis vectors and the test point $\mathbf{x} = [2, 3]^\top$ by Euclidean distance $\sqrt{13}$.

### Visual Assets & Setup
- Shear matrix $S = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$, Rotation matrix $R = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$.
- Composed matrices:
  - $SR = \begin{bmatrix} 1 & -1 \\ 1 & 0 \end{bmatrix}$ (Rotation first, Shear second).
  - $RS = \begin{bmatrix} 0 & -1 \\ 1 & 1 \end{bmatrix}$ (Shear first, Rotation second).
- Test probe point $\mathbf{x} = [2, 3]^\top$:
  - $(SR)\mathbf{x} = [-1, 2]^\top$ (Gold dot & arrow).
  - $(RS)\mathbf{x} = [-3, 5]^\top$ (Orange/Red dot & arrow).
- Divergence vector connecting $(SR)\mathbf{x}$ to $(RS)\mathbf{x}$ of length $\|\Delta\| = \sqrt{13} \approx 3.61$.
- Commutator badge: $[S, R] = SR - RS = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \neq 0$, with $\text{Tr}([S, R]) = 0$.

### Choreography & Timeline
| Time (s) | Phase | Camera / Canvas Action | HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| **1:02–1:08** | Order 1: Rotation then Shear ($SR$) | Reset to standard grid. Probe point $\mathbf{x} = [2, 3]^\top$ appears. $R$ rotates space $90^\circ$ counter-clockwise; then $S$ shears horizontally. $\hat{\imath} \to [1, 1]^\top$, probe lands at $(SR)\mathbf{x} = [-1, 2]^\top$ marked with gold beacon. | UL: "Act 4: Non-Commutative Stress Test" <br> UR: Order 1: $SR = \begin{bmatrix} 1 & -1 \\ 1 & 0 \end{bmatrix}$ | Basis destination under $SR$: $\hat{\imath} \to [1, 1]^\top, \hat{\jmath} \to [-1, 0]^\top$ |
| **1:08–1:15** | Order 2: Shear then Rotation ($RS$) | Ghost trails of Order 1 remain. System resets to initial state. Now $S$ shears first; then $R$ rotates $90^\circ$. $\hat{\imath} \to [0, 1]^\top$, probe lands at $(RS)\mathbf{x} = [-3, 5]^\top$ marked with red beacon. | UR: Order 2: $RS = \begin{bmatrix} 0 & -1 \\ 1 & 1 \end{bmatrix}$ | Basis destination under $RS$: $\hat{\imath} \to [0, 1]^\top, \hat{\jmath} \to [-1, 1]^\top$ |
| **1:15–1:21** | Divergence Vector & Commutator | Red dashed divergence vector bridges the two outcomes: from $[-1, 2]^\top$ to $[-3, 5]^\top$. Length $\|\Delta\| = \sqrt{13} \approx 3.61$ pulses in soft red. Commutator panel appears: $[S, R] = SR - RS \neq 0$. | LL: "Divergence: $\|\Delta\| = \sqrt{13} \approx 3.61$" <br> UR: $[S, R] = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \ne 0$ | Spatial non-commutativity: operations distort axes before subsequent operations act |
| **1:21–1:26** | Cyclic Trace & Determinant Invariants | Final summary card highlights: while $SR \neq RS$ spatially, $\det(SR) = \det(RS) = 1.0$ and $\text{Tr}([S, R]) = 0$. Space orientation and area scaling commute even when transformations do not. | Center Card: "Matrix Multiplication is Geometric Composition: Generally Non-Commutative ($AB \neq BA$)" | Determinant multiplicativity and trace cyclic invariance hold universally |
