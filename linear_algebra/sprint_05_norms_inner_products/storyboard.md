# Part 2: 4-Act Visual Storyboard — Norms ($L_1, L_2, L_\infty$), Inner Products & Cauchy-Schwarz

## Scene Specifications & Global Canvas Layout
- **Resolution & Frame Rate**: 1920x1080 (`-qh` / `-pqh`), 60 fps (`16:9` aspect ratio).
- **Background Color**: Deep Slate Dark (`#1C1C24`).
- **Coordinate System**: Centered `NumberPlane` spanning $x \in [-6, 6]$, $y \in [-4, 4]$ with `x_length=9.6, y_length=6.4` (`0.8` scene units per math unit), dimmed axis grid (`#414868`), and a persistent white origin anchor dot $\mathbf{0} = [0, 0]^\top$.
- **HUD Panels (Rule 3: Zero Overlap)**: Rounded obsidian containers (`#16161E`, 88% opacity, `#414868` border, `z_index=10`) pinned strictly to viewport corners:
  - `UL`: Act title & guiding question.
  - `UR`: Live algebraic readout (`DecimalNumber(..., edge_to_fix=LEFT)` pre-sized to worst-case negative/wide string) and comparison equations.
  - `DL`: Invariant badges (norm ordering, Cholesky whitening, triangle/parallelogram verdicts).
  - `DR`: Signed dot-product readout and $\cos\omega \in [-1, 1]$ Cauchy-Schwarz saturation gauge.

---

## Act 1: Concrete Spatial Puzzle — Measuring $\mathbf{x} = [3, 4]^\top$ & Ruler Projection (`Act1SpatialPuzzle`)
- **Duration**: ~20s (`0:00` – `0:20`)
- **Pedagogical Objective**: Confront the viewer with three distinct notions of length on a single 3-4-5 vector $\mathbf{x} = [3, 4]^\top$, establish the ordering $\|\mathbf{x}\|_\infty \le \|\mathbf{x}\|_2 \le \|\mathbf{x}\|_1$, and preview $\mathbf{x}^\top\mathbf{y}$ as projecting $\mathbf{x}$ onto $\mathbf{y}$'s ruler.

### Visual Assets & Setup
- Primary vector $\mathbf{x}_{\text{puzzle}} = [3, 4]^\top$ (`VEC_COLOR`, `#F7D96F`, label placed `LEFT` of tip to clear `UR` HUD).
- Taxicab $L_1$ path `(0,0) -> (3,0) -> (3,4)` (`I_HAT_COLOR`, `#83C167`, total length `7`).
- Chebyshev $L_\infty$ dominant vertical bar `(0,0) -> (0,4)` + dashed guide `(0,4) -> (3,4)` (`J_HAT_COLOR`, `#FC6255`, length `4`).
- Ruler vector $\mathbf{y}_{\text{puzzle}} = [2, 1]^\top$ (`I_HAT_COLOR`), ruler line with tick marks at $k\mathbf{y}_{\text{puzzle}}$ ($k \in \{-2,-1,0,1,2,3\}$), projection $\operatorname{proj}_{\mathbf{y}}(\mathbf{x}) = [4, 2]^\top$ (`TARGET_COLOR`), residual $[-1, 2]^\top$, and `safe_right_angle`.

### Choreography & Timeline
| Time (s) | Phase | Camera / Canvas Action | HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| 0:00–0:04 | Setup & Vector | Draw `NumberPlane`, origin $\mathbf{0}$, and grow $\mathbf{x}_{\text{puzzle}} = [3, 4]^\top$. | `UL`: *"Act 1: Spatial Puzzle — Three Ways to Measure Length"*. | $\mathbf{x} = [3, 4]^\top$ |
| 0:04–0:10 | Three Norms | Trace $L_1$ staircase ($3+4=7$), flash $L_2$ hypotenuse ($5$), highlight $L_\infty$ bar ($4$). | `UR`: $\|\mathbf{x}\|_1=7, \|\mathbf{x}\|_2=5, \|\mathbf{x}\|_\infty=4$. `DL`: $\|\mathbf{x}\|_\infty \le \|\mathbf{x}\|_2 \le \|\mathbf{x}\|_1$. | $4 \le 5 \le 7$; $\operatorname{rms}(\mathbf{x}) = 5/\sqrt{2}$ |
| 0:10–0:16 | Ruler Projection | Dim $L_1/L_\infty$ paths; draw $\mathbf{y}_{\text{puzzle}}=[2,1]^\top$ and ticked ruler line; drop perpendicular dashed residual to $[4, 2]^\top$ with `RightAngle`. | `DR`: $\mathbf{x}^\top \mathbf{y} = 10 = 2\|\mathbf{y}\|_2^2$. | $\mathbf{r} = [-1, 2]^\top \perp [2, 1]^\top$ ($\mathbf{r}^\top \mathbf{y} = 0$) |
| 0:16–0:20 | Pause & Bridge | Highlight thick projection bar `(0,0)->(4,2)` and pose morph question. | `DL`: *"How do unit balls morph across $p$?"* | Projection coordinate $\lambda = 2$ |

---

## Act 2: Dynamic Deformation — $L_p$ Unit Ball Morph & $\omega$-Ruler Sweep (`Act2DynamicDeformation`)
- **Duration**: ~26s (`0:20` – `0:46`)
- **Pedagogical Objective**: Morph $\|\mathbf{x}\|_p = 1$ continuously from $p=1$ (diamond) $\to p=2$ (circle) $\to p=\infty$ (square) while pinning axis vertices, then sweep $\mathbf{x}(\omega)$ around a tilted ruler $\mathbf{y}_{\text{ruler}} = [1.6, 1.2]^\top$ to reveal $\mathbf{x}^\top\mathbf{y}$ as signed projection.

### Visual Assets & Setup
- **Part A**: Faint ghost contours for $B_1$ diamond and $B_\infty$ square (`GHOST_COLOR`), pinned basis dots $(\pm 1,0), (0,\pm 1)$ (`TARGET_COLOR`), `alpha_tracker = ValueTracker(1.0)` ($\alpha = 1/p$), `always_redraw` unit ball (`LINE_COLOR`, 241 factored-max nodes via `set_points_as_corners`), moving diagonal corner dot $(2^{-1/p}, 2^{-1/p})$.
- **Part B**: Fixed $\mathbf{y}_{\text{ruler}} = [1.6, 1.2]^\top$ ($\|\mathbf{y}\|_2 = 2.0$) + blue ruler line; rotating $\mathbf{x}(\omega)$ of length `X_SWEEP_NORM = 2.5`; dynamic projection bar (`I_HAT_COLOR` if $>0$, `ERROR_COLOR` if $<0$), dashed drop line, `safe_right_angle`, and angle `Arc`.

### Choreography & Timeline
| Time (s) | Phase | Camera / Canvas Action | HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| 0:20–0:25 | $p=1 \to 2$ | Sweep $\alpha: 1.0 \to 0.5$; $B_1$ diamond inflates smoothly into $B_2$ circle; diagonal dot slides $(0.5,0.5) \to (0.707,0.707)$. | `UR`: pre-sized `DecimalNumber` sweeps $1.00 \to 2.00$. `DL`: *"L2 Circle: Rotation Invariant"*. | $B_1 \subseteq B_2$, axis vertices pinned |
| 0:25–0:32 | $p=2 \to \infty$ | Sweep $\alpha: 0.5 \to 0.05 \to 0.0$; $B_2$ circle inflates into exact $B_\infty$ square; diagonal dot reaches $(1, 1)$. | `UR` transitions cleanly to $p = \infty$ *(Chebyshev Square)*. | $B_1 \subseteq B_2 \subseteq B_\infty$ |
| 0:32–0:33 | Clear Part A | Explicitly `.clear_updaters()` and fade out Part A dynamic objects. | Transition to Part B HUDs. | Zero leftover updaters |
| 0:33–0:46 | $\omega$-Ruler Sweep | Sweep $\omega: 0 \to \pi/3 \to \pi/2 \to \pi \to \arccos(0.8)$; pause at $\pi/2$ (`0.00` orthogonal), $\pi$ (`-5.00` anti-parallel), and $\arccos(0.8)$ (`4.00`, 3-4-5 canonical triangle). | `UR` live `DecimalNumber` for $\cos\omega$ and $\mathbf{x}^\top\mathbf{y}$ (`-0.00` clamped). | $\mathbf{x}^\top\mathbf{y} = \|\mathbf{x}\|_2\|\mathbf{y}\|_2\cos\omega$ |

---

## Act 3: Algebraic Morph — Cauchy-Schwarz Gauge & SPD $A$-Ellipsoid (`Act3AlgebraicMorph`)
- **Duration**: ~22s (`0:46` – `1:08`)
- **Pedagogical Objective**: Visualize why Cauchy-Schwarz traps $|\mathbf{x}^\top\mathbf{y}| \le \|\mathbf{x}\|_2\|\mathbf{y}\|_2$ via the $\cos\omega \in [-1,1]$ gauge, then generalize inner products to SPD matrices $A_{\text{SPD}} = [[2,1],[1,2]]$ that warp Euclidean circles into $A$-ellipsoids.

### Visual Assets & Setup
- **Part A**: Dashed circle $\|\mathbf{x}\|_2 = 2.5$ (`GHOST_COLOR`), fixed $\mathbf{y}_{\text{ruler}}=[1.6,1.2]^\top$, rotating $\mathbf{x}(\omega)$, canonical reference `X_VEC=[1,2], Y_VEC=[2,1]` ($|\mathbf{x}^\top\mathbf{y}|=4 \le 5$), and `DR` pinned $\cos\omega \in [-1, +1]$ gauge bar.
- **Part B**: Homotopy `t_tracker: 0 -> 1` on $M(t) = (1-t)I + t A_{\text{SPD}}$ warping the unit circle into the $A$-unit ellipse $\mathbf{x}^\top A\mathbf{x} = 1$ (`SUBSPACE_COLOR`), plus vectors $\mathbf{u}_A = [1, 0]^\top$ and $\mathbf{v}_A = [-1, 2]^\top$.

### Choreography & Timeline
| Time (s) | Phase | Camera / Canvas Action | HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| 0:46–0:56 | Cauchy-Schwarz Gauge | Sweep $\omega: \arccos(0.8) \to 0 \to \pi \to \pi/3$; needle hits $+1.00$ and $-1.00$ on `DR` gauge with `Indicate` flash on collinearity. | `UL`: Cauchy-Schwarz inequality; `UR`: $\lvert\mathbf{x}^\top\mathbf{y}\rvert = 4 \le 5$ (canonical) & live bound. | $\lvert\mathbf{x}^\top\mathbf{y}\rvert = \|\mathbf{x}\|_2\|\mathbf{y}\|_2 \iff \text{collinear}$ |
| 0:56–1:08 | SPD $A$-Ellipsoid | Fade gauge; morph unit circle into tilted $A$-ellipse (`SUBSPACE_COLOR`); highlight $\mathbf{u}_A=[1,0]^\top$, $\mathbf{v}_A=[-1,2]^\top$. | `UR`: $\mathbf{u}^\top\mathbf{v} = -1 \ne 0$, yet $\mathbf{u}^\top A\mathbf{v} = 0$. `DL`: Cholesky $\|L^\top \mathbf{x}\|_2 = \|\mathbf{x}\|_A$. | $\langle \mathbf{x},\mathbf{y}\rangle_A = \mathbf{x}^\top A\mathbf{y}$ ($A=A^\top \succ 0$) |

---

## Act 4: Degenerate Stress Tests & Master Summary Card (`Act4DegenerateStressTest`)
- **Duration**: ~26s (`1:08` – `1:34`)
- **Pedagogical Objective**: Stress-test every defining axiom by exhibiting four mathematical breakdown modes and synthesizing all invariants on a final summary card.

### Visual Assets & Setup
- Failure 1: `lp_unit_ball_points(plane, 0.5, 241)` (`ERROR_COLOR`) inside dashed $B_1$ diamond, $\mathbf{u}_{\text{deg}}=[1,0], \mathbf{v}_{\text{deg}}=[0,1], \mathbf{u}_{\text{deg}}+\mathbf{v}_{\text{deg}}=[1,1]$.
- Failure 2: Parallelogram spanned by $\mathbf{u}_{\text{deg}}, \mathbf{v}_{\text{deg}}$ and diagonals $[1,1], [1,-1]$.
- Failure 3: $\mathbf{x} \to \mathbf{0}$ collapse onto white origin anchor.
- Failure 4: Parametric hyperbolas $x^2 - y^2 = \pm 1$ (`(\pm\cosh t, \sinh t)` / `(\sinh t, \pm\cosh t)`), null asymptotes $y = \pm x$, and $\mathbf{x}_{\text{indef}} = [1, 2]^\top$.
- Master Summary Card: Centered `make_hud_panel(..., padding=0.28)` at `z_index=20`.

### Choreography & Timeline
| Time (s) | Phase | Camera / Canvas Action | HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| 1:08–1:14 | $p=0.5$ Non-Convex | Draw pinched $B_{0.5}$ star inside $B_1$; draw $\mathbf{u}+\mathbf{v}=[1,1]$ shooting past diamond. | `UR`: $\|\mathbf{u}+\mathbf{v}\|_{0.5} = 4 > 1+1$ *(Triangle Inequality Fails)*. | Non-convex ball $\implies$ invalid norm |
| 1:14–1:19 | $L_1$ Parallelogram | Draw parallelogram & diagonals on $[1,0],[0,1]$. | `UR`: $L_1$: $2^2+2^2 = 8 \ne 2(1^2+1^2)=4$. | $L_1$ not induced by any inner product |
| 1:19–1:23 | $\mathbf{x} = \mathbf{0}$ Edge | Collapse $\mathbf{x}$ onto origin $\mathbf{0}$ next to $\mathbf{y}$. | `UR`: $\mathbf{0}^\top\mathbf{y}=0$ ($\mathbf{0}\perp\mathbf{y}$), $\cos\omega = 0/0$ undefined. | Angle requires nonzero vectors |
| 1:23–1:29 | Indefinite $A_{\text{indef}}$ | Plot hyperbolas $x^2-y^2=\pm 1$, asymptotes $y=\pm x$, and $\mathbf{x}_{\text{indef}}=[1,2]^\top$. | `UR`: $\mathbf{x}^\top A_{\text{indef}}\mathbf{x} = -3 < 0$ *(Positivity Fails)*. | Inner product requires $A \succ 0$ |
| 1:29–1:34 | Summary Card | Dim plane; present centered master summary card (`z_index=20`) and hold `2.5s`. | 4 Core Invariants + RMS/Ordering recap. | Synthesis complete |
