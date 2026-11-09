# Part 2: 4-Act Visual Storyboard — Vectors, Linear Combinations & Span

## Scene Specifications & Global Canvas Layout
- **Frame Resolution**: 1920x1080 (16:9 widescreen)
- **Background Color**: Dark slate (`#1C1C24`)
- **Coordinate Grid**: `NumberPlane` with x-range $[-5, 5]$, y-range $[-4, 4]$, faded grid lines (`opacity=0.25`)
- **Origin Anchor**: Permanent high-contrast white dot at $(0, 0)$ labeled $\mathbf{0}$
- **HUD Panel Configuration**:
  - `UL` (Upper-Left): Module & Act title badges (`make_hud_panel`)
  - `UR` (Upper-Right): Dynamic ValueTracker scalar readouts ($c_1, c_2$)
  - `LL` (Lower-Left): Subspace metric badge ($\dim(\text{span})$, Subspace Type)
  - `LR` (Lower-Right): Pedagogical cross-reference / Dual Bridge note

---

## Act 1: Concrete Spatial Puzzle (Can Two Fixed Arrows Reach Any 2D Target?)
- **Duration**: 0:00 – 0:18 (18.0 s)
- **Pedagogical Objective**: Establish vectors as concrete directed displacements; demonstrate linear combinations as tip-to-tail path concatenation; formulate reachability as solving for scalar multipliers.

### Visual Assets & Setup
- $\mathbf{v}_1 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$ rendered in `I_HAT_COLOR` (`#83C167`, Green).
- $\mathbf{v}_2 = \begin{bmatrix} -1 \\ 2 \end{bmatrix}$ rendered in `J_HAT_COLOR` (`#FC6255`, Red).
- Target point $\mathbf{b} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$ rendered in `TARGET_COLOR` (`#F9E498`, Gold).

### Choreography & Timeline
| Time (s) | Phase | Camera / Canvas Action | HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| **0:00–0:04** | Frame Setup & Vector Intro | NumberPlane fades in; origin dot $\mathbf{0}$ pulses. Vectors $\mathbf{v}_1$ and $\mathbf{v}_2$ grow from $(0,0)$ via `GrowArrow`. | UL: "Act 1: Spatial Puzzle" | Free vectors anchored at coordinate origin |
| **0:04–0:08** | Puzzle Introduction | Gold dot appears at $\mathbf{b} = [3, 4]^\top$ with expanding target ring (`Broadcast`). | UL: "Can scaling $\mathbf{v}_1, \mathbf{v}_2$ reach $\mathbf{b}$?" | Linear system consistency challenge $c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 = \mathbf{b}$ |
| **0:08–0:12** | Naive Attempt (Shortfall) | Ghost copy of $\mathbf{v}_2$ shifts tip-to-tail to tip of $\mathbf{v}_1$. Resultant arrow reaches $[1, 3]^\top$, missing $\mathbf{b}$. Red indicator flashes shortfall. | UR: $c_1 = 1.0, c_2 = 1.0 \implies [1, 3]^\top \ne \mathbf{b}$ | Arbitrary coefficients miss arbitrary targets |
| **0:12–0:18** | Exact Solution & Closure | Green arrow $\mathbf{v}_1$ dilates to $c_1 = 2 \implies [4, 2]^\top$. Red arrow $\mathbf{v}_2$ attached tip-to-tail ($c_2 = 1$). Combined tip lands exactly on $\mathbf{b} = [3, 4]^\top$. Dashed lines complete the parallelogram $\mathbf{v}_2 + 2\mathbf{v}_1$. Gold dot flashes (`Indicate`). | UL: "Solution: $2\mathbf{v}_1 + 1\mathbf{v}_2 = \mathbf{b}$" <br> UR: $c_1 = 2.0, c_2 = 1.0$ | Unique linear combination for non-collinear basis |

---

## Act 2: Dynamic Deformation & Ghost Space (Scalars as Dynamic Dials)
- **Duration**: 0:18 – 0:38 (20.0 s)
- **Pedagogical Objective**: Transition from static vector arithmetic to dynamic continuous space generation; illustrate degrees of freedom as independent scalar dials sweeping an oblique lattice.

### Visual Assets & Setup
- `ValueTracker` instances for $c_1 \in [-2.5, 2.5]$ and $c_2 \in [-2.0, 2.0]$.
- Live dynamic `DecimalNumber` readouts in UR panel.
- Dynamic resultant vector $\mathbf{v}(c_1, c_2) = c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2$ updated via `always_redraw`.
- Dashed trace lines and ghost grid lines (`stroke_opacity=0.35`).

### Choreography & Timeline
| Time (s) | Phase | Camera / Canvas Action | HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| **0:18–0:24** | 1D Line Sweep ($c_1$ active) | Lock $c_2 = 0$. ValueTracker sweeps $c_1$ from $-2.5$ to $+2.5$. The green arrow expands, contracts, passes through $\mathbf{0}$, inverts, leaving a dashed line trace $L_1$ along the direction of $\mathbf{v}_1$. | UR: $c_1 = t, c_2 = 0.0$ <br> LL: "1 Scalar $\implies$ 1D Line Subspace" | $\text{span}(\mathbf{v}_1) = \{t \mathbf{v}_1 \mid t \in \mathbb{R}\}$ passes through $\mathbf{0}$ |
| **0:24–0:30** | Parallel Shift ($c_2$ active) | Lock $c_1 = 1.0$. ValueTracker sweeps $c_2$ from $-2.0$ to $+2.0$. Red arrow attached to the tip of $\mathbf{v}_1$ sweeps out a parallel line $L_2' = \mathbf{v}_1 + c_2 \mathbf{v}_2$. Faint ghost lines illustrate affine parallel translation. | UR: $c_1 = 1.0, c_2 = s$ <br> LL: "Affine Translation: $\mathbf{v}_1 + \text{span}(\mathbf{v}_2)$" | Non-origin lines are affine flats, not subspaces |
| **0:30–0:38** | Oblique Mesh Expansion | Both $c_1, c_2$ sweep across discrete integer steps $\{-2, -1, 0, 1, 2\}$. A dynamic rhomboidal coordinate mesh blooms outward from the origin, covering the Cartesian plane with oblique grid cells. | UL: "Act 2: Oblique Coordinate Mesh" <br> LL: "2 Independent Vectors $\implies$ 2D Plane" | Change of basis: Cartesian grid deformed into oblique coordinates |

---

## Act 3: Color-Coded Algebraic Morph (Formula to Geometric Span Sheet)
- **Duration**: 0:38 – 0:56 (18.0 s)
- **Pedagogical Objective**: Connect symbolic linear algebra syntax directly to visual geometry; show how set-builder notation translates into a continuous subspace wash.

### Visual Assets & Setup
- Formatted `MathTex` formula with color-mapped substrings matching vector colors.
- Translucent `Polygon` / `Rectangle` sheet with `fill_color=SUBSPACE_COLOR` (`#9A72AC`, Purple) and `fill_opacity=0.25`.
- Textbook citation card linking to MML §2.6 and VMLS §5.1.

### Choreography & Timeline
| Time (s) | Phase | Camera / Canvas Action | HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| **0:38–0:44** | Symbolic Equation Morph | Center formula appears: $\mathbf{v} = {\color{#83C167}c_1 \mathbf{v}_1} + {\color{#FC6255}c_2 \mathbf{v}_2}$. Substrings pulse and emit colored rays pointing directly to their geometric arrows in the canvas. The $+$ sign highlights the tip-to-tail junction. | UL: "Act 3: Algebraic Morph" <br> Center: Color-coded linear combination | Vector addition and scalar scaling generate all reachable vectors |
| **0:44–0:50** | Set-Builder Transition | Formula transforms via `ReplacementTransform` into: $\text{span}(\mathbf{v}_1, \mathbf{v}_2) = \left\{ c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 \;\middle|\; c_1, c_2 \in \mathbb{R} \right\}$. | LR: "MML §2.6 Def 2.13 (Span)" | The span is the minimal linear subspace containing the set |
| **0:50–0:56** | Continuous Span Wash | The discrete rhomboidal grid dissolves into a continuous translucent purple wash that floods the entire visible viewport. A glowing origin dot remains visible through the wash. | LL: "$\text{span}(\mathbf{v}_1, \mathbf{v}_2) = \mathbb{R}^2$ <br> $\dim(\text{span}) = 2$ (Full Rank)" | Generating set spans complete ambient space $\mathbb{R}^2$ |

---

## Act 4: Degenerate / Edge-Case Stress Test (Collinear Collapse & Rank Loss)
- **Duration**: 0:56 – 1:18 (22.0 s)
- **Pedagogical Objective**: Expose mathematical boundaries; demonstrate collinear collapse from 2D plane to 1D line; demonstrate inconsistency in linear systems when target escapes the column span; demonstrate trivial 0D subspace.

### Visual Assets & Setup
- Dynamic rotation/scaling of $\mathbf{v}_2$ to become collinear with $\mathbf{v}_1$: $\mathbf{v}_2 \to \begin{bmatrix} 4 \\ 2 \end{bmatrix} = 2 \mathbf{v}_1$.
- 2D translucent wash compressing into a 1D line mobject along $y = 0.5x$.
- Unreachable target point $\mathbf{b}_{\text{unreachable}} = \begin{bmatrix} 1 \\ 4 \end{bmatrix}$ with red error animation (`Wiggle`).
- Contraction to zero vector $\mathbf{v}_1, \mathbf{v}_2 \to \mathbf{0}$.

### Choreography & Timeline
| Time (s) | Phase | Camera / Canvas Action | HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| **0:56–1:02** | Collinear Transition | HUD Warning banner flashes amber: "Degeneracy Stress Test". Vector $\mathbf{v}_2$ smoothly rotates from $[-1, 2]^\top$ into alignment with $\mathbf{v}_1$, settling at $[4, 2]^\top = 2 \mathbf{v}_1$. | UL: "Warning: Linear Dependence" <br> UR: $\mathbf{v}_2 = 2 \mathbf{v}_1$ | Linear dependence: $\exists c \ne \mathbf{0}$ such that $\sum c_i \mathbf{v}_i = \mathbf{0}$ |
| **1:02–1:07** | Subspace Collapse | The 2D purple wash visibly compresses from both sides and collapses into a single 1D purple line $y = 0.5x$ passing through $(0,0)$. | LL: "$\dim(\text{span}) = 1$ (Rank Collapse!)" <br> Subspace: 1D Line | Collinear vectors span only a 1D line; rank drops to 1 |
| **1:07–1:12** | Unreachable Target Probe | Gold target appears off the line at $\mathbf{b}_{\text{unreachable}} = [1, 4]^\top$. Sliders $c_1, c_2$ rapidly oscillate, but the chained arrow is trapped on the line $y = 0.5x$. Target performs a red `Wiggle`. | UL: "Target Unreachable!" <br> UR: No valid $(c_1, c_2)$ | Inconsistent system: $\mathbf{b} \notin \text{span}(\mathbf{v}_1, \mathbf{v}_2)$ |
| **1:12–1:15** | GATE Consistency Invariant | Inconsistency badge pops up displaying Rouché-Capelli rank failure: $\text{rank}(A) = 1 \ne \text{rank}([A \mid \mathbf{b}]) = 2 \implies 0 \text{ Solutions}$. | LR: "GATE Shortcut: Rouché-Capelli Rank Test" | Augmented matrix rank strictly exceeds coefficient matrix rank |
| **1:15–1:18** | Trivial Subspace Collapse | Vectors $\mathbf{v}_1, \mathbf{v}_2$ shrink to length 0. The 1D line contracts into the single origin dot $\mathbf{0}$. Final HUD badge: $\text{span}(\{\mathbf{0}\}) = \{\mathbf{0}\} \; (\dim = 0)$. Screen fades to dark slate. | UL: "Trivial Subspace: $\{\mathbf{0}\}$" <br> LL: "$\dim(\text{span}) = 0$" | $\text{span}(\mathbf{0}) = \{\mathbf{0}\}$ is the only 0-dimensional subspace |
