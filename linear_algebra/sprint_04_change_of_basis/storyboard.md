# Part 2: 4-Act Visual Storyboard — Change of Basis & Similarity Transformations

## Scene Specifications & Global Canvas Layout
- **Frame Resolution**: 1920x1080 (16:9 widescreen)
- **Background Color**: Dark slate (`#1C1C24`)
- **Coordinate Grids**:
  - Standard Grid: `NumberPlane` with x-range $[-6, 6]$, y-range $[-4, 4]$, clamped scale `x_length=9.6, y_length=6.4` (0.8 scale factor protecting viewport $[-7.11, 7.11] \times [-4.0, 4.0]$)
  - Eigen-Grid: Tilted coordinate plane aligned with columns of $P = [\mathbf{v}_1 \mid \mathbf{v}_2]$ where $\mathbf{v}_1 = [1, 1]^\top$ and $\mathbf{v}_2 = [-1, 1]^\top$, rendered in gold and teal dashed lines
- **Origin Anchor**: Permanent high-contrast white dot at $(0, 0)$ labeled $\mathbf{0}$
- **HUD Panel Configuration**:
  - `UL` (Upper-Left): Module & Act title badges (`make_hud_panel`)
  - `UR` (Upper-Right): Dynamic Matrix readout $A$, basis matrix $P$, diagonal matrix $D$
  - `LL` (Lower-Left): Geometric invariants badge ($\text{Tr}(A) = 4$, $\det(A) = 3$, eigenvalues)
  - `LR` (Lower-Right): Pedagogical cross-reference / Dual Bridge note

---

## Act 1: Concrete Spatial Puzzle (Coupled Shear/Stretch vs Decoupled Scaling)
- **Duration**: 0:00 – 0:18 (18.0 s)
- **Pedagogical Objective**: Pose the core dilemma: When applying matrix $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ to standard coordinates, both axes stretch and shear simultaneously, rotating arbitrary vectors like $\mathbf{x} = [2, 0]^\top$ into $[4, 2]^\top$. Is there a special coordinate system where this complex 2D distortion decouples into simple, independent 1D scalings?

### Visual Assets & Setup
- Standard unit basis vectors: $\hat{\imath} = [1, 0]^\top$ (`I_HAT_COLOR` `#83C167`, Green), $\hat{\jmath} = [0, 1]^\top$ (`J_HAT_COLOR` `#FC6255`, Red).
- Transformed columns: $\mathbf{a}_1 = [2, 1]^\top$, $\mathbf{a}_2 = [1, 2]^\top$.
- Test vector: $\mathbf{x} = [2, 0]^\top$ (`VEC_COLOR` `#F7D96F`, Yellow).
- Destination target: $A\mathbf{x} = [4, 2]^\top$ (`TARGET_COLOR` `#F9E498`, Gold beacon).
- Rotation angle arc between $\mathbf{x}$ and $A\mathbf{x}$ highlighting cross-axis rotation.

### Choreography & Timeline
| Time (s) | Phase | Camera / Canvas Action | HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| **0:00–0:04** | Frame Setup & Basis Vectors | Standard NumberPlane fades in. Unit arrows $\hat{\imath}$ (green) and $\hat{\jmath}$ (red) emerge from origin $\mathbf{0}$. Input vector $\mathbf{x} = [2, 0]^\top$ extends along the x-axis in yellow. | UL: "Act 1: Spatial Puzzle" <br> UR: Standard Basis $\{\hat{\imath}, \hat{\jmath}\}$ | Standard orthonormal reference frame spans $\mathbb{R}^2$ |
| **0:04–0:09** | Direct Transformation & Coupling | Matrix $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ acts. Unit vectors warp into $\mathbf{a}_1 = [2, 1]^\top$ and $\mathbf{a}_2 = [1, 2]^\top$. Test vector $\mathbf{x}$ stretches to length $\sqrt{20}$ and rotates upward by $\theta \approx 26.57^\circ$ to target $[4, 2]^\top$. | UR: $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ <br> UL: "Coupled Deformation: Stretch + Rotation" | Vector rotates: $\mathbf{x} \times A\mathbf{x} = 2(2) - 0(4) = 4 \ne 0$ |
| **0:09–0:14** | The Decoupling Challenge | Faint circular arc highlights angular deviation $\Delta \theta$. Both coordinate directions are entangled; altering $x$ affects $y$-output and vice-versa. | UL: "Can we find axes with ZERO rotation?" | Off-diagonal entries $A_{12} = A_{21} = 1$ cause cross-coupling |
| **0:14–0:18** | Mystery Invariant Rays Preview | Faint diagonal rays along $y = x$ and $y = -x$ pulse in gold and teal. The question is posed: What happens to vectors anchored directly along these special directions? | UL: "Special Invariant Directions?" <br> LL: $\text{Tr}(A)=4, \det(A)=3$ | Eigen-directions preserve vector orientation under mapping |

---

## Act 2: Dynamic Deformation & Ghost Space (The Invariant Eigen-Grid)
- **Duration**: 0:18 – 0:42 (24.0 s)
- **Pedagogical Objective**: Introduce the eigen-basis $P = [\mathbf{v}_1 \mid \mathbf{v}_2]$ with $\mathbf{v}_1 = [1, 1]^\top$ ($\lambda_1 = 3$) and $\mathbf{v}_2 = [-1, 1]^\top$ ($\lambda_2 = 1$). Animate continuous homotopy $M(t) = (1-t)I + tA$, demonstrating that while the standard Cartesian grid shears and tilts, the eigen-grid lines slide along themselves without rotating.

### Visual Assets & Setup
- Ghost standard Cartesian grid in faint background (`stroke_opacity=0.20`).
- Eigenvectors: $\mathbf{v}_1 = [1, 1]^\top$ (Green `#83C167`) along line $y = x$, $\mathbf{v}_2 = [-1, 1]^\top$ (Red `#FC6255`) along line $y = -x$.
- Eigen-Grid: Oblique coordinate plane spanned by $\mathbf{v}_1$ and $\mathbf{v}_2$ (`stroke_color=YELLOW_C`, dashed lines).
- Homotopy parameter $t \in [0, 1]$ driving $M(t) = (1-t)I + tA$.
- Stretch badges along eigen-axes showing exact scale factors: $3\times$ along $\mathbf{v}_1$, $1\times$ along $\mathbf{v}_2$.

### Choreography & Timeline
| Time (s) | Phase | Camera / Canvas Action | HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| **0:18–0:24** | Eigenvectors & Eigen-Grid Overlay | Standard grid dims to ghost scaffolding. Orthogonal eigenvectors $\mathbf{v}_1 = [1, 1]^\top$ and $\mathbf{v}_2 = [-1, 1]^\top$ grow along the diagonal lines $y=x$ and $y=-x$. Eigen-grid lines populate the canvas. | UL: "Act 2: Invariant Eigen-Grid" <br> UR: $P = [\mathbf{v}_1 \mid \mathbf{v}_2] = \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}$ | Orthogonality: $\mathbf{v}_1 \cdot \mathbf{v}_2 = 1(-1) + 1(1) = 0$ |
| **0:24–0:34** | Continuous Space Homotopy | ValueTracker sweeps $t: 0 \to 1$. Standard grid lines shear and tilt relative to Cartesian axes. However, vector $\mathbf{v}_1$ elongates along its line from length $\sqrt{2}$ to $3\sqrt{2}$ with zero rotation. Vector $\mathbf{v}_2$ remains completely stationary at length $\sqrt{2}$. | UR: $A\mathbf{v}_1 = 3\mathbf{v}_1, \quad A\mathbf{v}_2 = 1\mathbf{v}_2$ <br> LL: $\lambda_1 = 3, \lambda_2 = 1$ | Zero rotation on eigen-axes: $\mathbf{v}_i \times A\mathbf{v}_i = 0$ |
| **0:34–0:42** | Decoupled Geometry Revelation | Orthogonal right-angle marker between $\mathbf{v}_1$ and $\mathbf{v}_2$ stays perpendicular throughout deformation. The eigen-grid shows that transformation $A$ is simply non-uniform scaling along rotated orthogonal axes. | UL: "Pure Diagonal Stretch along Eigen-Axes" <br> LL: "Decoupled 1D Actions: $\lambda_1=3, \lambda_2=1$" | Spectral Theorem: Real symmetric matrices have orthogonal eigenvectors |

---

## Act 3: Color-Coded Algebraic Morph (The 3-Step Similarity Dance $P^{-1} \to D \to P$)
- **Duration**: 0:42 – 1:04 (22.0 s)
- **Pedagogical Objective**: Visually decompose similarity transformation $A = P D P^{-1}$ into a sequential 3-step dance on test vector $\mathbf{x} = [2, 0]^\top$: (1) $P^{-1}$ standard-to-eigen change of basis, (2) $D$ pure axis scaling, and (3) $P$ restoration to standard frame, demonstrating exact convergence with $A\mathbf{x} = [4, 2]^\top$.

### Visual Assets & Setup
- 3-Stage transformation sequence:
  - Stage 1 ($P^{-1}$): Standard coordinates $[2, 0]^\top \to$ Eigen-coordinates $[1, -1]_B$.
  - Stage 2 ($D$): Diagonal stretch $[1, -1]_B \to [3(1), 1(-1)]_B = [3, -1]_B$.
  - Stage 3 ($P$): Basis reconstruction $3\mathbf{v}_1 - 1\mathbf{v}_2 = [4, 2]^\top$.
- Color-coded equation morph: $A\mathbf{x} = \mathbf{P} \mathbf{D} \mathbf{P}^{-1} \mathbf{x}$.
- Invariant metric panel: Trace ($4$) and Determinant ($3$) conserved across all representations.

### Choreography & Timeline
| Time (s) | Phase | Camera / Canvas Action | HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| **0:42–0:48** | Stage 1: Change of Basis ($P^{-1}$) | Test vector $\mathbf{x} = [2, 0]^\top$ appears. Grid coordinates re-index into eigen-grid components. Projection lines drop onto $\mathbf{v}_1$ and $\mathbf{v}_2$, showing $\mathbf{x} = 1\mathbf{v}_1 - 1\mathbf{v}_2$. Coordinate readout reads $[1, -1]_B$. | UL: "Act 3: Similarity Transformation" <br> UR: Step 1 ($P^{-1}$): $[\mathbf{x}]_B = P^{-1}\mathbf{x} = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$ | Invertibility: $\det(P) = 2 \ne 0 \implies P^{-1}$ exists uniquely |
| **0:48–0:55** | Stage 2: Decoupled Diagonal Scaling ($D$) | Eigen-coordinate along $\mathbf{v}_1$ stretches $3\times$ from $1$ to $3$. Eigen-coordinate along $\mathbf{v}_2$ scales by $1\times$, staying at $-1$. The intermediate vector in eigen-coordinates becomes $[3, -1]_B$. | UR: Step 2 ($D$): $D[\mathbf{x}]_B = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}\begin{bmatrix} 1 \\ -1 \end{bmatrix} = \begin{bmatrix} 3 \\ -1 \end{bmatrix}$ | Diagonal decoupling: $D_{ij} = \lambda_i \delta_{ij}$; zero cross-talk |
| **0:55–1:04** | Stage 3: Return to Standard ($P$) & Convergence | Basis restoration $P$ converts eigen-components back: $3\mathbf{v}_1 - 1\mathbf{v}_2 = 3[1, 1]^\top - [-1, 1]^\top = [4, 2]^\top$. The reconstructed arrow snaps into the golden target beacon $[4, 2]^\top$. Trace and Determinant panels highlight. | UR: Step 3 ($P$): $P(D P^{-1}\mathbf{x}) = A\mathbf{x} = \begin{bmatrix} 4 \\ 2 \end{bmatrix}$ <br> LL: $\text{Tr}(P^{-1}AP) = \text{Tr}(A) = 4$ <br> $\det(P^{-1}AP) = \det(A) = 3$ | Similarity equivalence: $P D P^{-1} \equiv A$; cyclic trace and determinant invariance |

---

## Act 4: Degenerate & Edge-Case Stress Test (The Defective Matrix Collapse)
- **Duration**: 1:04 – 1:28 (24.0 s)
- **Pedagogical Objective**: Stress-test the boundaries of diagonalizability. Introduce defective shear matrix $M_{\text{def}} = \begin{bmatrix} 2 & 1 \\ 0 & 2 \end{bmatrix}$ with repeated eigenvalue $\lambda = 2$. Demonstrate visually why an eigen-grid cannot be formed: all eigenvectors collapse onto the 1D horizontal line $\text{span}([1, 0]^\top)$ (geometric multiplicity 1 < algebraic multiplicity 2), causing the similarity transformation to fail.

### Visual Assets & Setup
- Defective matrix: $M_{\text{def}} = \begin{bmatrix} 2 & 1 \\ 0 & 2 \end{bmatrix}$.
- Repeated eigenvalue $\lambda = 2$ with characteristic polynomial $(\lambda - 2)^2 = 0$.
- Single eigenvector $\mathbf{v} = [1, 0]^\top$ along the x-axis.
- Probing a candidate second vector $\mathbf{u} = [0, 1]^\top$: showing $M_{\text{def}}\mathbf{u} = [1, 2]^\top$, which rotates and shears away from $\mathbf{u}$ by $\Delta \theta \ne 0$.
- Collapsed eigen-grid warning card: Singular eigenvector matrix $\det([\mathbf{v}_1 \mid \mathbf{v}_2]) = 0$.

### Choreography & Timeline
| Time (s) | Phase | Camera / Canvas Action | HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| **1:04–1:10** | Defective Matrix Introduction | Canvas clears. Defective shear matrix $M_{\text{def}} = \begin{bmatrix} 2 & 1 \\ 0 & 2 \end{bmatrix}$ appears. Eigenvalue equation shows double root $\lambda_1 = \lambda_2 = 2$ (algebraic multiplicity 2). | UL: "Act 4: Defective Matrix Stress Test" <br> UR: $M_{\text{def}} = \begin{bmatrix} 2 & 1 \\ 0 & 2 \end{bmatrix}, \quad (\lambda - 2)^2 = 0$ | Algebraic multiplicity = 2 |
| **1:10–1:18** | Eigenspace Collapse to 1D Line | Solving $(M_{\text{def}} - 2I)\mathbf{v} = \mathbf{0}$ yields $\begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}\mathbf{v} = \mathbf{0} \implies y = 0$. Eigenvector arrow $\mathbf{v} = [1, 0]^\top$ expands on the horizontal line. A probing off-axis vector $\mathbf{u} = [0, 1]^\top$ transforms to $[1, 2]^\top$, shearing significantly. No second independent eigenvector exists! | UR: $(M_{\text{def}} - 2I)\mathbf{v} = \mathbf{0} \implies \text{Eigenspace} = \text{span}([1, 0]^\top)$ <br> LL: Geometric Multiplicity = $1 < 2$ | Geometric multiplicity $\text{nullity}(M-2I) = 1$; eigenspace is only 1D |
| **1:18–1:28** | The Non-Diagonalizability Verdict | The attempted eigen-grid collapses into a single horizontal line. Warning badge flashes: "Eigen-basis Incomplete — Cannot Invert $P$!" Master summary card highlights the fundamental theorem: A matrix is diagonalizable if and only if geometric multiplicity equals algebraic multiplicity for all eigenvalues. | Center HUD Card: "Non-Diagonalizable: Geometric Multiplicity (1) < Algebraic Multiplicity (2). Requires Jordan Canonical Form." | Incomplete eigenvector span prevents similarity diagonalization |
