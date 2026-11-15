# Part 2: 4-Act Visual Storyboard — Column Space, Null Space & Rank-Nullity in 3D

## Canvas & 3D Camera Configuration
- **Resolution**: $1920 \times 1080$ (16:9 widescreen, 60 fps production / 15 fps preview)
- **Background Color**: `#1C1C24` (Deep obsidian slate)
- **3D Coordinate Engine**: ManimCE `ThreeDScene` with `ThreeDAxes(x_range=[-4, 4, 1], y_range=[-4, 4, 1], z_range=[-4, 4, 1])`
- **Initial Camera Orientation**: $\phi = 70^\circ$, $\theta = -45^\circ$, distance $d = 20.0$
- **Dynamic Camera Action**: `self.begin_ambient_camera_rotation(rate=0.08)` to communicate three-dimensional spatial depth
- **HUD Pinning Rule**: All 2D text, mathematical formulas, and status cards are strictly viewport-pinned via `self.add_fixed_in_frame_mobjects(panel)` with dark backing `RoundedRectangle` cards (`fill_opacity=0.92`, `stroke_width=1.5`, `z_index=10`)
- **Semantic Color Palette**:
  - Basis Column 1 ($\mathbf{a}_1$): Green `#83C167`
  - Basis Column 2 ($\mathbf{a}_2$): Red `#FC6255`
  - Dependent Column 3 ($\mathbf{a}_3$): Blue `#58C4DD`
  - Column Space Plane $\mathcal{C}(A)$: Teal `#2A9D8F` (surface opacity 0.35, grid stroke 1.2)
  - Null Space Line $\mathcal{N}(A)$: Bright Orange `#FF9E64`
  - Affine Solution Line $\mathbf{x}_p + \mathcal{N}(A)$: Golden Yellow `#F7D96F`
  - Particular Solution $\mathbf{x}_p$: Yellow `#F7D96F`
  - Reachable Target $\mathbf{b} = [1, 2, 3]^\top$: Pure White / Gold `#FFFFFF`
  - Inconsistent Target $\mathbf{b}_{\text{unreach}} = [1, 0, 0]^\top$: Coral Error Red `#F7768E`
  - Origin Marker $\mathbf{0}$: White Dot `#FFFFFF`

---

## Act 1: Concrete Spatial Puzzle (0:00 – 0:20)
**Pedagogical Objective**: Establish the 3D coordinate system and formulate the reachability puzzle: Can matrix $A$ reach any point in 3D space, specifically target $\mathbf{b} = [1, 2, 3]^\top$?

### Act 1 Choreography Table
| Time (s) | Phase | Camera / 3D Canvas Action | Pinned HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| 0:00–0:04 | Orientation Setup | 3D axes fade in with ambient rotation ($\phi=70^\circ, \theta=-45^\circ$). Origin dot $\mathbf{0}$ pulses. | Top-left HUD card: `"Sprint 3: Column & Null Space in 3D"`. | Origin anchor: $A\mathbf{0} = \mathbf{0}$. |
| 0:04–0:10 | Column Vectors Reveal | Three 3D arrows grow from origin: $\mathbf{a}_1 = [1, 0, 1]^\top$ (Green), $\mathbf{a}_2 = [0, 1, 1]^\top$ (Red), $\mathbf{a}_3 = [1, 1, 2]^\top$ (Blue). | Matrix card UR: $A = [\mathbf{a}_1 \mid \mathbf{a}_2 \mid \mathbf{a}_3] = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix}$. | Matrix column definition $A\mathbf{e}_j = \mathbf{a}_j$. |
| 0:10–0:15 | Target Point Puzzle | White sphere appears at target $\mathbf{b} = [1, 2, 3]^\top$. Dotted beacon line connects $\mathbf{b}$ to origin. | Bottom-left HUD question: `"Can A x = b reach any 3D target?"` | Linear combination reachability $x_1\mathbf{a}_1 + x_2\mathbf{a}_2 + x_3\mathbf{a}_3 = \mathbf{b}$. |
| 0:15–0:20 | Dependency Clue | Column 3 flashes: $\mathbf{a}_1$ and $\mathbf{a}_2$ tip-to-tail parallelogram completes, perfectly matching $\mathbf{a}_3$. | Matrix card updates: `"\mathbf{a}_3 = \mathbf{a}_1 + \mathbf{a}_2 \implies \text{Linear Dependence!"}`. | Redundancy condition: $\det(A) = 0$, $\text{rank}(A) \le 2$. |

---

## Act 2: Dynamic Deformation & Ghost Space (0:20 – 0:45)
**Pedagogical Objective**: Animate continuous 3D volume collapse under matrix $A$ via homotopy $M(t) = (1-t)I + tA$, revealing the 2D planar sheet $\mathcal{C}(A)$ defined by $x + y - z = 0$.

### Act 2 Choreography Table
| Time (s) | Phase | Camera / 3D Canvas Action | Pinned HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| 0:20–0:25 | 3D Coordinate Box | A semi-transparent 3D coordinate cube $[-2, 2]^3$ and ghost Cartesian grid appear in pale blue. | HUD card: `"Continuous Homotopy: M(t) = (1-t)I + t A"`. Slider $t: 0 \to 1$. | Initial state: $t=0 \implies M(0) = I_3$, $\det(M(0)) = 1$. |
| 0:25–0:35 | The 3D Volume Flattening | Continuous sweep $t \in [0, 1]$. 3D coordinate cube flattens into an oblique 2D parallelogram surface; 3D volume squashes to zero. | Determinant tracker UL: `\det(M(t)) \to 0.0`. Height dimension shrinks to 0. | Volume scaling: $\text{Vol}(M(t)V) = \det(M(t))\text{Vol}(V) \to 0$. |
| 0:35–0:40 | Column Space Plane Sheet | Translucent teal 2D planar sheet $\mathcal{C}(A)$ expands across the viewport through the origin. Columns $\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3$ lie fully embedded in it. | HUD panel: `"\mathcal{C}(A) = \text{span}(\mathbf{a}_1, \mathbf{a}_2) \subset \mathbb{R}^3"`. Plane equation: `$x + y - z = 0$`. | Planar manifold: $\dim(\mathcal{C}(A)) = \text{rank}(A) = 2$. Normal $\mathbf{n} = [1, 1, -1]^\top$. |
| 0:40–0:45 | Target In-Plane Verification | Target beacon $\mathbf{b} = [1, 2, 3]^\top$ lies flush on the teal plane sheet. Right-angle marker with normal $\mathbf{n}$ confirms $\mathbf{n} \cdot \mathbf{b} = 0$. | Status badge: `"\mathbf{b} \in \mathcal{C}(A) \implies \text{Target Reachable!}"`. | Consistency condition: $1 + 2 - 3 = 0$. |

---

## Act 3: Color-Coded Algebraic Morph (0:45 – 1:05)
**Pedagogical Objective**: Visualize the null space line $\mathcal{N}(A)$ marching into $\mathbf{0}$, demonstrate the orthogonal complement $\mathcal{N}(A) = (\mathcal{C}(A^\top))^\perp$, and reveal the Rank-Nullity Theorem ($2 + 1 = 3$).

### Act 3 Choreography Table
| Time (s) | Phase | Camera / 3D Canvas Action | Pinned HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| 0:45–0:50 | Null Line Emergence | Bright orange line illuminates along direction $\mathbf{v}_{\text{null}} = [1, 1, -1]^\top$. Multiple orange sample vectors sprout along the line. | HUD card UR: `"\mathcal{N}(A) = \{\mathbf{x} \mid A\mathbf{x} = \mathbf{0}\}"`. Direction: `\mathbf{v}_{\text{null}} = [1, 1, -1]^\top`. | Kernel definition: $A\mathbf{v}_{\text{null}} = \mathbf{0}$. |
| 0:50–0:56 | The Null March | Orange vectors along $t \mathbf{v}_{\text{null}}$ contract and slide simultaneously along the line, collapsing into the white origin dot $\mathbf{0}$. | HUD alert: `"Entire 1D Line Collapses to Point 0!"`. | Dimensional loss: $1\text{D} \to 0\text{D}$. |
| 0:56–1:00 | Orthogonal Complement Marker | Right-angle elbow marker connects null line $\mathbf{v}_{\text{null}}$ to row space plane. Normal vector $\mathbf{n}$ aligns with $\mathbf{v}_{\text{null}}$. | HUD formula: `"\mathcal{N}(A) = (\mathcal{C}(A^\top))^\perp"`. Direct sum: `"\mathbb{R}^3 = \mathcal{C}(A^\top) \oplus \mathcal{N}(A)"`. | Fundamental Theorem of Linear Algebra: row space $\perp$ null space. |
| 1:00–1:05 | Rank-Nullity Synthesis | Pinned master equation morphs and illuminates: $\text{rank}(A) + \text{nullity}(A) = 3$. Teals and oranges highlight the dimension terms. | Centered HUD Banner: `"\underbrace{\text{rank}(A)}_{2\text{D (Plane)}} + \underbrace{\text{nullity}(A)}_{1\text{D (Line)}} = \underbrace{3}_{3\text{D Space}}"`. | Conservation of Dimension: $2 + 1 = 3$. |

---

## Act 4: Degenerate / Edge-Case Stress Test (1:05 – 1:25)
**Pedagogical Objective**: Animate the affine solution set $\mathbf{x}_p + \mathcal{N}(A)$ collapsing to target $\mathbf{b}$, then stress-test an unreachable target off the plane to demonstrate Rouché-Capelli rank failure.

### Act 4 Choreography Table
| Time (s) | Phase | Camera / 3D Canvas Action | Pinned HUD / Text Status | Mathematical Invariant |
|---|---|---|---|---|
| 1:05–1:10 | Particular Solution $\mathbf{x}_p$ | Yellow vector $\mathbf{x}_p = [1, 2, 0]^\top$ appears from origin. Shifted affine line $\mathbf{x}_p + t\mathbf{v}_{\text{null}}$ illuminates in gold parallel to the null line. | HUD card LR: `"\text{Affine Solutions: } \mathbf{x} = \mathbf{x}_p + t \mathbf{v}_{\text{null}}"`. | Affine manifold: parallel translation of linear subspace. |
| 1:10–1:15 | Many-to-One Funnel | Three distinct sample points on affine line ($\mathbf{x}_p, \mathbf{x}_p + \mathbf{v}_{\text{null}}, \mathbf{x}_{\text{LN}}$) map through $A$ simultaneously. All three trajectory arrows land on the single target $\mathbf{b} = [1, 2, 3]^\top$. | HUD badge: `"\forall \mathbf{x} \in \mathbf{x}_p + \mathcal{N}(A): A\mathbf{x} = \mathbf{b}"`. | Superposition: $A(\mathbf{x}_p + t\mathbf{v}_{\text{null}}) = A\mathbf{x}_p + t A\mathbf{v}_{\text{null}} = \mathbf{b} + \mathbf{0} = \mathbf{b}$. |
| 1:15–1:20 | Unreachable Off-Plane Target | Coral red sphere appears off the plane at $\mathbf{b}_{\text{unreach}} = [1, 0, 0]^\top$. Dashed perpendicular drop indicates non-zero distance $\sqrt{3}/3$. | HUD warning: `"Off-Plane Target: \mathbf{b}_{\text{unreach}} = [1, 0, 0]^\top"`. Normal test: `$1 + 0 - 0 = 1 \neq 0$`. | Inconsistency: $\mathbf{b} \notin \mathcal{C}(A)$. |
| 1:20–1:25 | Rouché-Capelli Inconsistency | Red beacon flashes error indicator. Augmented matrix ranks display side-by-side: $\text{rank}(A) = 2$ vs $\text{rank}([A \mid \mathbf{b}_{\text{unreach}}]) = 3$. | Final HUD Banner: `"\text{rank}(A) = 2 < \text{rank}([A \mid \mathbf{b}]) = 3 \implies 0 \text{ Solutions (Inconsistent)}"`. | Rouché-Capelli Theorem complete solvability classification. |
