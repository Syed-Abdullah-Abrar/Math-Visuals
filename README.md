# Mathematical & Computational Foundations

Intuitive, first-principles repository combining geometric visualizations, formal mathematical models, and code implementations across Mathematics and Computer Science.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Linear Algebra Curriculum](#1-linear-algebra-curriculum)
   - [Sprint 1: Vectors, Linear Combinations & Span](#sprint-1-vectors-linear-combinations--span)
   - [Curriculum Roadmap](#curriculum-roadmap)
3. [Scratchpad & Prototyping](#2-scratchpad--prototyping)
4. [Foundational Philosophy & Blueprint](#3-foundational-philosophy--blueprint)
   - [Part 1: The Core Philosophy (The Unified Field)](#part-1-the-core-philosophy-the-unified-field)
   - [Part 2: The Geometric Translation of Technical Landscapes](#part-2-the-geometric-translation-of-technical-landscapes)
   - [Part 3: The Focus Matrix (Prioritizing via Cognitive Constraints)](#part-3-the-focus-matrix-prioritizing-via-cognitive-constraints)
   - [Part 4: The 2-Year Execution Pipeline](#part-4-the-2-year-execution-pipeline)
   - [Part 5: The System Integration (Your Obsidian Engine)](#part-5-the-system-integration-your-obsidian-engine)

---

## Repository Structure

```
math_cs/
├── README.md                     # Master blueprint, table of contents, and curriculum roadmap
├── .gitignore                    # Python, virtualenv, and Manim cache rules
├── linear_algebra/               # Geometric linear algebra curriculum
│   ├── README.md                 # Curriculum module index & sprint roadmap
│   └── sprint_01_vectors_span/   # Sprint 1: Vectors, Linear Combinations & Span
│       ├── README.md             # Master companion guide (protocol integration)
│       ├── brief.md              # Part 1: Geometric Consensus Brief
│       ├── storyboard.md         # Part 2: 4-Act Visual Storyboard
│       ├── scene_vectors_span.py # Part 3: Production ManimCE script
│       └── tweaks_prompts.md     # Part 4: Interactive Tweaks & Intuition Prompts
├── tests/                        # Automated verification test suite
│   └── test_sprint01_invariants.py # Invariant & render validation tests
└── scratchpad_codes/             # Prototyping & numerical verification scripts
    └── matrix_expansion.py       # Quadratic form expansions (symmetric vs general)
```

---

## 1. Linear Algebra Curriculum

> **Status**: Active Curriculum  
> **Primary References**:  
> - *Mathematics for Machine Learning* (MML Ch. 2.4–2.6) — Deisenroth, Faisal, Ong  
> - *Introduction to Applied Linear Algebra* (VMLS Ch. 1, 5) — Boyd, Vandenberghe  
> - *Immersive Linear Algebra* (ILA Ch. 1–2, 5, 8) — Ström, Åström, Akenine-Möller  

Linear algebra operates through spatial transformations. We treat matrices not as numeric tables, but as continuous coordinate deformations that stretch, rotate, and shear space while keeping the origin fixed.

### Sprint 1: Vectors, Linear Combinations & Span

Sprint 1 establishes the atomic generating mechanism of linear spaces in [`linear_algebra/sprint_01_vectors_span/`](linear_algebra/sprint_01_vectors_span/). Every sprint follows a strict 4-part deliverable protocol:

1. **Part 1: Geometric Consensus Brief** ([`brief.md`](linear_algebra/sprint_01_vectors_span/brief.md))  
   Synthesizes the core physical mechanics:
   - **Geometric Action**: Spatial vector addition as displacement concatenation (parallelogram law), continuous scalar scaling dials, and oblique mesh generation.
   - **Geometric Invariant**: Strict origin anchoring ($\mathbf{0} \in \text{span}(S)$), minimal subspace containment, and dimension collapse under collinearity.
   - **Algebraic-to-Visual Dictionary**: Direct mapping from scalar multipliers, vector sums, and spans to visual sliders, arrows, and translucent planar sheets.
   - **Dual Bridge**: Links theory to Transformer self-attention (attention outputs as explicit convex combinations $\sum \alpha_i \mathbf{v}_i$) and GATE CS/DA linear system consistency ($A\mathbf{x} = \mathbf{b} \iff \mathbf{b} \in \text{col}(A)$).

2. **Part 2: 4-Act Visual Storyboard** ([`storyboard.md`](linear_algebra/sprint_01_vectors_span/storyboard.md))  
   Defines frame-by-frame spatial choreography:
   - **Act 1: Concrete Spatial Puzzle**: Reaching target vector $\mathbf{b} = [3, 4]^\top$ via basis vectors $\mathbf{v}_1 = [2, 1]^\top$ and $\mathbf{v}_2 = [-1, 2]^\top$.
   - **Act 2: Dynamic Deformation & Ghost Space**: Sliders sweep $c_1, c_2 \in [-2.5, 2.5]$, tiling the 2D Cartesian plane into an oblique mesh.
   - **Act 3: Color-Coded Algebraic Morph**: Formula $\mathbf{v} = c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2$ morphs into arrows, set-builder notation, and a continuous translucent 2D span sheet.
   - **Act 4: Degenerate Stress Test**: Rotating $\mathbf{v}_2$ into collinearity with $\mathbf{v}_1$ collapses the 2D plane into a 1D line ($\det = 0, \text{rank} = 1$); reducing to $\mathbf{0}$ collapses the span to a 0D point.

3. **Part 3: Production ManimCE Code** ([`scene_vectors_span.py`](linear_algebra/sprint_01_vectors_span/scene_vectors_span.py))  
   Production animation script written for Manim Community Edition v0.18+ (verified on v0.21.0):
   - Adheres to the 3Blue1Brown semantic color palette (`#83C167` for $\mathbf{v}_1$, `#FC6255` for $\mathbf{v}_2$, `#58C4DD` for $\mathbf{b}$, `#9A72AC` for subspaces).
   - Real-time HUD telemetry panels showing coordinates, coefficients, determinant, and subspace rank.
   - Dynamic parameter choreography via `ValueTracker` and `always_redraw`.
   - Exact numerical calculations via NumPy.

4. **Part 4: Interactive Tweaks & Intuition Prompts** ([`tweaks_prompts.md`](linear_algebra/sprint_01_vectors_span/tweaks_prompts.md))  
   Interactive parameter modification guide and 4 conceptual stress-test challenges probing affine flats, negative cones, softmax temperatures, and null space dimensionality.

Master companion guide and integration instructions are located in [`linear_algebra/sprint_01_vectors_span/README.md`](linear_algebra/sprint_01_vectors_span/README.md).

#### Render & Verification Quickstart

```bash
# Activate project virtual environment
source .venv/bin/activate

# Render Sprint 1 Full Master Narrative (preview quality, 480p15)
manim -ql linear_algebra/sprint_01_vectors_span/scene_vectors_span.py VectorSpanScene

# Render individual acts
manim -ql linear_algebra/sprint_01_vectors_span/scene_vectors_span.py Act1SpatialPuzzle
manim -ql linear_algebra/sprint_01_vectors_span/scene_vectors_span.py Act2DynamicDeformation
manim -ql linear_algebra/sprint_01_vectors_span/scene_vectors_span.py Act3AlgebraicMorph
manim -ql linear_algebra/sprint_01_vectors_span/scene_vectors_span.py Act4DegenerateStressTest

# Execute automated invariant test suite
python -m pytest tests/test_sprint01_invariants.py
```

### Curriculum Roadmap

| Sprint | Topic | Geometric Focus | References | Status |
|---|---|---|---|---|
| **01** | **Vectors, Linear Combinations & Span** | Oblique grid tiling, displacement concatenation, collinear collapse | MML §2.4–2.6, VMLS Ch. 1, 5, ILA Ch. 1–2 | **Active** |
| **02** | **Linear Transformations & Matrix Action** | Grid deformation, origin preservation, basis vectors as column destinations | MML §2.7, VMLS Ch. 5, 7, ILA Ch. 3–4 | Planned |
| **03** | **Determinants & Orientation Scaling** | Signed area/volume expansion factors, orientation reversal, null-space collapse | MML §4.1, VMLS Ch. 5, ILA Ch. 6 | Planned |
| **04** | **Systems of Linear Equations** | Intersecting hyperplanes, column span inclusion ($Ax = b$), Rouché-Capelli rank test | MML §2.3, VMLS Ch. 8, ILA Ch. 5 | Planned |
| **05** | **Vector Subspaces & Fundamental Theorem** | Four fundamental subspaces ($\text{Col}, \text{Null}, \text{Row}, \text{Left Null}$), rank-nullity theorem | MML §2.4, Strang FTLA | Planned |
| **06** | **Orthogonality & Projections** | Orthogonal complements, Gram-Schmidt orthogonalization, least squares projection | MML §3.1–3.4, VMLS Ch. 5, 12, ILA Ch. 7 | Planned |
| **07** | **Spectral Theory (Eigenvalues & Eigenvectors)** | Invariant axes of transformation, characteristic equation, spectral decomposition | MML §4.2–4.4, ILA Ch. 8 | Planned |
| **08** | **Singular Value Decomposition (SVD)** | Rotation-scaling-rotation hyper-ellipse factorization, low-rank approximation | MML §4.5, VMLS Ch. 18 | Planned |

---

## 2. Scratchpad & Prototyping

Isolated verification scripts for fast algebraic and numerical checks prior to animation implementation:
- [matrix_expansion.py](scratchpad_codes/matrix_expansion.py): Verifies quadratic forms $(\mathbf{a} - \mathbf{b})^\top \mathbf{M} (\mathbf{a} - \mathbf{b})$ under symmetric and non-symmetric matrices $\mathbf{M}$.

---

## 3. Foundational Philosophy & Blueprint

To anchor everything into a single, permanent architecture, we synthesize the core philosophy, the geometric translations, the contrarian focus strategies, and the two-year execution plan into one unified blueprint.

Mathematics and computer science fuse into a single, intuitive language that operates through spatial and mechanical reasoning.

---

### Part 1: The Core Philosophy (The Unified Field)

The foundational mistake of modern technical education is treating math as abstract symbol manipulation and computer science as writing lines of text code. To achieve muscle memory, you must adopt a single, elegant worldview:

* **Mathematics** is the study of pure structure, relationship, and transformation. It is the abstract blueprint of how things can move, warp, scale, and balance across arbitrary dimensions.
* **Computer Science** is the physical manifestation of that blueprint. It is the art of using physical matter (silicon, electricity) and time to build, automate, and navigate those mathematical structures.

When you look at software code, you are not looking at text; you are looking at a machine that constructs a geometric landscape in memory. When you look at an equation, you are looking at a description of how that landscape bends or flows.

---

### Part 2: The Geometric Translation of Technical Landscapes

To stop pushing your brain to calculate, offload symbolic logic onto your visual and spatial systems:

#### 1. Formal Logic & Discrete Math (Spatial Boundaries)
* **The Symbols**: $\land$ (AND), $\lor$ (OR), $\neg$ (NOT), $\implies$ (Implication).
* **The Geometry**: Logic is the science of spatial containment and boundaries. A proposition or data state is a designated region inside a universal coordinate space.
* **The Intuition**: The logical statement $A \implies B$ ("If A, then B") simply means the region A is entirely enclosed inside the larger boundary of region B. Modus Ponens (knowing that if you are in A, you must be in B) is just realizing that if you are physically standing inside the inner circle, you are automatically standing inside the outer circle.

#### 2. Linear Algebra (Warping Grid Lines)
* **The Symbols**: Matrices, Vectors, Determinants, Eigenvalues.
* **The Geometry**: Linear algebra is the visual art of stretching, shearing, compressing, and rotating space while ensuring the origin stays fixed and parallel grid lines remain straight.
* **The Intuition**: A matrix is not a grid of numbers; it is a recipe for a dynamic space warp. Multiplying a vector by a matrix means watching the entire coordinate system deform, pulling your vector along with it. The determinant measures how much an area or volume scales up or shrinks down during that warp. Eigenvectors are the special, unyielding pathways in space that do not change their physical direction when the universe warps around them—they simply stretch or shrink along their original axes.

#### 3. Calculus & Machine Learning Optimization (Mountain Terrains)
* **The Symbols**: Gradients, Loss Functions, Backpropagation, Derivatives.
* **The Geometry**: Machine learning is the exploration of high-dimensional physical topography.
* **The Intuition**: A machine learning model's loss function creates a literal landscape of mountains, valleys, and deep ravines. The gradient vector acts like a physical compass needle pointing directly up the steepest cliff face. Training a neural network using gradient descent is not an abstract arithmetic loop; it is the physical sensation of an object rolling down a rugged mountain path, navigating momentum, trying to locate the lowest possible valley floor (global minimum) without getting caught in a shallow pothole (local minimum).

#### 4. Probability & Causal AI (Hyper-Dimensional Volumes)
* **The Symbols**: Joint Probability $P(X, Y)$, Conditional Probability $P(X \mid Y)$, Marginalization.
* **The Geometry**: Probability is the geometry of slicing, dicing, and projecting smoke clouds.
* **The Intuition**: A complex joint probability distribution is like a high-dimensional cloud of smoke suspended inside a glass box. Conditioning on an event ($P(X \mid Y)$) means taking a flat, razor-sharp geometric slice right through that smoke at a specific coordinate. Marginalization (summing out a variable) is the act of turning off a dimension and looking at the flat shadow that the high-dimensional smoke cloud casts onto a lower-dimensional wall.

#### 5. Systems Architecture & Version Control (Flows, Allocations, and Trees)
* **Git**: Git is not a checklist of file backups; it is a Directed Acyclic Graph (DAG) growing across time. Branches are literal forks in a physical road. Running a git rebase means picking up a detached branch structure, lifting it by its base node, and physically sliding it over to paste it onto the tip of another node.
* **Operating Systems (Virtual Memory)**: The kernel acts like a magical optical illusionist. It convinces every single running process (tenant) that it owns a flat, continuous, unbroken infinity of space (linear memory address space). Behind the scenes, the kernel is secretly chopping their belongings into uniform physical chunks (pages) and scattering them across completely non-contiguous rooms inside the physical hardware chips (RAM).

---

### Part 3: The Focus Matrix (Prioritizing via Cognitive Constraints)

Because human energy, time, and focus are finite, prioritize deep mechanical anchors over superficial signals:

| What to Focus on MORE (Deep Anchors) | What to Focus on LESS (Superficial Signals) |
|---|---|
| **First-Principles Mimicry**: Building micro-versions of core engines (custom malloc, minimal Git, primitive compiler shell) from scratch. | **LeetCode Puzzle Grinding**: Memorizing isolated template tricks to solve hundreds of isolated competitive programming puzzles out of context. |
| **Structural & Paradigm Mastery**: Understanding the deep mathematical architecture behind a tool (e.g., how an asynchronous event loop works conceptually). | **Framework & API Chasing**: Memorizing specific, rapidly changing syntax quirks of trending libraries, wrappers, and high-level tools. |
| **Visual/Dynamic Prototyping**: Forcing yourself to animate, slide, and sketch mathematical parameters using interactive engines. | **Monolithic Note Encyclopedias**: Transcribing vast textbook chapters into exhaustive, dead walls of text inside your Obsidian vault. |

---

### Part 4: The 2-Year Execution Pipeline

A four-semester sequence taking you from raw logical gates to high-dimensional neural network optimization:

#### Year 1: Constructing the Spatial Infrastructure
* **Semester 1 (Months 1–6): The Digital Spine**
  * *Focus*: Demystify the leap from formal logic symbols to electrical signals and physical hardware memory.
  * *Canon*: *Book of Proof* (Hammack), *Code* (Petzold), *Computer Systems: A Programmer's Perspective* (Bryant & O'Hallaron).
  * *Projects*: Build an 8-bit Arithmetic Logic Unit (ALU) out of virtual NAND gates. Write a custom memory manager (`malloc` and `free`) in raw C99 to manipulate hardware byte arrays directly.
* **Semester 2 (Months 7–12): The Warp & The Topography**
  * *Focus*: Shift from sequential data layouts to spatial transformations (Linear Algebra) and fluid, continuous change (Calculus).
  * *Canon*: *Linear Algebra Done Right* (Axler), *Introduction to Algorithms* (CLRS).
  * *Tools*: GeoGebra for visual slider experimentation, Manim (Python) for programmatic animations.
  * *Projects*: Build a 3D wireframe graphics rasterizer from scratch in Python or C without libraries (custom projection and rotation matrices). Build a binary graph/network database engine with direct disk persistence.

#### Year 2: Achieving the Deep Synthesis
* **Semester 3 (Months 13–18): Randomness & Time Realities**
  * *Focus*: Master the geometry of high-dimensional probability and kernel orchestration of multi-processing space and time.
  * *Canon*: *Introduction to Probability* (Blitzstein & Hwang), *Operating Systems: Three Easy Pieces* (OSTEP).
  * *Projects*: Write a functional command-line Linux shell in C handling process forks (`fork()`), execution maps, and pipes. Build a mini-Git version control tool from scratch using SHA-1 content-addressable DAG nodes.
* **Semester 4 (Months 19–24): The Intelligence Continuum**
  * *Focus*: Fuse systems, calculus, linear algebra, and data networks to build machine learning intelligence from first principles.
  * *Canon*: *Mathematics for Machine Learning* (Deisenroth), *Deep Learning* (Goodfellow).
  * *Projects*: Write a scalar-valued automatic differentiation engine (`micrograd` style) from scratch in pure Python. Upgrade it to a Tensor/Matrix framework and write a visual tracer mapping gradient descent across a multi-dimensional loss surface.

---

### Part 5: The System Integration (Your Obsidian Engine)

Convert your notes into a **Triple-Anchor Intuition Matrix**. Every note for a core mathematical theorem or computer science concept contains three linked elements:

1. **The Human Narrative**: The historical frustration or bottleneck that forced the invention of the concept.
2. **The Geometric Blueprint**: An embedded visual asset (GeoGebra link, Manim animation snippet, or canvas sketch) showing how the concept deforms or behaves spatially.
3. **The Code Manifestation**: A minimal, clean, first-principles code snippet (in C, Rust, or Python) demonstrating the backbone execution of that structure.
