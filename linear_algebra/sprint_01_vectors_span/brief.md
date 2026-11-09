# Part 1: Geometric Consensus Brief — Vectors, Linear Combinations & Span

## 1. Geometric Action
A vector $\mathbf{v} \in \mathbb{R}^2$ represents a directed displacement arrow rooted at the origin $(0, 0)$. Scalar multiplication $c \mathbf{v}$ scales vector length along its line of action; negative scalars invert orientation by 180° through the origin. Vector addition $\mathbf{u} + \mathbf{v}$ concatenates spatial displacements: translating the tail of $\mathbf{v}$ to the tip of $\mathbf{u}$ defines a path identical to the diagonal of the completed parallelogram. 

When two non-parallel vectors $\mathbf{v}_1, \mathbf{v}_2$ scale continuously ($c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2$), they generate an oblique coordinate grid. Sweeping the scalar coefficients $c_1, c_2 \in \mathbb{R}$ across all real numbers dynamically reaches and tiles the complete two-dimensional Euclidean plane $\mathbb{R}^2$.

## 2. Geometric Invariants
- **Origin Anchor**: Every linear subspace contains the zero vector $\mathbf{0}$. Setting $c_1 = c_2 = 0$ yields $\mathbf{0} \in \text{span}(S)$. A line or plane translated away from the origin is an affine flat, not a linear subspace.
- **Subspace Closure**: Any linear combination of vectors inside $\text{span}(S)$ remains strictly inside $\text{span}(S)$ (closed under addition and scalar multiplication).
- **Dimensional Collapse**: If $\mathbf{v}_1$ and $\mathbf{v}_2$ are linearly dependent ($\mathbf{v}_2 = k \mathbf{v}_1$), directional degrees of freedom collapse from 2 to 1. The reachable span flattens into a 1D line through $\mathbf{0}$. If $\mathbf{v}_1 = \mathbf{v}_2 = \mathbf{0}$, the span collapses to the 0D trivial subspace $\{\mathbf{0}\}$.

## 3. Algebraic-to-Visual Dictionary
| Algebraic Concept | Visual Representation | Geometric Semantics |
|---|---|---|
| Scalar $c_i \in \mathbb{R}$ | Dynamic slider / dial | Continuous line dilation and reversal |
| Vector $\mathbf{v} \in \mathbb{R}^2$ | Directed arrow from origin | Spatial displacement from $(0, 0)$ to $(x, y)$ |
| Linear Combination $\sum c_i \mathbf{v}_i$ | Chained tip-to-tail path | Parallelogram diagonal composite traversal |
| Span $\text{span}(\mathbf{v}_1, \dots, \mathbf{v}_k)$ | Translucent colored sheet | Complete reachable subspace manifold |
| Linear Dependence | Collinear vector alignment | Redundant arrow locked to existing axis of motion |
| Basis $B = \{\mathbf{v}_1, \mathbf{v}_2\}$ | Minimal coordinate scaffolding | Unique coordinate address pair $(c_1, c_2)$ for every point |

## 4. Dual Bridge
### Frontier ML: Attention as Convex Combinations & Latent Arithmetic
In Transformer Self-Attention, each token representation is an explicit convex combination of value vectors:
$$\text{Attention}(Q, K, V) = \sum_{j=1}^N \alpha_j V_j, \quad \text{where } \alpha_j = \frac{\exp(q_i^\top k_j / \sqrt{d_k})}{\sum_{m=1}^N \exp(q_i^\top k_m / \sqrt{d_k})}$$
Because attention weights satisfy $\alpha_j \ge 0$ and $\sum_j \alpha_j = 1$, context representations are mathematically bounded inside the convex hull of the value vectors. In latent feature representations (Word2Vec, diffusion latents, LLM residual streams), semantic manipulations operate through vector addition along learned directional linear subspaces ($\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}$).

### GATE CS/DA Shortcut: System Consistency & Rank Collapse
The matrix equation $A \mathbf{x} = \mathbf{b}$ states that $\mathbf{b}$ is a linear combination of the columns of $A$: $\sum x_j \mathbf{a}_j = \mathbf{b}$. By the Rouché-Capelli Theorem, a solution exists if and only if $\mathbf{b}$ lies within the column span of $A$:
$$\mathbf{b} \in \text{Col}(A) \iff \text{rank}(A) = \text{rank}([A \mid \mathbf{b}])$$
If columns are linearly dependent, $\text{rank}(A) < n$, forcing $\det(A) = 0$. The column span collapses into a lower-dimensional subspace, guaranteeing either infinite solutions (if $\mathbf{b}$ lies in the collapsed span) or zero solutions (if $\mathbf{b}$ lies outside the collapsed span).
