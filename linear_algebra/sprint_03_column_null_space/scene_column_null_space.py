"""
Sprint 3: Column Space, Null Space & Rank-Nullity Theorem in 3D (ManimCE v0.18+)
Curriculum: Linear Algebra & Theoretical CS / Machine Learning Foundations
References: MML Ch. 2.6, VMLS Ch. 10, Kreyszig Ch. 7, ILA Ch. 6

This production script implements the 4-act pedagogical narrative in 3D:
- Act 1: Concrete Spatial Puzzle (3D Reachability Puzzle, Columns a1, a2, a3 & Target b)
- Act 2: Dynamic Deformation & Ghost Space (3D Volume Flattening to 2D Planar Sheet C(A))
- Act 3: Color-Coded Algebraic Morph (1D Null Line N(A) Marching into Origin, Rank-Nullity 2+1=3)
- Act 4: Degenerate Stress Test (Affine Solutions x_p + N(A) & Rouché-Capelli Inconsistency)

Render Commands:
    Preview Low-Res:   manim -ql linear_algebra/sprint_03_column_null_space/scene_column_null_space.py ColumnNullSpaceScene
    Production Hi-Res: manim -qh linear_algebra/sprint_03_column_null_space/scene_column_null_space.py ColumnNullSpaceScene
    Single Act Test:   manim -ql linear_algebra/sprint_03_column_null_space/scene_column_null_space.py Act1SpatialPuzzle
"""

from __future__ import annotations

import math
import numpy as np
from manim import *

# -----------------------------------------------------------------------------
# Rule 1: 3Blue1Brown Semantic Color Palette
# -----------------------------------------------------------------------------
I_HAT_COLOR = "#83C167"       # Green C    - Column 1 (a1)
J_HAT_COLOR = "#FC6255"       # Red C      - Column 2 (a2)
K_HAT_COLOR = "#58C4DD"       # Blue C     - Column 3 (a3 = a1 + a2)
SUBSPACE_COLOR = "#2A9D8F"    # Teal       - Column Space C(A) 2D plane sheet
NULL_COLOR = "#FF9E64"        # Orange     - Null Space N(A) 1D line & vectors
AFFINE_COLOR = "#F7D96F"      # Yellow     - Affine solution line x_p + N(A)
ORIGIN_COLOR = "#FFFFFF"      # White      - Immovable origin anchor (0, 0, 0)
TARGET_COLOR = "#F9E498"      # Gold       - Reachable target b = [1, 2, 3]^T
ERROR_COLOR = "#F7768E"       # Soft Red   - Unreachable off-plane target & rank mismatch
BG_COLOR = "#1C1C24"          # Dark Slate - High-contrast viewport background
PANEL_BG = "#16161E"          # Obsidian   - HUD panel background container
PANEL_BORDER = "#414868"      # Slate-Teal - HUD panel container border
TEXT_COLOR = "#ECEFF4"        # Snow       - Primary typography
MUTED_TEXT = "#7AA2F7"        # Light Blue - Secondary badges & annotations

# -----------------------------------------------------------------------------
# Mathematical Constants & Assertions (Rule 6: Exact NumPy Precision)
# -----------------------------------------------------------------------------
A_MAT = np.array([
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 2.0]
])

A1_COORDS = np.array([1.0, 0.0, 1.0])
A2_COORDS = np.array([0.0, 1.0, 1.0])
A3_COORDS = np.array([1.0, 1.0, 2.0])
V_NULL = np.array([1.0, 1.0, -1.0])
NORMAL_VEC = np.array([1.0, 1.0, -1.0])
TARGET_B = np.array([1.0, 2.0, 3.0])
X_P = np.array([1.0, 2.0, 0.0])
X_LN = np.array([0.0, 1.0, 1.0])
B_UNREACH = np.array([1.0, 0.0, 0.0])

# Pre-flight mathematical consistency verification
assert np.allclose(A_MAT @ np.array([1.0, 0.0, 0.0]), A1_COORDS), "a1 must be A @ e1"
assert np.allclose(A_MAT @ np.array([0.0, 1.0, 0.0]), A2_COORDS), "a2 must be A @ e2"
assert np.allclose(A_MAT @ np.array([0.0, 0.0, 1.0]), A3_COORDS), "a3 must be A @ e3"
assert np.allclose(A3_COORDS, A1_COORDS + A2_COORDS), "a3 must equal a1 + a2"
assert np.isclose(np.linalg.det(A_MAT), 0.0), "det(A) must be 0.0"
assert np.linalg.matrix_rank(A_MAT) == 2, "Matrix A must have rank exactly 2"
assert np.allclose(A_MAT @ V_NULL, np.zeros(3)), "A @ v_null must be 0"
assert np.allclose(A_MAT @ X_P, TARGET_B), "A @ x_p must equal target b"
assert np.allclose(A_MAT @ X_LN, TARGET_B), "A @ x_LN must equal target b"
assert np.isclose(np.dot(X_LN, V_NULL), 0.0), "x_LN must be strictly orthogonal to v_null"
assert np.isclose(np.linalg.norm(X_LN), np.sqrt(2.0)), "||x_LN|| must be sqrt(2)"
assert np.linalg.matrix_rank(np.column_stack([A_MAT, TARGET_B])) == 2, "rank([A|b]) must be 2"
assert np.linalg.matrix_rank(np.column_stack([A_MAT, B_UNREACH])) == 3, "rank([A|b_unreach]) must be 3"


# -----------------------------------------------------------------------------
# Rule 3: Zero Text Overlap HUD Panel Factory (Pinned in Viewport)
# -----------------------------------------------------------------------------
def make_hud_panel(
    content: Mobject,
    padding: float = 0.18,
    bg_color: str = PANEL_BG,
    bg_opacity: float = 0.92,
    stroke_color: str = PANEL_BORDER,
    stroke_width: float = 1.5,
    corner_radius: float = 0.12,
) -> VGroup:
    """Wraps any mobject in a dark rounded rectangle with high z-index."""
    bg = RoundedRectangle(
        corner_radius=corner_radius,
        width=content.width + 2 * padding,
        height=content.height + 2 * padding,
        fill_color=bg_color,
        fill_opacity=bg_opacity,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
    )
    bg.move_to(content.get_center())
    panel = VGroup(bg, content)
    panel.set_z_index(10)
    return panel


# -----------------------------------------------------------------------------
# Base Scene Class (ThreeDScene Subclass with 3D Axes & Viewport HUD Pinning)
# -----------------------------------------------------------------------------
class BaseColumnNullSpaceScene(ThreeDScene):
    """Master base class providing 3D coordinate space, camera orbits, and HUD pinning."""

    def setup(self):
        super().setup()
        self.camera.background_color = BG_COLOR
        # Standard 3D camera orientation for oblique depth
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.08)

    def create_axes_3d(self) -> ThreeDAxes:
        """Rule 2: 3D Axes clamped to viewport dimensions."""
        axes = ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[-4, 4, 1],
            x_length=7.0,
            y_length=7.0,
            z_length=5.0,
            axis_config={
                "color": "#4C566A",
                "stroke_width": 1.5,
                "include_ticks": True,
                "tick_size": 0.08,
            },
        )
        return axes

    def pin_hud(self, panel: Mobject, corner=UL, buff: float = 0.25):
        """Rule 3: Enforce add_fixed_in_frame_mobjects for all 2D HUD cards."""
        panel.to_corner(corner, buff=buff)
        self.add_fixed_in_frame_mobjects(panel)

    def unpin_hud(self, *panels: Mobject):
        """Cleanly removes fixed-in-frame HUD mobjects from camera frame."""
        self.remove_fixed_in_frame_mobjects(*panels)

    # -------------------------------------------------------------------------
    # Act 1: Concrete Spatial Puzzle (3D Reachability & Column Vectors)
    # -------------------------------------------------------------------------
    def run_act1(self):
        axes = self.create_axes_3d()
        origin_dot = Dot3D(point=axes.c2p(0, 0, 0), radius=0.08, color=ORIGIN_COLOR)

        # Title HUD Panel
        title_text = VGroup(
            Text("Sprint 3: Column Space & Null Space in 3D", font_size=20, color=TEXT_COLOR, weight=BOLD),
            Text("Act 1: Spatial Reachability Puzzle", font_size=14, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        title_panel = make_hud_panel(title_text)
        self.pin_hud(title_panel, corner=UL)

        # Matrix Panel
        matrix_math = MathTex(
            r"A = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix}",
            font_size=20,
            color=TEXT_COLOR,
        )
        matrix_panel = make_hud_panel(matrix_math)
        self.pin_hud(matrix_panel, corner=UR)

        self.play(Create(axes), FadeIn(origin_dot), run_time=1.0)

        # Column vectors a1, a2, a3
        arrow_a1 = Arrow3D(start=axes.c2p(0, 0, 0), end=axes.c2p(*A1_COORDS), color=I_HAT_COLOR)
        arrow_a2 = Arrow3D(start=axes.c2p(0, 0, 0), end=axes.c2p(*A2_COORDS), color=J_HAT_COLOR)
        arrow_a3 = Arrow3D(start=axes.c2p(0, 0, 0), end=axes.c2p(*A3_COORDS), color=K_HAT_COLOR)

        self.play(FadeIn(arrow_a1), FadeIn(arrow_a2), run_time=0.8)
        self.play(FadeIn(arrow_a3), run_time=0.8)

        # Target point b = [1, 2, 3]^T
        target_dot = Dot3D(point=axes.c2p(*TARGET_B), radius=0.10, color=TARGET_COLOR)
        target_arrow = Arrow3D(start=axes.c2p(0, 0, 0), end=axes.c2p(*TARGET_B), color=TARGET_COLOR)

        # Puzzle question panel
        puzzle_text = VGroup(
            MathTex(r"\text{Target } \mathbf{b} = [1, 2, 3]^\top", font_size=18, color=TARGET_COLOR),
            Text("Can A reach ANY target in 3D space?", font_size=14, color=TEXT_COLOR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        puzzle_panel = make_hud_panel(puzzle_text)
        self.pin_hud(puzzle_panel, corner=DL)

        self.play(FadeIn(target_dot), FadeIn(target_arrow), run_time=0.8)
        self.wait(0.8)

        # Dependency reveal: a3 = a1 + a2 tip-to-tail
        dep_line = Line3D(start=axes.c2p(*A1_COORDS), end=axes.c2p(*A3_COORDS), color=J_HAT_COLOR)
        dep_badge_text = MathTex(r"\mathbf{a}_3 = \mathbf{a}_1 + \mathbf{a}_2 \implies \text{Rank Deficient!}", font_size=18, color=MUTED_TEXT)
        dep_badge_panel = make_hud_panel(dep_badge_text)
        self.pin_hud(dep_badge_panel, corner=DR)

        self.play(Create(dep_line), run_time=0.8)
        self.wait(1.0)

        # Clean up Act 1
        self.unpin_hud(title_panel, matrix_panel, puzzle_panel, dep_badge_panel)
        self.play(
            FadeOut(arrow_a1), FadeOut(arrow_a2), FadeOut(arrow_a3),
            FadeOut(target_dot), FadeOut(target_arrow), FadeOut(dep_line),
            FadeOut(axes), FadeOut(origin_dot),
            run_time=0.8
        )

    # -------------------------------------------------------------------------
    # Act 2: Dynamic Deformation & Ghost Space (The 2D Planar Collapse)
    # -------------------------------------------------------------------------
    def run_act2(self):
        axes = self.create_axes_3d()
        origin_dot = Dot3D(point=axes.c2p(0, 0, 0), radius=0.08, color=ORIGIN_COLOR)

        title_text = VGroup(
            Text("Act 2: 3D Space Flattens to 2D Plane Sheet", font_size=20, color=TEXT_COLOR, weight=BOLD),
            Text("Continuous Homotopy: M(t) = (1-t)I + tA", font_size=14, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        title_panel = make_hud_panel(title_text)
        self.pin_hud(title_panel, corner=UL)

        self.play(Create(axes), FadeIn(origin_dot), run_time=0.8)

        # Ghost coordinate wireframe cube [-1.5, 1.5]^3
        cube_lines = VGroup()
        corners = [
            [-1.5, -1.5, -1.5], [1.5, -1.5, -1.5], [1.5, 1.5, -1.5], [-1.5, 1.5, -1.5],
            [-1.5, -1.5, 1.5], [1.5, -1.5, 1.5], [1.5, 1.5, 1.5], [-1.5, 1.5, 1.5]
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        for e1, e2 in edges:
            l = Line3D(start=axes.c2p(*corners[e1]), end=axes.c2p(*corners[e2]), color="#434C5E")
            cube_lines.add(l)

        self.play(FadeIn(cube_lines), run_time=0.8)

        # Flattening into Column Space 2D Plane: z = x + y <=> x + y - z = 0
        plane_sheet = Surface(
            lambda u, v: axes.c2p(u, v, u + v),
            u_range=[-2.2, 2.2],
            v_range=[-2.2, 2.2],
            fill_opacity=0.35,
            color=SUBSPACE_COLOR,
            resolution=(16, 16),
        )

        plane_info_text = VGroup(
            MathTex(r"\mathcal{C}(A) = \text{span}(\mathbf{a}_1, \mathbf{a}_2) \subset \mathbb{R}^3", font_size=18, color=SUBSPACE_COLOR),
            MathTex(r"\text{Plane Equation: } x + y - z = 0", font_size=16, color=TEXT_COLOR),
            MathTex(r"\dim(\mathcal{C}(A)) = \text{rank}(A) = 2", font_size=16, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        plane_info_panel = make_hud_panel(plane_info_text)
        self.pin_hud(plane_info_panel, corner=UR)

        self.play(FadeOut(cube_lines), Create(plane_sheet), run_time=1.2)

        # Embedded column arrows and reachable target b on plane
        arrow_a1 = Arrow3D(start=axes.c2p(0, 0, 0), end=axes.c2p(*A1_COORDS), color=I_HAT_COLOR)
        arrow_a2 = Arrow3D(start=axes.c2p(0, 0, 0), end=axes.c2p(*A2_COORDS), color=J_HAT_COLOR)
        target_dot = Dot3D(point=axes.c2p(*TARGET_B), radius=0.10, color=TARGET_COLOR)
        target_arrow = Arrow3D(start=axes.c2p(0, 0, 0), end=axes.c2p(*TARGET_B), color=TARGET_COLOR)

        self.play(FadeIn(arrow_a1), FadeIn(arrow_a2), FadeIn(target_dot), FadeIn(target_arrow), run_time=0.8)

        # In-plane target verification badge
        status_text = MathTex(
            r"\mathbf{b} \in \mathcal{C}(A) \implies 1 + 2 - 3 = 0 \text{ (Reachable!)}",
            font_size=18,
            color=TARGET_COLOR,
        )
        status_panel = make_hud_panel(status_text)
        self.pin_hud(status_panel, corner=DR)

        self.wait(1.2)

        # Clean up Act 2
        self.unpin_hud(title_panel, plane_info_panel, status_panel)
        self.play(
            FadeOut(plane_sheet), FadeOut(arrow_a1), FadeOut(arrow_a2),
            FadeOut(target_dot), FadeOut(target_arrow),
            FadeOut(axes), FadeOut(origin_dot),
            run_time=0.8
        )

    # -------------------------------------------------------------------------
    # Act 3: Color-Coded Algebraic Morph (Null Line Collapse & Rank-Nullity)
    # -------------------------------------------------------------------------
    def run_act3(self):
        axes = self.create_axes_3d()
        origin_dot = Dot3D(point=axes.c2p(0, 0, 0), radius=0.09, color=ORIGIN_COLOR)

        title_text = VGroup(
            Text("Act 3: The Null Space & Rank-Nullity Theorem", font_size=20, color=TEXT_COLOR, weight=BOLD),
            Text("1D Line Collapses to Point 0 | 2 + 1 = 3", font_size=14, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        title_panel = make_hud_panel(title_text)
        self.pin_hud(title_panel, corner=UL)

        self.play(Create(axes), FadeIn(origin_dot), run_time=0.8)

        # Null space line: t * [1, 1, -1]^T
        null_line = Line3D(
            start=axes.c2p(-2.5, -2.5, 2.5),
            end=axes.c2p(2.5, 2.5, -2.5),
            color=NULL_COLOR,
        )
        null_vec_pos = Arrow3D(start=axes.c2p(0, 0, 0), end=axes.c2p(*V_NULL), color=NULL_COLOR)
        null_vec_neg = Arrow3D(start=axes.c2p(0, 0, 0), end=axes.c2p(*(-V_NULL)), color=NULL_COLOR)

        null_info_text = VGroup(
            MathTex(r"\mathcal{N}(A) = \{\mathbf{x} \in \mathbb{R}^3 \mid A\mathbf{x} = \mathbf{0}\}", font_size=18, color=NULL_COLOR),
            MathTex(r"\text{Kernel Vector: } \mathbf{v}_{\text{null}} = [1, 1, -1]^\top", font_size=16, color=TEXT_COLOR),
            MathTex(r"\dim(\mathcal{N}(A)) = \text{nullity}(A) = 1", font_size=16, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        null_info_panel = make_hud_panel(null_info_text)
        self.pin_hud(null_info_panel, corner=UR)

        self.play(Create(null_line), FadeIn(null_vec_pos), FadeIn(null_vec_neg), run_time=1.0)

        # The Null March: vectors contracting into origin
        self.play(
            null_vec_pos.animate.scale(0.01, about_point=axes.c2p(0, 0, 0)),
            null_vec_neg.animate.scale(0.01, about_point=axes.c2p(0, 0, 0)),
            run_time=1.2,
            rate_func=rush_into,
        )
        self.remove(null_vec_pos, null_vec_neg)

        # Orthogonal complement indicator: N(A) = (C(A^T))^\perp
        ortho_text = MathTex(
            r"\mathcal{N}(A) = (\mathcal{C}(A^\top))^\perp \iff \mathbf{v}_{\text{null}} \cdot \mathbf{row}_i = 0",
            font_size=17,
            color=MUTED_TEXT,
        )
        ortho_panel = make_hud_panel(ortho_text)
        self.pin_hud(ortho_panel, corner=DL)

        self.wait(0.8)

        # Master Rank-Nullity Theorem Card
        rank_nullity_math = MathTex(
            r"\underbrace{\text{rank}(A)}_{2\text{D (Plane)}} + \underbrace{\text{nullity}(A)}_{1\text{D (Line)}} = \underbrace{3}_{\text{Domain }\mathbb{R}^3}",
            font_size=20,
            color=TEXT_COLOR,
        )
        # Color coding parts of equation
        rank_nullity_math[0][0:7].set_color(SUBSPACE_COLOR)
        rank_nullity_math[0][8:18].set_color(NULL_COLOR)
        rank_nullity_panel = make_hud_panel(rank_nullity_math, padding=0.25)
        self.pin_hud(rank_nullity_panel, corner=DR)

        self.wait(1.5)

        # Clean up Act 3
        self.unpin_hud(title_panel, null_info_panel, ortho_panel, rank_nullity_panel)
        self.play(FadeOut(null_line), FadeOut(axes), FadeOut(origin_dot), run_time=0.8)

    # -------------------------------------------------------------------------
    # Act 4: Degenerate / Edge-Case Stress Test (Affine Solutions & Inconsistency)
    # -------------------------------------------------------------------------
    def run_act4(self):
        axes = self.create_axes_3d()
        origin_dot = Dot3D(point=axes.c2p(0, 0, 0), radius=0.08, color=ORIGIN_COLOR)

        title_text = VGroup(
            Text("Act 4: Affine Solution Line & Inconsistent Systems", font_size=20, color=TEXT_COLOR, weight=BOLD),
            Text("Solutions: x_p + N(A) | Rouché-Capelli Rank Mismatch", font_size=14, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        title_panel = make_hud_panel(title_text)
        self.pin_hud(title_panel, corner=UL)

        self.play(Create(axes), FadeIn(origin_dot), run_time=0.8)

        # 1. Particular solution x_p = [1, 2, 0]^T and affine line x_p + t * v_null
        arrow_xp = Arrow3D(start=axes.c2p(0, 0, 0), end=axes.c2p(*X_P), color=AFFINE_COLOR)
        affine_line = Line3D(
            start=axes.c2p(*(X_P - 2.5 * V_NULL)),
            end=axes.c2p(*(X_P + 2.5 * V_NULL)),
            color=AFFINE_COLOR,
        )
        # Target point b on canvas
        target_dot = Dot3D(point=axes.c2p(*TARGET_B), radius=0.10, color=TARGET_COLOR)
        target_arrow = Arrow3D(start=axes.c2p(0, 0, 0), end=axes.c2p(*TARGET_B), color=TARGET_COLOR)

        affine_info_text = VGroup(
            MathTex(r"\text{Solution Set: } \mathbf{x} = \mathbf{x}_p + t \mathbf{v}_{\text{null}}", font_size=17, color=AFFINE_COLOR),
            MathTex(r"\text{Particular: } \mathbf{x}_p = [1, 2, 0]^\top", font_size=15, color=TEXT_COLOR),
            MathTex(r"\text{Least-Norm: } \mathbf{x}_{\text{LN}} = [0, 1, 1]^\top \perp \mathcal{N}(A)", font_size=15, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        affine_info_panel = make_hud_panel(affine_info_text)
        self.pin_hud(affine_info_panel, corner=UR)

        self.play(FadeIn(arrow_xp), Create(affine_line), FadeIn(target_dot), FadeIn(target_arrow), run_time=1.0)

        # Funnel message: all points on affine line map to target b
        funnel_text = MathTex(
            r"\forall \mathbf{x} \in \mathbf{x}_p + \mathcal{N}(A): A\mathbf{x} = A\mathbf{x}_p + \mathbf{0} = \mathbf{b}",
            font_size=17,
            color=TARGET_COLOR,
        )
        funnel_panel = make_hud_panel(funnel_text)
        self.pin_hud(funnel_panel, corner=DL)
        self.wait(1.0)

        # 2. Inconsistent unreachable target probe b_unreach = [1, 0, 0]^T
        unreach_dot = Dot3D(point=axes.c2p(*B_UNREACH), radius=0.10, color=ERROR_COLOR)
        unreach_arrow = Arrow3D(start=axes.c2p(0, 0, 0), end=axes.c2p(*B_UNREACH), color=ERROR_COLOR)

        rouche_capelli_text = VGroup(
            MathTex(r"\text{Off-Plane Target: } \mathbf{b}_{\text{unreach}} = [1, 0, 0]^\top", font_size=17, color=ERROR_COLOR),
            MathTex(r"\text{Rank Mismatch: } \text{rank}(A) = 2 < \text{rank}([A \mid \mathbf{b}]) = 3", font_size=16, color=TEXT_COLOR),
            Text("0 Solutions: Inconsistent System (Rouché-Capelli)", font_size=14, color=ERROR_COLOR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        rouche_capelli_panel = make_hud_panel(rouche_capelli_text)
        self.pin_hud(rouche_capelli_panel, corner=DR)

        self.play(FadeIn(unreach_dot), FadeIn(unreach_arrow), run_time=0.8)
        self.wait(1.5)

        # Clean up Act 4
        self.unpin_hud(title_panel, affine_info_panel, funnel_panel, rouche_capelli_panel)
        self.play(
            FadeOut(arrow_xp), FadeOut(affine_line), FadeOut(target_dot), FadeOut(target_arrow),
            FadeOut(unreach_dot), FadeOut(unreach_arrow), FadeOut(axes), FadeOut(origin_dot),
            run_time=0.8
        )


# -----------------------------------------------------------------------------
# Individual Act Executable Scenes (for Rapid Modular Testing & AST Verification)
# -----------------------------------------------------------------------------
class Act1SpatialPuzzle(BaseColumnNullSpaceScene):
    """Executes Act 1: 3D Spatial Reachability Puzzle."""
    def construct(self):
        self.run_act1()


class Act2DynamicDeformation(BaseColumnNullSpaceScene):
    """Executes Act 2: Dynamic 3D to 2D Planar Collapse."""
    def construct(self):
        self.run_act2()


class Act3AlgebraicMorph(BaseColumnNullSpaceScene):
    """Executes Act 3: Null Space 1D Line & Rank-Nullity Theorem."""
    def construct(self):
        self.run_act3()


class Act4DegenerateStressTest(BaseColumnNullSpaceScene):
    """Executes Act 4: Affine Solution Line & Rouché-Capelli Inconsistency."""
    def construct(self):
        self.run_act4()


# -----------------------------------------------------------------------------
# Master Unified Scene (Full 4-Act Master Curriculum Production)
# -----------------------------------------------------------------------------
class ColumnNullSpaceScene(BaseColumnNullSpaceScene):
    """Master production scene rendering the complete 4-Act Sprint 3 module."""
    def construct(self):
        self.run_act1()
        self.run_act2()
        self.run_act3()
        self.run_act4()
