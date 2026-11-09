"""
Sprint 1: Vectors, Linear Combinations & Span (ManimCE v0.18+)
Curriculum: Linear Algebra & Theoretical CS / Machine Learning Foundations
References: MML Ch. 2.4-2.6, VMLS Ch. 1 & 5, ILA Ch. 1-2

This production script implements the 4-act pedagogical narrative:
- Act 1: Concrete Spatial Puzzle (Solving 2*v1 + 1*v2 = b)
- Act 2: Dynamic Deformation & Ghost Space (Sweeping c1, c2 dials, oblique mesh)
- Act 3: Color-Coded Algebraic Morph (Formula morph to geometry, 2D continuous span wash)
- Act 4: Degenerate Stress Test (Collinear collapse to 1D line, Rouché-Capelli rank failure, 0D point)

Render Commands:
    Preview Low-Res:   manim -ql scene_vectors_span.py VectorSpanScene
    Production Hi-Res: manim -qh scene_vectors_span.py VectorSpanScene
    Single Act Test:   manim -ql scene_vectors_span.py Act1SpatialPuzzle
"""

from __future__ import annotations

import numpy as np
from manim import *

# -----------------------------------------------------------------------------
# Rule 1: 3Blue1Brown Semantic Color Palette
# -----------------------------------------------------------------------------
I_HAT_COLOR = "#83C167"       # Green C  - Basis vector v1 / x-projection
J_HAT_COLOR = "#FC6255"       # Red C    - Basis vector v2 / y-projection
VEC_COLOR = "#F7D96F"         # Yellow   - Resultant linear combination sum
TARGET_COLOR = "#F9E498"      # Gold A   - Target objective point b
SUBSPACE_COLOR = "#9A72AC"    # Purple   - Reachable span / subspace wash
LINE_COLOR = "#58C4DD"        # Blue C   - Subspace line trace
ORIGIN_COLOR = "#FFFFFF"      # White    - Subspace origin invariant anchor (0, 0)
ERROR_COLOR = "#F7768E"       # Soft Red - Inconsistency / rank mismatch indicator
BG_COLOR = "#1C1C24"          # Slate    - High-contrast dark viewport background
PANEL_BG = "#16161E"          # Obsidian - HUD container background fill
PANEL_BORDER = "#414868"      # Slate-Teal- HUD container outline stroke
TEXT_COLOR = "#ECEFF4"        # Snow     - High-contrast text typography
MUTED_TEXT = "#7AA2F7"        # Light Blue- Secondary badges and formula labels

# -----------------------------------------------------------------------------
# Mathematical Constants & Assertions (Rule 6: Exact NumPy Calculations)
# -----------------------------------------------------------------------------
V1_COORDS = np.array([2.0, 1.0])
V2_COORDS = np.array([-1.0, 2.0])
TARGET_COORDS = np.array([3.0, 4.0])
UNREACHABLE_COORDS = np.array([1.0, 4.0])

# Pre-flight mathematical consistency verification
A_BASIS = np.column_stack([V1_COORDS, V2_COORDS])
SOL_COEFFS = np.linalg.solve(A_BASIS, TARGET_COORDS)
assert np.allclose(SOL_COEFFS, [2.0, 1.0]), "Act 1 basis solution must be c1=2, c2=1"
assert np.isclose(np.linalg.det(A_BASIS), 5.0), "Basis matrix determinant must be 5.0"
assert np.linalg.matrix_rank(A_BASIS) == 2, "Basis vectors must have full rank 2"


# -----------------------------------------------------------------------------
# Rule 3: Zero Text Overlap HUD Panel Factory
# -----------------------------------------------------------------------------
def make_hud_panel(
    content: Mobject,
    padding: float = 0.2,
    bg_color: str = PANEL_BG,
    bg_opacity: float = 0.88,
    stroke_color: str = PANEL_BORDER,
    stroke_width: float = 1.5,
    corner_radius: float = 0.12,
) -> VGroup:
    """Wraps any mobject inside a high-contrast bounded container."""
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
    return VGroup(bg, content)


# -----------------------------------------------------------------------------
# Base Scene Class (Shared Geometry, Palette, and Modular Act Execution)
# -----------------------------------------------------------------------------
class BaseVectorSpanScene(Scene):
    """Base class providing shared infrastructure and modular act methods."""

    def setup(self):
        super().setup()
        self.camera.background_color = BG_COLOR

    def create_plane(self) -> NumberPlane:
        """Rule 2: Coordinate system with custom ghost planes & 0.8 unit scale."""
        return NumberPlane(
            x_range=[-6, 6, 1],
            y_range=[-4, 4, 1],
            x_length=9.6,  # 0.8 scale factor protects viewport boundaries
            y_length=6.4,
            background_line_style={
                "stroke_color": TEAL_E,
                "stroke_width": 1.2,
                "stroke_opacity": 0.35,
            },
            faded_line_style={
                "stroke_color": TEAL_E,
                "stroke_width": 0.8,
                "stroke_opacity": 0.15,
            },
            axis_config={
                "stroke_color": WHITE,
                "stroke_width": 2.0,
                "include_numbers": False,
            },
        )

    def create_origin_anchor(self, plane: NumberPlane) -> VGroup:
        """Rule 5: Subspace origin invariant anchor dot and label."""
        dot = Dot(plane.c2p(0, 0), radius=0.08, color=ORIGIN_COLOR)
        label = MathTex(r"\mathbf{0}", font_size=22, color=ORIGIN_COLOR).next_to(
            dot, DL, buff=0.1
        )
        return VGroup(dot, label)

    def create_vector_arrow(
        self,
        plane: NumberPlane,
        start_coords: np.ndarray,
        end_coords: np.ndarray,
        color: str,
        stroke_width: float = 4.0,
    ) -> Arrow:
        """Rule 4: Explicit buff=0 ensures continuous tip-to-tail chaining."""
        return Arrow(
            plane.c2p(*start_coords),
            plane.c2p(*end_coords),
            buff=0,
            color=color,
            stroke_width=stroke_width,
            max_tip_length_to_length_ratio=0.25,
        )

    # -------------------------------------------------------------------------
    # Act 1: Concrete Spatial Puzzle
    # -------------------------------------------------------------------------
    def run_act1(self, plane: NumberPlane, origin_group: VGroup, is_standalone: bool = False):
        # 1. Setup Header & Objective HUD
        title_text = VGroup(
            Text("Act 1: Concrete Spatial Puzzle", font_size=22, color=TEXT_COLOR, weight=BOLD),
            Text("Can two fixed arrows reach any target point?", font_size=16, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        header_panel = make_hud_panel(title_text).to_corner(UL, buff=0.35)

        v1_arrow = self.create_vector_arrow(plane, np.array([0.0, 0.0]), V1_COORDS, I_HAT_COLOR)
        v1_lbl = MathTex(r"\mathbf{v}_1 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}", font_size=20, color=I_HAT_COLOR).next_to(
            plane.c2p(*V1_COORDS), DR, buff=0.15
        )

        v2_arrow = self.create_vector_arrow(plane, np.array([0.0, 0.0]), V2_COORDS, J_HAT_COLOR)
        v2_lbl = MathTex(r"\mathbf{v}_2 = \begin{bmatrix} -1 \\ 2 \end{bmatrix}", font_size=20, color=J_HAT_COLOR).next_to(
            plane.c2p(*V2_COORDS), UL, buff=0.15
        )

        target_dot = Dot(plane.c2p(*TARGET_COORDS), radius=0.11, color=TARGET_COLOR)
        target_lbl = MathTex(r"\mathbf{b} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}", font_size=20, color=TARGET_COLOR).next_to(
            target_dot, UR, buff=0.15
        )

        self.play(FadeIn(header_panel, shift=DOWN), run_time=1.0)
        self.play(
            GrowArrow(v1_arrow),
            FadeIn(v1_lbl),
            GrowArrow(v2_arrow),
            FadeIn(v2_lbl),
            run_time=1.2,
        )
        self.play(FadeIn(target_dot, scale=1.5), FadeIn(target_lbl), run_time=0.8)
        self.play(Indicate(target_dot, color=TARGET_COLOR), run_time=0.8)
        self.wait(0.5)

        # 2. Attempt 1: Naive concatenation (Shortfall)
        shortfall_hud = make_hud_panel(
            Text("Attempt 1: 1·v1 + 1·v2 (Shortfall)", font_size=16, color=ERROR_COLOR)
        ).to_corner(DR, buff=0.35)
        
        v2_tip_to_tail = self.create_vector_arrow(
            plane, V1_COORDS, V1_COORDS + V2_COORDS, J_HAT_COLOR, stroke_width=3.5
        )
        naive_sum = self.create_vector_arrow(
            plane, np.array([0.0, 0.0]), V1_COORDS + V2_COORDS, VEC_COLOR, stroke_width=3.5
        )

        self.play(FadeIn(shortfall_hud, shift=UP), run_time=0.6)
        self.play(v2_arrow.animate.move_to(v2_tip_to_tail.get_center()), run_time=1.2)
        self.play(Create(naive_sum), run_time=1.0)
        self.play(Wiggle(naive_sum), run_time=0.8)
        self.wait(0.6)

        # 3. Attempt 2: Exact Solution (2*v1 + 1*v2 = b)
        solution_hud = make_hud_panel(
            Text("Attempt 2: 2·v1 + 1·v2 = b (Exact Solution)", font_size=16, color=TEXT_COLOR)
        ).to_corner(DR, buff=0.35)

        v1_scaled = self.create_vector_arrow(plane, np.array([0.0, 0.0]), 2.0 * V1_COORDS, I_HAT_COLOR)
        v2_solved = self.create_vector_arrow(
            plane, 2.0 * V1_COORDS, TARGET_COORDS, J_HAT_COLOR
        )
        sol_vec = self.create_vector_arrow(
            plane, np.array([0.0, 0.0]), TARGET_COORDS, VEC_COLOR, stroke_width=4.5
        )

        # Parallelogram closure lines
        p_line1 = DashedLine(
            plane.c2p(0, 0), plane.c2p(*V2_COORDS), color=J_HAT_COLOR, stroke_opacity=0.4
        )
        p_line2 = DashedLine(
            plane.c2p(*V2_COORDS), plane.c2p(*TARGET_COORDS), color=I_HAT_COLOR, stroke_opacity=0.4
        )

        self.play(
            FadeOut(shortfall_hud),
            FadeIn(solution_hud),
            FadeOut(naive_sum),
            Transform(v1_arrow, v1_scaled),
            Transform(v2_arrow, v2_solved),
            run_time=1.5,
        )
        self.play(Create(sol_vec), Create(p_line1), Create(p_line2), run_time=1.2)
        self.play(Indicate(target_dot, scale_factor=1.3), run_time=0.8)
        self.wait(1.0)

        # Cleanup ephemerals if orchestrating master scene
        if not is_standalone:
            self.play(
                FadeOut(header_panel),
                FadeOut(solution_hud),
                FadeOut(target_dot),
                FadeOut(target_lbl),
                FadeOut(v1_lbl),
                FadeOut(v2_lbl),
                FadeOut(sol_vec),
                FadeOut(p_line1),
                FadeOut(p_line2),
                v1_arrow.animate.become(self.create_vector_arrow(plane, np.array([0.0, 0.0]), V1_COORDS, I_HAT_COLOR)),
                v2_arrow.animate.become(self.create_vector_arrow(plane, np.array([0.0, 0.0]), V2_COORDS, J_HAT_COLOR)),
                run_time=1.2,
            )
        return v1_arrow, v2_arrow

    # -------------------------------------------------------------------------
    # Act 2: Dynamic Deformation & Ghost Space
    # -------------------------------------------------------------------------
    def run_act2(
        self,
        plane: NumberPlane,
        origin_group: VGroup,
        v1_arrow: Arrow,
        v2_arrow: Arrow,
        is_standalone: bool = False,
    ):
        header_text = VGroup(
            Text("Act 2: Dynamic Deformation & Ghost Space", font_size=22, color=TEXT_COLOR, weight=BOLD),
            Text("Continuous dials sweeping through all linear combinations", font_size=16, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        header_panel = make_hud_panel(header_text).to_corner(UL, buff=0.35)
        header_panel.set_z_index(10)

        # ValueTrackers and DecimalNumber HUD Readouts (Rule 4: Zero LaTeX recompilation)
        c1_tracker = ValueTracker(1.0)
        c2_tracker = ValueTracker(0.0)

        dial_title = Text("Coefficients", font_size=14, color=MUTED_TEXT)
        c1_label = MathTex(r"c_1:", font_size=18, color=I_HAT_COLOR)
        c1_num = DecimalNumber(1.00, num_decimal_places=2, color=I_HAT_COLOR)
        c1_num.add_updater(lambda m: m.set_value(c1_tracker.get_value()))

        c2_label = MathTex(r"c_2:", font_size=18, color=J_HAT_COLOR)
        c2_num = DecimalNumber(0.00, num_decimal_places=2, color=J_HAT_COLOR)
        c2_num.add_updater(lambda m: m.set_value(c2_tracker.get_value()))

        c1_group = VGroup(c1_label, c1_num).arrange(RIGHT, buff=0.1)
        c2_group = VGroup(c2_label, c2_num).arrange(RIGHT, buff=0.1)
        dial_panel_content = VGroup(dial_title, c1_group, c2_group).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        dial_panel = make_hud_panel(dial_panel_content).to_corner(UR, buff=0.35)
        dial_panel.set_z_index(10)

        # Dynamic swept arrows using always_redraw
        sweep_v1 = always_redraw(
            lambda: Arrow(
                plane.c2p(0, 0),
                plane.c2p(*(c1_tracker.get_value() * V1_COORDS)),
                buff=0,
                color=I_HAT_COLOR,
                stroke_width=4.0,
            )
        )
        sweep_v2 = always_redraw(
            lambda: Arrow(
                plane.c2p(*(c1_tracker.get_value() * V1_COORDS)),
                plane.c2p(*(c1_tracker.get_value() * V1_COORDS + c2_tracker.get_value() * V2_COORDS)),
                buff=0,
                color=J_HAT_COLOR,
                stroke_width=4.0,
            )
        )

        self.play(FadeIn(header_panel, shift=DOWN), FadeIn(dial_panel, shift=DOWN), run_time=1.0)
        self.add(sweep_v1, sweep_v2)
        self.remove(v1_arrow, v2_arrow)

        # 1. 1D Line Sweep (c1 sweeps [-2.5, 2.5], c2=0)
        line_1d = DashedLine(
            plane.c2p(*(-2.5 * V1_COORDS)),
            plane.c2p(*(2.5 * V1_COORDS)),
            color=I_HAT_COLOR,
            stroke_opacity=0.35,
        )
        badge_1d = make_hud_panel(
            Text("span(v1): 1D Line Subspace", font_size=16, color=I_HAT_COLOR)
        ).to_corner(DL, buff=0.35)

        self.play(FadeIn(badge_1d, shift=UP), Create(line_1d), run_time=0.8)
        self.play(c1_tracker.animate.set_value(2.5), run_time=1.5, rate_func=smooth)
        self.play(c1_tracker.animate.set_value(-2.0), run_time=2.0, rate_func=smooth)
        self.play(c1_tracker.animate.set_value(1.0), run_time=1.2, rate_func=smooth)
        self.wait(0.5)

        # 2. 2D Surface Sweep (c2 sweeps [-2.0, 2.0])
        badge_2d = make_hud_panel(
            Text("Oblique Mesh: span(v1, v2) = R²", font_size=16, color=TEXT_COLOR)
        ).to_corner(DL, buff=0.35)

        self.play(ReplacementTransform(badge_1d, badge_2d), run_time=0.8)
        self.play(c2_tracker.animate.set_value(1.8), run_time=1.5, rate_func=smooth)
        self.play(c2_tracker.animate.set_value(-1.8), run_time=2.0, rate_func=smooth)
        self.play(c2_tracker.animate.set_value(1.0), run_time=1.2, rate_func=smooth)
        self.wait(0.5)

        # 3. Discrete Oblique Mesh Generation
        mesh_lines = VGroup()
        for i in range(-2, 3):
            # Lines parallel to v2
            start_pt = i * V1_COORDS - 2.5 * V2_COORDS
            end_pt = i * V1_COORDS + 2.5 * V2_COORDS
            mesh_lines.add(
                DashedLine(plane.c2p(*start_pt), plane.c2p(*end_pt), color=J_HAT_COLOR, stroke_opacity=0.25)
            )
            # Lines parallel to v1
            start_pt2 = -2.5 * V1_COORDS + i * V2_COORDS
            end_pt2 = 2.5 * V1_COORDS + i * V2_COORDS
            mesh_lines.add(
                DashedLine(plane.c2p(*start_pt2), plane.c2p(*end_pt2), color=I_HAT_COLOR, stroke_opacity=0.25)
            )

        mesh_lines.set_z_index(-1)
        self.play(Create(mesh_lines), run_time=1.5)
        self.wait(1.0)

        # Transition / Cleanup
        sweep_v1.clear_updaters()
        sweep_v2.clear_updaters()
        self.remove(sweep_v1, sweep_v2)

        v1_restored = self.create_vector_arrow(plane, np.array([0.0, 0.0]), V1_COORDS, I_HAT_COLOR)
        v2_restored = self.create_vector_arrow(plane, np.array([0.0, 0.0]), V2_COORDS, J_HAT_COLOR)
        self.add(v1_restored, v2_restored)

        if not is_standalone:
            self.play(
                FadeOut(header_panel),
                FadeOut(dial_panel),
                FadeOut(badge_2d),
                FadeOut(line_1d),
                FadeOut(mesh_lines),
                run_time=1.0,
            )
        return v1_restored, v2_restored

    # -------------------------------------------------------------------------
    # Act 3: Color-Coded Algebraic Morph
    # -------------------------------------------------------------------------
    def run_act3(
        self,
        plane: NumberPlane,
        origin_group: VGroup,
        v1_arrow: Arrow,
        v2_arrow: Arrow,
        is_standalone: bool = False,
    ):
        header_text = VGroup(
            Text("Act 3: Color-Coded Algebraic Morph", font_size=22, color=TEXT_COLOR, weight=BOLD),
            Text("Bridging formal set-builder algebra to infinite geometric sheet", font_size=16, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        header_panel = make_hud_panel(header_text).to_corner(UL, buff=0.35)

        # Mathematical morph formula
        eq_linear_comb = MathTex(
            r"\mathbf{v}", r"=", r"c_1 \mathbf{v}_1", r"+", r"c_2 \mathbf{v}_2",
            font_size=32,
        )
        eq_linear_comb[2].set_color(I_HAT_COLOR)
        eq_linear_comb[4].set_color(J_HAT_COLOR)
        panel_eq = make_hud_panel(eq_linear_comb).to_corner(UR, buff=0.35)

        eq_span_def = MathTex(
            r"\text{span}(\mathbf{v}_1, \mathbf{v}_2) = \left\{ c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 \;\middle|\; c_1, c_2 \in \mathbb{R} \right\}",
            font_size=28,
        )
        eq_span_def.set_color_by_tex(r"c_1 \mathbf{v}_1", I_HAT_COLOR)
        eq_span_def.set_color_by_tex(r"c_2 \mathbf{v}_2", J_HAT_COLOR)
        panel_span = make_hud_panel(eq_span_def).to_corner(UR, buff=0.35)

        self.play(FadeIn(header_panel, shift=DOWN), FadeIn(panel_eq, shift=DOWN), run_time=1.0)
        self.wait(1.0)

        # Morph formula to set-builder span definition
        self.play(ReplacementTransform(panel_eq, panel_span), run_time=1.5)
        self.wait(0.5)

        # Continuous 2D Span Wash
        span_sheet = Rectangle(
            width=16.0,
            height=10.0,
            fill_color=SUBSPACE_COLOR,
            fill_opacity=0.25,
            stroke_width=0,
        )
        dim_badge = make_hud_panel(
            Text("Subspace: span(v1, v2) = R² | Dim = 2 | Full Rank", font_size=16, color=TEXT_COLOR)
        ).to_corner(DL, buff=0.35)

        # Note the origin anchor remains prominent
        self.play(FadeIn(span_sheet), FadeIn(dim_badge, shift=UP), run_time=1.5)
        self.play(Indicate(origin_group[0], scale_factor=1.4, color=ORIGIN_COLOR), run_time=0.8)
        self.wait(1.0)

        if not is_standalone:
            self.play(
                FadeOut(header_panel),
                FadeOut(panel_span),
                FadeOut(dim_badge),
                run_time=1.0,
            )
        return v1_arrow, v2_arrow, span_sheet

    # -------------------------------------------------------------------------
    # Act 4: Degenerate Stress Test (Collinear & Zero Vector Collapse)
    # -------------------------------------------------------------------------
    def run_act4(
        self,
        plane: NumberPlane,
        origin_group: VGroup,
        v1_arrow: Arrow,
        v2_arrow: Arrow,
        span_sheet: Rectangle,
        is_standalone: bool = False,
    ):
        header_text = VGroup(
            Text("Act 4: Degenerate Stress Test", font_size=22, color=TEXT_COLOR, weight=BOLD),
            Text("Subspace collapse: What happens when the arrows align?", font_size=16, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        header_panel = make_hud_panel(header_text).to_corner(UL, buff=0.35)

        self.play(FadeIn(header_panel, shift=DOWN), run_time=1.0)
        self.wait(0.5)

        # 1. Collinear Transition: v2 rotates to 2*v1 = [4, 2]
        v2_collinear = self.create_vector_arrow(
            plane, np.array([0.0, 0.0]), 2.0 * V1_COORDS, J_HAT_COLOR
        )
        collinear_line = Line(
            plane.c2p(-5.0, -2.5),
            plane.c2p(5.0, 2.5),
            color=SUBSPACE_COLOR,
            stroke_width=4.0,
        )
        collapse_badge = make_hud_panel(
            Text("Collinear Collapse: det(A)=0 | rank(A)=1 | Dim = 1", font_size=16, color=ERROR_COLOR)
        ).to_corner(DL, buff=0.35)
        collapse_badge.set_z_index(10)

        self.play(
            Transform(v2_arrow, v2_collinear),
            Transform(span_sheet, collinear_line),
            FadeIn(collapse_badge, shift=UP),
            run_time=2.0,
            rate_func=rush_into,
        )
        self.wait(0.8)

        # 2. Impossibility Probe: Target b_unreachable = [1, 4]
        unreachable_dot = Dot(plane.c2p(*UNREACHABLE_COORDS), radius=0.11, color=ERROR_COLOR)
        unreachable_lbl = MathTex(
            r"\mathbf{b}_{\text{unreachable}} = \begin{bmatrix} 1 \\ 4 \end{bmatrix}",
            font_size=20,
            color=ERROR_COLOR,
        ).next_to(unreachable_dot, UR, buff=0.15)

        rouche_capelli_msg = VGroup(
            MathTex(r"\mathbf{b} \notin \text{span}(\mathbf{v}_1, \mathbf{v}_2)", font_size=20, color=ERROR_COLOR),
            MathTex(r"\text{rank}(A) = 1 \ne \text{rank}([A \mid \mathbf{b}]) = 2", font_size=18, color=TEXT_COLOR),
            Text("Rouché-Capelli: Inconsistent (0 Solutions)", font_size=14, color=ERROR_COLOR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        rouche_panel = make_hud_panel(rouche_capelli_msg).to_corner(UR, buff=0.35)
        rouche_panel.set_z_index(10)

        self.play(FadeIn(unreachable_dot), FadeIn(unreachable_lbl), FadeIn(rouche_panel, shift=DOWN), run_time=1.0)
        self.play(Wiggle(unreachable_dot), run_time=0.8)
        self.wait(1.0)

        # 3. Trivial Subspace Collapse (v1, v2 -> 0)
        trivial_badge = make_hud_panel(
            Text("Trivial Subspace: span(0) = {0} | Dim = 0", font_size=16, color=ORIGIN_COLOR)
        ).to_corner(DL, buff=0.35)

        self.play(
            FadeOut(unreachable_dot),
            FadeOut(unreachable_lbl),
            FadeOut(rouche_panel),
            ReplacementTransform(collapse_badge, trivial_badge),
            v1_arrow.animate.become(Dot(plane.c2p(0, 0), radius=0.08, color=I_HAT_COLOR)),
            v2_arrow.animate.become(Dot(plane.c2p(0, 0), radius=0.08, color=J_HAT_COLOR)),
            span_sheet.animate.become(Dot(plane.c2p(0, 0), radius=0.08, color=SUBSPACE_COLOR)),
            run_time=2.0,
            rate_func=rush_into,
        )
        self.play(Indicate(origin_group[0], scale_factor=1.6, color=ORIGIN_COLOR), run_time=1.0)
        self.wait(1.5)

        # Outro Hold & Final Summary
        summary_box_content = VGroup(
            Text("Linear Span Hierarchy & Invariants", font_size=20, color=TEXT_COLOR, weight=BOLD),
            Text("• 2D Plane: Linearly Independent (rank=2, det≠0)", font_size=15, color=I_HAT_COLOR),
            Text("• 1D Line: Collinear / Dependent (rank=1, det=0)", font_size=15, color=SUBSPACE_COLOR),
            Text("• 0D Point: Zero Vector Trivial Subspace (rank=0)", font_size=15, color=ORIGIN_COLOR),
            Text("• Invariant: Origin 0 is strictly contained in every linear subspace", font_size=15, color=VEC_COLOR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        summary_panel = make_hud_panel(summary_box_content, padding=0.28, bg_opacity=1.0).move_to(ORIGIN)
        summary_panel.set_z_index(10)

        self.play(
            FadeOut(header_panel),
            FadeOut(trivial_badge),
            FadeOut(origin_group),
            FadeOut(v1_arrow),
            FadeOut(v2_arrow),
            FadeOut(span_sheet),
            run_time=0.8,
        )
        self.play(
            FadeIn(summary_panel, scale=1.1),
            run_time=1.5,
        )
        self.wait(2.5)


# Alias for compatibility
VectorSpanBase = BaseVectorSpanScene


# -----------------------------------------------------------------------------
# Modular Executable Scene Classes
# -----------------------------------------------------------------------------
class Act1SpatialPuzzle(BaseVectorSpanScene):
    """Standalone executable scene for Act 1."""
    def construct(self):
        plane = self.create_plane()
        origin_group = self.create_origin_anchor(plane)
        self.add(plane, origin_group)
        self.run_act1(plane, origin_group, is_standalone=True)


class Act2DynamicDeformation(BaseVectorSpanScene):
    """Standalone executable scene for Act 2."""
    def construct(self):
        plane = self.create_plane()
        origin_group = self.create_origin_anchor(plane)
        v1 = self.create_vector_arrow(plane, np.array([0.0, 0.0]), V1_COORDS, I_HAT_COLOR)
        v2 = self.create_vector_arrow(plane, np.array([0.0, 0.0]), V2_COORDS, J_HAT_COLOR)
        self.add(plane, origin_group, v1, v2)
        self.run_act2(plane, origin_group, v1, v2, is_standalone=True)


class Act3AlgebraicMorph(BaseVectorSpanScene):
    """Standalone executable scene for Act 3."""
    def construct(self):
        plane = self.create_plane()
        origin_group = self.create_origin_anchor(plane)
        v1 = self.create_vector_arrow(plane, np.array([0.0, 0.0]), V1_COORDS, I_HAT_COLOR)
        v2 = self.create_vector_arrow(plane, np.array([0.0, 0.0]), V2_COORDS, J_HAT_COLOR)
        self.add(plane, origin_group, v1, v2)
        self.run_act3(plane, origin_group, v1, v2, is_standalone=True)


class Act4DegenerateStressTest(BaseVectorSpanScene):
    """Standalone executable scene for Act 4."""
    def construct(self):
        plane = self.create_plane()
        origin_group = self.create_origin_anchor(plane)
        v1 = self.create_vector_arrow(plane, np.array([0.0, 0.0]), V1_COORDS, I_HAT_COLOR)
        v2 = self.create_vector_arrow(plane, np.array([0.0, 0.0]), V2_COORDS, J_HAT_COLOR)
        span_sheet = Rectangle(
            width=16.0,
            height=10.0,
            fill_color=SUBSPACE_COLOR,
            fill_opacity=0.25,
            stroke_width=0,
        )
        self.add(plane, origin_group, span_sheet, v1, v2)
        self.run_act4(plane, origin_group, v1, v2, span_sheet, is_standalone=True)


class VectorSpanScene(BaseVectorSpanScene):
    """Master unified scene seamlessly orchestrating all 4 acts."""
    def construct(self):
        plane = self.create_plane()
        origin_group = self.create_origin_anchor(plane)
        self.add(plane, origin_group)

        # Act 1: Concrete Spatial Puzzle
        v1, v2 = self.run_act1(plane, origin_group, is_standalone=False)

        # Act 2: Dynamic Deformation & Ghost Space
        v1, v2 = self.run_act2(plane, origin_group, v1, v2, is_standalone=False)

        # Act 3: Color-Coded Algebraic Morph
        v1, v2, span_sheet = self.run_act3(plane, origin_group, v1, v2, is_standalone=False)

        # Act 4: Degenerate Stress Test
        self.run_act4(plane, origin_group, v1, v2, span_sheet, is_standalone=False)
