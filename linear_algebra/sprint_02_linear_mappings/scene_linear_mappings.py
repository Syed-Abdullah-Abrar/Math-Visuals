"""
Sprint 2: Linear Mappings & Matrix Multiplication as Composition (ManimCE v0.18+)
Curriculum: Linear Algebra & Theoretical CS / Machine Learning Foundations
References: MML Ch. 2.5, VMLS Ch. 6, 7 & 10, Kreyszig Ch. 7, ILA Ch. 6

This production script implements the 4-act pedagogical narrative:
- Act 1: Concrete Spatial Puzzle (Tracking Basis Destinations i_hat -> a1, j_hat -> a2)
- Act 2: Dynamic Deformation & Ghost Space (Continuous Homotopy M(t) = (1-t)I + tA, Parallel Grid Preservation)
- Act 3: Color-Coded Algebraic Morph (Matrix Multiplication as Sequential Composition BA)
- Act 4: Degenerate Stress Test (Geometric Non-Commutativity SR != RS via Rotation and Shear)

Render Commands:
    Preview Low-Res:   manim -ql scene_linear_mappings.py LinearMappingsScene
    Production Hi-Res: manim -qh scene_linear_mappings.py LinearMappingsScene
    Single Act Test:   manim -ql scene_linear_mappings.py Act1SpatialPuzzle
"""

from __future__ import annotations

import math
import numpy as np
from manim import *

# -----------------------------------------------------------------------------
# Rule 1: 3Blue1Brown Semantic Color Palette
# -----------------------------------------------------------------------------
I_HAT_COLOR = "#83C167"       # Green C   - Basis vector i_hat / Column 1 (a1)
J_HAT_COLOR = "#FC6255"       # Red C     - Basis vector j_hat / Column 2 (a2)
VEC_COLOR = "#F7D96F"         # Yellow    - Arbitrary input / resultant vector x
TARGET_COLOR = "#F9E498"      # Gold A    - Transformed destination point Ax
SUBSPACE_COLOR = "#9A72AC"    # Purple    - Unit square / transformed parallelogram area
LINE_COLOR = "#58C4DD"        # Blue C    - Deformed grid lines trace
ORIGIN_COLOR = "#FFFFFF"      # White     - Universal origin invariant anchor (0, 0)
ERROR_COLOR = "#F7768E"       # Soft Red  - Non-commutativity discrepancy / divergence
BG_COLOR = "#1C1C24"          # Slate     - High-contrast dark viewport background
PANEL_BG = "#16161E"          # Obsidian  - HUD container background fill
PANEL_BORDER = "#414868"      # Slate-Teal- HUD container outline stroke
TEXT_COLOR = "#ECEFF4"        # Snow      - High-contrast typography
MUTED_TEXT = "#7AA2F7"        # Light Blue- Secondary badges and formula labels

# -----------------------------------------------------------------------------
# Mathematical Constants & Assertions (Rule 6: Exact NumPy Calculations)
# -----------------------------------------------------------------------------
A_MAT = np.array([[2.0, 0.0], [1.0, 2.0]])
A1_COORDS = np.array([2.0, 1.0])
A2_COORDS = np.array([0.0, 2.0])
PUZZLE_X = np.array([1.5, 1.0])
PUZZLE_TARGET = A_MAT @ PUZZLE_X  # [3.0, 3.5]

# Shear and Rotation matrices for composition & non-commutativity
S_MAT = np.array([[1.0, 1.0], [0.0, 1.0]])
R_MAT = np.array([[0.0, -1.0], [1.0, 0.0]])
SR_MAT = S_MAT @ R_MAT  # [[1, -1], [1, 0]]
RS_MAT = R_MAT @ S_MAT  # [[0, -1], [1, 1]]
STRESS_X = np.array([2.0, 3.0])
DEST_SR = SR_MAT @ STRESS_X  # [-1.0, 2.0]
DEST_RS = RS_MAT @ STRESS_X  # [-3.0, 5.0]
COMMUTATOR = SR_MAT - RS_MAT  # [[1, 0], [0, -1]]

# Pre-flight mathematical consistency verification
assert np.allclose(A_MAT @ np.array([1.0, 0.0]), A1_COORDS), "a1 must be A @ [1, 0]"
assert np.allclose(A_MAT @ np.array([0.0, 1.0]), A2_COORDS), "a2 must be A @ [0, 1]"
assert np.isclose(np.linalg.det(A_MAT), 4.0), "det(A) must be 4.0"
assert np.linalg.matrix_rank(A_MAT) == 2, "Matrix A must have full rank 2"
assert np.allclose(PUZZLE_TARGET, [3.0, 3.5]), "Target point must be [3.0, 3.5]"
assert not np.allclose(SR_MAT, RS_MAT), "SR must not equal RS"
assert np.isclose(np.linalg.det(SR_MAT), 1.0), "det(SR) must be 1.0"
assert np.isclose(np.linalg.det(RS_MAT), 1.0), "det(RS) must be 1.0"
assert np.isclose(np.trace(COMMUTATOR), 0.0), "Trace of commutator must be zero"
assert np.isclose(np.linalg.norm(DEST_SR - DEST_RS), np.sqrt(13.0)), "Discrepancy must be sqrt(13)"


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
    """Wraps any mobject inside a high-contrast bounded container with z-index 10."""
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
# Base Scene Class (Shared Geometry, Palette, and Modular Act Execution)
# -----------------------------------------------------------------------------
class BaseLinearMappingsScene(Scene):
    """Base class providing shared infrastructure and modular act methods."""

    def setup(self):
        super().setup()
        self.camera.background_color = BG_COLOR

    def create_plane(self) -> NumberPlane:
        """Rule 2: Coordinate system with custom ghost planes & 0.8 unit scale."""
        return NumberPlane(
            x_range=[-6, 6, 1],
            y_range=[-5, 5, 1],
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

    def create_ghost_plane(self) -> NumberPlane:
        """Faint background reference grid that remains static during deformations."""
        return NumberPlane(
            x_range=[-6, 6, 1],
            y_range=[-5, 5, 1],
            x_length=9.6,
            y_length=6.4,
            background_line_style={
                "stroke_color": "#414868",
                "stroke_width": 1.0,
                "stroke_opacity": 0.25,
            },
            faded_line_style={
                "stroke_color": "#414868",
                "stroke_width": 0.6,
                "stroke_opacity": 0.10,
            },
            axis_config={
                "stroke_color": "#7AA2F7",
                "stroke_width": 1.2,
                "include_numbers": False,
            },
        )

    def create_origin_anchor(self, plane: NumberPlane) -> VGroup:
        """Rule 5: Subspace origin invariant anchor dot and label."""
        dot = Dot(plane.c2p(0, 0), radius=0.08, color=ORIGIN_COLOR)
        label = MathTex(r"\mathbf{0}", font_size=22, color=ORIGIN_COLOR).next_to(
            dot, DL, buff=0.1
        )
        group = VGroup(dot, label)
        group.set_z_index(5)
        return group

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
    # ACT 1: Concrete Spatial Puzzle
    # -------------------------------------------------------------------------
    def run_act1(
        self, plane: NumberPlane, origin_group: VGroup, is_standalone: bool = False
    ) -> tuple[Arrow, Arrow]:
        """Act 1: Poses the reachability challenge, tracking basis transformation."""
        title = Text("Act 1: Spatial Puzzle — Tracking Basis Destinations", font_size=20, color=TEXT_COLOR)
        title_panel = make_hud_panel(title).to_corner(UL, buff=0.3)

        # Standard basis vectors i_hat = [1, 0]^T and j_hat = [0, 1]^T
        i_hat = self.create_vector_arrow(plane, np.zeros(2), np.array([1.0, 0.0]), I_HAT_COLOR)
        j_hat = self.create_vector_arrow(plane, np.zeros(2), np.array([0.0, 1.0]), J_HAT_COLOR)
        i_label = MathTex(r"\hat{\imath}", font_size=20, color=I_HAT_COLOR).next_to(i_hat, DOWN, buff=0.1)
        j_label = MathTex(r"\hat{\jmath}", font_size=20, color=J_HAT_COLOR).next_to(j_hat, LEFT, buff=0.1)

        # Arbitrary test vector x = 1.5 i_hat + 1.0 j_hat
        x_vec = self.create_vector_arrow(plane, np.zeros(2), PUZZLE_X, VEC_COLOR)
        x_label = MathTex(r"\mathbf{x} = 1.5\hat{\imath} + \hat{\jmath}", font_size=20, color=VEC_COLOR).next_to(
            x_vec.get_end(), UR, buff=0.1
        )

        # Target destination point Ax = [3.0, 3.5]^T
        target_dot = Dot(plane.c2p(*PUZZLE_TARGET), radius=0.10, color=TARGET_COLOR)
        target_ring = Circle(radius=0.25, color=TARGET_COLOR).move_to(target_dot.get_center())
        target_label = MathTex(r"T(\mathbf{x}) = ?", font_size=20, color=TARGET_COLOR).next_to(
            target_dot, UR, buff=0.15
        )

        ur_badge_text = MathTex(
            r"T(\mathbf{x}) = 1.5 T(\hat{\imath}) + 1.0 T(\hat{\jmath})",
            font_size=18,
            color=TEXT_COLOR,
        )
        ur_panel = make_hud_panel(ur_badge_text).to_corner(UR, buff=0.3)

        self.play(FadeIn(title_panel), FadeIn(ur_panel), run_time=0.8)
        self.play(GrowArrow(i_hat), FadeIn(i_label), GrowArrow(j_hat), FadeIn(j_label), run_time=1.0)
        self.play(GrowArrow(x_vec), FadeIn(x_label), run_time=1.0)
        self.play(FadeIn(target_dot), Create(target_ring), FadeIn(target_label), run_time=0.8)
        self.wait(0.5)

        # Basis vectors transform to a1 = [2, 1]^T, a2 = [0, 2]^T
        a1_arrow = self.create_vector_arrow(plane, np.zeros(2), A1_COORDS, I_HAT_COLOR)
        a2_arrow = self.create_vector_arrow(plane, np.zeros(2), A2_COORDS, J_HAT_COLOR)
        a1_label = MathTex(r"\mathbf{a}_1 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}", font_size=18, color=I_HAT_COLOR).next_to(
            a1_arrow.get_end(), RIGHT, buff=0.15
        )
        a2_label = MathTex(r"\mathbf{a}_2 = \begin{bmatrix} 0 \\ 2 \end{bmatrix}", font_size=18, color=J_HAT_COLOR).next_to(
            a2_arrow.get_end(), UP, buff=0.15
        )

        matrix_badge = MathTex(
            r"A = [\mathbf{a}_1 \mid \mathbf{a}_2] = \begin{bmatrix} 2 & 0 \\ 1 & 2 \end{bmatrix}",
            font_size=18,
            color=TEXT_COLOR,
        )
        matrix_panel = make_hud_panel(matrix_badge).to_corner(UR, buff=0.3)

        self.play(
            Transform(i_hat, a1_arrow),
            Transform(i_label, a1_label),
            Transform(j_hat, a2_arrow),
            Transform(j_label, a2_label),
            Transform(ur_panel, matrix_panel),
            run_time=1.4,
        )

        # Show linear combination 1.5 a1 + 1.0 a2 reaching target
        scaled_a1 = self.create_vector_arrow(plane, np.zeros(2), 1.5 * A1_COORDS, I_HAT_COLOR, stroke_width=3.5)
        shifted_a2 = self.create_vector_arrow(plane, 1.5 * A1_COORDS, PUZZLE_TARGET, J_HAT_COLOR, stroke_width=3.5)
        comb_vec = self.create_vector_arrow(plane, np.zeros(2), PUZZLE_TARGET, VEC_COLOR, stroke_width=4.5)

        solved_badge = MathTex(
            r"A\mathbf{x} = 1.5\begin{bmatrix} 2 \\ 1 \end{bmatrix} + 1.0\begin{bmatrix} 0 \\ 2 \end{bmatrix} = \begin{bmatrix} 3.0 \\ 3.5 \end{bmatrix}",
            font_size=17,
            color=TEXT_COLOR,
        )
        solved_panel = make_hud_panel(solved_badge).to_corner(DL, buff=0.3)

        solved_target_label = MathTex(r"A\mathbf{x} = [3.0, 3.5]^\top", font_size=18, color=TARGET_COLOR).next_to(
            target_dot, UR, buff=0.15
        )

        self.play(
            Create(scaled_a1),
            Create(shifted_a2),
            Transform(x_vec, comb_vec),
            Transform(target_label, solved_target_label),
            FadeIn(solved_panel),
            run_time=1.5,
        )
        self.play(Indicate(target_dot, color=VEC_COLOR), Flash(target_dot, color=TARGET_COLOR), run_time=1.0)
        self.wait(1.0)

        # Cleanup ephemeral elements
        ephemeral = [
            title_panel, ur_panel, matrix_panel, solved_panel,
            scaled_a1, shifted_a2, x_vec, x_label,
            target_dot, target_ring, target_label, solved_target_label,
            i_label, j_label,
        ]
        if is_standalone:
            self.play(*[FadeOut(m) for m in ephemeral if m in self.mobjects], run_time=1.0)
            self.wait(0.5)
            return i_hat, j_hat
        else:
            self.play(*[FadeOut(m) for m in ephemeral if m in self.mobjects], run_time=0.8)
            return i_hat, j_hat

    # -------------------------------------------------------------------------
    # ACT 2: Dynamic Deformation & Ghost Space
    # -------------------------------------------------------------------------
    def run_act2(
        self,
        plane: NumberPlane,
        origin_group: VGroup,
        v1: Arrow,
        v2: Arrow,
        is_standalone: bool = False,
    ) -> tuple[NumberPlane, Arrow, Arrow, Polygon]:
        """Act 2: Continuous matrix interpolation M(t) = (1-t)I + tA with ghost grid."""
        title = Text("Act 2: Dynamic Deformation & Ghost Grid Invariance", font_size=20, color=TEXT_COLOR)
        title_panel = make_hud_panel(title).to_corner(UL, buff=0.3)

        # Create static ghost plane behind deforming grid
        ghost_plane = self.create_ghost_plane()
        ghost_plane.set_z_index(-2)

        # Reset basis arrows to standard frame for continuous deformation
        i_unit = np.array([1.0, 0.0])
        j_unit = np.array([0.0, 1.0])
        v1_reset = self.create_vector_arrow(plane, np.zeros(2), i_unit, I_HAT_COLOR)
        v2_reset = self.create_vector_arrow(plane, np.zeros(2), j_unit, J_HAT_COLOR)

        self.play(
            FadeIn(title_panel),
            FadeIn(ghost_plane),
            Transform(v1, v1_reset),
            Transform(v2, v2_reset),
            run_time=1.0,
        )

        # ValueTracker for homotopy parameter t in [0, 1]
        t_tracker = ValueTracker(0.0)

        # Dynamic unit square / parallelogram polygon referencing plane's deformed unit coordinates
        def get_current_poly():
            p0 = plane.c2p(0, 0)
            p1 = plane.c2p(1, 0)
            p2 = plane.c2p(1, 1)
            p3 = plane.c2p(0, 1)
            poly = Polygon(p0, p1, p2, p3, color=SUBSPACE_COLOR, fill_color=SUBSPACE_COLOR, fill_opacity=0.30, stroke_width=2.0)
            poly.set_z_index(1)
            return poly

        unit_poly = always_redraw(get_current_poly)

        # Dynamic arrow updaters referencing plane's deformed unit coordinates
        def get_current_v1():
            arr = self.create_vector_arrow(plane, np.zeros(2), np.array([1.0, 0.0]), I_HAT_COLOR, stroke_width=4.0)
            arr.set_z_index(3)
            return arr

        def get_current_v2():
            arr = self.create_vector_arrow(plane, np.zeros(2), np.array([0.0, 1.0]), J_HAT_COLOR, stroke_width=4.0)
            arr.set_z_index(3)
            return arr

        dyn_v1 = always_redraw(get_current_v1)
        dyn_v2 = always_redraw(get_current_v2)

        # Dynamic readouts: Area = det(M(t)) = (1+t)^2
        area_num = DecimalNumber(1.0, num_decimal_places=2, font_size=20, color=SUBSPACE_COLOR)
        area_num.add_updater(lambda d: d.set_value((1.0 + t_tracker.get_value()) ** 2))
        area_label = MathTex(r"\text{Area } |\det(M(t))| = ", font_size=18, color=TEXT_COLOR)
        area_group = VGroup(area_label, area_num).arrange(RIGHT, buff=0.1)
        area_panel = make_hud_panel(area_group).to_corner(DL, buff=0.3)

        matrix_label = MathTex(
            r"M(t) = (1-t)I + tA = \begin{bmatrix} 1+t & 0 \\ t & 1+t \end{bmatrix}",
            font_size=18,
            color=TEXT_COLOR,
        )
        matrix_panel = make_hud_panel(matrix_label).to_corner(UR, buff=0.3)

        self.add(unit_poly, dyn_v1, dyn_v2)
        self.remove(v1, v2)
        self.play(FadeIn(area_panel), FadeIn(matrix_panel), run_time=0.8)

        # Smooth continuous space deformation
        # We deform both the number plane and sweep t_tracker simultaneously
        self.play(
            t_tracker.animate.set_value(1.0),
            plane.animate.apply_matrix(A_MAT),
            run_time=3.0,
            rate_func=linear,
        )
        self.wait(1.0)

        # Teardown dynamic updaters cleanly (Rule 4)
        area_num.clear_updaters()
        unit_poly.clear_updaters()
        dyn_v1.clear_updaters()
        dyn_v2.clear_updaters()

        # Create persistent static objects for Act 3
        final_poly = get_current_poly()
        final_v1 = get_current_v1()
        final_v2 = get_current_v2()
        self.add(final_poly, final_v1, final_v2)
        self.remove(unit_poly, dyn_v1, dyn_v2)

        invariance_note = Text(
            "Parallel & Evenly Spaced Grid Lines Preserved — det(A) = 4.0",
            font_size=17,
            color=TEXT_COLOR,
        )
        note_panel = make_hud_panel(invariance_note).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(note_panel), run_time=0.8)
        self.wait(1.5)

        ephemeral = [title_panel, area_panel, matrix_panel, note_panel, ghost_plane]
        if is_standalone:
            self.play(*[FadeOut(m) for m in ephemeral if m in self.mobjects], run_time=1.0)
            self.wait(0.5)
            return plane, final_v1, final_v2, final_poly
        else:
            self.play(*[FadeOut(m) for m in ephemeral if m in self.mobjects], run_time=0.8)
            return plane, final_v1, final_v2, final_poly

    # -------------------------------------------------------------------------
    # ACT 3: Color-Coded Algebraic Morph
    # -------------------------------------------------------------------------
    def run_act3(
        self,
        plane: NumberPlane,
        origin_group: VGroup,
        v1: Arrow,
        v2: Arrow,
        unit_poly: Polygon,
        is_standalone: bool = False,
    ) -> tuple[NumberPlane, Arrow, Arrow]:
        """Act 3: Sequential composition BA (A applied first, followed by B)."""
        title = Text("Act 3: Matrix Multiplication as Sequential Composition", font_size=20, color=TEXT_COLOR)
        title_panel = make_hud_panel(title).to_corner(UL, buff=0.3)

        # B is 90° counter-clockwise rotation matrix R
        b_label = MathTex(
            r"(B \circ A)(\mathbf{x}) = B(A\mathbf{x}) = (BA)\mathbf{x}",
            font_size=19,
            color=TEXT_COLOR,
        )
        b_panel = make_hud_panel(b_label).to_corner(UR, buff=0.3)

        det_rule = MathTex(
            r"\det(BA) = \det(B)\det(A) = 1 \times 4 = 4",
            font_size=18,
            color=TEXT_COLOR,
        )
        det_panel = make_hud_panel(det_rule).to_corner(DL, buff=0.3)

        self.play(FadeIn(title_panel), FadeIn(b_panel), FadeIn(det_panel), run_time=0.8)
        self.wait(1.0)

        # Apply transformation B (90° rotation) on top of deformed plane
        # BA columns: B @ a1 = [-1, 2]^T, B @ a2 = [-2, 0]^T
        ba_a1 = R_MAT @ A1_COORDS  # [-1.0, 2.0]
        ba_a2 = R_MAT @ A2_COORDS  # [-2.0, 0.0]

        composed_badge = MathTex(
            r"BA = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 1 & 2 \end{bmatrix} = \begin{bmatrix} -1 & -2 \\ 2 & 0 \end{bmatrix}",
            font_size=17,
            color=TEXT_COLOR,
        )
        composed_panel = make_hud_panel(composed_badge).to_corner(UR, buff=0.3)

        self.play(
            plane.animate.apply_matrix(R_MAT),
            unit_poly.animate.apply_matrix(R_MAT),
            v1.animate.apply_matrix(R_MAT),
            v2.animate.apply_matrix(R_MAT),
            Transform(b_panel, composed_panel),
            run_time=2.0,
        )
        self.wait(1.5)

        ephemeral = [title_panel, b_panel, composed_panel, det_panel, unit_poly]
        if is_standalone:
            self.play(*[FadeOut(m) for m in ephemeral if m in self.mobjects], run_time=1.0)
            self.wait(0.5)
            return plane, v1, v2
        else:
            self.play(*[FadeOut(m) for m in ephemeral if m in self.mobjects], run_time=0.8)
            return plane, v1, v2

    # -------------------------------------------------------------------------
    # ACT 4: Degenerate Stress Test (Non-Commutativity BA != AB)
    # -------------------------------------------------------------------------
    def run_act4(
        self,
        plane: NumberPlane,
        origin_group: VGroup,
        is_standalone: bool = False,
    ):
        """Act 4: Geometrically proving non-commutativity SR != RS via rotation and shear."""
        title = Text("Act 4: Non-Commutativity Stress Test", font_size=18, color=TEXT_COLOR)
        title_panel = make_hud_panel(title).to_corner(UL, buff=0.3)

        # Fresh standard plane for clean non-commutativity demonstration
        fresh_plane = self.create_plane()
        fresh_origin = self.create_origin_anchor(fresh_plane)

        self.play(
            FadeIn(title_panel),
            ReplacementTransform(plane, fresh_plane),
            ReplacementTransform(origin_group, fresh_origin),
            run_time=1.0,
        )

        # Test point x = [2, 3]^T
        probe_dot = Dot(fresh_plane.c2p(*STRESS_X), radius=0.10, color=VEC_COLOR)
        probe_label = MathTex(r"\mathbf{x} = [2, 3]^\top", font_size=18, color=VEC_COLOR).next_to(
            probe_dot, UR, buff=0.1
        )
        probe_vec = self.create_vector_arrow(fresh_plane, np.zeros(2), STRESS_X, VEC_COLOR)

        self.play(GrowArrow(probe_vec), FadeIn(probe_dot), FadeIn(probe_label), run_time=1.0)
        self.wait(0.5)

        # ---------------------------------------------------------------------
        # Order 1: SR (Rotation first, Shear second)
        # ---------------------------------------------------------------------
        # (SR)x = [-1, 2]^T
        dest_sr_dot = Dot(fresh_plane.c2p(*DEST_SR), radius=0.11, color=TARGET_COLOR)
        dest_sr_label = MathTex(r"(SR)\mathbf{x} = [-1, 2]^\top", font_size=18, color=TARGET_COLOR).next_to(
            dest_sr_dot, LEFT, buff=0.15
        )
        dest_sr_arrow = self.create_vector_arrow(fresh_plane, np.zeros(2), DEST_SR, TARGET_COLOR, stroke_width=3.5)

        order1_badge = MathTex(
            r"\text{Order 1 } (SR): \text{Rotate } 90^\circ \to \text{Shear}",
            font_size=17,
            color=TARGET_COLOR,
        )
        order1_panel = make_hud_panel(order1_badge).to_corner(UR, buff=0.3)

        self.play(
            FadeIn(order1_panel),
            FadeOut(title_panel),
            Transform(probe_vec, dest_sr_arrow),
            Transform(probe_dot, dest_sr_dot),
            Transform(probe_label, dest_sr_label),
            run_time=1.4,
        )
        self.play(Flash(dest_sr_dot, color=TARGET_COLOR), run_time=0.6)
        self.wait(0.8)

        # Save Order 1 as permanent beacon
        beacon_sr = VGroup(dest_sr_arrow.copy(), dest_sr_dot.copy(), dest_sr_label.copy())

        # ---------------------------------------------------------------------
        # Order 2: RS (Shear first, Rotation second)
        # ---------------------------------------------------------------------
        # (RS)x = [-3, 5]^T
        dest_rs_dot = Dot(fresh_plane.c2p(*DEST_RS), radius=0.11, color=ERROR_COLOR)
        dest_rs_label = MathTex(r"(RS)\mathbf{x} = [-3, 5]^\top", font_size=18, color=ERROR_COLOR).next_to(
            dest_rs_dot, DL, buff=0.15
        )
        dest_rs_arrow = self.create_vector_arrow(fresh_plane, np.zeros(2), DEST_RS, ERROR_COLOR, stroke_width=3.5)

        order2_badge = MathTex(
            r"\text{Order 2 } (RS): \text{Shear } \to \text{Rotate } 90^\circ",
            font_size=17,
            color=ERROR_COLOR,
        )
        order2_panel = make_hud_panel(order2_badge).next_to(order1_panel, DOWN, buff=0.2)

        self.add(beacon_sr)
        self.play(
            FadeIn(order2_panel),
            Create(dest_rs_arrow),
            FadeIn(dest_rs_dot),
            FadeIn(dest_rs_label),
            run_time=1.4,
        )
        self.play(Flash(dest_rs_dot, color=ERROR_COLOR), run_time=0.6)
        self.wait(0.8)

        # ---------------------------------------------------------------------
        # Spatial Divergence Vector connecting [-1, 2]^T and [-3, 5]^T
        # ---------------------------------------------------------------------
        divergence_line = DashedLine(
            fresh_plane.c2p(*DEST_SR),
            fresh_plane.c2p(*DEST_RS),
            color=ERROR_COLOR,
            stroke_width=3.0,
            dash_length=0.1,
        )
        dist_badge = MathTex(
            r"\|\Delta\| = \sqrt{13} \approx 3.61 \ne 0",
            font_size=18,
            color=ERROR_COLOR,
        )
        dist_panel = make_hud_panel(dist_badge).to_corner(DL, buff=0.3)

        commutator_badge = MathTex(
            r"[S, R] = SR - RS = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \ne \mathbf{0}",
            font_size=17,
            color=TEXT_COLOR,
        )
        commutator_panel = make_hud_panel(commutator_badge).next_to(order2_panel, DOWN, buff=0.2)

        self.play(
            Create(divergence_line),
            FadeIn(dist_panel),
            FadeIn(commutator_panel),
            run_time=1.2,
        )
        self.play(Indicate(divergence_line, color=WHITE), run_time=0.8)
        self.wait(1.2)

        # Summary Master Card
        summary_title = Text("Geometric Matrix Non-Commutativity Invariant", font_size=20, color=TEXT_COLOR)
        summary_body = MathTex(
            r"SR \neq RS \quad \Longleftrightarrow \quad \text{Sequential Spatial Warping is Order-Sensitive}",
            font_size=18,
            color=TEXT_COLOR,
        )
        summary_invariants = MathTex(
            r"\text{Yet Area Commutes: } \det(SR) = \det(RS) = 1.0 \quad \text{and} \quad \text{Tr}([S, R]) = 0",
            font_size=17,
            color=MUTED_TEXT,
        )
        summary_content = VGroup(summary_title, summary_body, summary_invariants).arrange(DOWN, buff=0.2)
        summary_panel = make_hud_panel(summary_content, padding=0.3)
        summary_panel.set_z_index(20)

        self.play(FadeIn(summary_panel, scale=1.1), run_time=1.5)
        self.wait(2.5)


# -----------------------------------------------------------------------------
# Modular Executable Scene Classes
# -----------------------------------------------------------------------------
class Act1SpatialPuzzle(BaseLinearMappingsScene):
    """Standalone executable scene for Act 1."""

    def construct(self):
        plane = self.create_plane()
        origin_group = self.create_origin_anchor(plane)
        self.add(plane, origin_group)
        self.run_act1(plane, origin_group, is_standalone=True)


class Act2DynamicDeformation(BaseLinearMappingsScene):
    """Standalone executable scene for Act 2."""

    def construct(self):
        plane = self.create_plane()
        origin_group = self.create_origin_anchor(plane)
        v1 = self.create_vector_arrow(plane, np.zeros(2), np.array([1.0, 0.0]), I_HAT_COLOR)
        v2 = self.create_vector_arrow(plane, np.zeros(2), np.array([0.0, 1.0]), J_HAT_COLOR)
        self.add(plane, origin_group, v1, v2)
        self.run_act2(plane, origin_group, v1, v2, is_standalone=True)


class Act3AlgebraicMorph(BaseLinearMappingsScene):
    """Standalone executable scene for Act 3."""

    def construct(self):
        plane = self.create_plane()
        origin_group = self.create_origin_anchor(plane)
        # Initialize in deformed state A
        plane.apply_matrix(A_MAT)
        v1 = self.create_vector_arrow(plane, np.zeros(2), np.array([1.0, 0.0]), I_HAT_COLOR)
        v2 = self.create_vector_arrow(plane, np.zeros(2), np.array([0.0, 1.0]), J_HAT_COLOR)
        p0 = plane.c2p(0, 0)
        p1 = plane.c2p(1, 0)
        p2 = plane.c2p(1, 1)
        p3 = plane.c2p(0, 1)
        unit_poly = Polygon(p0, p1, p2, p3, color=SUBSPACE_COLOR, fill_color=SUBSPACE_COLOR, fill_opacity=0.30, stroke_width=2.0)
        self.add(plane, origin_group, unit_poly, v1, v2)
        self.run_act3(plane, origin_group, v1, v2, unit_poly, is_standalone=True)


class Act4NonCommutativeStressTest(BaseLinearMappingsScene):
    """Standalone executable scene for Act 4."""

    def construct(self):
        plane = self.create_plane()
        origin_group = self.create_origin_anchor(plane)
        self.add(plane, origin_group)
        self.run_act4(plane, origin_group, is_standalone=True)


class LinearMappingsScene(BaseLinearMappingsScene):
    """Master unified scene seamlessly orchestrating all 4 acts."""

    def construct(self):
        plane = self.create_plane()
        origin_group = self.create_origin_anchor(plane)
        self.add(plane, origin_group)

        # Act 1: Concrete Spatial Puzzle
        v1, v2 = self.run_act1(plane, origin_group, is_standalone=False)

        # Act 2: Dynamic Deformation & Ghost Space
        plane, v1, v2, unit_poly = self.run_act2(plane, origin_group, v1, v2, is_standalone=False)

        # Act 3: Color-Coded Algebraic Morph
        plane, v1, v2 = self.run_act3(plane, origin_group, v1, v2, unit_poly, is_standalone=False)

        # Act 4: Degenerate Stress Test
        self.run_act4(plane, origin_group, is_standalone=False)
