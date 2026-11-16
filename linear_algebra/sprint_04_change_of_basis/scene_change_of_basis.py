"""
Sprint 4: Change of Basis & Similarity Transformations P^-1 A P (ManimCE v0.18+)
Curriculum: Linear Algebra & Theoretical CS / Machine Learning Foundations
References: MML Ch. 2.7, VMLS Ch. 6 & 10, Kreyszig Ch. 8, ILA Ch. 6

This production script implements the 4-act pedagogical narrative:
- Act 1: Concrete Spatial Puzzle (Coupled Shear/Stretch vs Decoupled Scaling)
- Act 2: Dynamic Deformation & Ghost Space (The Invariant Eigen-Grid Scaffolding)
- Act 3: Color-Coded Algebraic Morph (The 3-Step Similarity Dance P^-1 -> D -> P)
- Act 4: Degenerate Stress Test (Defective Matrix Non-Diagonalizability & Jordan Form)

Render Commands:
    Preview Low-Res:   manim -ql linear_algebra/sprint_04_change_of_basis/scene_change_of_basis.py ChangeOfBasisScene
    Production Hi-Res: manim -qh linear_algebra/sprint_04_change_of_basis/scene_change_of_basis.py ChangeOfBasisScene
    Single Act Test:   manim -ql linear_algebra/sprint_04_change_of_basis/scene_change_of_basis.py Act1SpatialPuzzle
"""

from __future__ import annotations

import math
import numpy as np
from manim import *

# -----------------------------------------------------------------------------
# Rule 1: 3Blue1Brown Semantic Color Palette
# -----------------------------------------------------------------------------
I_HAT_COLOR = "#83C167"       # Green C    - Basis vector i_hat / Eigenvector 1 (v1)
J_HAT_COLOR = "#FC6255"       # Red C      - Basis vector j_hat / Eigenvector 2 (v2)
VEC_COLOR = "#F7D96F"         # Yellow     - Arbitrary test vector x
TARGET_COLOR = "#F9E498"      # Gold A     - Transformed destination point Ax
SUBSPACE_COLOR = "#2A9D8F"    # Teal       - Eigen-subspace scaffolding
EIGEN_GRID_COLOR = "#E0AF68"  # Amber      - Oblique eigen-grid coordinate lines
GHOST_COLOR = "#414868"       # Slate Gray - Ghost standard Cartesian grid lines
ERROR_COLOR = "#F7768E"       # Soft Red   - Defective matrix shear & discrepancy
BG_COLOR = "#1C1C24"          # Dark Slate - High-contrast dark viewport background
PANEL_BG = "#16161E"          # Obsidian   - HUD panel container background fill
PANEL_BORDER = "#414868"      # Slate Teal - HUD panel container outline stroke
TEXT_COLOR = "#ECEFF4"        # Snow       - High-contrast primary typography
MUTED_TEXT = "#7AA2F7"        # Light Blue - Secondary badges & annotations

# -----------------------------------------------------------------------------
# Mathematical Constants & Assertions (Rule 6: Exact NumPy Calculations)
# Required AST names: V1, V2, EIGEN_V1, EIGEN_V2, P_MATRIX, A_MATRIX
# -----------------------------------------------------------------------------
V1 = np.array([1.0, 1.0])
V2 = np.array([-1.0, 1.0])
EIGEN_V1 = V1
EIGEN_V2 = V2
A_MATRIX = np.array([[2.0, 1.0], [1.0, 2.0]])
P_MATRIX = np.column_stack([V1, V2])  # [[1, -1], [1, 1]]
D_MATRIX = np.diag([3.0, 1.0])
P_INV_MATRIX = 0.5 * np.array([[1.0, 1.0], [-1.0, 1.0]])

# Alternate standard references
A_MAT = A_MATRIX
P_MAT = P_MATRIX
D_MAT = D_MATRIX
P_INV_MAT = P_INV_MATRIX
M_DEF = np.array([[2.0, 1.0], [0.0, 2.0]])

TEST_X = np.array([2.0, 0.0])
TARGET_AX = A_MAT @ TEST_X  # [4.0, 2.0]

# Pre-flight mathematical consistency verification
assert np.allclose(A_MAT @ V1, 3.0 * V1), "A @ v1 must equal 3*v1"
assert np.allclose(A_MAT @ V2, 1.0 * V2), "A @ v2 must equal 1*v2"
assert np.allclose(P_MAT @ D_MAT @ P_INV_MAT, A_MAT), "P @ D @ P^-1 must equal A"
assert np.isclose(np.linalg.det(A_MAT), 3.0), "det(A) must be 3.0"
assert np.isclose(np.trace(A_MAT), 4.0), "Tr(A) must be 4.0"
assert np.isclose(np.linalg.det(P_MAT), 2.0), "det(P) must be 2.0"
assert np.isclose(np.dot(V1, V2), 0.0), "Eigenvectors must be orthogonal"
assert np.allclose(TARGET_AX, np.array([4.0, 2.0])), "Target must be [4.0, 2.0]"
assert np.allclose(P_INV_MAT @ TEST_X, np.array([1.0, -1.0])), "P^-1 @ x must be [1, -1]"
assert np.allclose(D_MAT @ np.array([1.0, -1.0]), np.array([3.0, -1.0])), "D @ coords must be [3, -1]"
assert np.allclose(P_MAT @ np.array([3.0, -1.0]), TARGET_AX), "P @ coords must be [4, 2]"
assert np.linalg.matrix_rank(M_DEF - 2.0 * np.eye(2)) == 1, "M_def nullity must be 1"


# -----------------------------------------------------------------------------
# Rule 3: Zero Text Overlap HUD Panel Factory
# -----------------------------------------------------------------------------
def make_hud_panel(
    content: Mobject,
    padding: float = 0.18,
    bg_color: str = PANEL_BG,
    bg_opacity: float = 0.88,
    stroke_color: str = PANEL_BORDER,
    stroke_width: float = 1.5,
    corner_radius: float = 0.12,
) -> VGroup:
    """Wraps any mobject inside a high-contrast rounded rectangular panel with z-index 10."""
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
# Base Scene Class (Shared Geometry, Viewport, and Modular Act Methods)
# -----------------------------------------------------------------------------
class BaseChangeOfBasisScene(Scene):
    """Base class providing shared infrastructure and modular act implementations."""

    def setup(self):
        super().setup()
        self.camera.background_color = BG_COLOR

    def create_plane(self) -> NumberPlane:
        """Rule 2: Coordinate plane clamped to safe viewport boundaries (scale 0.8)."""
        return NumberPlane(
            x_range=[-6, 6, 1],
            y_range=[-4, 4, 1],
            x_length=9.6,  # 0.8 scale protects viewport [-7.11, 7.11] x [-4.0, 4.0]
            y_length=6.4,
            background_line_style={
                "stroke_color": TEAL_E,
                "stroke_width": 1.2,
                "stroke_opacity": 0.45,
            },
            faded_line_style={
                "stroke_color": TEAL_E,
                "stroke_width": 0.6,
                "stroke_opacity": 0.20,
            },
        )

    def create_ghost_plane(self) -> NumberPlane:
        """Static ghost Cartesian plane for visual reference."""
        return NumberPlane(
            x_range=[-6, 6, 1],
            y_range=[-4, 4, 1],
            x_length=9.6,
            y_length=6.4,
            background_line_style={
                "stroke_color": GHOST_COLOR,
                "stroke_width": 1.0,
                "stroke_opacity": 0.30,
            },
            faded_line_style={
                "stroke_color": GHOST_COLOR,
                "stroke_width": 0.5,
                "stroke_opacity": 0.15,
            },
        )

    def create_origin_anchor(self, plane: NumberPlane) -> VGroup:
        """Rule 5: Immovable origin anchor with high-contrast label."""
        dot = Dot(plane.c2p(0, 0), radius=0.08, color=WHITE)
        label = MathTex(r"\mathbf{0}", font_size=18, color=WHITE).next_to(dot, DL, buff=0.1)
        group = VGroup(dot, label)
        group.set_z_index(8)
        return group

    def create_vector_arrow(
        self,
        plane: NumberPlane,
        start_coords: np.ndarray,
        end_coords: np.ndarray,
        color: str,
        stroke_width: float = 3.5,
    ) -> Arrow:
        """Helper to create visually consistent arrows in coordinate plane space."""
        return Arrow(
            plane.c2p(*start_coords),
            plane.c2p(*end_coords),
            buff=0,
            color=color,
            stroke_width=stroke_width,
            max_tip_length_to_length_ratio=0.25,
            max_stroke_width_to_length_ratio=5.0,
        )

    def create_eigen_grid(self, plane: NumberPlane, length: float = 5.0) -> VGroup:
        """Creates dashed eigen-grid coordinate scaffolding along lines y = x and y = -x."""
        lines = VGroup()
        # Main eigen-axes
        axis1 = DashedLine(
            plane.c2p(-length, -length),
            plane.c2p(length, length),
            color=I_HAT_COLOR,
            stroke_width=1.5,
            dash_length=0.15,
        )
        axis2 = DashedLine(
            plane.c2p(length, -length),
            plane.c2p(-length, length),
            color=J_HAT_COLOR,
            stroke_width=1.5,
            dash_length=0.15,
        )
        lines.add(axis1, axis2)
        # Parallel scaffolding grid lines
        for offset in [-3.0, -1.5, 1.5, 3.0]:
            p1_start = np.array([-length, -length]) + offset * np.array([-0.5, 0.5])
            p1_end = np.array([length, length]) + offset * np.array([-0.5, 0.5])
            l1 = DashedLine(
                plane.c2p(*p1_start),
                plane.c2p(*p1_end),
                color=EIGEN_GRID_COLOR,
                stroke_width=0.8,
                stroke_opacity=0.35,
                dash_length=0.12,
            )
            p2_start = np.array([length, -length]) + offset * np.array([0.5, 0.5])
            p2_end = np.array([-length, length]) + offset * np.array([0.5, 0.5])
            l2 = DashedLine(
                plane.c2p(*p2_start),
                plane.c2p(*p2_end),
                color=EIGEN_GRID_COLOR,
                stroke_width=0.8,
                stroke_opacity=0.35,
                dash_length=0.12,
            )
            lines.add(l1, l2)
        lines.set_z_index(3)
        return lines

    # -------------------------------------------------------------------------
    # Act 1: Concrete Spatial Puzzle (Coupled vs Decoupled)
    # -------------------------------------------------------------------------
    def run_act1(self, plane: NumberPlane, origin_group: VGroup, is_standalone: bool = False):
        """Act 1: Demonstrates coupled shear/stretch in standard basis vs decoupled desire."""
        # HUD Panels
        title_text = Text("Act 1: Spatial Puzzle — Coupled Axes", font_size=20, color=TEXT_COLOR)
        title_panel = make_hud_panel(title_text).to_corner(UL, buff=0.3)

        basis_math = MathTex(
            r"\text{Standard Basis: } \hat{\imath} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \; \hat{\jmath} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}",
            font_size=18,
            color=TEXT_COLOR,
        )
        basis_panel = make_hud_panel(basis_math).to_corner(UR, buff=0.3)

        self.play(FadeIn(title_panel), FadeIn(basis_panel), run_time=1.0)

        # Standard basis arrows
        i_vec = self.create_vector_arrow(plane, np.zeros(2), np.array([1.0, 0.0]), I_HAT_COLOR)
        j_vec = self.create_vector_arrow(plane, np.zeros(2), np.array([0.0, 1.0]), J_HAT_COLOR)
        i_label = MathTex(r"\hat{\imath}", font_size=18, color=I_HAT_COLOR).next_to(i_vec.get_end(), DR, buff=0.1)
        j_label = MathTex(r"\hat{\jmath}", font_size=18, color=J_HAT_COLOR).next_to(j_vec.get_end(), UL, buff=0.1)

        self.play(
            GrowArrow(i_vec),
            GrowArrow(j_vec),
            FadeIn(i_label),
            FadeIn(j_label),
            run_time=1.2,
        )
        self.wait(0.5)

        # Arbitrary test vector x = [2, 0]
        x_vec = self.create_vector_arrow(plane, np.zeros(2), TEST_X, VEC_COLOR, stroke_width=4.0)
        x_label = MathTex(r"\mathbf{x} = \begin{bmatrix} 2 \\ 0 \end{bmatrix}", font_size=18, color=VEC_COLOR).next_to(
            x_vec.get_end(), UR, buff=0.1
        )
        target_dot = Dot(plane.c2p(*TARGET_AX), radius=0.09, color=TARGET_COLOR)
        target_ring = Circle(radius=0.22, color=TARGET_COLOR, stroke_width=1.8).move_to(plane.c2p(*TARGET_AX))
        target_beacon = VGroup(target_dot, target_ring)
        target_label = MathTex(
            r"A\mathbf{x} = \begin{bmatrix} 4 \\ 2 \end{bmatrix}",
            font_size=18,
            color=TARGET_COLOR,
        ).next_to(target_beacon, UR, buff=0.15)

        self.play(GrowArrow(x_vec), FadeIn(x_label), run_time=1.0)
        self.play(FadeIn(target_beacon), FadeIn(target_label), run_time=0.8)
        self.wait(0.6)

        # Direct transformation under matrix A
        matrix_math = MathTex(
            r"A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \implies \begin{aligned} A\hat{\imath} &= [2, 1]^\top \\ A\hat{\jmath} &= [1, 2]^\top \end{aligned}",
            font_size=17,
            color=TEXT_COLOR,
        )
        matrix_panel = make_hud_panel(matrix_math).to_corner(UR, buff=0.3)

        a1_target = A_MAT @ np.array([1.0, 0.0])
        a2_target = A_MAT @ np.array([0.0, 1.0])

        new_i_vec = self.create_vector_arrow(plane, np.zeros(2), a1_target, I_HAT_COLOR)
        new_j_vec = self.create_vector_arrow(plane, np.zeros(2), a2_target, J_HAT_COLOR)
        new_x_vec = self.create_vector_arrow(plane, np.zeros(2), TARGET_AX, VEC_COLOR, stroke_width=4.0)

        coupled_text = Text("Coupled Warping: Both Stretch & Rotation", font_size=17, color=ERROR_COLOR)
        coupled_panel = make_hud_panel(coupled_text).to_corner(DL, buff=0.3)

        invariants_math = MathTex(
            r"\text{Tr}(A) = 4, \quad \det(A) = 3",
            font_size=17,
            color=MUTED_TEXT,
        )
        invariants_panel = make_hud_panel(invariants_math).next_to(coupled_panel, UP, buff=0.2)

        self.play(
            ReplacementTransform(basis_panel, matrix_panel),
            Transform(i_vec, new_i_vec),
            Transform(j_vec, new_j_vec),
            Transform(x_vec, new_x_vec),
            i_label.animate.next_to(new_i_vec.get_end(), DR, buff=0.1),
            j_label.animate.next_to(new_j_vec.get_end(), UL, buff=0.1),
            x_label.animate.next_to(new_x_vec.get_end(), UR, buff=0.15),
            FadeIn(coupled_panel),
            FadeIn(invariants_panel),
            run_time=2.0,
            rate_func=smooth,
        )
        self.play(Indicate(target_beacon, color=WHITE), run_time=0.8)
        self.wait(1.0)

        # Pulse question: Are there axes where rotation is ZERO?
        puzzle_question = Text("Can we find axes with ZERO rotation?", font_size=17, color=TEXT_COLOR)
        puzzle_panel = make_hud_panel(puzzle_question).to_corner(DL, buff=0.3)
        self.play(ReplacementTransform(coupled_panel, puzzle_panel), run_time=1.0)
        self.wait(1.5)

        if is_standalone:
            self.play(
                FadeOut(title_panel),
                FadeOut(matrix_panel),
                FadeOut(puzzle_panel),
                FadeOut(invariants_panel),
                FadeOut(i_vec),
                FadeOut(j_vec),
                FadeOut(x_vec),
                FadeOut(i_label),
                FadeOut(j_label),
                FadeOut(x_label),
                FadeOut(target_beacon),
                FadeOut(target_label),
                run_time=1.0,
            )
            return

        # Return active mobjects for continuity
        cleanup_group = VGroup(
            title_panel, matrix_panel, puzzle_panel, invariants_panel,
            i_vec, j_vec, x_vec, i_label, j_label, x_label, target_beacon, target_label
        )
        return cleanup_group

    # -------------------------------------------------------------------------
    # Act 2: Dynamic Deformation & Ghost Space (The Invariant Eigen-Grid)
    # -------------------------------------------------------------------------
    def run_act2(self, plane: NumberPlane, origin_group: VGroup, is_standalone: bool = False):
        """Act 2: Reveals the invariant eigen-grid and animates pure elongation without rotation."""
        title_text = Text("Act 2: Invariant Eigen-Grid & Zero Rotation", font_size=20, color=TEXT_COLOR)
        title_panel = make_hud_panel(title_text).to_corner(UL, buff=0.3)

        eigen_math = MathTex(
            r"P = [\mathbf{v}_1 \mid \mathbf{v}_2] = \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}, \quad \begin{aligned} A\mathbf{v}_1 &= 3\mathbf{v}_1 \\ A\mathbf{v}_2 &= 1\mathbf{v}_2 \end{aligned}",
            font_size=17,
            color=TEXT_COLOR,
        )
        eigen_panel = make_hud_panel(eigen_math).to_corner(UR, buff=0.3)

        self.play(FadeIn(title_panel), FadeIn(eigen_panel), run_time=1.0)

        # Create eigen-grid scaffolding
        eigen_grid = self.create_eigen_grid(plane)
        self.play(Create(eigen_grid), run_time=1.5)

        # Eigenvectors v1 = [1, 1], v2 = [-1, 1]
        v1_arrow = self.create_vector_arrow(plane, np.zeros(2), V1, I_HAT_COLOR, stroke_width=4.0)
        v2_arrow = self.create_vector_arrow(plane, np.zeros(2), V2, J_HAT_COLOR, stroke_width=4.0)
        v1_label = MathTex(r"\mathbf{v}_1 \; (\lambda_1 = 3)", font_size=18, color=I_HAT_COLOR).next_to(
            v1_arrow.get_end(), UR, buff=0.15
        )
        v2_label = MathTex(r"\mathbf{v}_2 \; (\lambda_2 = 1)", font_size=18, color=J_HAT_COLOR).next_to(
            v2_arrow.get_end(), UL, buff=0.15
        )

        self.play(
            GrowArrow(v1_arrow),
            GrowArrow(v2_arrow),
            FadeIn(v1_label),
            FadeIn(v2_label),
            run_time=1.2,
        )

        # Right angle marker between orthogonal eigenvectors
        p_origin = plane.c2p(0, 0)
        p_v1 = plane.c2p(*(0.4 * V1))
        p_v2 = plane.c2p(*(0.4 * V2))
        p_corner = plane.c2p(*(0.4 * (V1 + V2)))
        right_angle = Polygon(
            p_origin, p_v1, p_corner, p_v2,
            color=SUBSPACE_COLOR,
            fill_color=SUBSPACE_COLOR,
            fill_opacity=0.35,
            stroke_width=1.5,
        )
        right_angle.set_z_index(5)
        self.play(FadeIn(right_angle), run_time=0.8)
        self.wait(0.5)

        # Continuous Homotopy M(t) = (1-t)I + tA
        t_tracker = ValueTracker(0.0)

        t_readout = MathTex(
            r"M(t) = (1-t)I + tA \quad (t = 0.00)",
            font_size=17,
            color=MUTED_TEXT,
        )
        t_panel = make_hud_panel(t_readout).to_corner(DL, buff=0.3)
        self.play(FadeIn(t_panel), run_time=0.8)

        # Dynamic arrows tracking homotopy
        def get_v1_tip():
            t = t_tracker.get_value()
            scale = 1.0 + 2.0 * t  # stretches 1 -> 3
            return plane.c2p(*(scale * V1))

        def get_v2_tip():
            t = t_tracker.get_value()
            scale = 1.0  # stays constant at 1
            return plane.c2p(*(scale * V2))

        dyn_v1 = always_redraw(
            lambda: Arrow(
                plane.c2p(0, 0),
                get_v1_tip(),
                buff=0,
                color=I_HAT_COLOR,
                stroke_width=4.0,
                max_tip_length_to_length_ratio=0.22,
            )
        )
        dyn_v2 = always_redraw(
            lambda: Arrow(
                plane.c2p(0, 0),
                get_v2_tip(),
                buff=0,
                color=J_HAT_COLOR,
                stroke_width=4.0,
                max_tip_length_to_length_ratio=0.22,
            )
        )

        self.remove(v1_arrow, v2_arrow)
        self.add(dyn_v1, dyn_v2)

        # Animate continuous sweep t: 0 -> 1
        self.play(
            t_tracker.animate.set_value(1.0),
            run_time=3.0,
            rate_func=smooth,
        )
        self.wait(0.8)

        # Update HUD panel to completed state
        decoupled_badge = Text(
            "Pure Invariant Dilation: Zero Rotation Along Eigen-Axes",
            font_size=16,
            color=TEXT_COLOR,
        )
        decoupled_panel = make_hud_panel(decoupled_badge).to_corner(DL, buff=0.3)
        self.play(ReplacementTransform(t_panel, decoupled_panel), run_time=1.0)
        self.wait(1.5)

        if is_standalone:
            self.play(
                FadeOut(title_panel),
                FadeOut(eigen_panel),
                FadeOut(decoupled_panel),
                FadeOut(eigen_grid),
                FadeOut(dyn_v1),
                FadeOut(dyn_v2),
                FadeOut(v1_label),
                FadeOut(v2_label),
                FadeOut(right_angle),
                run_time=1.0,
            )
            return

        cleanup_group = VGroup(
            title_panel, eigen_panel, decoupled_panel, eigen_grid,
            dyn_v1, dyn_v2, v1_label, v2_label, right_angle
        )
        return cleanup_group

    # -------------------------------------------------------------------------
    # Act 3: Color-Coded Algebraic Morph (The 3-Step Similarity Dance)
    # -------------------------------------------------------------------------
    def run_act3(self, plane: NumberPlane, origin_group: VGroup, is_standalone: bool = False):
        """Act 3: Step-by-step visual sequence P^-1 -> D -> P reproducing A = P D P^-1."""
        title_text = Text("Act 3: Similarity Transformation P^-1 A P", font_size=20, color=TEXT_COLOR)
        title_panel = make_hud_panel(title_text).to_corner(UL, buff=0.3)

        formula_math = MathTex(
            r"A = P D P^{-1} \iff P^{-1} A P = D",
            font_size=18,
            color=TEXT_COLOR,
        )
        formula_panel = make_hud_panel(formula_math).to_corner(UR, buff=0.3)

        self.play(FadeIn(title_panel), FadeIn(formula_panel), run_time=1.0)

        # Draw eigen-axes scaffolding
        axis1 = DashedLine(plane.c2p(-4, -4), plane.c2p(4, 4), color=I_HAT_COLOR, stroke_width=1.5)
        axis2 = DashedLine(plane.c2p(4, -4), plane.c2p(-4, 4), color=J_HAT_COLOR, stroke_width=1.5)
        self.play(Create(axis1), Create(axis2), run_time=1.0)

        # Initial test vector x = [2, 0]
        x_arrow = self.create_vector_arrow(plane, np.zeros(2), TEST_X, VEC_COLOR, stroke_width=4.0)
        x_label = MathTex(r"\mathbf{x} = \begin{bmatrix} 2 \\ 0 \end{bmatrix}", font_size=18, color=VEC_COLOR).next_to(
            x_arrow.get_end(), UR, buff=0.1
        )
        self.play(GrowArrow(x_arrow), FadeIn(x_label), run_time=1.0)
        self.wait(0.5)

        # Step 1: P^-1 standard-to-eigen change of basis
        step1_text = MathTex(
            r"\text{Step 1 } (P^{-1}): \quad [\mathbf{x}]_B = P^{-1}\mathbf{x} = \begin{bmatrix} 1 \\ -1 \end{bmatrix}_B",
            font_size=17,
            color=TEXT_COLOR,
        )
        step1_panel = make_hud_panel(step1_text).to_corner(DL, buff=0.3)

        # Projection lines onto eigen-axes
        proj1 = DashedLine(plane.c2p(*TEST_X), plane.c2p(*V1), color=I_HAT_COLOR, stroke_width=1.8)
        proj2 = DashedLine(plane.c2p(*TEST_X), plane.c2p(*(-V2)), color=J_HAT_COLOR, stroke_width=1.8)
        comp1 = self.create_vector_arrow(plane, np.zeros(2), V1, I_HAT_COLOR, stroke_width=3.0)
        comp2 = self.create_vector_arrow(plane, V1, TEST_X, J_HAT_COLOR, stroke_width=3.0)

        self.play(
            FadeIn(step1_panel),
            Create(proj1),
            Create(proj2),
            GrowArrow(comp1),
            GrowArrow(comp2),
            run_time=1.8,
        )
        self.wait(1.0)

        # Step 2: D decoupled diagonal scaling
        step2_text = MathTex(
            r"\text{Step 2 } (D): \quad D[\mathbf{x}]_B = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}\begin{bmatrix} 1 \\ -1 \end{bmatrix} = \begin{bmatrix} 3 \\ -1 \end{bmatrix}_B",
            font_size=17,
            color=TEXT_COLOR,
        )
        step2_panel = make_hud_panel(step2_text).to_corner(DL, buff=0.3)

        scaled_comp1 = self.create_vector_arrow(plane, np.zeros(2), 3.0 * V1, I_HAT_COLOR, stroke_width=3.0)
        scaled_comp2 = self.create_vector_arrow(plane, 3.0 * V1, 3.0 * V1 - V2, J_HAT_COLOR, stroke_width=3.0)

        self.play(
            ReplacementTransform(step1_panel, step2_panel),
            Transform(comp1, scaled_comp1),
            Transform(comp2, scaled_comp2),
            FadeOut(proj1),
            FadeOut(proj2),
            run_time=2.0,
            rate_func=smooth,
        )
        self.wait(1.0)

        # Step 3: P eigen-to-standard restoration
        step3_text = MathTex(
            r"\text{Step 3 } (P): \quad P \begin{bmatrix} 3 \\ -1 \end{bmatrix} = 3\mathbf{v}_1 - 1\mathbf{v}_2 = \begin{bmatrix} 4 \\ 2 \end{bmatrix} = A\mathbf{x}",
            font_size=17,
            color=TARGET_COLOR,
        )
        step3_panel = make_hud_panel(step3_text).to_corner(DL, buff=0.3)

        target_arrow = self.create_vector_arrow(plane, np.zeros(2), TARGET_AX, TARGET_COLOR, stroke_width=4.5)
        target_beacon = Dot(plane.c2p(*TARGET_AX), radius=0.09, color=TARGET_COLOR)

        self.play(
            ReplacementTransform(step2_panel, step3_panel),
            ReplacementTransform(x_arrow, target_arrow),
            FadeIn(target_beacon),
            run_time=1.8,
        )
        self.play(Indicate(target_arrow, color=WHITE), run_time=0.8)
        self.wait(1.0)

        # Master Invariants Badge
        invariants_math = MathTex(
            r"\text{Similarity Invariants: } \text{Tr}(P^{-1}AP) = \text{Tr}(A) = 4, \quad \det(P^{-1}AP) = \det(A) = 3",
            font_size=16,
            color=MUTED_TEXT,
        )
        invariants_panel = make_hud_panel(invariants_math).next_to(step3_panel, UP, buff=0.2, aligned_edge=LEFT)
        self.play(FadeIn(invariants_panel), run_time=1.0)
        self.wait(1.5)

        if is_standalone:
            self.play(
                FadeOut(title_panel),
                FadeOut(formula_panel),
                FadeOut(step3_panel),
                FadeOut(invariants_panel),
                FadeOut(axis1),
                FadeOut(axis2),
                FadeOut(comp1),
                FadeOut(comp2),
                FadeOut(target_arrow),
                FadeOut(target_beacon),
                FadeOut(x_label),
                run_time=1.0,
            )
            return

        cleanup_group = VGroup(
            title_panel, formula_panel, step3_panel, invariants_panel,
            axis1, axis2, comp1, comp2, target_arrow, target_beacon, x_label
        )
        return cleanup_group

    # -------------------------------------------------------------------------
    # Act 4: Degenerate Stress Test (Defective Matrix Collapse)
    # -------------------------------------------------------------------------
    def run_act4(self, plane: NumberPlane, origin_group: VGroup, is_standalone: bool = False):
        """Act 4: Shows defective shear matrix with geometric multiplicity 1 failing diagonalization."""
        title_text = Text("Act 4: Defective Matrix Stress Test", font_size=20, color=TEXT_COLOR)
        title_panel = make_hud_panel(title_text).to_corner(UL, buff=0.3)

        defective_math = MathTex(
            r"M_{\text{def}} = \begin{bmatrix} 2 & 1 \\ 0 & 2 \end{bmatrix}, \quad p(\lambda) = (\lambda - 2)^2 = 0",
            font_size=17,
            color=ERROR_COLOR,
        )
        defective_panel = make_hud_panel(defective_math).to_corner(UR, buff=0.3)

        self.play(FadeIn(title_panel), FadeIn(defective_panel), run_time=1.0)

        # Single eigenspace line along x-axis
        eigenspace_line = Line(
            plane.c2p(-5, 0),
            plane.c2p(5, 0),
            color=I_HAT_COLOR,
            stroke_width=3.5,
        )
        eigenspace_label = MathTex(
            r"\text{Eigenspace } E(2) = \text{span}([1, 0]^\top) \quad (\text{Dim} = 1)",
            font_size=17,
            color=I_HAT_COLOR,
        ).next_to(eigenspace_line, DOWN, buff=0.2)

        v_arrow = self.create_vector_arrow(plane, np.zeros(2), np.array([1.0, 0.0]), I_HAT_COLOR, stroke_width=4.0)

        self.play(
            Create(eigenspace_line),
            GrowArrow(v_arrow),
            FadeIn(eigenspace_label),
            run_time=1.5,
        )
        self.wait(0.5)

        # Test candidate vertical vector u = [0, 1] to see if it can serve as second eigenvector
        u_arrow = self.create_vector_arrow(plane, np.zeros(2), np.array([0.0, 1.0]), J_HAT_COLOR, stroke_width=3.5)
        u_label = MathTex(r"\mathbf{u} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}", font_size=17, color=J_HAT_COLOR).next_to(
            u_arrow.get_end(), UL, buff=0.1
        )
        self.play(GrowArrow(u_arrow), FadeIn(u_label), run_time=1.0)
        self.wait(0.5)

        # Apply M_def to u: M_def @ [0, 1] = [1, 2]
        mu_arrow = self.create_vector_arrow(plane, np.zeros(2), np.array([1.0, 2.0]), ERROR_COLOR, stroke_width=4.0)
        mu_label = MathTex(
            r"M_{\text{def}}\mathbf{u} = \begin{bmatrix} 1 \\ 2 \end{bmatrix} \ne \lambda \mathbf{u}",
            font_size=17,
            color=ERROR_COLOR,
        ).next_to(mu_arrow.get_end(), UR, buff=0.1)

        shear_arc = CurvedArrow(plane.c2p(0.0, 0.8), plane.c2p(0.4, 0.8), color=ERROR_COLOR)

        self.play(
            Transform(u_arrow, mu_arrow),
            ReplacementTransform(u_label, mu_label),
            Create(shear_arc),
            run_time=1.8,
            rate_func=smooth,
        )
        self.wait(1.0)

        # Multiplicity deficiency badge
        mult_badge = MathTex(
            r"\text{Algebraic Multiplicity} = 2 \quad > \quad \text{Geometric Multiplicity} = 1",
            font_size=17,
            color=TEXT_COLOR,
        )
        mult_panel = make_hud_panel(mult_badge).to_corner(DL, buff=0.3)
        self.play(FadeIn(mult_panel), run_time=1.0)
        self.wait(1.2)

        # Master Summary Verdict Card
        verdict_title = Text("Non-Diagonalizable: Eigenspace Deficient", font_size=20, color=ERROR_COLOR)
        verdict_body = MathTex(
            r"\det([\mathbf{v}_1 \mid \mathbf{v}_2]) = 0 \implies \text{No invertible basis } P \text{ exists to diagonalize } M_{\text{def}}",
            font_size=17,
            color=TEXT_COLOR,
        )
        verdict_jordan = Text(
            "Requires Jordan Canonical Form with generalized eigenvectors.",
            font_size=15,
            color=MUTED_TEXT,
        )
        verdict_content = VGroup(verdict_title, verdict_body, verdict_jordan).arrange(DOWN, buff=0.2)
        verdict_panel = make_hud_panel(verdict_content, padding=0.25)
        verdict_panel.set_z_index(20)

        self.play(FadeIn(verdict_panel, scale=1.08), run_time=1.5)
        self.wait(2.5)

        if is_standalone:
            self.play(
                FadeOut(title_panel),
                FadeOut(defective_panel),
                FadeOut(mult_panel),
                FadeOut(verdict_panel),
                FadeOut(eigenspace_line),
                FadeOut(eigenspace_label),
                FadeOut(v_arrow),
                FadeOut(u_arrow),
                FadeOut(mu_label),
                FadeOut(shear_arc),
                run_time=1.0,
            )
            return

        cleanup_group = VGroup(
            title_panel, defective_panel, mult_panel, verdict_panel,
            eigenspace_line, eigenspace_label, v_arrow, u_arrow, mu_label, shear_arc
        )
        return cleanup_group


# -----------------------------------------------------------------------------
# Modular Executable Scene Classes
# -----------------------------------------------------------------------------
class Act1SpatialPuzzle(BaseChangeOfBasisScene):
    """Standalone executable scene for Act 1."""

    def construct(self):
        plane = self.create_plane()
        origin_group = self.create_origin_anchor(plane)
        self.add(plane, origin_group)
        self.run_act1(plane, origin_group, is_standalone=True)


class Act2DynamicDeformation(BaseChangeOfBasisScene):
    """Standalone executable scene for Act 2."""

    def construct(self):
        plane = self.create_plane()
        origin_group = self.create_origin_anchor(plane)
        self.add(plane, origin_group)
        self.run_act2(plane, origin_group, is_standalone=True)


class Act3AlgebraicMorph(BaseChangeOfBasisScene):
    """Standalone executable scene for Act 3."""

    def construct(self):
        plane = self.create_plane()
        origin_group = self.create_origin_anchor(plane)
        self.add(plane, origin_group)
        self.run_act3(plane, origin_group, is_standalone=True)


class Act4DegenerateStressTest(BaseChangeOfBasisScene):
    """Standalone executable scene for Act 4."""

    def construct(self):
        plane = self.create_plane()
        origin_group = self.create_origin_anchor(plane)
        self.add(plane, origin_group)
        self.run_act4(plane, origin_group, is_standalone=True)


class ChangeOfBasisScene(BaseChangeOfBasisScene):
    """Master unified scene seamlessly orchestrating all 4 acts."""

    def construct(self):
        plane = self.create_plane()
        origin_group = self.create_origin_anchor(plane)
        self.add(plane, origin_group)

        # Act 1: Spatial Puzzle
        g1 = self.run_act1(plane, origin_group, is_standalone=False)
        self.play(FadeOut(g1), run_time=0.8)

        # Act 2: Dynamic Deformation & Invariant Eigen-Grid
        g2 = self.run_act2(plane, origin_group, is_standalone=False)
        self.play(FadeOut(g2), run_time=0.8)

        # Act 3: 3-Step Similarity Dance P^-1 -> D -> P
        g3 = self.run_act3(plane, origin_group, is_standalone=False)
        self.play(FadeOut(g3), run_time=0.8)

        # Act 4: Defective Matrix Stress Test
        g4 = self.run_act4(plane, origin_group, is_standalone=False)
        self.play(FadeOut(g4), run_time=0.8)
