"""
Sprint 6: Orthogonal Projections onto 1D Lines & k-D Subspaces (ManimCE v0.21.0)
Curriculum: Linear Algebra & Theoretical CS / Machine Learning Foundations
References: MML Ch. 3.6, 3.8, 9.2.3, 10.3; VMLS Ch. 11.5, 12.1-12.3, 15; ILA Ch. 3, 6

This production script implements the 4-act pedagogical narrative in 3D:
- Act 1: Concrete Spatial Puzzle (1D Line Projection MML Ex. 3.10 & Idempotence pi^2 = pi)
- Act 2: Dynamic Deformation (Sliding y_hat = Ax(s) across U = C(A) with Pinned 2D Corner Error Graph)
- Act 3: Color-Coded Algebraic Morph (Normal Equations -> P = A(A^T A)^{-1} A^T, I-P, Pythagoras)
- Act 4: Degenerate Stress Test (Affine Flats, Rank-Deficient A^+ & Ridge, Lauchli kappa^2, Oblique P_ob)

Render Commands:
    Preview Low-Res:   .venv/bin/manim -pql linear_algebra/sprint_06_orthogonal_projections/scene_orthogonal_projections.py OrthogonalProjectionsScene
    Headless Low-Res:  .venv/bin/manim -ql linear_algebra/sprint_06_orthogonal_projections/scene_orthogonal_projections.py OrthogonalProjectionsScene
    Production Hi-Res: .venv/bin/manim -pqh linear_algebra/sprint_06_orthogonal_projections/scene_orthogonal_projections.py OrthogonalProjectionsScene
    Headless Hi-Res:   .venv/bin/manim -qh linear_algebra/sprint_06_orthogonal_projections/scene_orthogonal_projections.py OrthogonalProjectionsScene
"""

from __future__ import annotations

import math
import numpy as np
from manim import *

# -----------------------------------------------------------------------------
# Rule 1: 3Blue1Brown Semantic Color Palette
# -----------------------------------------------------------------------------
I_HAT_COLOR = "#83C167"       # Green      - Basis column a1
J_HAT_COLOR = "#FC6255"       # Red        - Basis column a2
PROJ_COLOR = "#58C4DD"        # Blue       - Orthogonal projection y_hat = P y
SUBSPACE_COLOR = "#2A9D8F"    # Teal       - Target subspace line / plane U = C(A)
NULL_COLOR = "#FF9E64"        # Orange     - Rank-1 collapse / Ridge rescue
AFFINE_COLOR = "#F7D96F"      # Yellow     - Affine flat L = x0 + U
RIGHT_ANGLE_COLOR = "#F7D96F" # Yellow     - 3D right-angle elbow indicator
ORIGIN_COLOR = "#FFFFFF"      # White      - Immovable origin anchor (0, 0, 0)
TARGET_COLOR = "#F9E498"      # Gold       - Off-subspace target vector y
ERROR_COLOR = "#F7768E"       # Coral Red  - Residual vector r = y - y_hat & oblique error
BG_COLOR = "#1C1C24"          # Dark Slate - High-contrast viewport background
PANEL_BG = "#16161E"          # Obsidian   - HUD panel background container
PANEL_BORDER = "#414868"      # Slate-Teal - HUD panel container border
TEXT_COLOR = "#ECEFF4"        # Snow       - Primary typography
MUTED_TEXT = "#7AA2F7"        # Light Blue - Secondary badges & annotations

# -----------------------------------------------------------------------------
# Mathematical Constants & Pre-Flight Assertions (Rule 6: Exact NumPy Precision)
# -----------------------------------------------------------------------------
ORIGIN_3D = np.array([0.0, 0.0, 0.0])

# Act 1: 1D Line Projection (MML Ex. 3.10 direction + exact integer target)
B_1D = np.array([1.0, 2.0, 2.0])
X_1D = np.array([3.0, 3.0, 0.0])
LAMBDA_1D = float(np.dot(B_1D, X_1D) / np.dot(B_1D, B_1D))
P_1D = np.outer(B_1D, B_1D) / np.dot(B_1D, B_1D)
Y_HAT_1D = P_1D @ X_1D
R_1D = X_1D - Y_HAT_1D

# Acts 2 & 3: 2D Subspace Plane (MML Ex. 3.11) + exact integer projection/residual
A_MAT = np.array([
    [1.0, 0.0],
    [1.0, 1.0],
    [1.0, 2.0],
])
A1_COL = A_MAT[:, 0]
A2_COL = A_MAT[:, 1]
N_PLANE = np.array([1.0, -2.0, 1.0])
Y_TARGET = np.array([3.0, -1.0, 1.0])
X_HAT = np.linalg.solve(A_MAT.T @ A_MAT, A_MAT.T @ Y_TARGET)
Y_HAT = A_MAT @ X_HAT
R_OPT = Y_TARGET - Y_HAT
P_2D = A_MAT @ np.linalg.inv(A_MAT.T @ A_MAT) @ A_MAT.T
P_COMP = np.eye(3) - P_2D

# Orthonormal basis for plane U (x - 2y + z = 0) used for bounded 3D Surface
Q_PLANE_1 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
Q_PLANE_2 = np.array([-1.0, 0.0, 1.0]) / np.sqrt(2.0)

# Act 4: Affine Flat, Rank-Deficient, Lauchli & Oblique Projector
X0_AFF = np.array([0.0, 0.0, 1.0])
Y_AFF = np.array([3.0, -1.0, 2.0])
Y_AFF_PROJ = X0_AFF + P_2D @ (Y_AFF - X0_AFF)

A_DEG = np.array([
    [1.0, 2.0],
    [1.0, 2.0],
    [1.0, 2.0],
])
P_DEG = A_DEG @ np.linalg.pinv(A_DEG)

P_OB = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [-1.0, 2.0, 0.0],
])
Y_OB_TEST = np.array([1.0, 1.0, 3.0])

# Pre-flight mathematical consistency verification
assert np.isclose(LAMBDA_1D, 1.0), "1D scalar coordinate must equal 1.0"
assert np.allclose(Y_HAT_1D, [1.0, 2.0, 2.0]), "1D projection must equal [1, 2, 2]^T"
assert np.allclose(R_1D, [2.0, 1.0, -2.0]), "1D residual must equal [2, 1, -2]^T"
assert np.isclose(np.dot(Y_HAT_1D, R_1D), 0.0), "1D residual must be orthogonal to b"
assert np.allclose(P_1D @ P_1D, P_1D) and np.allclose(P_1D.T, P_1D), "P_1D must be symmetric & idempotent"
assert np.isclose(np.trace(P_1D), 1.0) and np.linalg.matrix_rank(P_1D) == 1, "rank(P_1D)=Tr(P_1D)=1"

assert np.allclose(X_HAT, [2.0, -1.0]), "Optimal x_hat must equal [2, -1]^T"
assert np.allclose(Y_HAT, [2.0, 1.0, 0.0]), "Optimal y_hat must equal [2, 1, 0]^T"
assert np.allclose(R_OPT, [1.0, -2.0, 1.0]), "Optimal residual must equal [1, -2, 1]^T"
assert np.allclose(A_MAT.T @ R_OPT, 0.0), "Residual must satisfy A^T r = 0"
assert np.isclose(np.linalg.norm(Y_TARGET) ** 2, np.linalg.norm(Y_HAT) ** 2 + np.linalg.norm(R_OPT) ** 2)
assert np.allclose(P_2D @ P_2D, P_2D) and np.allclose(P_2D.T, P_2D), "P_2D must be symmetric & idempotent"
assert np.isclose(np.trace(P_2D), 2.0) and np.linalg.matrix_rank(P_2D) == 2, "rank(P_2D)=Tr(P_2D)=2"
assert np.allclose(np.sort(np.linalg.eigvalsh(P_2D)), [0.0, 1.0, 1.0]), "spec(P_2D) == {0, 1, 1}"
assert np.isclose(np.linalg.norm(P_2D, 2), 1.0), "||P_2D||_2 must be 1.0"
assert np.allclose(P_COMP, np.outer(N_PLANE, N_PLANE) / np.dot(N_PLANE, N_PLANE)), "I - P_2D == n n^T / ||n||^2"
assert np.allclose(P_2D @ P_COMP, 0.0), "P(I - P) == 0"
assert np.isclose(np.dot(Q_PLANE_1, Q_PLANE_2), 0.0) and np.isclose(np.dot(Q_PLANE_1, N_PLANE), 0.0)
assert np.isclose(np.dot(Q_PLANE_2, N_PLANE), 0.0)

for _s in (-2.0, -1.0, 0.0, 1.0, 1.5):
    _xs = np.array([2.0 + _s, -1.0 - _s])
    assert np.isclose(np.linalg.norm(Y_TARGET - A_MAT @ _xs), math.sqrt(6.0 + 2.0 * _s * _s))

assert np.allclose(Y_AFF_PROJ, [2.0, 1.0, 1.0]), "Affine projection must equal [2, 1, 1]^T"
assert np.isclose(np.linalg.det(A_DEG.T @ A_DEG), 0.0), "A_DEG^T A_DEG must be singular"
assert np.allclose(P_DEG, np.ones((3, 3)) / 3.0), "Rank-deficient projector must be 1/3 * 1 1^T"
assert np.allclose(P_OB @ P_OB, P_OB) and not np.allclose(P_OB.T, P_OB), "P_OB idempotent & asymmetric"
assert np.isclose(np.linalg.norm(P_OB, 2), np.sqrt(6.0)), "||P_OB||_2 == sqrt(6)"
assert np.linalg.norm(Y_OB_TEST - P_OB @ Y_OB_TEST) > np.linalg.norm(Y_OB_TEST - P_2D @ Y_OB_TEST)


# -----------------------------------------------------------------------------
# Rule 3: Zero Text Overlap HUD Panel Factory & 3D Geometry Helpers
# -----------------------------------------------------------------------------
def make_hud_panel(
    content: Mobject,
    padding: float = 0.16,
    bg_color: str = PANEL_BG,
    bg_opacity: float = 0.92,
    stroke_color: str = PANEL_BORDER,
    stroke_width: float = 1.5,
    corner_radius: float = 0.12,
    min_width: float = 0.0,
) -> VGroup:
    """Wraps any mobject in a dark rounded rectangle with high z-index."""
    w = max(content.width + 2 * padding, min_width)
    h = content.height + 2 * padding
    bg = RoundedRectangle(
        corner_radius=corner_radius,
        width=w,
        height=h,
        fill_color=bg_color,
        fill_opacity=bg_opacity,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
    )
    bg.move_to(content.get_center())
    panel = VGroup(bg, content)
    panel.set_z_index(10)
    return panel


def create_3d_right_angle(
    axes: ThreeDAxes,
    vertex: np.ndarray,
    dir1: np.ndarray,
    dir2: np.ndarray,
    size: float = 0.28,
    color: str = RIGHT_ANGLE_COLOR,
) -> VMobject:
    """Constructs a crisp 3D right-angle elbow without calling XY-only RightAngle."""
    p0 = np.array(axes.c2p(*vertex), dtype=float)
    p1 = np.array(axes.c2p(*(vertex + dir1)), dtype=float)
    p2 = np.array(axes.c2p(*(vertex + dir2)), dtype=float)
    u1 = p1 - p0
    u1 = u1 / (np.linalg.norm(u1) + 1e-12)
    u2 = p2 - p0
    u2 = u2 / (np.linalg.norm(u2) + 1e-12)
    elbow = VMobject(color=color, stroke_width=3.5, shade_in_3d=False)
    elbow.set_points_as_corners([
        p0 + size * u1,
        p0 + size * (u1 + u2),
        p0 + size * u2,
    ])
    return elbow


def safe_arrow_3d(
    axes: ThreeDAxes,
    start_coords: np.ndarray,
    end_coords: np.ndarray,
    color: str,
    resolution: tuple[int, int] = (2, 10),
) -> Mobject:
    """Short-length-safe Arrow3D avoiding cone inversion when dist < height."""
    p_start = np.array(axes.c2p(*start_coords), dtype=float)
    p_end = np.array(axes.c2p(*end_coords), dtype=float)
    dist = float(np.linalg.norm(p_end - p_start))
    if dist < 0.08:
        return Dot3D(point=p_start, radius=0.05, color=color, resolution=(6, 6))
    tip_h = min(0.24, 0.35 * dist)
    tip_r = min(0.065, 0.28 * tip_h)
    return Arrow3D(
        start=p_start,
        end=p_end,
        color=color,
        height=tip_h,
        base_radius=tip_r,
        resolution=resolution,
    )


def safe_line_3d(
    axes: ThreeDAxes,
    start_coords: np.ndarray,
    end_coords: np.ndarray,
    color: str,
    thickness: float = 0.02,
    resolution: tuple[int, int] = (2, 8),
) -> Mobject:
    """Zero-length-safe Line3D with low resolution for fast 60fps dynamic updaters."""
    p_start = np.array(axes.c2p(*start_coords), dtype=float)
    p_end = np.array(axes.c2p(*end_coords), dtype=float)
    dist = float(np.linalg.norm(p_end - p_start))
    if dist < 0.05:
        return Dot3D(point=p_start, radius=0.04, color=color, resolution=(6, 6))
    return Line3D(
        start=p_start,
        end=p_end,
        thickness=thickness,
        color=color,
        resolution=resolution,
    )


# -----------------------------------------------------------------------------
# Base Scene Class (ThreeDScene Subclass with 3D Axes & Viewport HUD Pinning)
# -----------------------------------------------------------------------------
class BaseOrthogonalProjectionsScene(ThreeDScene):
    """Master base class providing 3D coordinate space, camera orbits, and HUD pinning."""

    def setup(self):
        super().setup()
        self.camera.background_color = BG_COLOR
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.08)

    def create_axes_3d(self) -> ThreeDAxes:
        """Rule 2: 3D Axes clamped to [-4, 4]^3 viewport dimensions."""
        return ThreeDAxes(
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

    def create_subspace_plane(
        self,
        axes: ThreeDAxes,
        offset: np.ndarray = ORIGIN_3D,
        fill_color: str = SUBSPACE_COLOR,
        checkerboard_colors: list[str] | bool = (SUBSPACE_COLOR, "#217A70"),
        fill_opacity: float = 0.35,
    ) -> Surface:
        """Builds translucent 2D plane U = C(A) (or shifted L = x0 + U) inside [-4, 4]^3."""
        return Surface(
            lambda u, v: axes.c2p(*(offset + u * Q_PLANE_1 + v * Q_PLANE_2)),
            u_range=[-2.3, 2.4],
            v_range=[-3.5, 2.4],
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            checkerboard_colors=checkerboard_colors,
            stroke_color=fill_color,
            stroke_width=0.4,
            resolution=(6, 6),
        )

    def pin_hud(self, panel: Mobject, corner=UL, buff: float = 0.22):
        """Rule 3: Enforce add_fixed_in_frame_mobjects for all 2D HUD cards."""
        panel.to_corner(corner, buff=buff)
        self.add_fixed_in_frame_mobjects(panel)

    def unpin_hud(self, *panels: Mobject, run_time: float = 0.45):
        """FadeOut while pinned, then remove from both fixed-in-frame set and scene."""
        if not panels:
            return
        self.play(*[FadeOut(p) for p in panels], run_time=run_time)
        self.remove_fixed_in_frame_mobjects(*panels)
        self.remove(*panels)

    # -------------------------------------------------------------------------
    # Act 1: Concrete Spatial Puzzle (1D Line Projection & Idempotence)
    # -------------------------------------------------------------------------
    def run_act1(self):
        axes = self.create_axes_3d()
        origin_dot = Dot3D(point=axes.c2p(0, 0, 0), radius=0.08, color=ORIGIN_COLOR)

        title_text = VGroup(
            Text("Sprint 6: Orthogonal Projections", font_size=19, color=TEXT_COLOR, weight=BOLD),
            Text("Act 1: 1D Line Projection & Idempotence", font_size=14, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        title_panel = make_hud_panel(title_text)
        self.pin_hud(title_panel, corner=UL)

        formula_1d = VGroup(
            MathTex(
                r"\mathbf{b} = [1, 2, 2]^\top,\quad \mathbf{x} = [3, 3, 0]^\top",
                font_size=17,
                color=TEXT_COLOR,
            ),
            MathTex(
                r"\lambda = \frac{\mathbf{b}^\top \mathbf{x}}{\|\mathbf{b}\|^2} = \frac{9}{9} = 1",
                font_size=17,
                color=PROJ_COLOR,
            ),
            MathTex(
                r"P_1 = \frac{\mathbf{b}\mathbf{b}^\top}{\|\mathbf{b}\|^2} = \frac{1}{9}\begin{bmatrix}1&2&2\\2&4&4\\2&4&4\end{bmatrix}",
                font_size=15,
                color=SUBSPACE_COLOR,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.07)
        formula_panel = make_hud_panel(formula_1d)
        self.pin_hud(formula_panel, corner=UR)

        self.play(Create(axes), FadeIn(origin_dot), run_time=0.9)

        # 1D line U = span(b) and target vector x_1D
        line_1d = safe_line_3d(axes, -1.5 * B_1D, 1.8 * B_1D, SUBSPACE_COLOR, thickness=0.022)
        b_arrow = safe_arrow_3d(axes, ORIGIN_3D, B_1D, SUBSPACE_COLOR)
        x_arrow = safe_arrow_3d(axes, ORIGIN_3D, X_1D, TARGET_COLOR)
        x_dot = Dot3D(point=axes.c2p(*X_1D), radius=0.09, color=TARGET_COLOR)

        self.play(Create(line_1d), FadeIn(b_arrow), FadeIn(x_arrow), FadeIn(x_dot), run_time=1.0)
        self.wait(1.4)

        # Slide scalar lambda * b from 0 to 1 along line_1d using lightweight in-place 3D segments
        lam_tracker = ValueTracker(0.0)
        proj_dot = Dot3D(point=axes.c2p(0, 0, 0), radius=0.09, color=PROJ_COLOR, resolution=(6, 6))
        proj_dot.add_updater(lambda m: m.move_to(axes.c2p(*(lam_tracker.get_value() * B_1D))))

        proj_line = VMobject(color=PROJ_COLOR, stroke_width=5.5, shade_in_3d=False)
        proj_line.add_updater(
            lambda m: m.set_points_as_corners([
                axes.c2p(0, 0, 0),
                axes.c2p(*(max(lam_tracker.get_value(), 0.02) * B_1D)),
            ])
        )
        res_line_1d = VMobject(color=ERROR_COLOR, stroke_width=5.0, shade_in_3d=False)
        res_line_1d.add_updater(
            lambda m: m.set_points_as_corners([
                axes.c2p(*(lam_tracker.get_value() * B_1D)),
                axes.c2p(*X_1D),
            ])
        )
        proj_line.update()
        res_line_1d.update()
        self.add(proj_dot, proj_line, res_line_1d)
        self.play(lam_tracker.animate.set_value(1.0), run_time=2.8, rate_func=smooth)

        proj_dot.clear_updaters()
        proj_line.clear_updaters()
        res_line_1d.clear_updaters()

        # Lock static 3D elbow and idempotence badge
        elbow_1d = create_3d_right_angle(axes, Y_HAT_1D, -B_1D, R_1D, size=0.30, color=RIGHT_ANGLE_COLOR)
        idem_text = VGroup(
            MathTex(r"\pi_U(\mathbf{x}) = [1, 2, 2]^\top,\quad \mathbf{r} = [2, 1, -2]^\top", font_size=16, color=TEXT_COLOR),
            MathTex(r"\pi^2 = \pi \iff P_1^2 = P_1,\quad \operatorname{Tr}(P_1) = 1", font_size=16, color=PROJ_COLOR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        idem_panel = make_hud_panel(idem_text)
        self.pin_hud(idem_panel, corner=DL)

        self.play(Create(elbow_1d), run_time=0.8)
        self.play(proj_dot.animate.scale(1.35), run_time=0.45)
        self.play(proj_dot.animate.scale(1.0 / 1.35), run_time=0.45)
        self.wait(1.8)

        # Transition puzzle preview for 2D subspace
        puzzle_text = VGroup(
            Text("Next: 2D Subspace U = C(A)", font_size=15, color=SUBSPACE_COLOR, weight=BOLD),
            MathTex(r"\text{Where on } U \text{ is } \|\mathbf{y} - \hat{\mathbf{y}}\|_2 \text{ minimized?}", font_size=16, color=TARGET_COLOR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        puzzle_panel = make_hud_panel(puzzle_text)
        self.pin_hud(puzzle_panel, corner=DR)
        self.wait(4.0)

        # Clean up Act 1
        self.unpin_hud(title_panel, formula_panel, idem_panel, puzzle_panel, run_time=0.5)
        self.play(
            FadeOut(line_1d), FadeOut(b_arrow), FadeOut(x_arrow), FadeOut(x_dot),
            FadeOut(proj_dot), FadeOut(proj_line), FadeOut(res_line_1d), FadeOut(elbow_1d),
            FadeOut(axes), FadeOut(origin_dot),
            run_time=0.7,
        )

    # -------------------------------------------------------------------------
    # Act 2: Dynamic Deformation (Sliding y_hat = Ax(s) + Corner Error Graph)
    # -------------------------------------------------------------------------
    def run_act2(self):
        axes = self.create_axes_3d()
        origin_dot = Dot3D(point=axes.c2p(0, 0, 0), radius=0.08, color=ORIGIN_COLOR)

        title_text = VGroup(
            Text("Act 2: Minimum Distance on k-D Subspace", font_size=19, color=TEXT_COLOR, weight=BOLD),
            Text("Sliding y_hat = Ax(s) vs. Error Graph", font_size=14, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        title_panel = make_hud_panel(title_text)
        self.pin_hud(title_panel, corner=UL)

        ortho_info = VGroup(
            MathTex(r"A = \begin{bmatrix}1&0\\1&1\\1&2\end{bmatrix},\quad \mathbf{y} = \begin{bmatrix}3\\-1\\1\end{bmatrix}", font_size=16, color=TEXT_COLOR),
            MathTex(r"U = \mathcal{C}(A): x - 2y + z = 0", font_size=15, color=SUBSPACE_COLOR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        ortho_panel = make_hud_panel(ortho_info)
        self.pin_hud(ortho_panel, corner=UR)

        # Pinned 2D Corner Error Graph (DR)
        err_axes = Axes(
            x_range=[-2.0, 1.5, 0.5],
            y_range=[0.0, 4.0, 1.0],
            x_length=2.8,
            y_length=1.7,
            tips=False,
            axis_config={"color": "#7AA2F7", "stroke_width": 1.5, "include_ticks": True, "tick_size": 0.05},
        )
        err_title = Text("Error d(s) = ||y - Ax(s)||_2", font_size=14, color=ERROR_COLOR)
        err_min_lbl = MathTex(r"\min d(0) = \sqrt{6} \approx 2.449", font_size=14, color=PROJ_COLOR)
        err_group = VGroup(err_title, err_axes, err_min_lbl).arrange(DOWN, buff=0.07)
        err_panel = make_hud_panel(err_group, padding=0.14)
        self.pin_hud(err_panel, corner=DR, buff=0.20)

        # Live DecimalNumber readout (DL)
        s_tracker = ValueTracker(-2.0)
        s_lbl = MathTex(r"s =", font_size=18, color=TEXT_COLOR)
        s_num = DecimalNumber(-2.0, num_decimal_places=2, font_size=18, color=PROJ_COLOR)
        s_row = VGroup(s_lbl, s_num).arrange(RIGHT, buff=0.12)

        d_lbl = MathTex(r"\|\mathbf{y} - \hat{\mathbf{y}}\|_2 =", font_size=17, color=TEXT_COLOR)
        d_num = DecimalNumber(math.sqrt(14.0), num_decimal_places=3, font_size=18, color=ERROR_COLOR)
        d_row = VGroup(d_lbl, d_num).arrange(RIGHT, buff=0.12)

        dl_content = VGroup(s_row, d_row).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        dl_panel = make_hud_panel(dl_content, padding=0.16, min_width=2.65)
        self.pin_hud(dl_panel, corner=DL, buff=0.22)

        # Dynamic DecimalNumber updaters (re-pin new digit glyphs every update!)
        def update_s_num(mob: DecimalNumber):
            mob.set_value(s_tracker.get_value())
            self.camera.add_fixed_in_frame_mobjects(mob)

        def update_d_num(mob: DecimalNumber):
            sv = s_tracker.get_value()
            mob.set_value(math.sqrt(6.0 + 2.0 * sv * sv))
            self.camera.add_fixed_in_frame_mobjects(mob)

        s_num.add_updater(update_s_num)
        d_num.add_updater(update_d_num)

        # Leaf VMobject error curve & moving Dot (pinned once, updated in-place)
        err_curve = VMobject(color=ERROR_COLOR, stroke_width=2.8, shade_in_3d=False)
        err_curve.set_z_index(12)
        err_dot = Dot(point=err_axes.c2p(-2.0, math.sqrt(14.0)), radius=0.055, color=PROJ_COLOR)
        err_dot.set_z_index(13)

        def update_err_curve(mob: VMobject):
            sv = s_tracker.get_value()
            s_end = max(sv, -1.98)
            n_pts = max(4, int((s_end + 2.0) * 24) + 2)
            s_vals = np.linspace(-2.0, s_end, n_pts)
            pts = [err_axes.c2p(float(t), math.sqrt(6.0 + 2.0 * t * t)) for t in s_vals]
            mob.set_points_as_corners(pts)

        def update_err_dot(mob: Dot):
            sv = s_tracker.get_value()
            mob.move_to(err_axes.c2p(sv, math.sqrt(6.0 + 2.0 * sv * sv)))

        update_err_curve(err_curve)
        update_err_dot(err_dot)
        self.add_fixed_in_frame_mobjects(err_curve, err_dot)
        err_curve.add_updater(update_err_curve)
        err_dot.add_updater(update_err_dot)

        # 3D Subspace Plane, basis columns, and target y
        plane_sheet = self.create_subspace_plane(axes, fill_opacity=0.35)
        arrow_a1 = safe_arrow_3d(axes, ORIGIN_3D, A1_COL, I_HAT_COLOR)
        arrow_a2 = safe_arrow_3d(axes, ORIGIN_3D, A2_COL, J_HAT_COLOR)
        target_dot = Dot3D(point=axes.c2p(*Y_TARGET), radius=0.10, color=TARGET_COLOR)
        target_arrow = safe_arrow_3d(axes, ORIGIN_3D, Y_TARGET, TARGET_COLOR)

        self.play(
            Create(axes), FadeIn(origin_dot), Create(plane_sheet),
            FadeIn(arrow_a1), FadeIn(arrow_a2), FadeIn(target_dot), FadeIn(target_arrow),
            run_time=1.3,
        )
        self.wait(0.8)

        # Sliding candidate y_hat(s) = A [2+s, -1-s]^T = [2+s, 1, -s]^T
        def cand_coords(sv: float) -> np.ndarray:
            return A_MAT @ np.array([2.0 + sv, -1.0 - sv])

        cand_dot = Dot3D(point=axes.c2p(*cand_coords(-2.0)), radius=0.09, color=PROJ_COLOR, resolution=(6, 6))
        cand_dot.add_updater(lambda m: m.move_to(axes.c2p(*cand_coords(s_tracker.get_value()))))
        cand_line = VMobject(color=PROJ_COLOR, stroke_width=5.0, shade_in_3d=False)
        cand_line.add_updater(
            lambda m: m.set_points_as_corners([
                axes.c2p(0, 0, 0),
                axes.c2p(*cand_coords(s_tracker.get_value())),
            ])
        )
        res_line = VMobject(color=ERROR_COLOR, stroke_width=5.5, shade_in_3d=False)
        res_line.add_updater(
            lambda m: m.set_points_as_corners([
                axes.c2p(*cand_coords(s_tracker.get_value())),
                axes.c2p(*Y_TARGET),
            ])
        )
        cand_line.update()
        res_line.update()
        self.add(cand_dot, cand_line, res_line)

        # Sweep from s = -2.0 down to s = 0.0 (global minimum!)
        self.play(s_tracker.animate.set_value(0.0), run_time=5.0, rate_func=smooth)

        # Illuminate 3D right-angle elbows and Orthogonality Principle badge at s = 0
        elbow_1 = create_3d_right_angle(axes, Y_HAT, -Y_HAT, R_OPT, size=0.28, color=RIGHT_ANGLE_COLOR)
        elbow_2 = create_3d_right_angle(axes, Y_HAT, A1_COL, R_OPT, size=0.28, color=RIGHT_ANGLE_COLOR)
        orth_badge = make_hud_panel(
            MathTex(r"A^\top (\mathbf{y} - A\hat{\mathbf{x}}) = \mathbf{0} \implies \text{Global Min!}", font_size=16, color=RIGHT_ANGLE_COLOR),
            padding=0.14,
        )
        orth_badge.next_to(ortho_panel, DOWN, buff=0.14, aligned_edge=RIGHT)
        self.add_fixed_in_frame_mobjects(orth_badge)

        self.play(Create(elbow_1), Create(elbow_2), FadeIn(orth_badge), run_time=0.9)
        self.wait(2.4)

        # Overshoot to s = 1.5 then snap back to s = 0.0
        self.play(FadeOut(elbow_1), FadeOut(elbow_2), run_time=0.4)
        self.play(s_tracker.animate.set_value(1.5), run_time=2.6, rate_func=smooth)
        self.wait(0.8)
        self.play(s_tracker.animate.set_value(0.0), run_time=2.0, rate_func=smooth)
        self.play(FadeIn(elbow_1), FadeIn(elbow_2), run_time=0.5)
        self.wait(3.0)

        # Clear ALL updaters before teardown
        for m in (cand_dot, cand_line, res_line, err_curve, err_dot, s_num, d_num):
            m.clear_updaters()

        self.unpin_hud(title_panel, ortho_panel, orth_badge, err_panel, dl_panel, err_curve, err_dot, run_time=0.5)
        self.play(
            FadeOut(plane_sheet), FadeOut(arrow_a1), FadeOut(arrow_a2),
            FadeOut(target_dot), FadeOut(target_arrow),
            FadeOut(cand_dot), FadeOut(cand_line), FadeOut(res_line),
            FadeOut(elbow_1), FadeOut(elbow_2),
            FadeOut(axes), FadeOut(origin_dot),
            run_time=0.7,
        )

    # -------------------------------------------------------------------------
    # Act 3: Color-Coded Algebraic Morph (Normal Equations, P, I-P, Pythagoras)
    # -------------------------------------------------------------------------
    def run_act3(self):
        axes = self.create_axes_3d()
        origin_dot = Dot3D(point=axes.c2p(0, 0, 0), radius=0.08, color=ORIGIN_COLOR)

        title_text = VGroup(
            Text("Act 3: Normal Equations & Projector P", font_size=19, color=TEXT_COLOR, weight=BOLD),
            Text("Orthogonal Complement (I - P) & Pythagoras", font_size=14, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        title_panel = make_hud_panel(title_text)
        self.pin_hud(title_panel, corner=UL)

        plane_sheet = self.create_subspace_plane(axes, fill_opacity=0.35)
        target_arrow = safe_arrow_3d(axes, ORIGIN_3D, Y_TARGET, TARGET_COLOR)
        proj_arrow = safe_arrow_3d(axes, ORIGIN_3D, Y_HAT, PROJ_COLOR)
        res_arrow = safe_arrow_3d(axes, Y_HAT, Y_TARGET, ERROR_COLOR)
        comp_line = safe_line_3d(axes, -1.3 * N_PLANE, 1.5 * N_PLANE, ERROR_COLOR, thickness=0.018)
        comp_arrow = safe_arrow_3d(axes, ORIGIN_3D, R_OPT, ERROR_COLOR)
        elbow = create_3d_right_angle(axes, Y_HAT, -Y_HAT, R_OPT, size=0.30, color=RIGHT_ANGLE_COLOR)

        self.play(
            Create(axes), FadeIn(origin_dot), Create(plane_sheet),
            FadeIn(target_arrow), FadeIn(proj_arrow), FadeIn(res_arrow),
            Create(elbow),
            run_time=1.2,
        )

        # Step 1: Orthogonality -> Normal Equations
        eq_step1 = make_hud_panel(
            VGroup(
                MathTex(r"A^\top (\mathbf{y} - A\hat{\mathbf{x}}) = \mathbf{0}", font_size=19, color=TARGET_COLOR),
                MathTex(r"\implies A^\top A\hat{\mathbf{x}} = A^\top \mathbf{y}", font_size=19, color=PROJ_COLOR),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        )
        self.pin_hud(eq_step1, corner=UR)
        self.wait(2.8)

        # Step 2: Closed-form projector P & complement I - P
        self.unpin_hud(eq_step1, run_time=0.35)
        eq_step2 = make_hud_panel(
            VGroup(
                MathTex(r"P = A(A^\top A)^{-1}A^\top = \frac{1}{6}\begin{bmatrix}5&2&-1\\2&2&2\\-1&2&5\end{bmatrix}", font_size=15, color=PROJ_COLOR),
                MathTex(r"I - P = \frac{1}{6}\begin{bmatrix}1&-2&1\\-2&4&-2\\1&-2&1\end{bmatrix} \to U^\perp", font_size=15, color=ERROR_COLOR),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        )
        self.pin_hud(eq_step2, corner=UR)
        self.play(Create(comp_line), FadeIn(comp_arrow), run_time=1.0)
        self.wait(2.4)

        # DL: Pythagorean Theorem Card
        pyth_card = make_hud_panel(
            VGroup(
                MathTex(r"\|\mathbf{y}\|_2^2 = \|P\mathbf{y}\|_2^2 + \|(I - P)\mathbf{y}\|_2^2", font_size=17, color=TEXT_COLOR),
                MathTex(r"11 = 5 + 6 \quad (P(I - P) = \mathbf{0})", font_size=16, color=RIGHT_ANGLE_COLOR),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.07)
        )
        self.pin_hud(pyth_card, corner=DL)
        self.wait(2.2)

        # DR: Projector Spectral & Algebraic Invariants Card
        inv_card = make_hud_panel(
            VGroup(
                MathTex(r"P^2 = P,\quad P^\top = P,\quad \|P\|_2 = 1", font_size=16, color=PROJ_COLOR),
                MathTex(r"\operatorname{Tr}(P) = \operatorname{rank}(P) = 2,\quad \lambda \in \{0, 1\}", font_size=16, color=SUBSPACE_COLOR),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.07)
        )
        self.pin_hud(inv_card, corner=DR)
        self.wait(4.5)

        # Clean up Act 3
        self.unpin_hud(title_panel, eq_step2, pyth_card, inv_card, run_time=0.5)
        self.play(
            FadeOut(plane_sheet), FadeOut(target_arrow), FadeOut(proj_arrow),
            FadeOut(res_arrow), FadeOut(comp_line), FadeOut(comp_arrow), FadeOut(elbow),
            FadeOut(axes), FadeOut(origin_dot),
            run_time=0.7,
        )

    # -------------------------------------------------------------------------
    # Act 4: Degenerate Stress Test (Affine, Rank-Deficient, Lauchli, Oblique)
    # -------------------------------------------------------------------------
    def run_act4(self):
        axes = self.create_axes_3d()
        origin_dot = Dot3D(point=axes.c2p(0, 0, 0), radius=0.08, color=ORIGIN_COLOR)

        title_text = VGroup(
            Text("Act 4: Affine, Rank-Deficient & Oblique", font_size=19, color=TEXT_COLOR, weight=BOLD),
            Text("Edge Cases & Numerical Conditioning", font_size=14, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        title_panel = make_hud_panel(title_text)
        self.pin_hud(title_panel, corner=UL)

        self.play(Create(axes), FadeIn(origin_dot), run_time=0.8)

        # 4A: Affine Subspace Shift L = x0 + U
        plane_u = self.create_subspace_plane(axes, fill_opacity=0.22)
        plane_l = self.create_subspace_plane(
            axes, offset=X0_AFF, fill_color=AFFINE_COLOR, checkerboard_colors=(AFFINE_COLOR, "#C9A83B"), fill_opacity=0.32
        )
        x0_arrow = safe_arrow_3d(axes, ORIGIN_3D, X0_AFF, AFFINE_COLOR)
        y_aff_dot = Dot3D(point=axes.c2p(*Y_AFF), radius=0.09, color=TARGET_COLOR)
        y_aff_proj_dot = Dot3D(point=axes.c2p(*Y_AFF_PROJ), radius=0.09, color=PROJ_COLOR)
        aff_res_line = safe_line_3d(axes, Y_AFF_PROJ, Y_AFF, ERROR_COLOR, thickness=0.022)
        aff_elbow = create_3d_right_angle(axes, Y_AFF_PROJ, -(Y_AFF_PROJ - X0_AFF), Y_AFF - Y_AFF_PROJ, size=0.28)

        aff_card = make_hud_panel(
            VGroup(
                Text("Affine Flat: L = x0 + U", font_size=15, color=AFFINE_COLOR, weight=BOLD),
                MathTex(r"\pi_L(\mathbf{x}) = \mathbf{x}_0 + P(\mathbf{x} - \mathbf{x}_0)", font_size=16, color=TEXT_COLOR),
                MathTex(r"\pi_L(\mathbf{0}) \ne \mathbf{0} \text{ (not linear!)}", font_size=15, color=MUTED_TEXT),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        )
        self.pin_hud(aff_card, corner=UR)

        self.play(
            Create(plane_u), Create(plane_l), FadeIn(x0_arrow),
            FadeIn(y_aff_dot), FadeIn(y_aff_proj_dot), Create(aff_res_line), Create(aff_elbow),
            run_time=1.3,
        )
        self.wait(2.8)

        # 4B: Rank-Deficient Collapse & Lauchli Ill-Conditioning
        self.play(
            FadeOut(plane_l), FadeOut(x0_arrow), FadeOut(y_aff_dot),
            FadeOut(y_aff_proj_dot), FadeOut(aff_res_line), FadeOut(aff_elbow),
            run_time=0.6,
        )
        self.unpin_hud(aff_card, run_time=0.35)

        line_deg = safe_line_3d(axes, -2.2 * A1_COL, 2.2 * A1_COL, NULL_COLOR, thickness=0.025)
        y_deg_proj = P_DEG @ Y_TARGET
        deg_target_dot = Dot3D(point=axes.c2p(*Y_TARGET), radius=0.09, color=TARGET_COLOR)
        deg_proj_dot = Dot3D(point=axes.c2p(*y_deg_proj), radius=0.09, color=NULL_COLOR)
        deg_res_line = safe_line_3d(axes, y_deg_proj, Y_TARGET, ERROR_COLOR, thickness=0.022)

        rank_card = make_hud_panel(
            VGroup(
                Text("Rank-Deficient A_deg (det(A^T A)=0)", font_size=14, color=NULL_COLOR, weight=BOLD),
                MathTex(r"P = A A^+,\quad \hat{\mathbf{x}}_{\text{RLS}} = (A^\top A + \lambda I)^{-1}A^\top\mathbf{y}", font_size=15, color=TEXT_COLOR),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        )
        self.pin_hud(rank_card, corner=UR)

        lauchli_card = make_hud_panel(
            VGroup(
                Text("Lauchli Normal Eqs Trap", font_size=14, color=ERROR_COLOR, weight=BOLD),
                MathTex(r"\kappa(A_\varepsilon^\top A_\varepsilon) = \kappa(A_\varepsilon)^2 \approx 2\times 10^{16}", font_size=15, color=TEXT_COLOR),
                Text("Use QR / SVD instead of A^T A!", font_size=14, color=PROJ_COLOR),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        )
        self.pin_hud(lauchli_card, corner=DL)

        self.play(Create(line_deg), FadeIn(deg_target_dot), FadeIn(deg_proj_dot), Create(deg_res_line), run_time=1.1)
        self.wait(3.2)

        # 4C: Oblique vs Orthogonal Contrast
        self.play(
            FadeOut(line_deg), FadeOut(deg_target_dot), FadeOut(deg_proj_dot), FadeOut(deg_res_line),
            run_time=0.5,
        )
        self.unpin_hud(rank_card, lauchli_card, run_time=0.35)

        y_ob_proj = P_OB @ Y_OB_TEST
        y_orth_proj = P_2D @ Y_OB_TEST
        ob_target_dot = Dot3D(point=axes.c2p(*Y_OB_TEST), radius=0.09, color=TARGET_COLOR)
        ob_proj_dot = Dot3D(point=axes.c2p(*y_ob_proj), radius=0.09, color=ERROR_COLOR)
        orth_proj_dot = Dot3D(point=axes.c2p(*y_orth_proj), radius=0.09, color=PROJ_COLOR)

        ob_ray = safe_line_3d(axes, Y_OB_TEST, y_ob_proj, ERROR_COLOR, thickness=0.025)
        orth_ray = safe_line_3d(axes, Y_OB_TEST, y_orth_proj, PROJ_COLOR, thickness=0.025)
        orth_elbow_ob = create_3d_right_angle(axes, y_orth_proj, -y_orth_proj, Y_OB_TEST - y_orth_proj, size=0.25)

        oblique_card = make_hud_panel(
            VGroup(
                Text("Oblique vs. Orthogonal Projector", font_size=14, color=ERROR_COLOR, weight=BOLD),
                MathTex(r"P_{\text{ob}}^2 = P_{\text{ob}},\ P_{\text{ob}}^\top \ne P_{\text{ob}},\ \|P_{\text{ob}}\|_2 = \sqrt{6} > 1", font_size=14, color=TEXT_COLOR),
                MathTex(r"\|\mathbf{r}_{\text{ob}}\|_2 = 2.0 > \|\mathbf{r}_{\text{orth}}\|_2 = \frac{\sqrt{6}}{3} \approx 0.816", font_size=14, color=PROJ_COLOR),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        )
        self.pin_hud(oblique_card, corner=DR)

        self.play(
            FadeIn(ob_target_dot), FadeIn(ob_proj_dot), FadeIn(orth_proj_dot),
            Create(ob_ray), Create(orth_ray), Create(orth_elbow_ob),
            run_time=1.3,
        )
        self.wait(4.8)

        # Clean up Act 4
        self.unpin_hud(title_panel, oblique_card, run_time=0.5)
        self.play(
            FadeOut(plane_u), FadeOut(ob_target_dot), FadeOut(ob_proj_dot), FadeOut(orth_proj_dot),
            FadeOut(ob_ray), FadeOut(orth_ray), FadeOut(orth_elbow_ob),
            FadeOut(axes), FadeOut(origin_dot),
            run_time=0.7,
        )


# -----------------------------------------------------------------------------
# Individual Act Executable Scenes (for Rapid Modular Testing & AST Verification)
# -----------------------------------------------------------------------------
class Act1SpatialPuzzle(BaseOrthogonalProjectionsScene):
    """Executes Act 1: 1D Line Projection & Idempotence."""
    def construct(self):
        self.run_act1()


class Act2DynamicDeformation(BaseOrthogonalProjectionsScene):
    """Executes Act 2: Sliding Candidate & Pinned 2D Corner Error Graph."""
    def construct(self):
        self.run_act2()


class Act3AlgebraicMorph(BaseOrthogonalProjectionsScene):
    """Executes Act 3: Normal Equations, Projector P, I-P & Pythagoras."""
    def construct(self):
        self.run_act3()


class Act4DegenerateStressTest(BaseOrthogonalProjectionsScene):
    """Executes Act 4: Affine, Rank-Deficient, Lauchli & Oblique Edge Cases."""
    def construct(self):
        self.run_act4()


# -----------------------------------------------------------------------------
# Master Unified Scene (Full 4-Act Master Curriculum Production)
# -----------------------------------------------------------------------------
class OrthogonalProjectionsScene(BaseOrthogonalProjectionsScene):
    """Master production scene rendering the complete 4-Act Sprint 6 module."""
    def construct(self):
        self.run_act1()
        self.run_act2()
        self.run_act3()
        self.run_act4()
