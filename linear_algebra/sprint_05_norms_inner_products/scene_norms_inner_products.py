"""
Sprint 5: Norms (L1, L2, L_inf), Inner Products & Cauchy-Schwarz (ManimCE v0.18+)
Curriculum: Linear Algebra & Theoretical CS / Machine Learning Foundations
References: MML Ch. 3.1-3.4 & 3.8.1, VMLS Ch. 3-4, ILA Ch. 3, Kreyszig Ch. 7-8

This production script implements the 4-act pedagogical narrative:
- Act 1: Concrete Spatial Puzzle (L1=7, L2=5, L_inf=4 on x=[3,4] & y-Ruler Preview)
- Act 2: Dynamic Deformation (Continuous Lp Unit Ball p=1->2->inf & Signed Dot-Product Sweep)
- Act 3: Algebraic Morph (Cauchy-Schwarz cos(omega) Gauge & SPD A-Ellipsoid Inner Product)
- Act 4: Degenerate Stress Test (p=0.5 Star, L1 Parallelogram Failure, Zero Vector, Indefinite A)

Render Commands:
    Preview Low-Res:    .venv/bin/manim -pql linear_algebra/sprint_05_norms_inner_products/scene_norms_inner_products.py NormsInnerProductsScene
    Headless Low-Res:   .venv/bin/manim -ql  linear_algebra/sprint_05_norms_inner_products/scene_norms_inner_products.py NormsInnerProductsScene
    Preview Hi-Res:     .venv/bin/manim -pqh linear_algebra/sprint_05_norms_inner_products/scene_norms_inner_products.py NormsInnerProductsScene
    Production Hi-Res:  .venv/bin/manim -qh  linear_algebra/sprint_05_norms_inner_products/scene_norms_inner_products.py NormsInnerProductsScene
    Single Act Test:    .venv/bin/manim -ql  linear_algebra/sprint_05_norms_inner_products/scene_norms_inner_products.py Act1SpatialPuzzle
"""

from __future__ import annotations

import numpy as np
from manim import *

# -----------------------------------------------------------------------------
# Rule 1: 3Blue1Brown Semantic Color Palette
# -----------------------------------------------------------------------------
I_HAT_COLOR = "#83C167"       # Green C    - Vector y / L1 taxicab path / positive projection
J_HAT_COLOR = "#FC6255"       # Red C      - Vector v / L_inf Chebyshev bar
VEC_COLOR = "#F7D96F"         # Yellow     - Primary vector x / Euclidean L2 hypotenuse
TARGET_COLOR = "#F9E498"      # Gold A     - Projection tip proj_y(x) / Cauchy-Schwarz equality
SUBSPACE_COLOR = "#2A9D8F"    # Teal       - SPD A-unit ellipse & projection span
LINE_COLOR = "#58C4DD"        # Blue C     - Morphing Lp unit ball ||x||_p = 1 & y-ruler
GHOST_COLOR = "#414868"       # Slate Gray - Ghost Cartesian plane & inscribed/circumscribed bounds
EIGEN_GRID_COLOR = "#E0AF68"  # Amber      - Angle arc omega & cos(omega) gauge
ORIGIN_COLOR = "#FFFFFF"      # White      - Origin anchor 0
ERROR_COLOR = "#F7768E"       # Soft Red   - Negative projection, p=0.5 star, L1/indef failures
BG_COLOR = "#1C1C24"          # Dark Slate - High-contrast dark viewport background
PANEL_BG = "#16161E"          # Obsidian   - HUD panel container background fill
PANEL_BORDER = "#414868"      # Slate Teal - HUD panel container outline stroke
TEXT_COLOR = "#ECEFF4"        # Snow       - High-contrast primary typography
MUTED_TEXT = "#7AA2F7"        # Light Blue - Secondary badges & annotations

# -----------------------------------------------------------------------------
# Harmonized Mathematical Constants & Pre-Flight Assertions (Rule 6)
# -----------------------------------------------------------------------------
X_PUZZLE = np.array([3.0, 4.0])
Y_PUZZLE = np.array([2.0, 1.0])
PROJ_PUZZLE = (float(np.dot(X_PUZZLE, Y_PUZZLE)) / float(np.dot(Y_PUZZLE, Y_PUZZLE))) * Y_PUZZLE
RESIDUAL_PUZZLE = X_PUZZLE - PROJ_PUZZLE

X_VEC = np.array([1.0, 2.0])
Y_VEC = np.array([2.0, 1.0])
PROJ_VEC = (float(np.dot(X_VEC, Y_VEC)) / float(np.dot(Y_VEC, Y_VEC))) * Y_VEC
RESIDUAL_VEC = X_VEC - PROJ_VEC

Y_RULER = np.array([1.6, 1.2])  # ||y||_2 = 2.0, tilted atan2(1.2, 1.6) = 36.87 deg
U_RULER = Y_RULER / np.linalg.norm(Y_RULER)
THETA_RULER = float(np.arctan2(Y_RULER[1], Y_RULER[0]))
X_SWEEP_NORM = 2.5

A_SPD = np.array([[2.0, 1.0], [1.0, 2.0]])
U_A_ORTH = np.array([1.0, 0.0])
V_A_ORTH = np.array([-1.0, 2.0])

U_DEG = np.array([1.0, 0.0])
V_DEG = np.array([0.0, 1.0])
QUASI_P = 0.5
A_INDEF = np.array([[1.0, 0.0], [0.0, -1.0]])
X_INDEF = np.array([1.0, 2.0])

# Pre-flight mathematical consistency verification
assert np.isclose(np.linalg.norm(X_PUZZLE, ord=1), 7.0), "||X_PUZZLE||_1 must be 7.0"
assert np.isclose(np.linalg.norm(X_PUZZLE, ord=2), 5.0), "||X_PUZZLE||_2 must be 5.0"
assert np.isclose(np.linalg.norm(X_PUZZLE, ord=np.inf), 4.0), "||X_PUZZLE||_inf must be 4.0"
assert np.isclose(float(np.dot(X_PUZZLE, Y_PUZZLE)), 10.0), "X_PUZZLE^T Y_PUZZLE must be 10.0"
assert np.allclose(PROJ_PUZZLE, np.array([4.0, 2.0])), "PROJ_PUZZLE must be [4.0, 2.0]"
assert np.allclose(RESIDUAL_PUZZLE, np.array([-1.0, 2.0])), "RESIDUAL_PUZZLE must be [-1.0, 2.0]"
assert np.isclose(float(np.dot(RESIDUAL_PUZZLE, Y_PUZZLE)), 0.0), "Residual must be orthogonal to Y_PUZZLE"

assert np.isclose(float(np.dot(X_VEC, Y_VEC)), 4.0), "X_VEC^T Y_VEC must be 4.0"
assert np.isclose(np.linalg.norm(X_VEC, ord=2) * np.linalg.norm(Y_VEC, ord=2), 5.0), "||X_VEC||*||Y_VEC|| must be 5.0"
assert np.isclose(float(np.dot(X_VEC, Y_VEC)) / (np.linalg.norm(X_VEC) * np.linalg.norm(Y_VEC)), 0.8), "cos(omega) must be 0.8"
assert np.allclose(PROJ_VEC, np.array([1.6, 0.8])), "PROJ_VEC must be [1.6, 0.8]"
assert np.isclose(float(np.dot(RESIDUAL_VEC, Y_VEC)), 0.0), "RESIDUAL_VEC must be orthogonal to Y_VEC"

assert np.isclose(np.linalg.norm(Y_RULER, ord=2), 2.0), "||Y_RULER||_2 must be 2.0"
assert np.allclose(A_SPD, A_SPD.T) and np.all(np.linalg.eigvalsh(A_SPD) > 0.0), "A_SPD must be symmetric positive definite"
assert np.isclose(float(U_A_ORTH @ A_SPD @ V_A_ORTH), 0.0), "u and v must be A-orthogonal"
assert not np.isclose(float(U_A_ORTH @ V_A_ORTH), 0.0), "u and v must not be Euclidean-orthogonal"
assert np.isclose(float(np.sum(np.abs(U_DEG + V_DEG) ** QUASI_P) ** (1.0 / QUASI_P)), 4.0), "p=0.5 triangle inequality failure"
assert not np.isclose(
    np.linalg.norm(U_DEG + V_DEG, ord=1) ** 2 + np.linalg.norm(U_DEG - V_DEG, ord=1) ** 2,
    2.0 * (np.linalg.norm(U_DEG, ord=1) ** 2 + np.linalg.norm(V_DEG, ord=1) ** 2),
), "L1 must violate parallelogram law"
assert np.isclose(float(X_INDEF @ A_INDEF @ X_INDEF), -3.0) and float(X_INDEF @ A_INDEF @ X_INDEF) < 0.0, "A_INDEF must fail positivity"


# -----------------------------------------------------------------------------
# Rule 3 & Geometry Helpers
# -----------------------------------------------------------------------------
def snap_zero(val: float, tol: float = 1e-9) -> float:
    """Clamp tiny near-zero floats to +0.0 to avoid '-0.00' DecimalNumber glitches."""
    return 0.0 if abs(val) < tol else float(val)


def make_hud_panel(
    content: Mobject,
    padding: float = 0.18,
    bg_color: str = PANEL_BG,
    bg_opacity: float = 0.88,
    stroke_color: str = PANEL_BORDER,
    stroke_width: float = 1.5,
    corner_radius: float = 0.12,
    min_width: float = 0.0,
) -> VGroup:
    """Wrap any mobject inside a high-contrast rounded HUD panel with z-index 10."""
    bg = RoundedRectangle(
        corner_radius=corner_radius,
        width=max(content.width + 2 * padding, min_width),
        height=content.height + 2 * padding,
        fill_color=bg_color,
        fill_opacity=bg_opacity,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
    )
    bg.move_to(content.get_center())
    bg.set_z_index(10)
    content.set_z_index(11)
    panel = VGroup(bg, content)
    return panel


def lp_unit_ball_points(plane: NumberPlane, p: float, n_pts: int = 241) -> list[np.ndarray]:
    """Compute overflow-safe closed polygon vertices for ||x||_p = 1 in scene coords."""
    thetas = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=True)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    abs_c = np.abs(cos_t)
    abs_s = np.abs(sin_t)
    m = np.maximum(abs_c, abs_s)
    q = np.minimum(abs_c, abs_s) / m
    if np.isinf(p) or p > 1e4:
        r = 1.0 / m
    else:
        r = 1.0 / (m * ((1.0 + q ** p) ** (1.0 / p)))
    xs = r * cos_t
    ys = r * sin_t
    return [plane.c2p(float(x), float(y)) for x, y in zip(xs, ys)]


def spd_unit_ellipse_points(plane: NumberPlane, mat: np.ndarray, n_pts: int = 241) -> list[np.ndarray]:
    """Compute closed polygon vertices for x^T M x = 1 in scene coordinates."""
    thetas = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=True)
    u = np.column_stack([np.cos(thetas), np.sin(thetas)])
    quad = np.einsum("ij,jk,ik->i", u, mat, u)
    r = 1.0 / np.sqrt(np.maximum(quad, 1e-12))
    pts = u * r[:, None]
    return [plane.c2p(float(x), float(y)) for x, y in pts]


def safe_right_angle(
    plane: NumberPlane,
    y_dir_math: np.ndarray,
    x_pt: np.ndarray,
    proj_pt: np.ndarray,
    proj_scalar: float,
    length: float = 0.22,
) -> VMobject:
    """Build a crash-free RightAngle elbow between ruler line and residual drop line."""
    d_perp = float(np.linalg.norm(x_pt[:2] - proj_pt[:2]))
    if d_perp < 0.15:
        return VMobject()
    u_y_scene = plane.c2p(float(y_dir_math[0]), float(y_dir_math[1])) - plane.c2p(0.0, 0.0)
    u_norm = float(np.linalg.norm(u_y_scene[:2]))
    if u_norm < 1e-8:
        return VMobject()
    u_y_scene = u_y_scene / u_norm
    sgn = 1.0 if proj_scalar >= 0.0 else -1.0
    ref_pt = proj_pt - sgn * u_y_scene
    line1 = Line(proj_pt, ref_pt)
    line2 = Line(proj_pt, x_pt)
    return RightAngle(line1, line2, length=length, quadrant=(1, 1), color=TARGET_COLOR, stroke_width=2.0)


# -----------------------------------------------------------------------------
# Base Scene Class (Shared Setup, Planes, Vector Helpers, 4-Act Runners)
# -----------------------------------------------------------------------------
class BaseNormsInnerProductsScene(Scene):
    """Shared scene infrastructure and 4-act orchestration for Sprint 5."""

    def setup(self) -> None:
        self.camera.background_color = BG_COLOR

    def create_plane(self) -> NumberPlane:
        return NumberPlane(
            x_range=[-6, 6, 1],
            y_range=[-4, 4, 1],
            x_length=9.6,
            y_length=6.4,
            background_line_style={
                "stroke_color": GHOST_COLOR,
                "stroke_width": 1.0,
                "stroke_opacity": 0.45,
            },
            axis_config={
                "stroke_color": TEXT_COLOR,
                "stroke_width": 1.6,
                "stroke_opacity": 0.75,
            },
        )

    def create_origin_anchor(self, plane: NumberPlane) -> VGroup:
        dot = Dot(plane.c2p(0.0, 0.0), radius=0.06, color=ORIGIN_COLOR, z_index=8)
        lbl = MathTex(r"\mathbf{0}", font_size=24, color=ORIGIN_COLOR).next_to(dot, DL, buff=0.08)
        lbl.set_z_index(8)
        return VGroup(dot, lbl)

    def create_vector_arrow(
        self,
        plane: NumberPlane,
        start_coords: np.ndarray,
        end_coords: np.ndarray,
        color: str,
        stroke_width: float = 3.5,
    ) -> Arrow:
        return Arrow(
            plane.c2p(float(start_coords[0]), float(start_coords[1])),
            plane.c2p(float(end_coords[0]), float(end_coords[1])),
            buff=0.0,
            color=color,
            stroke_width=stroke_width,
            max_tip_length_to_length_ratio=0.14,
            max_stroke_width_to_length_ratio=6.0,
        )

    # -------------------------------------------------------------------------
    # Act 1: Concrete Spatial Puzzle
    # -------------------------------------------------------------------------
    def run_act1(
        self,
        plane: NumberPlane,
        origin_group: VGroup,
        is_standalone: bool = False,
    ) -> VGroup:
        title = Text("Act 1: Spatial Puzzle — Three Ways to Measure Length", font_size=22, color=TEXT_COLOR)
        title_panel = make_hud_panel(title, padding=0.14).to_corner(UL, buff=0.25)

        x_arrow = self.create_vector_arrow(plane, np.zeros(2), X_PUZZLE, VEC_COLOR, stroke_width=4.0)
        x_tip = plane.c2p(float(X_PUZZLE[0]), float(X_PUZZLE[1]))
        x_lbl = MathTex(r"\mathbf{x} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}", font_size=24, color=VEC_COLOR).next_to(
            x_arrow.get_center(), UL, buff=0.12
        )

        self.play(FadeIn(title_panel, shift=DOWN * 0.15), GrowArrow(x_arrow), Write(x_lbl), run_time=1.3)
        self.wait(0.4)

        # Taxicab L1 path: (0,0) -> (3,0) -> (3,4)
        l1_h = Line(plane.c2p(0.0, 0.0), plane.c2p(3.0, 0.0), color=I_HAT_COLOR, stroke_width=4.2)
        l1_v = Line(plane.c2p(3.0, 0.0), plane.c2p(3.0, 4.0), color=I_HAT_COLOR, stroke_width=4.2)
        l1_lbl_h = MathTex("3", font_size=24, color=I_HAT_COLOR).next_to(l1_h, DOWN, buff=0.10)
        l1_lbl_v = MathTex("4", font_size=24, color=I_HAT_COLOR).next_to(l1_v, RIGHT, buff=0.10)

        # L2 hypotenuse label
        l2_lbl = MathTex(r"\|\mathbf{x}\|_2 = 5", font_size=24, color=VEC_COLOR).move_to(
            plane.c2p(1.15, 2.35)
        )

        # Chebyshev L_inf bar: (0,0) -> (0,4) + dashed guide (0,4) -> (3,4)
        linf_bar = Line(plane.c2p(0.0, 0.0), plane.c2p(0.0, 4.0), color=J_HAT_COLOR, stroke_width=5.0)
        linf_guide = DashedLine(
            plane.c2p(0.0, 4.0), plane.c2p(3.0, 4.0), color=J_HAT_COLOR, stroke_width=2.2, dash_length=0.12
        )
        linf_lbl = MathTex(r"\max = 4", font_size=24, color=J_HAT_COLOR).next_to(linf_bar, LEFT, buff=0.12)

        # UR compact norm HUD (x >= 4.0 clears [3,4] tip at x=2.4)
        norm_tex = MathTex(
            r"\|\mathbf{x}\|_1 = 7", r"\;\|\mathbf{x}\|_2 = 5", r"\;\|\mathbf{x}\|_\infty = 4",
            font_size=22,
        )
        norm_tex[0].set_color(I_HAT_COLOR)
        norm_tex[1].set_color(VEC_COLOR)
        norm_tex[2].set_color(J_HAT_COLOR)
        norm_tex.arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        ur_panel = make_hud_panel(norm_tex, padding=0.14).to_corner(UR, buff=0.25)

        order_tex = MathTex(
            r"\|\mathbf{x}\|_\infty \le \|\mathbf{x}\|_2 \le \|\mathbf{x}\|_1 \quad (\text{rms}=\tfrac{5}{\sqrt{2}})",
            font_size=22,
            color=MUTED_TEXT,
        )
        dl_panel = make_hud_panel(order_tex, padding=0.14).to_corner(DL, buff=0.25)

        self.play(
            Create(l1_h),
            Create(l1_v),
            FadeIn(l1_lbl_h),
            FadeIn(l1_lbl_v),
            run_time=1.3,
        )
        self.play(FadeIn(l2_lbl), Create(linf_bar), Create(linf_guide), FadeIn(linf_lbl), run_time=1.2)
        self.play(FadeIn(ur_panel, shift=LEFT * 0.15), FadeIn(dl_panel, shift=UP * 0.15), run_time=0.9)
        self.wait(1.0)

        # Dim L1/L_inf guides and preview dot product as ruler projection onto y=[2,1]
        self.play(
            l1_h.animate.set_stroke(opacity=0.22),
            l1_v.animate.set_stroke(opacity=0.22),
            linf_bar.animate.set_stroke(opacity=0.22),
            linf_guide.animate.set_stroke(opacity=0.20),
            FadeOut(l1_lbl_h),
            FadeOut(l1_lbl_v),
            FadeOut(l2_lbl),
            FadeOut(linf_lbl),
            run_time=0.8,
        )

        # Ruler line along Y_PUZZLE = [2, 1]
        u_y_puz = Y_PUZZLE / np.linalg.norm(Y_PUZZLE)
        ruler_line = Line(
            plane.c2p(*(-5.5 * u_y_puz)),
            plane.c2p(*(5.5 * u_y_puz)),
            color=LINE_COLOR,
            stroke_width=2.2,
            stroke_opacity=0.65,
        )
        tick_marks = VGroup()
        perp_u = np.array([-u_y_puz[1], u_y_puz[0]])
        for k in (-2, -1, 1, 2):
            base = k * Y_PUZZLE
            t0 = plane.c2p(*(base - 0.12 * perp_u))
            t1 = plane.c2p(*(base + 0.12 * perp_u))
            tick_marks.add(Line(t0, t1, color=LINE_COLOR, stroke_width=2.0))

        y_arrow = self.create_vector_arrow(plane, np.zeros(2), Y_PUZZLE, I_HAT_COLOR, stroke_width=3.8)
        y_lbl = MathTex(r"\mathbf{y} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}", font_size=24, color=I_HAT_COLOR).next_to(
            plane.c2p(float(Y_PUZZLE[0]), float(Y_PUZZLE[1])), DOWN, buff=0.14
        )

        proj_pt = plane.c2p(float(PROJ_PUZZLE[0]), float(PROJ_PUZZLE[1]))
        proj_bar = Line(plane.c2p(0.0, 0.0), proj_pt, color=TARGET_COLOR, stroke_width=6.0)
        proj_dot = Dot(proj_pt, radius=0.07, color=TARGET_COLOR, z_index=7)
        proj_lbl = MathTex(r"\operatorname{proj}_{\mathbf{y}}(\mathbf{x}) = 2\mathbf{y}", font_size=22, color=TARGET_COLOR).next_to(
            proj_dot, DR, buff=0.10
        )
        drop_line = DashedLine(x_tip, proj_pt, color=TARGET_COLOR, stroke_width=2.2, dash_length=0.10)
        ra = safe_right_angle(plane, Y_PUZZLE, x_tip, proj_pt, proj_scalar=2.0, length=0.22)

        dr_tex = MathTex(
            r"\mathbf{x}^\top \mathbf{y} = 10 = 2\|\mathbf{y}\|_2^2",
            font_size=23,
            color=TARGET_COLOR,
        )
        dr_panel = make_hud_panel(dr_tex, padding=0.14).to_corner(DR, buff=0.25)

        self.play(Create(ruler_line), Create(tick_marks), GrowArrow(y_arrow), Write(y_lbl), run_time=1.1)
        self.play(Create(drop_line), Create(ra), Create(proj_bar), FadeIn(proj_dot), Write(proj_lbl), run_time=1.2)
        self.play(FadeIn(dr_panel, shift=LEFT * 0.15), run_time=0.8)
        self.wait(1.6)

        cleanup_group = VGroup(
            title_panel,
            ur_panel,
            dl_panel,
            dr_panel,
            x_arrow,
            x_lbl,
            l1_h,
            l1_v,
            linf_bar,
            linf_guide,
            ruler_line,
            tick_marks,
            y_arrow,
            y_lbl,
            proj_bar,
            proj_dot,
            proj_lbl,
            drop_line,
            ra,
        )
        if is_standalone:
            self.play(FadeOut(cleanup_group), run_time=0.7)
            return VGroup()
        return cleanup_group

    # -------------------------------------------------------------------------
    # Act 2: Continuous Lp Unit Ball Morph & Signed Dot-Product Ruler Sweep
    # -------------------------------------------------------------------------
    def run_act2(
        self,
        plane: NumberPlane,
        origin_group: VGroup,
        is_standalone: bool = False,
    ) -> VGroup:
        title = Text("Act 2: Lp Unit Ball Morph (p = 1 → 2 → ∞)", font_size=22, color=TEXT_COLOR)
        title_panel = make_hud_panel(title, padding=0.14).to_corner(UL, buff=0.25)

        # Ghost bounds: L1 diamond and L_inf square
        ghost_l1 = VMobject(stroke_color=GHOST_COLOR, stroke_width=2.0, stroke_opacity=0.55)
        ghost_l1.set_points_as_corners(lp_unit_ball_points(plane, 1.0, 241))
        ghost_linf = VMobject(stroke_color=GHOST_COLOR, stroke_width=2.0, stroke_opacity=0.55)
        ghost_linf.set_points_as_corners(lp_unit_ball_points(plane, np.inf, 241))

        pinned_dots = VGroup(
            *[
                Dot(plane.c2p(float(cx), float(cy)), radius=0.055, color=TARGET_COLOR, z_index=7)
                for cx, cy in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
            ]
        )

        # Track alpha = 1/p in [0, 1]
        alpha_tracker = ValueTracker(1.0)

        def _build_ball() -> VMobject:
            alpha = alpha_tracker.get_value()
            p_val = np.inf if alpha <= 1e-4 else 1.0 / alpha
            mob = VMobject(
                stroke_color=LINE_COLOR,
                stroke_width=4.2,
                fill_color=LINE_COLOR,
                fill_opacity=0.22,
            )
            mob.set_points_as_corners(lp_unit_ball_points(plane, p_val, 241))
            return mob

        def _build_diag_dot() -> Dot:
            alpha = alpha_tracker.get_value()
            coord = 2.0 ** (-alpha)
            return Dot(plane.c2p(coord, coord), radius=0.065, color=VEC_COLOR, z_index=8)

        ball_mob = always_redraw(_build_ball)
        diag_dot = always_redraw(_build_diag_dot)

        p_label = MathTex(r"p =", font_size=26, color=TEXT_COLOR)
        p_num = DecimalNumber(20.00, num_decimal_places=2, font_size=26, color=LINE_COLOR, edge_to_fix=LEFT)
        p_num.set_value(1.00)
        p_row = VGroup(p_label, p_num).arrange(RIGHT, buff=0.12)
        ur_p_panel = make_hud_panel(p_row, padding=0.14).to_corner(UR, buff=0.25)

        def _update_p_num(m: DecimalNumber) -> None:
            m.set_value(min(20.0, 1.0 / max(alpha_tracker.get_value(), 0.05)))
            m.set_z_index(12)

        p_num.set_z_index(12)
        p_num.add_updater(_update_p_num)

        dl_badge_1 = make_hud_panel(
            Text("L1 Diamond: Taxicab Sparsity Corners", font_size=20, color=I_HAT_COLOR),
            padding=0.14,
        ).to_corner(DL, buff=0.25)

        self.play(
            FadeIn(title_panel, shift=DOWN * 0.15),
            Create(ghost_l1),
            Create(ghost_linf),
            FadeIn(pinned_dots),
            FadeIn(ball_mob),
            FadeIn(diag_dot),
            FadeIn(ur_p_panel),
            FadeIn(dl_badge_1),
            run_time=1.1,
        )
        self.wait(0.6)

        # Morph p=1 -> p=2 (alpha: 1.0 -> 0.5)
        dl_badge_2 = make_hud_panel(
            Text("L2 Circle: Unique Rotation-Invariant Ball", font_size=20, color=VEC_COLOR),
            padding=0.14,
        ).to_corner(DL, buff=0.25)
        self.play(alpha_tracker.animate.set_value(0.5), ReplacementTransform(dl_badge_1, dl_badge_2), run_time=2.4, rate_func=smooth)
        self.wait(0.9)

        # Morph p=2 -> p=20 -> p=inf (alpha: 0.5 -> 0.05 -> 0.0)
        dl_badge_3 = make_hud_panel(
            Text("L_inf Square: Chebyshev Box (B1 ⊆ B2 ⊆ B_inf)", font_size=20, color=J_HAT_COLOR),
            padding=0.14,
        ).to_corner(DL, buff=0.25)
        self.play(alpha_tracker.animate.set_value(0.05), ReplacementTransform(dl_badge_2, dl_badge_3), run_time=2.0, rate_func=smooth)

        p_num.clear_updaters()
        p_inf_row = MathTex(r"p = \infty \;\text{(Chebyshev Square)}", font_size=23, color=LINE_COLOR)
        ur_inf_panel = make_hud_panel(p_inf_row, padding=0.14).to_corner(UR, buff=0.25)
        self.play(
            alpha_tracker.animate.set_value(0.0),
            ReplacementTransform(ur_p_panel, ur_inf_panel),
            run_time=1.0,
            rate_func=smooth,
        )
        self.wait(1.0)

        # Explicitly clear Part A updaters & remove before Part B
        ball_mob.clear_updaters()
        diag_dot.clear_updaters()
        part_a_objs = VGroup(ghost_l1, ghost_linf, pinned_dots, ball_mob, diag_dot, ur_inf_panel, dl_badge_3)
        self.play(FadeOut(part_a_objs), run_time=0.6)
        self.remove(ball_mob, diag_dot)

        # Part B: Live Dot-Product Ruler Sweep
        title_b = Text("Act 2B: Dot Product as Signed Ruler Projection", font_size=22, color=TEXT_COLOR)
        title_b_panel = make_hud_panel(title_b, padding=0.14).to_corner(UL, buff=0.25)

        ruler_line = Line(
            plane.c2p(*(-5.5 * U_RULER)),
            plane.c2p(*(5.5 * U_RULER)),
            color=LINE_COLOR,
            stroke_width=2.2,
            stroke_opacity=0.65,
        )
        y_arrow = self.create_vector_arrow(plane, np.zeros(2), Y_RULER, I_HAT_COLOR, stroke_width=4.0)
        y_lbl = MathTex(r"\mathbf{y} = [1.6,\, 1.2]^\top\;(\|\mathbf{y}\|_2=2)", font_size=21, color=I_HAT_COLOR).next_to(
            plane.c2p(float(Y_RULER[0]), float(Y_RULER[1])), DR, buff=0.12
        )

        omega_tracker = ValueTracker(0.0)
        origin_pt = plane.c2p(0.0, 0.0)

        def _x_math() -> np.ndarray:
            ang = THETA_RULER + omega_tracker.get_value()
            return X_SWEEP_NORM * np.array([np.cos(ang), np.sin(ang)])

        dyn_x = always_redraw(
            lambda: self.create_vector_arrow(plane, np.zeros(2), _x_math(), VEC_COLOR, stroke_width=4.0)
        )

        def _build_proj() -> VMobject:
            xm = _x_math()
            s = float(np.dot(xm, U_RULER))
            if abs(s) < 0.04:
                return VMobject()
            p_math = s * U_RULER
            col = I_HAT_COLOR if s > 0.0 else ERROR_COLOR
            return Line(origin_pt, plane.c2p(float(p_math[0]), float(p_math[1])), color=col, stroke_width=6.0)

        def _build_drop() -> VMobject:
            xm = _x_math()
            s = float(np.dot(xm, U_RULER))
            p_math = s * U_RULER
            if float(np.linalg.norm(xm - p_math)) < 0.05:
                return VMobject()
            return DashedLine(
                plane.c2p(float(xm[0]), float(xm[1])),
                plane.c2p(float(p_math[0]), float(p_math[1])),
                color=TARGET_COLOR,
                stroke_width=2.0,
                dash_length=0.10,
            )

        def _build_ra() -> VMobject:
            xm = _x_math()
            s = float(np.dot(xm, U_RULER))
            p_math = s * U_RULER
            return safe_right_angle(
                plane,
                Y_RULER,
                plane.c2p(float(xm[0]), float(xm[1])),
                plane.c2p(float(p_math[0]), float(p_math[1])),
                proj_scalar=s,
                length=0.20,
            )

        def _build_arc() -> VMobject:
            w = omega_tracker.get_value()
            if abs(w) < 0.08:
                return VMobject()
            return Arc(
                radius=0.55,
                start_angle=THETA_RULER,
                angle=w,
                arc_center=origin_pt,
                color=EIGEN_GRID_COLOR,
                stroke_width=2.6,
            )

        dyn_proj = always_redraw(_build_proj)
        dyn_drop = always_redraw(_build_drop)
        dyn_ra = always_redraw(_build_ra)
        dyn_arc = always_redraw(_build_arc)

        cos_lbl = MathTex(r"\cos\omega =", font_size=23, color=TEXT_COLOR)
        cos_num = DecimalNumber(-5.00, num_decimal_places=2, font_size=23, color=EIGEN_GRID_COLOR, edge_to_fix=LEFT)
        cos_num.set_value(1.00)
        row_cos = VGroup(cos_lbl, cos_num).arrange(RIGHT, buff=0.10)

        dot_lbl = MathTex(r"\mathbf{x}^\top\mathbf{y} =", font_size=23, color=TEXT_COLOR)
        dot_num = DecimalNumber(-5.00, num_decimal_places=2, font_size=23, color=TARGET_COLOR, edge_to_fix=LEFT)
        dot_num.set_value(5.00)
        row_dot = VGroup(dot_lbl, dot_num).arrange(RIGHT, buff=0.10)

        ur_b_content = VGroup(row_cos, row_dot).arrange(DOWN, aligned_edge=LEFT, buff=0.10)
        ur_b_panel = make_hud_panel(ur_b_content, padding=0.15).to_corner(UR, buff=0.25)

        def _update_cos_num(m: DecimalNumber) -> None:
            m.set_value(snap_zero(float(np.cos(omega_tracker.get_value()))))
            m.set_z_index(12)

        def _update_dot_num(m: DecimalNumber) -> None:
            m.set_value(snap_zero(X_SWEEP_NORM * 2.0 * float(np.cos(omega_tracker.get_value()))))
            m.set_z_index(12)

        cos_num.set_z_index(12)
        dot_num.set_z_index(12)
        cos_num.add_updater(_update_cos_num)
        dot_num.add_updater(_update_dot_num)

        dl_b_panel = make_hud_panel(
            MathTex(r"\mathbf{x}^\top\mathbf{y} = \|\mathbf{x}\|_2 \|\mathbf{y}\|_2 \cos\omega", font_size=22, color=MUTED_TEXT),
            padding=0.14,
        ).to_corner(DL, buff=0.25)

        self.play(
            ReplacementTransform(title_panel, title_b_panel),
            Create(ruler_line),
            GrowArrow(y_arrow),
            Write(y_lbl),
            FadeIn(dyn_x),
            FadeIn(dyn_proj),
            FadeIn(dyn_drop),
            FadeIn(dyn_ra),
            FadeIn(dyn_arc),
            FadeIn(ur_b_panel),
            FadeIn(dl_b_panel),
            run_time=1.0,
        )

        # Sweep omega: 0 -> PI/3 -> PI/2 (orthogonal pause) -> PI (anti-parallel) -> arccos(0.8) (3-4-5 triangle)
        self.play(omega_tracker.animate.set_value(PI / 3.0), run_time=1.5, rate_func=smooth)
        self.play(omega_tracker.animate.set_value(PI / 2.0), run_time=1.1, rate_func=smooth)
        self.wait(0.7)  # 0.00 orthogonality pause
        self.play(omega_tracker.animate.set_value(PI), run_time=1.6, rate_func=smooth)
        self.wait(0.7)  # -5.00 anti-parallel pause
        self.play(omega_tracker.animate.set_value(float(np.arccos(0.8))), run_time=1.5, rate_func=smooth)
        self.wait(0.9)

        # Clean up updaters explicitly
        for mob in (dyn_x, dyn_proj, dyn_drop, dyn_ra, dyn_arc, cos_num, dot_num):
            mob.clear_updaters()

        cleanup_group = VGroup(
            title_b_panel,
            ruler_line,
            y_arrow,
            y_lbl,
            dyn_x,
            dyn_proj,
            dyn_drop,
            dyn_ra,
            dyn_arc,
            ur_b_panel,
            dl_b_panel,
        )
        if is_standalone:
            self.play(FadeOut(cleanup_group), run_time=0.7)
            self.remove(dyn_x, dyn_proj, dyn_drop, dyn_ra, dyn_arc)
            return VGroup()
        return cleanup_group

    # -------------------------------------------------------------------------
    # Act 3: Cauchy-Schwarz Gauge & SPD A-Inner Product Ellipsoid
    # -------------------------------------------------------------------------
    def run_act3(
        self,
        plane: NumberPlane,
        origin_group: VGroup,
        is_standalone: bool = False,
    ) -> VGroup:
        title_a = Text("Act 3A: Cauchy-Schwarz Inequality & Collinear Saturation", font_size=22, color=TEXT_COLOR)
        title_panel = make_hud_panel(title_a, padding=0.14).to_corner(UL, buff=0.25)

        circle_pts = [
            plane.c2p(X_SWEEP_NORM * float(np.cos(t)), X_SWEEP_NORM * float(np.sin(t)))
            for t in np.linspace(0.0, 2.0 * np.pi, 181, endpoint=True)
        ]
        circle_path = VMobject(stroke_color=GHOST_COLOR, stroke_width=2.0, stroke_opacity=0.6)
        circle_path.set_points_as_corners(circle_pts)
        bound_circle = DashedVMobject(circle_path, num_dashes=48)

        ruler_line = Line(
            plane.c2p(*(-5.5 * U_RULER)),
            plane.c2p(*(5.5 * U_RULER)),
            color=LINE_COLOR,
            stroke_width=2.0,
            stroke_opacity=0.5,
        )
        y_arrow = self.create_vector_arrow(plane, np.zeros(2), Y_RULER, I_HAT_COLOR, stroke_width=3.8)
        origin_pt = plane.c2p(0.0, 0.0)

        omega_tracker = ValueTracker(float(np.arccos(0.8)))

        def _x_math() -> np.ndarray:
            ang = THETA_RULER + omega_tracker.get_value()
            return X_SWEEP_NORM * np.array([np.cos(ang), np.sin(ang)])

        dyn_x = always_redraw(lambda: self.create_vector_arrow(plane, np.zeros(2), _x_math(), VEC_COLOR, stroke_width=3.8))
        dyn_proj = always_redraw(
            lambda: Line(
                origin_pt,
                plane.c2p(*(float(np.dot(_x_math(), U_RULER)) * U_RULER)),
                color=I_HAT_COLOR if np.dot(_x_math(), U_RULER) >= 0.0 else ERROR_COLOR,
                stroke_width=5.5,
            )
            if abs(float(np.dot(_x_math(), U_RULER))) > 0.05
            else VMobject()
        )
        dyn_drop = always_redraw(
            lambda: DashedLine(
                plane.c2p(*_x_math()),
                plane.c2p(*(float(np.dot(_x_math(), U_RULER)) * U_RULER)),
                color=TARGET_COLOR,
                stroke_width=2.0,
                dash_length=0.10,
            )
            if float(np.linalg.norm(_x_math() - float(np.dot(_x_math(), U_RULER)) * U_RULER)) > 0.05
            else VMobject()
        )
        dyn_ra = always_redraw(
            lambda: safe_right_angle(
                plane,
                Y_RULER,
                plane.c2p(*_x_math()),
                plane.c2p(*(float(np.dot(_x_math(), U_RULER)) * U_RULER)),
                proj_scalar=float(np.dot(_x_math(), U_RULER)),
                length=0.20,
            )
        )

        # UR canonical Cauchy-Schwarz identity
        cs_eq = MathTex(r"|\mathbf{x}^\top\mathbf{y}| \le \|\mathbf{x}\|_2 \|\mathbf{y}\|_2", font_size=24, color=TARGET_COLOR)
        cs_ex = MathTex(r"\text{Ex: } |\mathbf{x}^\top\mathbf{y}| = 4 < 5 \;\;(\cos\omega = 0.8)", font_size=20, color=MUTED_TEXT)
        ur_cs_panel = make_hud_panel(VGroup(cs_eq, cs_ex).arrange(DOWN, aligned_edge=LEFT, buff=0.08), padding=0.14).to_corner(
            UR, buff=0.25
        )

        # DR pinned cos(omega) in [-1, +1] gauge
        gauge_bar = Line(LEFT * 1.4, RIGHT * 1.4, color=TEXT_COLOR, stroke_width=3.0)
        tick_m1 = Line(DOWN * 0.10, UP * 0.10, color=ERROR_COLOR, stroke_width=2.5).move_to(gauge_bar.get_left())
        tick_0 = Line(DOWN * 0.08, UP * 0.08, color=GHOST_COLOR, stroke_width=2.0).move_to(gauge_bar.get_center())
        tick_p1 = Line(DOWN * 0.10, UP * 0.10, color=I_HAT_COLOR, stroke_width=2.5).move_to(gauge_bar.get_right())
        lbl_m1 = MathTex("-1", font_size=18, color=ERROR_COLOR).next_to(tick_m1, DOWN, buff=0.06)
        lbl_p1 = MathTex("+1", font_size=18, color=I_HAT_COLOR).next_to(tick_p1, DOWN, buff=0.06)
        cos_hdr = MathTex(r"\cos\omega =", font_size=21, color=TEXT_COLOR)
        cos_val = DecimalNumber(-1.00, num_decimal_places=2, font_size=21, color=EIGEN_GRID_COLOR, edge_to_fix=LEFT)
        cos_val.set_value(0.80)
        cos_row = VGroup(cos_hdr, cos_val).arrange(RIGHT, buff=0.10)
        gauge_group = VGroup(cos_row, VGroup(gauge_bar, tick_m1, tick_0, tick_p1, lbl_m1, lbl_p1)).arrange(DOWN, buff=0.14)
        dr_gauge_panel = make_hud_panel(gauge_group, padding=0.14).to_corner(DR, buff=0.25)

        gauge_marker = Dot(
            gauge_bar.get_center() + RIGHT * (1.4 * 0.80),
            radius=0.08,
            color=EIGEN_GRID_COLOR,
            z_index=12,
        )
        def _update_cos_val(m: DecimalNumber) -> None:
            m.set_value(snap_zero(float(np.cos(omega_tracker.get_value()))))
            m.set_z_index(12)

        cos_val.set_z_index(12)
        gauge_marker.add_updater(
            lambda m: m.move_to(gauge_bar.get_center() + RIGHT * (1.4 * float(np.cos(omega_tracker.get_value()))))
        )
        cos_val.add_updater(_update_cos_val)

        self.play(
            FadeIn(title_panel, shift=DOWN * 0.15),
            Create(bound_circle),
            Create(ruler_line),
            GrowArrow(y_arrow),
            FadeIn(dyn_x),
            FadeIn(dyn_proj),
            FadeIn(dyn_drop),
            FadeIn(dyn_ra),
            FadeIn(ur_cs_panel),
            FadeIn(dr_gauge_panel),
            FadeIn(gauge_marker),
            run_time=1.1,
        )

        # Saturate at +1 (collinear) and -1 (anti-parallel)
        self.play(omega_tracker.animate.set_value(0.0), run_time=1.3, rate_func=smooth)
        self.play(Indicate(gauge_marker, color=TARGET_COLOR), Indicate(y_arrow, color=TARGET_COLOR), run_time=0.6)
        self.play(omega_tracker.animate.set_value(PI), run_time=1.7, rate_func=smooth)
        self.play(Indicate(gauge_marker, color=ERROR_COLOR), run_time=0.6)
        self.play(omega_tracker.animate.set_value(PI / 3.0), run_time=1.0, rate_func=smooth)

        for mob in (dyn_x, dyn_proj, dyn_drop, dyn_ra, gauge_marker, cos_val):
            mob.clear_updaters()

        part_a_objs = VGroup(
            bound_circle,
            ruler_line,
            y_arrow,
            dyn_x,
            dyn_proj,
            dyn_drop,
            dyn_ra,
            ur_cs_panel,
            dr_gauge_panel,
            gauge_marker,
        )
        self.play(FadeOut(part_a_objs), run_time=0.6)
        self.remove(dyn_x, dyn_proj, dyn_drop, dyn_ra, gauge_marker)

        # Part B: General SPD Inner Product <x, y>_A = x^T A y
        title_b = Text("Act 3B: SPD Inner Product A-Ellipsoid & A-Orthogonality", font_size=22, color=TEXT_COLOR)
        title_b_panel = make_hud_panel(title_b, padding=0.14).to_corner(UL, buff=0.25)

        euclid_circle = VMobject(stroke_color=GHOST_COLOR, stroke_width=2.0, stroke_opacity=0.6)
        euclid_circle.set_points_as_corners(spd_unit_ellipse_points(plane, np.eye(2), 241))
        euclid_circle_dashed = DashedVMobject(euclid_circle, num_dashes=48)

        t_tracker = ValueTracker(0.0)
        dyn_ellipse = always_redraw(
            lambda: VMobject(
                stroke_color=SUBSPACE_COLOR,
                stroke_width=4.2,
                fill_color=SUBSPACE_COLOR,
                fill_opacity=0.20,
            ).set_points_as_corners(
                spd_unit_ellipse_points(
                    plane,
                    (1.0 - t_tracker.get_value()) * np.eye(2) + t_tracker.get_value() * A_SPD,
                    241,
                )
            )
        )

        u_arrow = self.create_vector_arrow(plane, np.zeros(2), U_A_ORTH, I_HAT_COLOR, stroke_width=4.0)
        v_arrow = self.create_vector_arrow(plane, np.zeros(2), V_A_ORTH, J_HAT_COLOR, stroke_width=4.0)
        u_lbl = MathTex(r"\mathbf{u}=[1,0]^\top", font_size=22, color=I_HAT_COLOR).next_to(
            plane.c2p(1.0, 0.0), DR, buff=0.10
        )
        v_lbl = MathTex(r"\mathbf{v}=[-1,2]^\top", font_size=22, color=J_HAT_COLOR).next_to(
            plane.c2p(-1.0, 2.0), UL, buff=0.10
        )

        spd_tex = VGroup(
            MathTex(r"A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \succ 0", font_size=22, color=SUBSPACE_COLOR),
            MathTex(r"\mathbf{u}^\top \mathbf{v} = -1 \ne 0", font_size=21, color=ERROR_COLOR),
            MathTex(r"\langle \mathbf{u}, \mathbf{v}\rangle_A = \mathbf{u}^\top A\mathbf{v} = 0", font_size=22, color=TARGET_COLOR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        ur_b_panel = make_hud_panel(spd_tex, padding=0.14).to_corner(UR, buff=0.25)

        dl_b_panel = make_hud_panel(
            MathTex(r"A = L L^\top \implies \|\mathbf{x}\|_A = \|L^\top \mathbf{x}\|_2", font_size=22, color=MUTED_TEXT),
            padding=0.14,
        ).to_corner(DL, buff=0.25)

        self.play(
            ReplacementTransform(title_panel, title_b_panel),
            Create(euclid_circle_dashed),
            FadeIn(dyn_ellipse),
            run_time=0.9,
        )
        self.play(t_tracker.animate.set_value(1.0), run_time=2.2, rate_func=smooth)
        self.play(
            GrowArrow(u_arrow),
            GrowArrow(v_arrow),
            Write(u_lbl),
            Write(v_lbl),
            FadeIn(ur_b_panel),
            FadeIn(dl_b_panel),
            run_time=1.3,
        )
        self.wait(1.6)

        dyn_ellipse.clear_updaters()
        cleanup_group = VGroup(
            title_b_panel,
            euclid_circle_dashed,
            dyn_ellipse,
            u_arrow,
            v_arrow,
            u_lbl,
            v_lbl,
            ur_b_panel,
            dl_b_panel,
        )
        if is_standalone:
            self.play(FadeOut(cleanup_group), run_time=0.7)
            self.remove(dyn_ellipse)
            return VGroup()
        return cleanup_group

    # -------------------------------------------------------------------------
    # Act 4: Degenerate Stress Tests & Master Summary Card
    # -------------------------------------------------------------------------
    def run_act4(
        self,
        plane: NumberPlane,
        origin_group: VGroup,
        is_standalone: bool = False,
    ) -> VGroup:
        title = Text("Act 4: Boundary Stress Tests — When Axioms Break", font_size=22, color=TEXT_COLOR)
        title_panel = make_hud_panel(title, padding=0.14).to_corner(UL, buff=0.25)
        self.play(FadeIn(title_panel, shift=DOWN * 0.15), run_time=0.8)

        # Failure 1: p = 0.5 Non-Convex Star Quasi-Norm
        l1_ref = VMobject(stroke_color=GHOST_COLOR, stroke_width=2.0, stroke_opacity=0.6)
        l1_ref.set_points_as_corners(lp_unit_ball_points(plane, 1.0, 241))
        l1_dashed = DashedVMobject(l1_ref, num_dashes=40)

        star_05 = VMobject(stroke_color=ERROR_COLOR, stroke_width=4.0, fill_color=ERROR_COLOR, fill_opacity=0.22)
        star_05.set_points_as_corners(lp_unit_ball_points(plane, QUASI_P, 241))

        u_arr = self.create_vector_arrow(plane, np.zeros(2), U_DEG, I_HAT_COLOR, stroke_width=3.6)
        v_arr = self.create_vector_arrow(plane, np.zeros(2), V_DEG, J_HAT_COLOR, stroke_width=3.6)
        uv_arr = self.create_vector_arrow(plane, np.zeros(2), U_DEG + V_DEG, ERROR_COLOR, stroke_width=3.8)

        f1_tex = VGroup(
            Text("Failure 1: p = 0.5 Quasi-Norm", font_size=20, color=ERROR_COLOR),
            MathTex(r"\|\mathbf{u}+\mathbf{v}\|_{0.5} = 4 > 1 + 1", font_size=22, color=TEXT_COLOR),
            Text("Non-convex star breaks triangle ineq.", font_size=18, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        ur_f1 = make_hud_panel(f1_tex, padding=0.14).to_corner(UR, buff=0.25)

        self.play(
            Create(l1_dashed),
            Create(star_05),
            GrowArrow(u_arr),
            GrowArrow(v_arr),
            GrowArrow(uv_arr),
            FadeIn(ur_f1),
            run_time=1.3,
        )
        self.wait(1.3)

        # Failure 2: L1 Parallelogram Law Failure
        para_edges = VGroup(
            Line(plane.c2p(1.0, 0.0), plane.c2p(1.0, 1.0), color=GHOST_COLOR, stroke_width=2.5),
            Line(plane.c2p(0.0, 1.0), plane.c2p(1.0, 1.0), color=GHOST_COLOR, stroke_width=2.5),
            DashedLine(plane.c2p(1.0, 0.0), plane.c2p(0.0, 1.0), color=EIGEN_GRID_COLOR, stroke_width=2.5),
        )
        f2_tex = VGroup(
            Text("Failure 2: L1 Parallelogram Law", font_size=20, color=ERROR_COLOR),
            MathTex(r"\|u+v\|_1^2 + \|u-v\|_1^2 = 8 \ne 4", font_size=21, color=TEXT_COLOR),
            Text("L1 & L_inf have NO inner product", font_size=18, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        ur_f2 = make_hud_panel(f2_tex, padding=0.14).to_corner(UR, buff=0.25)

        self.play(
            FadeOut(star_05),
            FadeOut(l1_dashed),
            Create(para_edges),
            FadeOut(ur_f1, shift=UP * 0.08),
            FadeIn(ur_f2, shift=DOWN * 0.08),
            run_time=1.0,
        )
        self.wait(1.2)

        # Failure 3: x = 0 Edge Case
        zero_dot = Dot(plane.c2p(0.0, 0.0), radius=0.11, color=ERROR_COLOR, z_index=9)
        f3_tex = VGroup(
            Text("Failure 3: Zero Vector x = 0", font_size=20, color=ERROR_COLOR),
            MathTex(r"\mathbf{0}^\top\mathbf{y} = 0 \;\;(\mathbf{0}\perp\mathbf{y})", font_size=21, color=TEXT_COLOR),
            MathTex(r"\cos\omega = \tfrac{0}{0} \;\text{undefined}", font_size=21, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        ur_f3 = make_hud_panel(f3_tex, padding=0.14).to_corner(UR, buff=0.25)

        self.play(
            FadeOut(para_edges),
            FadeOut(uv_arr),
            FadeOut(u_arr),
            Transform(v_arr, zero_dot),
            FadeOut(ur_f2, shift=UP * 0.08),
            FadeIn(ur_f3, shift=DOWN * 0.08),
            run_time=1.0,
        )
        self.wait(1.0)

        # Failure 4: Indefinite Matrix A_indef = diag(1, -1)
        ts = np.linspace(-1.7, 1.7, 121)
        hyp_curves = VGroup()
        for cx_s, cy_s in [(np.cosh(ts), np.sinh(ts)), (-np.cosh(ts), np.sinh(ts)), (np.sinh(ts), np.cosh(ts)), (np.sinh(ts), -np.cosh(ts))]:
            pts = [plane.c2p(float(x), float(y)) for x, y in zip(cx_s, cy_s) if abs(x) <= 5.8 and abs(y) <= 3.8]
            if len(pts) >= 2:
                m = VMobject(stroke_color=ERROR_COLOR, stroke_width=2.8, stroke_opacity=0.85, fill_opacity=0.0)
                m.set_points_as_corners(pts)
                hyp_curves.add(m)
        null_1 = DashedLine(plane.c2p(-3.8, -3.8), plane.c2p(3.8, 3.8), color=GHOST_COLOR, stroke_width=2.0)
        null_2 = DashedLine(plane.c2p(-3.8, 3.8), plane.c2p(3.8, -3.8), color=GHOST_COLOR, stroke_width=2.0)
        x_indef_arr = self.create_vector_arrow(plane, np.zeros(2), X_INDEF, ERROR_COLOR, stroke_width=4.0)
        x_indef_lbl = MathTex(r"\mathbf{x}=[1,2]^\top", font_size=22, color=ERROR_COLOR).next_to(
            plane.c2p(float(X_INDEF[0]), float(X_INDEF[1])), RIGHT, buff=0.12
        )

        f4_tex = VGroup(
            Text("Failure 4: Indefinite A = diag(1,-1)", font_size=20, color=ERROR_COLOR),
            MathTex(r"\mathbf{x}^\top A_{\text{indef}} \mathbf{x} = -3 < 0", font_size=22, color=TEXT_COLOR),
            Text("Hyperbolas & null cone: not an IP", font_size=18, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        ur_f4 = make_hud_panel(f4_tex, padding=0.14).to_corner(UR, buff=0.25)

        self.play(
            FadeOut(v_arr),
            Create(null_1),
            Create(null_2),
            Create(hyp_curves),
            GrowArrow(x_indef_arr),
            Write(x_indef_lbl),
            FadeOut(ur_f3, shift=UP * 0.08),
            FadeIn(ur_f4, shift=DOWN * 0.08),
            run_time=1.2,
        )
        self.wait(1.3)

        # Master Summary Card (z_index = 20/21)
        summary_title = Text("Sprint 5 Verdict: Norms, Inner Products & Cauchy-Schwarz", font_size=25, color=TARGET_COLOR)
        summary_lines = VGroup(
            MathTex(r"1.\;\; \|\mathbf{x}\|_\infty \le \|\mathbf{x}\|_2 \le \|\mathbf{x}\|_1 \quad (B_1 \subseteq B_2 \subseteq B_\infty),\;\; \operatorname{rms}(\mathbf{x})=\|\mathbf{x}\|_2/\sqrt{n}", font_size=21, color=LINE_COLOR),
            MathTex(r"2.\;\; |\mathbf{x}^\top\mathbf{y}| \le \|\mathbf{x}\|_2\|\mathbf{y}\|_2 \;\;(\text{equality}\iff \text{collinear},\;\; \cos\omega\in[-1,1])", font_size=21, color=VEC_COLOR),
            MathTex(r"3.\;\; \langle\mathbf{x},\mathbf{y}\rangle_A = \mathbf{x}^\top A\mathbf{y} \text{ is an inner product} \iff A = A^\top = LL^\top \succ 0", font_size=21, color=SUBSPACE_COLOR),
            MathTex(r"4.\;\; p<1 \text{ fails }\triangle\text{-ineq};\; L_1, L_\infty \text{ fail parallelogram};\; A_{\text{indef}} \text{ fails positivity}", font_size=21, color=ERROR_COLOR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        summary_content = VGroup(summary_title, summary_lines).arrange(DOWN, buff=0.24)
        summary_card = make_hud_panel(summary_content, padding=0.28, bg_opacity=0.95, stroke_color=TARGET_COLOR, stroke_width=2.0)
        summary_card.move_to(ORIGIN)
        summary_card[0].set_z_index(20)
        summary_card[1].set_z_index(21)

        act4_geom = VGroup(title_panel, ur_f4, null_1, null_2, hyp_curves, x_indef_arr, x_indef_lbl)
        self.play(FadeOut(act4_geom), FadeIn(summary_card, scale=0.95), run_time=1.0)
        self.wait(2.5)

        cleanup_group = VGroup(summary_card)
        if is_standalone:
            self.play(FadeOut(cleanup_group), run_time=0.7)
            return VGroup()
        return cleanup_group


# -----------------------------------------------------------------------------
# Standalone Act Scenes & Composite Master Scene
# -----------------------------------------------------------------------------
class Act1SpatialPuzzle(BaseNormsInnerProductsScene):
    def construct(self) -> None:
        plane = self.create_plane()
        origin = self.create_origin_anchor(plane)
        self.play(Create(plane), FadeIn(origin), run_time=0.8)
        self.run_act1(plane, origin, is_standalone=True)


class Act2DynamicDeformation(BaseNormsInnerProductsScene):
    def construct(self) -> None:
        plane = self.create_plane()
        origin = self.create_origin_anchor(plane)
        self.play(Create(plane), FadeIn(origin), run_time=0.8)
        self.run_act2(plane, origin, is_standalone=True)


class Act3AlgebraicMorph(BaseNormsInnerProductsScene):
    def construct(self) -> None:
        plane = self.create_plane()
        origin = self.create_origin_anchor(plane)
        self.play(Create(plane), FadeIn(origin), run_time=0.8)
        self.run_act3(plane, origin, is_standalone=True)


class Act4DegenerateStressTest(BaseNormsInnerProductsScene):
    def construct(self) -> None:
        plane = self.create_plane()
        origin = self.create_origin_anchor(plane)
        self.play(Create(plane), FadeIn(origin), run_time=0.8)
        self.run_act4(plane, origin, is_standalone=True)


class NormsInnerProductsScene(BaseNormsInnerProductsScene):
    def construct(self) -> None:
        plane = self.create_plane()
        origin = self.create_origin_anchor(plane)
        self.play(Create(plane), FadeIn(origin), run_time=0.9)

        g1 = self.run_act1(plane, origin, is_standalone=False)
        self.play(FadeOut(g1), run_time=0.6)

        g2 = self.run_act2(plane, origin, is_standalone=False)
        self.play(FadeOut(g2), run_time=0.6)

        g3 = self.run_act3(plane, origin, is_standalone=False)
        self.play(FadeOut(g3), run_time=0.6)

        g4 = self.run_act4(plane, origin, is_standalone=False)
        self.play(FadeOut(g4), FadeOut(plane), FadeOut(origin), run_time=0.8)
