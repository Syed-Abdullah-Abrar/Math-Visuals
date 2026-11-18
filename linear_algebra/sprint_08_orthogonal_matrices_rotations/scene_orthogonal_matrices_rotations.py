"""
Sprint 8: Orthogonal Matrices, Rotations SO(n) & Givens/Householder (ManimCE v0.18+)
Curriculum: Linear Algebra & Theoretical CS / Machine Learning Foundations
References: MML Ch. 3.4 & 3.9, VMLS Ch. 7.1 & 10.4, Kreyszig Ch. 8.3 & 20.9, ILA Ch. 6

This production script implements the 4-act pedagogical narrative:
- Act 1: Concrete Spatial Puzzle (Isometry vs. Area-Preserving Shear Distortion)
- Act 2: Dynamic Deformation & Ghost Space (SO(2), Reflections, Cartan-Dieudonne & SO(3))
- Act 3: Color-Coded Algebraic Morph (Q^T Q = I, Givens Rotation & Householder Reflector)
- Act 4: Degenerate & Edge-Case Stress Test (Cancellation, Q^1000 Drift & Sparsity Tradeoff)

Render Commands:
    Preview Low-Res:   manim -pql linear_algebra/sprint_08_orthogonal_matrices_rotations/scene_orthogonal_matrices_rotations.py OrthogonalRotationsScene
    Headless Low-Res:  manim -ql linear_algebra/sprint_08_orthogonal_matrices_rotations/scene_orthogonal_matrices_rotations.py OrthogonalRotationsScene
    Production Hi-Res: manim -pqh linear_algebra/sprint_08_orthogonal_matrices_rotations/scene_orthogonal_matrices_rotations.py OrthogonalRotationsScene
    Headless Hi-Res:   manim -qh linear_algebra/sprint_08_orthogonal_matrices_rotations/scene_orthogonal_matrices_rotations.py OrthogonalRotationsScene
    Single Act Test:   manim -ql linear_algebra/sprint_08_orthogonal_matrices_rotations/scene_orthogonal_matrices_rotations.py Act1SpatialPuzzle
"""

from __future__ import annotations

import math
import numpy as np
from manim import *

# -----------------------------------------------------------------------------
# Rule 1: 3Blue1Brown Semantic Color Palette
# -----------------------------------------------------------------------------
I_HAT_COLOR = "#83C167"       # Green C    - Basis e1 / q1 / Exact Q invariant
J_HAT_COLOR = "#FC6255"       # Red C      - Basis e2 / q2 / y-axis
K_HAT_COLOR = "#58C4DD"       # Blue C     - z-axis / mirror line l1 / Givens arc
VEC_COLOR = "#F7D96F"         # Yellow     - Test vector / Householder v / mirror l2
TARGET_COLOR = "#F9E498"      # Gold A     - Destination beacon / canonical target
SUBSPACE_COLOR = "#9A72AC"    # Purple     - Unit circle S^1 / hyperplane v^perp
LINE_COLOR = "#58C4DD"        # Blue C     - Reference axis / Givens plane
GHOST_COLOR = "#414868"       # Slate Gray - Static ghost grid & ghost unit circle
ORIGIN_COLOR = "#FFFFFF"      # White      - Origin anchor (0, 0)
ERROR_COLOR = "#F7768E"       # Soft Red   - Shear distortion / det=-1 / drift / cancellation
BG_COLOR = "#1C1C24"          # Dark Slate - High-contrast dark viewport background
PANEL_BG = "#16161E"          # Obsidian   - HUD panel container background fill
PANEL_BORDER = "#414868"      # Slate Teal - HUD panel container outline stroke
TEXT_COLOR = "#ECEFF4"        # Snow       - High-contrast primary typography
MUTED_TEXT = "#7AA2F7"        # Light Blue - Secondary badges & annotations

# -----------------------------------------------------------------------------
# Rule 6: Module-Top NumPy Mathematical Constants & Pre-Flight Assertions
# -----------------------------------------------------------------------------
# 1. Exact Pythagorean SO(2) Rotation (3-4-5 triple: theta = atan2(4, 3) ~ 53.13 deg)
Q_PYTH = np.array([[0.6, -0.8],
                   [0.8,  0.6]], dtype=float)
THETA_PYTH = math.atan2(4.0, 3.0)
THETA_DEMO = math.pi / 3.0  # 60 deg
R_60 = np.array([[math.cos(THETA_DEMO), -math.sin(THETA_DEMO)],
                 [math.sin(THETA_DEMO),  math.cos(THETA_DEMO)]], dtype=float)

# 2. Non-Orthogonal Area-Preserving Shear (det=1, NOT orthogonal; angle -> 39.81 deg)
M_SHEAR = np.array([[1.0, 1.2],
                    [0.0, 1.0]], dtype=float)
TEST_W = np.array([0.9, 0.8], dtype=float)

# 3. 2D Reflections & Cartan-Dieudonne Composition: H2 @ H1 = R(2 * (alpha2 - alpha1))
ALPHA_1 = math.pi / 12.0   # 15 deg
ALPHA_2 = math.pi / 4.0    # 45 deg -> 2*(45-15) = 60 deg = pi/3
H_1 = np.array([[math.cos(2.0 * ALPHA_1),  math.sin(2.0 * ALPHA_1)],
                [math.sin(2.0 * ALPHA_1), -math.cos(2.0 * ALPHA_1)]], dtype=float)
H_2 = np.array([[math.cos(2.0 * ALPHA_2),  math.sin(2.0 * ALPHA_2)],
                [math.sin(2.0 * ALPHA_2), -math.cos(2.0 * ALPHA_2)]], dtype=float)
H_PYTH = np.array([[0.6, 0.8], [0.8, -0.6]], dtype=float)

# 4. 3D SO(3) Elementary Axis Rotations (MML §3.9.2 eqs. 3.77-3.78) & Kreyszig Ex. 1
R1_90 = np.array([[1.0, 0.0,  0.0],
                  [0.0, 0.0, -1.0],
                  [0.0, 1.0,  0.0]], dtype=float)
R2_90 = np.array([[ 0.0, 0.0, 1.0],
                  [ 0.0, 1.0, 0.0],
                  [-1.0, 0.0, 0.0]], dtype=float)
Q_KREYSZIG_3D = (1.0 / 3.0) * np.array([[ 2.0, 1.0,  2.0],
                                        [-2.0, 2.0,  1.0],
                                        [ 1.0, 2.0, -2.0]], dtype=float)

# 5. Givens Rotation G(0, 2, theta) on [3, 2, 4]^T and its 2D planar slice [1.2, 1.6]^T
X_GIVENS = np.array([3.0, 2.0, 4.0], dtype=float)
G_02 = np.array([[ 0.6, 0.0, 0.8],
                 [ 0.0, 1.0, 0.0],
                 [-0.8, 0.0, 0.6]], dtype=float)
X_GIV_2D = np.array([1.2, 1.6], dtype=float)
G_2D = np.array([[ 0.6, 0.8],
                 [-0.8, 0.6]], dtype=float)

# 6. Householder Reflectors: 3D canonical [1, 2, 2]^T, 2D planar [1.2, 1.6]^T & cancellation [1, 1e-9]^T
X_HH = np.array([1.0, 2.0, 2.0], dtype=float)
V_HH = np.array([4.0, 2.0, 2.0], dtype=float)
H_HH = np.eye(3) - 2.0 * np.outer(V_HH, V_HH) / np.dot(V_HH, V_HH)

X_HH_2D = np.array([1.2, 1.6], dtype=float)
V_HH_2D = np.array([3.2, 1.6], dtype=float)
H_HH_2D = np.eye(2) - 2.0 * np.outer(V_HH_2D, V_HH_2D) / np.dot(V_HH_2D, V_HH_2D)

X_CANCEL = np.array([1.0, 1e-9], dtype=float)

# 7. Repeated Power Drift Matrix (n = 1000)
THETA_DRIFT = 0.37
Q_DRIFT = np.array([[math.cos(THETA_DRIFT), -math.sin(THETA_DRIFT)],
                    [math.sin(THETA_DRIFT),  math.cos(THETA_DRIFT)]], dtype=float)
Q_DRIFT_N = 1000

# Pre-flight mathematical verification assertions
assert np.allclose(Q_PYTH.T @ Q_PYTH, np.eye(2), atol=1e-15), "Q_PYTH must be orthogonal"
assert np.isclose(np.linalg.det(Q_PYTH), 1.0, atol=1e-15), "Q_PYTH must have det=+1"
assert np.allclose(np.abs(np.linalg.eigvals(Q_PYTH)), 1.0, atol=1e-15), "Q_PYTH eigs must lie on S^1"
assert np.allclose(np.linalg.svd(Q_PYTH, compute_uv=False), [1.0, 1.0], atol=1e-15), "Q_PYTH singular values == 1"

assert np.isclose(np.linalg.det(M_SHEAR), 1.0, atol=1e-15), "M_SHEAR must preserve area (det=1)"
assert not np.allclose(M_SHEAR.T @ M_SHEAR, np.eye(2), atol=1e-6), "M_SHEAR must NOT be orthogonal"
assert math.isclose(
    math.degrees(math.acos(1.2 / math.sqrt(2.44))),
    39.8055710922652,
    rel_tol=1e-12,
), "M_SHEAR basis angle must be 39.8056 deg"

assert np.allclose(H_1 @ H_1, np.eye(2), atol=1e-15), "H_1 must be involution"
assert np.isclose(np.linalg.det(H_1), -1.0, atol=1e-15), "H_1 must have det=-1"
assert np.allclose(H_2 @ H_1, R_60, atol=1e-15), "Cartan-Dieudonne H2 @ H1 == R(60 deg)"
assert np.allclose(H_1 @ H_2, R_60.T, atol=1e-15), "Reversed H1 @ H2 == R(-60 deg)"

assert not np.allclose(R1_90 @ R2_90, R2_90 @ R1_90, atol=1e-6), "SO(3) rotations must not commute"
assert np.allclose((R2_90 @ R1_90) @ np.array([1.0, 0.0, 0.0]), [0.0, 0.0, -1.0], atol=1e-15)
assert np.allclose((R1_90 @ R2_90) @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-15)

assert np.allclose(G_02 @ X_GIVENS, [5.0, 2.0, 0.0], atol=1e-15), "G_02 must zero entry 2"
assert np.isclose(np.linalg.det(G_02), 1.0, atol=1e-15), "G_02 in SO(3)"
assert np.allclose(G_2D @ X_GIV_2D, [2.0, 0.0], atol=1e-15), "G_2D must zero entry 1"

assert np.allclose(H_HH @ X_HH, [-3.0, 0.0, 0.0], atol=1e-15), "H_HH must zero subcolumn"
assert np.isclose(np.linalg.det(H_HH), -1.0, atol=1e-15), "Householder det == -1"
assert np.allclose(H_HH_2D @ X_HH_2D, [-2.0, 0.0], atol=1e-15), "2D Householder target == [-2, 0]"
assert np.isclose(np.dot(V_HH_2D, 0.5 * (X_HH_2D + H_HH_2D @ X_HH_2D)), 0.0, atol=1e-15)


# -----------------------------------------------------------------------------
# Helper Functions: Rotation, Householder, Safe Angle/RightAngle, 3D Projection
# -----------------------------------------------------------------------------
def rot2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def rot_x_3d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def rot_y_3d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def safe_right_angle(
    p_origin: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    length: float = 0.22,
    color: str = SUBSPACE_COLOR,
    stroke_width: float = 2.0,
) -> VMobject:
    """Guarded RightAngle preventing zero-length or non-orthogonal crashes during sweeps."""
    d1 = p1[:2] - p_origin[:2]
    d2 = p2[:2] - p_origin[:2]
    n1 = float(np.linalg.norm(d1))
    n2 = float(np.linalg.norm(d2))
    if n1 < 1e-4 or n2 < 1e-4:
        return VMobject()
    u1 = d1 / n1
    u2 = d2 / n2
    if abs(float(np.dot(u1, u2))) > 1e-3:
        return VMobject()
    l1 = Line(p_origin, p_origin + np.array([u1[0], u1[1], 0.0]))
    l2 = Line(p_origin, p_origin + np.array([u2[0], u2[1], 0.0]))
    return RightAngle(l1, l2, length=length, quadrant=(1, 1), color=color, stroke_width=stroke_width)


def safe_angle_arc(
    p_origin: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    radius: float = 0.36,
    color: str = VEC_COLOR,
    stroke_width: float = 2.5,
) -> VMobject:
    """Draws shortest signed angle arc in [-pi, pi] without Manim Angle reflex flips."""
    d1 = p1[:2] - p_origin[:2]
    d2 = p2[:2] - p_origin[:2]
    if np.linalg.norm(d1) < 1e-4 or np.linalg.norm(d2) < 1e-4:
        return VMobject()
    a1 = math.atan2(float(d1[1]), float(d1[0]))
    a2 = math.atan2(float(d2[1]), float(d2[0]))
    delta = (a2 - a1 + math.pi) % (2.0 * math.pi) - math.pi
    if abs(delta) < 1e-3:
        return VMobject()
    return Arc(
        radius=radius,
        start_angle=a1,
        angle=delta,
        arc_center=p_origin,
        color=color,
        stroke_width=stroke_width,
    )


def project_iso_3d(p3: np.ndarray, center: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Orthographic isometric 3D->2D projection (az=-65 deg, el=22 deg; z points straight UP)."""
    az = math.radians(-65.0)
    el = math.radians(22.0)
    cam_right = np.array([-math.sin(az), math.cos(az), 0.0])
    cam_up = np.array([-math.sin(el) * math.cos(az), -math.sin(el) * math.sin(az), math.cos(el)])
    x2 = float(np.dot(cam_right, p3)) * scale
    y2 = float(np.dot(cam_up, p3)) * scale
    return center + np.array([x2, y2, 0.0])


# -----------------------------------------------------------------------------
# Rule 3: Zero Text Overlap HUD Panel Factory
# -----------------------------------------------------------------------------
def make_hud_panel(
    content: Mobject,
    padding: float = 0.16,
    bg_color: str = PANEL_BG,
    bg_opacity: float = 0.90,
    stroke_color: str = PANEL_BORDER,
    stroke_width: float = 1.5,
    corner_radius: float = 0.12,
    min_width: float = 0.0,
    min_height: float = 0.0,
) -> VGroup:
    """Wraps any mobject inside a high-contrast rounded rectangular panel with z-index 10."""
    w = max(content.width + 2.0 * padding, min_width)
    h = max(content.height + 2.0 * padding, min_height)
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
    content.set_z_index(11)
    return panel


# -----------------------------------------------------------------------------
# Base Scene Class (Shared Geometry, Viewport, and Modular Act Methods)
# -----------------------------------------------------------------------------
class BaseOrthogonalRotationsScene(Scene):
    """Base class providing strict isotropic planes, HUD helpers, and modular acts."""

    def setup(self):
        super().setup()
        self.camera.background_color = BG_COLOR

    def create_plane(self) -> NumberPlane:
        """Rule 2: Central isotropic NumberPlane (scale = 1.1) safe inside HUD band."""
        return NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-2.2, 2.2, 1],
            x_length=6.6,
            y_length=4.84,
            background_line_style={
                "stroke_color": TEAL_E,
                "stroke_width": 1.1,
                "stroke_opacity": 0.42,
            },
            faded_line_style={
                "stroke_color": TEAL_E,
                "stroke_width": 0.5,
                "stroke_opacity": 0.18,
            },
        ).shift(DOWN * 0.05)

    def create_split_plane(self, center: np.ndarray) -> NumberPlane:
        """Rule 2: Split-screen isotropic NumberPlane (scale = 0.9) for side-by-side contrasts."""
        return NumberPlane(
            x_range=[-2.5, 2.5, 1],
            y_range=[-2.0, 2.0, 1],
            x_length=4.5,
            y_length=3.6,
            background_line_style={
                "stroke_color": TEAL_E,
                "stroke_width": 1.0,
                "stroke_opacity": 0.38,
            },
            faded_line_style={
                "stroke_color": TEAL_E,
                "stroke_width": 0.5,
                "stroke_opacity": 0.16,
            },
        ).move_to(center)

    def create_orbit_plane(self, center: np.ndarray) -> NumberPlane:
        """Isotropic orbit plane (scale = 0.55) for Act 4 Q^1000 spiral comparison."""
        return NumberPlane(
            x_range=[-3.5, 3.5, 1],
            y_range=[-3.5, 3.5, 1],
            x_length=3.85,
            y_length=3.85,
            background_line_style={
                "stroke_color": TEAL_E,
                "stroke_width": 0.9,
                "stroke_opacity": 0.35,
            },
            faded_line_style={
                "stroke_color": TEAL_E,
                "stroke_width": 0.4,
                "stroke_opacity": 0.15,
            },
        ).move_to(center)

    def create_origin_anchor(self, plane: NumberPlane) -> VGroup:
        """Rule 5: Immovable origin anchor with high-contrast label."""
        dot = Dot(plane.c2p(0, 0), radius=0.075, color=ORIGIN_COLOR)
        label = MathTex(r"\mathbf{0}", font_size=17, color=ORIGIN_COLOR).next_to(dot, DL, buff=0.08)
        group = VGroup(dot, label)
        group.set_z_index(8)
        return group

    def create_ghost_circle(
        self,
        plane: NumberPlane,
        radius: float = 1.0,
        color: str = GHOST_COLOR,
        stroke_width: float = 2.0,
    ) -> DashedVMobject:
        """Dashed reference circle of given radius in plane coordinates."""
        r_scene = float(np.linalg.norm(plane.c2p(radius, 0) - plane.c2p(0, 0)))
        circ = Circle(radius=r_scene, color=color, stroke_width=stroke_width).move_to(plane.c2p(0, 0))
        dashed = DashedVMobject(circ, num_dashes=36, dashed_ratio=0.55)
        dashed.set_z_index(2)
        return dashed

    def create_vector_arrow(
        self,
        plane: NumberPlane,
        start_coords: np.ndarray,
        end_coords: np.ndarray,
        color: str,
        stroke_width: float = 3.5,
    ) -> Arrow:
        """Consistent vector arrow in coordinate plane space."""
        return Arrow(
            plane.c2p(float(start_coords[0]), float(start_coords[1])),
            plane.c2p(float(end_coords[0]), float(end_coords[1])),
            buff=0,
            color=color,
            stroke_width=stroke_width,
            max_tip_length_to_length_ratio=0.22,
            max_stroke_width_to_length_ratio=5.0,
        )

    def create_chiral_flag(
        self,
        plane: NumberPlane,
        q1: np.ndarray,
        q2: np.ndarray,
        det_sign: float,
    ) -> VGroup:
        """Visualizes orientation (CCW green vs CW coral) via chiral triangle + directed arc."""
        col = I_HAT_COLOR if det_sign >= 0 else ERROR_COLOR
        o = plane.c2p(0, 0)
        p_a = plane.c2p(*(0.65 * q1))
        p_b = plane.c2p(*(0.32 * q1 + 0.62 * q2))
        tri = Polygon(o, p_a, p_b, fill_color=col, fill_opacity=0.22, stroke_color=col, stroke_width=1.4)
        arc = safe_angle_arc(o, plane.c2p(*q1), plane.c2p(*q2), radius=0.45, color=col, stroke_width=2.6)
        tip_dir = q2 - 0.35 * q1
        tip_dot = Dot(plane.c2p(*(0.45 * tip_dir / max(np.linalg.norm(tip_dir), 1e-6))), radius=0.045, color=col)
        return VGroup(tri, arc, tip_dot)

    # -------------------------------------------------------------------------
    # Act 1: Concrete Spatial Puzzle (Isometry vs. Area-Preserving Shear)
    # -------------------------------------------------------------------------
    def run_act1(self, is_standalone: bool = False) -> VGroup | None:
        """Act 1: Contrasts det=1 shear M_SHEAR against rigid orthogonal Q_PYTH."""
        title_content = VGroup(
            Text("Act 1: Isometry vs. Shear Distortion", font_size=18, color=TEXT_COLOR, weight=BOLD),
            Text("det(A) = 1 preserves area, NOT lengths or angles", font_size=13, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        title_panel = make_hud_panel(title_content).to_corner(UL, buff=0.25)

        ur_math = MathTex(
            r"M_{\text{shear}} = \begin{bmatrix} 1 & 1.2 \\ 0 & 1 \end{bmatrix}"
            r"\;\text{ vs }\;"
            r"Q = \begin{bmatrix} 0.6 & -0.8 \\ 0.8 & 0.6 \end{bmatrix}",
            font_size=16,
            color=TEXT_COLOR,
        )
        ur_panel = make_hud_panel(ur_math).to_corner(UR, buff=0.25)

        left_center = LEFT * 3.15 + DOWN * 0.12
        right_center = RIGHT * 3.15 + DOWN * 0.12
        plane_l = self.create_split_plane(left_center)
        plane_r = self.create_split_plane(right_center)
        orig_l = self.create_origin_anchor(plane_l)
        orig_r = self.create_origin_anchor(plane_r)
        ghost_l = self.create_ghost_circle(plane_l, radius=1.0)
        ghost_r = self.create_ghost_circle(plane_r, radius=1.0)

        lbl_l = Text("Non-Orthogonal Shear (det = 1)", font_size=15, color=ERROR_COLOR, weight=BOLD).next_to(
            plane_l, UP, buff=0.14
        )
        lbl_r = Text("Orthogonal Q in SO(2) (det = 1)", font_size=15, color=I_HAT_COLOR, weight=BOLD).next_to(
            plane_r, UP, buff=0.14
        )

        self.play(
            FadeIn(title_panel),
            FadeIn(ur_panel),
            Create(plane_l),
            Create(plane_r),
            FadeIn(orig_l),
            FadeIn(orig_r),
            Create(ghost_l),
            Create(ghost_r),
            FadeIn(lbl_l),
            FadeIn(lbl_r),
            run_time=1.0,
        )

        t_tracker = ValueTracker(0.0)
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])

        def left_mat() -> np.ndarray:
            t = t_tracker.get_value()
            return (1.0 - t) * np.eye(2) + t * M_SHEAR

        def right_mat() -> np.ndarray:
            t = t_tracker.get_value()
            return rot2d(t * THETA_PYTH)

        # Deformed left circle & rigid right circle
        circ_l = always_redraw(
            lambda: ParametricFunction(
                lambda s: plane_l.c2p(*(left_mat() @ np.array([math.cos(s), math.sin(s)]))),
                t_range=[0.0, TAU],
                color=ERROR_COLOR,
                stroke_width=2.6,
            ).set_z_index(4)
        )
        circ_r = always_redraw(
            lambda: ParametricFunction(
                lambda s: plane_r.c2p(*(right_mat() @ np.array([math.cos(s), math.sin(s)]))),
                t_range=[0.0, TAU],
                color=SUBSPACE_COLOR,
                stroke_width=2.6,
            ).set_z_index(4)
        )

        # Basis and test vectors
        u_l = always_redraw(lambda: self.create_vector_arrow(plane_l, np.zeros(2), left_mat() @ e1, I_HAT_COLOR, 3.2))
        v_l = always_redraw(lambda: self.create_vector_arrow(plane_l, np.zeros(2), left_mat() @ e2, J_HAT_COLOR, 3.2))
        w_l = always_redraw(lambda: self.create_vector_arrow(plane_l, np.zeros(2), left_mat() @ TEST_W, VEC_COLOR, 3.2))
        ang_l = always_redraw(
            lambda: safe_angle_arc(
                plane_l.c2p(0, 0),
                plane_l.c2p(*(left_mat() @ e1)),
                plane_l.c2p(*(left_mat() @ e2)),
                radius=0.32,
                color=ERROR_COLOR,
                stroke_width=2.4,
            )
        )

        u_r = always_redraw(lambda: self.create_vector_arrow(plane_r, np.zeros(2), right_mat() @ e1, I_HAT_COLOR, 3.2))
        v_r = always_redraw(lambda: self.create_vector_arrow(plane_r, np.zeros(2), right_mat() @ e2, J_HAT_COLOR, 3.2))
        w_r = always_redraw(lambda: self.create_vector_arrow(plane_r, np.zeros(2), right_mat() @ TEST_W, VEC_COLOR, 3.2))
        ra_r = always_redraw(
            lambda: safe_right_angle(
                plane_r.c2p(0, 0),
                plane_r.c2p(*(right_mat() @ e1)),
                plane_r.c2p(*(right_mat() @ e2)),
                length=0.22,
                color=SUBSPACE_COLOR,
                stroke_width=2.2,
            )
        )

        # Pre-sized DecimalNumber HUDs (DL & DR)
        dl_norm_num = DecimalNumber(1.20, num_decimal_places=2, font_size=16, color=VEC_COLOR, edge_to_fix=LEFT).set_z_index(12)
        dl_ang_num = DecimalNumber(90.00, num_decimal_places=2, font_size=16, color=ERROR_COLOR, edge_to_fix=LEFT).set_z_index(12)
        dl_norm_num.add_updater(lambda m: m.set_value(float(np.linalg.norm(left_mat() @ TEST_W))).set_z_index(12))

        def _left_angle_deg() -> float:
            a = left_mat() @ e1
            b = left_mat() @ e2
            cos_v = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            return math.degrees(math.acos(max(-1.0, min(1.0, cos_v))))

        dl_ang_num.add_updater(lambda m: m.set_value(_left_angle_deg()).set_z_index(12))

        dl_row1 = VGroup(
            MathTex(r"\|M\mathbf{w}\|_2 =", font_size=16, color=TEXT_COLOR),
            dl_norm_num,
        ).arrange(RIGHT, buff=0.10)
        dl_row2 = VGroup(
            MathTex(r"\angle(M\mathbf{e}_1, M\mathbf{e}_2) =", font_size=16, color=TEXT_COLOR),
            dl_ang_num,
            MathTex(r"^\circ", font_size=16, color=ERROR_COLOR),
        ).arrange(RIGHT, buff=0.06)
        dl_content = VGroup(dl_row1, dl_row2).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        dl_panel = make_hud_panel(dl_content, min_width=4.35, stroke_color=ERROR_COLOR).to_corner(DL, buff=0.25)

        dr_norm_num = DecimalNumber(1.20, num_decimal_places=2, font_size=16, color=VEC_COLOR, edge_to_fix=LEFT).set_z_index(12)
        dr_ang_num = DecimalNumber(90.00, num_decimal_places=2, font_size=16, color=I_HAT_COLOR, edge_to_fix=LEFT).set_z_index(12)
        dr_norm_num.add_updater(lambda m: m.set_value(float(np.linalg.norm(right_mat() @ TEST_W))).set_z_index(12))
        dr_row1 = VGroup(
            MathTex(r"\|Q\mathbf{w}\|_2 =", font_size=16, color=TEXT_COLOR),
            dr_norm_num,
            MathTex(r"\;(\sigma_1=\sigma_2=1)", font_size=15, color=I_HAT_COLOR),
        ).arrange(RIGHT, buff=0.08)
        dr_row2 = VGroup(
            MathTex(r"\angle(Q\mathbf{e}_1, Q\mathbf{e}_2) =", font_size=16, color=TEXT_COLOR),
            dr_ang_num,
            MathTex(r"^\circ\;\;(Q^\top Q = I)", font_size=15, color=I_HAT_COLOR),
        ).arrange(RIGHT, buff=0.06)
        dr_content = VGroup(dr_row1, dr_row2).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        dr_panel = make_hud_panel(dr_content, min_width=4.75, stroke_color=I_HAT_COLOR).to_corner(DR, buff=0.25)

        self.add(circ_l, circ_r, u_l, v_l, w_l, ang_l, u_r, v_r, w_r, ra_r)
        self.play(FadeIn(dl_panel), FadeIn(dr_panel), run_time=0.7)

        # Sweep shear vs geodesic rotation
        self.play(t_tracker.animate.set_value(1.0), run_time=3.5, rate_func=smooth)
        self.wait(1.0)

        verdict = make_hud_panel(
            MathTex(
                r"\det(M)=1 \text{ is NOT enough!}\quad\text{Rigid isometry }\iff Q^\top Q = I_n",
                font_size=17,
                color=TARGET_COLOR,
            ),
            stroke_color=TARGET_COLOR,
        ).to_edge(DOWN, buff=1.55)
        self.play(FadeIn(verdict), run_time=0.6)
        self.wait(1.6)

        # Explicit updater cleanup
        for mob in [circ_l, circ_r, u_l, v_l, w_l, ang_l, u_r, v_r, w_r, ra_r, dl_norm_num, dl_ang_num, dr_norm_num]:
            mob.clear_updaters()

        cleanup = VGroup(
            title_panel, ur_panel, dl_panel, dr_panel, verdict,
            plane_l, plane_r, orig_l, orig_r, ghost_l, ghost_r, lbl_l, lbl_r,
            circ_l, circ_r, u_l, v_l, w_l, ang_l, u_r, v_r, w_r, ra_r,
        )
        if is_standalone:
            self.play(FadeOut(cleanup), run_time=0.7)
            return None
        return cleanup

    # -------------------------------------------------------------------------
    # Act 2: Dynamic Deformation & Ghost Space (SO(2), Reflections, SO(3))
    # -------------------------------------------------------------------------
    def run_act2(self, is_standalone: bool = False) -> VGroup | None:
        """Act 2: SO(2) sweep, det=-1 reflection, Cartan-Dieudonne, and SO(3) non-commutativity."""
        title_content = VGroup(
            Text("Act 2: SO(2), Reflections & 3D SO(3)", font_size=18, color=TEXT_COLOR, weight=BOLD),
            Text("Orientation det = +1 vs -1, Cartan-Dieudonne, non-Abelian 3D", font_size=13, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        title_panel = make_hud_panel(title_content).to_corner(UL, buff=0.25)

        ur_math = MathTex(
            r"R(\theta) = \begin{bmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{bmatrix} \in SO(2)",
            font_size=16,
            color=TEXT_COLOR,
        )
        ur_panel = make_hud_panel(ur_math).to_corner(UR, buff=0.25)

        plane = self.create_plane()
        origin_group = self.create_origin_anchor(plane)
        ghost_circ = self.create_ghost_circle(plane, radius=1.0)
        live_circ = Circle(
            radius=float(np.linalg.norm(plane.c2p(1, 0) - plane.c2p(0, 0))),
            color=SUBSPACE_COLOR,
            stroke_width=2.2,
        ).move_to(plane.c2p(0, 0)).set_z_index(3)

        det_num = DecimalNumber(1.00, num_decimal_places=2, show_ellipsis=False, font_size=17, color=I_HAT_COLOR, edge_to_fix=LEFT)
        dl_row1 = VGroup(MathTex(r"\det(Q) =", font_size=17, color=TEXT_COLOR), det_num).arrange(RIGHT, buff=0.10)
        dl_status = Text("Orientation: CCW (+1)", font_size=14, color=I_HAT_COLOR)
        dl_panel = make_hud_panel(
            VGroup(dl_row1, dl_status).arrange(DOWN, aligned_edge=LEFT, buff=0.08),
            min_width=3.6,
            stroke_color=I_HAT_COLOR,
        ).to_corner(DL, buff=0.25)

        dr_panel = make_hud_panel(
            Text("SO(2): Connected Component (det = +1)", font_size=14, color=TEXT_COLOR),
            min_width=4.3,
        ).to_corner(DR, buff=0.25)

        self.play(
            FadeIn(title_panel),
            FadeIn(ur_panel),
            FadeIn(dl_panel),
            FadeIn(dr_panel),
            Create(plane),
            FadeIn(origin_group),
            Create(ghost_circ),
            Create(live_circ),
            run_time=0.9,
        )

        # Beat 1: Continuous SO(2) sweep
        theta_tracker = ValueTracker(0.0)
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])

        q1_mob = always_redraw(
            lambda: self.create_vector_arrow(plane, np.zeros(2), rot2d(theta_tracker.get_value()) @ e1, I_HAT_COLOR, 3.6)
        )
        q2_mob = always_redraw(
            lambda: self.create_vector_arrow(plane, np.zeros(2), rot2d(theta_tracker.get_value()) @ e2, J_HAT_COLOR, 3.6)
        )
        ra_mob = always_redraw(
            lambda: safe_right_angle(
                plane.c2p(0, 0),
                plane.c2p(*(rot2d(theta_tracker.get_value()) @ e1)),
                plane.c2p(*(rot2d(theta_tracker.get_value()) @ e2)),
                length=0.24,
                color=SUBSPACE_COLOR,
            )
        )
        flag_mob = always_redraw(
            lambda: self.create_chiral_flag(
                plane,
                rot2d(theta_tracker.get_value()) @ e1,
                rot2d(theta_tracker.get_value()) @ e2,
                +1.0,
            )
        )
        self.add(q1_mob, q2_mob, ra_mob, flag_mob)
        self.play(theta_tracker.animate.set_value(2.0 * math.pi / 3.0), run_time=2.0, rate_func=smooth)
        self.play(theta_tracker.animate.set_value(0.0), run_time=1.3, rate_func=smooth)

        for mob in [q1_mob, q2_mob, ra_mob, flag_mob]:
            mob.clear_updaters()
        self.remove(q1_mob, q2_mob, ra_mob, flag_mob)

        # Beat 2: Improper Reflection H(alpha_1) passing through det = 0
        u_mir1 = np.array([math.cos(ALPHA_1), math.sin(ALPHA_1)])
        n_mir1 = np.array([-math.sin(ALPHA_1), math.cos(ALPHA_1)])
        mir1_line = DashedLine(
            plane.c2p(*(-2.6 * u_mir1)),
            plane.c2p(*(2.6 * u_mir1)),
            color=LINE_COLOR,
            stroke_width=2.2,
            dash_length=0.12,
        )
        mir1_lbl = MathTex(r"\ell_1 \;(\alpha_1=15^\circ)", font_size=15, color=LINE_COLOR).move_to(
            plane.c2p(*(2.15 * u_mir1 + 0.28 * n_mir1))
        )

        new_ur = make_hud_panel(
            MathTex(
                r"H(\alpha_1) = \begin{bmatrix}\cos 2\alpha_1 & \sin 2\alpha_1 \\ \sin 2\alpha_1 & -\cos 2\alpha_1\end{bmatrix} \notin SO(2)",
                font_size=15,
                color=ERROR_COLOR,
            ),
            stroke_color=ERROR_COLOR,
        ).to_corner(UR, buff=0.25)
        new_dr = make_hud_panel(
            Text("det = 0 barrier separates det = +1 and det = -1", font_size=14, color=ERROR_COLOR),
            min_width=4.5,
            stroke_color=ERROR_COLOR,
        ).to_corner(DR, buff=0.25)

        self.play(Create(mir1_line), FadeIn(mir1_lbl), FadeOut(ur_panel), FadeOut(dr_panel), run_time=0.45)
        ur_panel = new_ur
        dr_panel = new_dr
        self.play(FadeIn(ur_panel), FadeIn(dr_panel), run_time=0.4)

        s_perp = ValueTracker(1.0)

        def refl_basis(vec: np.ndarray, s: float) -> np.ndarray:
            par = float(np.dot(vec, u_mir1)) * u_mir1
            perp = float(np.dot(vec, n_mir1)) * n_mir1
            return par + s * perp

        det_num.add_updater(lambda m: m.set_value(s_perp.get_value()).set_z_index(12))
        q1_ref = always_redraw(lambda: self.create_vector_arrow(plane, np.zeros(2), refl_basis(e1, s_perp.get_value()), I_HAT_COLOR, 3.6))
        q2_ref = always_redraw(lambda: self.create_vector_arrow(plane, np.zeros(2), refl_basis(e2, s_perp.get_value()), J_HAT_COLOR, 3.6))
        flag_ref = always_redraw(
            lambda: self.create_chiral_flag(
                plane,
                refl_basis(e1, s_perp.get_value()),
                refl_basis(e2, s_perp.get_value()),
                s_perp.get_value(),
            )
        )
        ra_ref = always_redraw(
            lambda: safe_right_angle(
                plane.c2p(0, 0),
                plane.c2p(*refl_basis(e1, s_perp.get_value())),
                plane.c2p(*refl_basis(e2, s_perp.get_value())),
                length=0.24,
                color=SUBSPACE_COLOR,
            )
        )
        self.add(q1_ref, q2_ref, flag_ref, ra_ref)
        self.play(s_perp.animate.set_value(-1.0), run_time=2.4, rate_func=smooth)

        for mob in [det_num, q1_ref, q2_ref, flag_ref, ra_ref]:
            mob.clear_updaters()

        dl_cw_panel = make_hud_panel(
            VGroup(
                MathTex(r"\det(H_1) = -1.00", font_size=17, color=ERROR_COLOR),
                Text("Orientation: CW (-1) FLIPPED", font_size=14, color=ERROR_COLOR),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.08),
            min_width=3.6,
            stroke_color=ERROR_COLOR,
        ).to_corner(DL, buff=0.25)
        self.play(FadeOut(dl_panel), run_time=0.25)
        dl_panel = dl_cw_panel
        self.play(FadeIn(dl_panel), run_time=0.3)
        self.wait(0.6)

        self.remove(q1_ref, q2_ref, flag_ref, ra_ref)

        # Beat 3: Cartan-Dieudonne Double Reflection H2 @ H1 = R(2 * Delta_alpha) = R(60 deg)
        u_mir2 = np.array([math.cos(ALPHA_2), math.sin(ALPHA_2)])
        n_mir2 = np.array([-math.sin(ALPHA_2), math.cos(ALPHA_2)])
        mir2_line = DashedLine(
            plane.c2p(*(-2.5 * u_mir2)),
            plane.c2p(*(2.5 * u_mir2)),
            color=VEC_COLOR,
            stroke_width=2.2,
            dash_length=0.12,
        )
        mir2_lbl = MathTex(r"\ell_2 \;(\alpha_2=45^\circ)", font_size=15, color=VEC_COLOR).move_to(
            plane.c2p(*(1.85 * u_mir2 + 0.35 * n_mir2))
        )
        da_arc = safe_angle_arc(plane.c2p(0, 0), plane.c2p(*(1.2 * u_mir1)), plane.c2p(*(1.2 * u_mir2)), radius=0.95, color=TARGET_COLOR)

        x0 = np.array([1.55, 0.15])
        x1 = H_1 @ x0
        x2 = H_2 @ x1  # == R_60 @ x0

        arr_x0 = self.create_vector_arrow(plane, np.zeros(2), x0, TEXT_COLOR, 3.2)
        arr_x1 = self.create_vector_arrow(plane, np.zeros(2), x1, ERROR_COLOR, 3.2)
        arr_x2 = self.create_vector_arrow(plane, np.zeros(2), x2, I_HAT_COLOR, 3.6)
        lbl_x0 = MathTex(r"\mathbf{x}", font_size=16, color=TEXT_COLOR).next_to(arr_x0.get_end(), DR, buff=0.06)
        lbl_x1 = MathTex(r"H_1\mathbf{x}", font_size=16, color=ERROR_COLOR).next_to(arr_x1.get_end(), UP, buff=0.06)
        lbl_x2 = MathTex(r"H_2 H_1\mathbf{x} = R(60^\circ)\mathbf{x}", font_size=15, color=I_HAT_COLOR).next_to(
            arr_x2.get_end(), UR, buff=0.06
        )
        sweep_arc = safe_angle_arc(plane.c2p(0, 0), plane.c2p(*x0), plane.c2p(*x2), radius=0.65, color=I_HAT_COLOR, stroke_width=2.8)

        cd_ur = make_hud_panel(
            MathTex(
                r"H_2 H_1 = R(2\Delta\alpha) = R(60^\circ) \in SO(2)",
                font_size=16,
                color=I_HAT_COLOR,
            ),
            stroke_color=I_HAT_COLOR,
        ).to_corner(UR, buff=0.25)
        cd_dl = make_hud_panel(
            VGroup(
                MathTex(r"\det(H_2 H_1) = +1.00", font_size=17, color=I_HAT_COLOR),
                Text("Orientation: CCW (+1) RESTORED", font_size=14, color=I_HAT_COLOR),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.08),
            min_width=3.6,
            stroke_color=I_HAT_COLOR,
        ).to_corner(DL, buff=0.25)
        cd_dr = make_hud_panel(
            MathTex(r"\det(H_2 H_1) = (-1)(-1) = +1 \;\text{(CCW Restored)}", font_size=15, color=I_HAT_COLOR),
            min_width=4.5,
            stroke_color=I_HAT_COLOR,
        ).to_corner(DR, buff=0.25)

        self.play(
            Create(mir2_line),
            FadeIn(mir2_lbl),
            Create(da_arc),
            FadeOut(ur_panel),
            FadeOut(dl_panel),
            FadeOut(dr_panel),
            run_time=0.5,
        )
        ur_panel, dl_panel, dr_panel = cd_ur, cd_dl, cd_dr
        self.play(FadeIn(ur_panel), FadeIn(dl_panel), FadeIn(dr_panel), run_time=0.4)
        self.play(GrowArrow(arr_x0), FadeIn(lbl_x0), run_time=0.6)
        self.play(TransformFromCopy(arr_x0, arr_x1), FadeIn(lbl_x1), run_time=0.8)
        self.play(TransformFromCopy(arr_x1, arr_x2), FadeIn(lbl_x2), Create(sweep_arc), run_time=0.9)
        self.wait(1.0)

        beat23_objs = VGroup(
            plane, origin_group, ghost_circ, live_circ,
            mir1_line, mir1_lbl, mir2_line, mir2_lbl, da_arc,
            arr_x0, arr_x1, arr_x2, lbl_x0, lbl_x1, lbl_x2, sweep_arc,
        )
        self.play(FadeOut(beat23_objs), FadeOut(ur_panel), FadeOut(dl_panel), FadeOut(dr_panel), run_time=0.55)

        # Beat 4: 2D Commutativity vs. 3D SO(3) Non-Commutativity (Projected 3D Cubes)
        so3_ur = make_hud_panel(
            VGroup(
                MathTex(r"\text{2D: } R(\alpha)R(\beta) = R(\beta)R(\alpha)", font_size=15, color=I_HAT_COLOR),
                MathTex(r"\text{3D: } R_2(90^\circ)R_1(90^\circ) \ne R_1(90^\circ)R_2(90^\circ)", font_size=15, color=ERROR_COLOR),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.06),
            stroke_color=PANEL_BORDER,
        ).to_corner(UR, buff=0.25)
        so3_dl = make_hud_panel(
            MathTex(r"(R_2 R_1)\mathbf{e}_1 = [0,\; 0,\; -1]^\top", font_size=16, color=TARGET_COLOR),
            min_width=4.0,
            stroke_color=TARGET_COLOR,
        ).to_corner(DL, buff=0.25)
        so3_dr = make_hud_panel(
            MathTex(r"(R_1 R_2)\mathbf{e}_1 = [0,\; 1,\; 0]^\top \;\;(\|\Delta\|_2 = \sqrt{2})", font_size=16, color=ERROR_COLOR),
            min_width=4.6,
            stroke_color=ERROR_COLOR,
        ).to_corner(DR, buff=0.25)

        ur_panel, dl_panel, dr_panel = so3_ur, so3_dl, so3_dr
        self.play(FadeIn(ur_panel), FadeIn(dl_panel), FadeIn(dr_panel), run_time=0.45)

        c_left = LEFT * 3.1 + DOWN * 0.15
        c_right = RIGHT * 3.1 + DOWN * 0.15
        lbl_3d_l = Text("Order A: R_x(90 deg) then R_y(90 deg)", font_size=15, color=TARGET_COLOR, weight=BOLD).move_to(
            LEFT * 3.1 + UP * 1.75
        )
        lbl_3d_r = Text("Order B: R_y(90 deg) then R_x(90 deg)", font_size=15, color=ERROR_COLOR, weight=BOLD).move_to(
            RIGHT * 3.1 + UP * 1.75
        )
        self.play(FadeIn(lbl_3d_l), FadeIn(lbl_3d_r), run_time=0.4)

        a_l = ValueTracker(0.0)
        b_l = ValueTracker(0.0)
        b_r = ValueTracker(0.0)
        a_r = ValueTracker(0.0)

        cube_verts = np.array([
            [-0.55, -0.55, -0.55], [0.55, -0.55, -0.55], [0.55, 0.55, -0.55], [-0.55, 0.55, -0.55],
            [-0.55, -0.55,  0.55], [0.55, -0.55,  0.55], [0.55, 0.55,  0.55], [-0.55, 0.55,  0.55],
        ], dtype=float)
        cube_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        plus_x_face = [1, 2, 6, 5]

        e1_lbl_tmpl_l = MathTex(r"\mathbf{e}_1", font_size=16, color=TARGET_COLOR)
        e1_lbl_tmpl_r = MathTex(r"\mathbf{e}_1", font_size=16, color=ERROR_COLOR)

        def build_cube_and_triad(R3: np.ndarray, center: np.ndarray, e1_col: str, e1_lbl_tmpl: Mobject) -> VGroup:
            pts2 = [project_iso_3d(R3 @ v, center, scale=1.15) for v in cube_verts]
            grp = VGroup()
            face_poly = Polygon(
                *[pts2[idx] for idx in plus_x_face],
                fill_color=e1_col,
                fill_opacity=0.22,
                stroke_width=0.0,
            )
            grp.add(face_poly)
            for i_e, j_e in cube_edges:
                grp.add(Line(pts2[i_e], pts2[j_e], color=GHOST_COLOR, stroke_width=1.6))
            o2 = project_iso_3d(np.zeros(3), center, scale=1.15)
            axes_spec = [
                (np.array([1.15, 0.0, 0.0]), e1_col, 4.0),
                (np.array([0.0, 1.0, 0.0]), J_HAT_COLOR, 2.8),
                (np.array([0.0, 0.0, 1.0]), K_HAT_COLOR, 2.8),
            ]
            for vec3, col, sw in axes_spec:
                tip2 = project_iso_3d(R3 @ vec3, center, scale=1.15)
                grp.add(Arrow(o2, tip2, buff=0, color=col, stroke_width=sw, max_tip_length_to_length_ratio=0.22))
            e1_lbl_pos = project_iso_3d(R3 @ np.array([1.35, 0.0, 0.0]), center, scale=1.15)
            grp.add(e1_lbl_tmpl.copy().move_to(e1_lbl_pos))
            grp.add(Dot(o2, radius=0.06, color=ORIGIN_COLOR))
            return grp

        cube_l = always_redraw(lambda: build_cube_and_triad(rot_y_3d(b_l.get_value()) @ rot_x_3d(a_l.get_value()), c_left, TARGET_COLOR, e1_lbl_tmpl_l))
        cube_r = always_redraw(lambda: build_cube_and_triad(rot_x_3d(a_r.get_value()) @ rot_y_3d(b_r.get_value()), c_right, ERROR_COLOR, e1_lbl_tmpl_r))
        self.add(cube_l, cube_r)

        # Step 1: Left applies R_x(90 deg), Right applies R_y(90 deg)
        self.play(a_l.animate.set_value(PI / 2.0), b_r.animate.set_value(PI / 2.0), run_time=1.4, rate_func=smooth)
        self.wait(0.3)
        # Step 2: Left applies R_y(90 deg), Right applies R_x(90 deg)
        self.play(b_l.animate.set_value(PI / 2.0), a_r.animate.set_value(PI / 2.0), run_time=1.4, rate_func=smooth)
        self.wait(1.4)

        cube_l.clear_updaters()
        cube_r.clear_updaters()

        cleanup = VGroup(
            title_panel, ur_panel, dl_panel, dr_panel,
            lbl_3d_l, lbl_3d_r, cube_l, cube_r,
        )
        if is_standalone:
            self.play(FadeOut(cleanup), run_time=0.7)
            return None
        return cleanup

    # -------------------------------------------------------------------------
    # Act 3: Color-Coded Algebraic Morph (Q^T Q = I, Givens & Householder)
    # -------------------------------------------------------------------------
    def run_act3(self, is_standalone: bool = False) -> VGroup | None:
        """Act 3: Q^T Q = I -> Q^-1 = Q^T, Givens circular zeroing, and Householder bounce."""
        title_content = VGroup(
            Text("Act 3: Q^T Q = I, Givens & Householder", font_size=18, color=TEXT_COLOR, weight=BOLD),
            Text("Two orthogonal ways to zero entries without losing length", font_size=13, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        title_panel = make_hud_panel(title_content).to_corner(UL, buff=0.25)

        ur_eq1 = make_hud_panel(
            MathTex(
                r"Q^\top Q = \begin{bmatrix} \mathbf{q}_1^\top \\ \mathbf{q}_2^\top \end{bmatrix}"
                r"\begin{bmatrix} \mathbf{q}_1 & \mathbf{q}_2 \end{bmatrix} = I_2 \implies Q^{-1} = Q^\top",
                font_size=15,
                color=TEXT_COLOR,
            )
        ).to_corner(UR, buff=0.25)

        plane = self.create_plane()
        origin_group = self.create_origin_anchor(plane)
        ghost_r1 = self.create_ghost_circle(plane, radius=1.0)

        q1 = Q_PYTH[:, 0]
        q2 = Q_PYTH[:, 1]
        arr_q1 = self.create_vector_arrow(plane, np.zeros(2), q1, I_HAT_COLOR, 3.4)
        arr_q2 = self.create_vector_arrow(plane, np.zeros(2), q2, J_HAT_COLOR, 3.4)
        lbl_q1 = MathTex(r"\mathbf{q}_1", font_size=16, color=I_HAT_COLOR).next_to(arr_q1.get_end(), UR, buff=0.06)
        lbl_q2 = MathTex(r"\mathbf{q}_2", font_size=16, color=J_HAT_COLOR).next_to(arr_q2.get_end(), UL, buff=0.06)
        ra_q = safe_right_angle(plane.c2p(0, 0), plane.c2p(*q1), plane.c2p(*q2), length=0.22, color=SUBSPACE_COLOR)

        x_b1 = np.array([1.4, 0.4])
        qx_b1 = Q_PYTH @ x_b1
        arr_xb1 = self.create_vector_arrow(plane, np.zeros(2), x_b1, VEC_COLOR, 3.4)
        arr_qxb1 = self.create_vector_arrow(plane, np.zeros(2), qx_b1, TARGET_COLOR, 3.4)
        lbl_xb1 = MathTex(r"\mathbf{x}", font_size=16, color=VEC_COLOR).next_to(arr_xb1.get_end(), RIGHT, buff=0.08)
        lbl_qxb1 = MathTex(r"Q\mathbf{x}", font_size=16, color=TARGET_COLOR).next_to(arr_qxb1.get_end(), UP, buff=0.08)

        dl_b1 = make_hud_panel(
            MathTex(r"(\det Q)^2 = \det(Q^\top Q) = 1 \implies \det Q = \pm 1", font_size=15, color=MUTED_TEXT),
            min_width=4.6,
        ).to_corner(DL, buff=0.25)
        dr_b1 = make_hud_panel(
            MathTex(r"\mathbf{x} \xrightarrow{Q} Q\mathbf{x} \xrightarrow{Q^\top} Q^\top(Q\mathbf{x}) = \mathbf{x}", font_size=15, color=I_HAT_COLOR),
            min_width=4.4,
            stroke_color=I_HAT_COLOR,
        ).to_corner(DR, buff=0.25)

        self.play(
            FadeIn(title_panel),
            FadeIn(ur_eq1),
            FadeIn(dl_b1),
            FadeIn(dr_b1),
            Create(plane),
            FadeIn(origin_group),
            Create(ghost_r1),
            GrowArrow(arr_q1),
            GrowArrow(arr_q2),
            FadeIn(lbl_q1),
            FadeIn(lbl_q2),
            Create(ra_q),
            run_time=1.0,
        )
        self.play(GrowArrow(arr_xb1), FadeIn(lbl_xb1), run_time=0.6)
        self.play(TransformFromCopy(arr_xb1, arr_qxb1), FadeIn(lbl_qxb1), run_time=0.8)
        arr_back = self.create_vector_arrow(plane, np.zeros(2), x_b1, I_HAT_COLOR, 3.4)
        self.play(Transform(arr_qxb1, arr_back), FadeOut(lbl_qxb1), run_time=0.8)
        self.wait(0.6)

        self.play(
            FadeOut(VGroup(ghost_r1, arr_q1, arr_q2, lbl_q1, lbl_q2, ra_q, arr_xb1, lbl_xb1, arr_qxb1)),
            run_time=0.5,
        )

        # Beat 2: Givens Rotation G(0, 2, theta) -> Planar slice x_giv2d = [1.2, 1.6] -> [2.0, 0.0]
        ghost_r2 = self.create_ghost_circle(plane, radius=2.0, color=LINE_COLOR, stroke_width=2.0)
        giv_ur = make_hud_panel(
            MathTex(
                r"G = \begin{bmatrix} c & -s \\ s & c \end{bmatrix} = \begin{bmatrix} 0.6 & 0.8 \\ -0.8 & 0.6 \end{bmatrix} \in SO(2)",
                font_size=15,
                color=LINE_COLOR,
            ),
            stroke_color=LINE_COLOR,
        ).to_corner(UR, buff=0.25)

        phi_tracker = ValueTracker(0.0)
        phi_target = -math.atan2(X_GIV_2D[1], X_GIV_2D[0])  # -atan2(1.6, 1.2) ~ -53.13 deg

        def giv_vec() -> np.ndarray:
            return rot2d(phi_tracker.get_value()) @ X_GIV_2D

        giv_arrow = always_redraw(lambda: self.create_vector_arrow(plane, np.zeros(2), giv_vec(), VEC_COLOR, 3.8))
        giv_drop_x = always_redraw(
            lambda: DashedLine(
                plane.c2p(float(giv_vec()[0]), float(giv_vec()[1])),
                plane.c2p(float(giv_vec()[0]), 0.0),
                color=ERROR_COLOR,
                stroke_width=2.0,
                dash_length=0.08,
            )
        )
        giv_dot = always_redraw(lambda: Dot(plane.c2p(*giv_vec()), radius=0.07, color=TARGET_COLOR))

        xj_num = DecimalNumber(1.60, num_decimal_places=2, font_size=16, color=ERROR_COLOR, edge_to_fix=LEFT).set_z_index(12)
        xj_num.add_updater(lambda m: m.set_value(float(giv_vec()[1])).set_z_index(12))
        r_num = DecimalNumber(2.00, num_decimal_places=2, font_size=16, color=I_HAT_COLOR, edge_to_fix=LEFT).set_z_index(12)

        dl_giv = make_hud_panel(
            VGroup(
                VGroup(MathTex(r"\text{Givens target } x_j =", font_size=16, color=TEXT_COLOR), xj_num).arrange(RIGHT, buff=0.08),
                VGroup(MathTex(r"r = \|\mathbf{x}\|_2 =", font_size=16, color=TEXT_COLOR), r_num).arrange(RIGHT, buff=0.08),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.08),
            min_width=4.0,
            stroke_color=LINE_COLOR,
        ).to_corner(DL, buff=0.25)

        dr_giv = make_hud_panel(
            MathTex(r"G_{02}[3,2,4]^\top = [5,2,0]^\top\;\;(n-2\text{ coords untouched})", font_size=14, color=LINE_COLOR),
            min_width=4.8,
            stroke_color=LINE_COLOR,
        ).to_corner(DR, buff=0.25)

        self.play(
            Create(ghost_r2),
            FadeOut(ur_eq1),
            FadeOut(dl_b1),
            FadeOut(dr_b1),
            run_time=0.4,
        )
        ur_eq1, dl_b1, dr_b1 = giv_ur, dl_giv, dr_giv
        self.play(FadeIn(ur_eq1), FadeIn(dl_b1), FadeIn(dr_b1), run_time=0.4)
        self.add(giv_arrow, giv_drop_x, giv_dot)
        self.play(phi_tracker.animate.set_value(phi_target), run_time=2.5, rate_func=smooth)
        self.wait(0.8)

        for mob in [giv_arrow, giv_drop_x, giv_dot, xj_num]:
            mob.clear_updaters()

        # Beat 3: Householder Reflection H = I - 2 v v^T / ||v||^2
        # v = [3.2, 1.6]^T, v^perp along [-1, 2]^T, target [-2.0, 0.0]^T, midpoint [-0.4, 0.8]^T
        self.remove(giv_arrow, giv_drop_x, giv_dot)

        v_unit = V_HH_2D / np.linalg.norm(V_HH_2D)
        v_perp_unit = np.array([-v_unit[1], v_unit[0]])
        hh_mirror = Line(
            plane.c2p(*(-2.25 * v_perp_unit)),
            plane.c2p(*(2.25 * v_perp_unit)),
            color=SUBSPACE_COLOR,
            stroke_width=2.6,
        )
        hh_mirror_lbl = MathTex(r"\mathbf{v}^\perp", font_size=16, color=SUBSPACE_COLOR).move_to(
            plane.c2p(*(1.85 * v_perp_unit + 0.25 * v_unit))
        )
        v_arrow = self.create_vector_arrow(plane, np.zeros(2), 1.3 * v_unit, VEC_COLOR, 3.2)
        v_lbl = MathTex(r"\hat{\mathbf{v}}", font_size=16, color=VEC_COLOR).next_to(v_arrow.get_end(), UR, buff=0.06)

        hx_target = H_HH_2D @ X_HH_2D  # [-2.0, 0.0]
        midpoint = 0.5 * (X_HH_2D + hx_target)  # [-0.4, 0.8]
        chord = DashedLine(plane.c2p(*X_HH_2D), plane.c2p(*hx_target), color=TARGET_COLOR, stroke_width=2.0, dash_length=0.1)
        ra_origin = safe_right_angle(plane.c2p(0, 0), plane.c2p(*v_unit), plane.c2p(*v_perp_unit), length=0.20, color=SUBSPACE_COLOR)
        ra_mid = safe_right_angle(plane.c2p(*midpoint), plane.c2p(*X_HH_2D), plane.c2p(*(midpoint + v_perp_unit)), length=0.20, color=TARGET_COLOR)

        hh_ur = make_hud_panel(
            MathTex(
                r"H = I - 2\frac{\mathbf{v}\mathbf{v}^\top}{\|\mathbf{v}\|_2^2},\;\; H^2=I,\;\; \det(H)=-1",
                font_size=15,
                color=SUBSPACE_COLOR,
            ),
            stroke_color=SUBSPACE_COLOR,
        ).to_corner(UR, buff=0.25)
        hh_dl = make_hud_panel(
            MathTex(r"\mathbf{v} = \mathbf{x} + \operatorname{sgn}(x_1)\|\mathbf{x}\|_2\mathbf{e}_1", font_size=15, color=VEC_COLOR),
            min_width=4.2,
            stroke_color=VEC_COLOR,
        ).to_corner(DL, buff=0.25)
        hh_dr = make_hud_panel(
            MathTex(r"H_{\text{3D}}[1,2,2]^\top = [-3,0,0]^\top\;\text{(1-step subcol zero)}", font_size=14, color=TARGET_COLOR),
            min_width=4.8,
            stroke_color=TARGET_COLOR,
        ).to_corner(DR, buff=0.25)

        s_hh = ValueTracker(0.0)
        proj_coeff = float(np.dot(V_HH_2D, X_HH_2D) / np.dot(V_HH_2D, V_HH_2D))
        hh_arrow = always_redraw(
            lambda: self.create_vector_arrow(
                plane,
                np.zeros(2),
                X_HH_2D - s_hh.get_value() * proj_coeff * V_HH_2D,
                TARGET_COLOR,
                3.8,
            )
        )
        hh_tip_dot = always_redraw(
            lambda: Dot(
                plane.c2p(*(X_HH_2D - s_hh.get_value() * proj_coeff * V_HH_2D)),
                radius=0.07,
                color=TARGET_COLOR,
            )
        )

        self.play(
            FadeOut(ur_eq1),
            FadeOut(dl_b1),
            FadeOut(dr_b1),
            Create(hh_mirror),
            FadeIn(hh_mirror_lbl),
            GrowArrow(v_arrow),
            FadeIn(v_lbl),
            Create(chord),
            Create(ra_origin),
            Create(ra_mid),
            run_time=0.6,
        )
        ur_eq1, dl_b1, dr_b1 = hh_ur, hh_dl, hh_dr
        self.play(FadeIn(ur_eq1), FadeIn(dl_b1), FadeIn(dr_b1), run_time=0.4)
        self.add(hh_arrow, hh_tip_dot)
        self.play(s_hh.animate.set_value(2.0), run_time=2.6, rate_func=smooth)
        self.wait(1.0)

        hh_arrow.clear_updaters()
        hh_tip_dot.clear_updaters()

        cleanup = VGroup(
            title_panel, ur_eq1, dl_b1, dr_b1,
            plane, origin_group, ghost_r2,
            hh_mirror, hh_mirror_lbl, v_arrow, v_lbl, chord, ra_origin, ra_mid, hh_arrow, hh_tip_dot,
        )
        if is_standalone:
            self.play(FadeOut(cleanup), run_time=0.7)
            return None
        return cleanup

    # -------------------------------------------------------------------------
    # Act 4: Degenerate & Edge-Case Stress Test
    # -------------------------------------------------------------------------
    def run_act4(self, is_standalone: bool = False) -> VGroup | None:
        """Act 4: Householder cancellation, Givens (0,0) guard, Q^1000 drift, and sparsity grid."""
        title_content = VGroup(
            Text("Act 4: Numerical Stability & Edge Cases", font_size=18, color=TEXT_COLOR, weight=BOLD),
            Text("Cancellation avoidance, Q^1000 drift, and Givens vs Householder", font_size=13, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        title_panel = make_hud_panel(title_content).to_corner(UL, buff=0.25)

        ur_panel = make_hud_panel(
            MathTex(r"\mathbf{x}_{\text{deg}} = [1.0,\; 10^{-9}]^\top,\;\;\epsilon_{\text{mach}} \approx 2.22\times 10^{-16}", font_size=15, color=TEXT_COLOR),
        ).to_corner(UR, buff=0.25)

        # Beat 1: Side-by-side Householder Sign Cancellation vs Constructive Sign + Givens Guard
        bad_card_content = VGroup(
            Text("Naive Sign: v = x - ||x|| e_1", font_size=16, color=ERROR_COLOR, weight=BOLD),
            MathTex(r"\|\mathbf{x}\|_2 = \sqrt{1 + 10^{-18}} \equiv 1.0 \text{ (float64)}", font_size=15, color=TEXT_COLOR),
            MathTex(r"\mathbf{v}_{\text{bad}} = [1 - 1,\; 10^{-9}]^\top = [0,\; 10^{-9}]^\top", font_size=15, color=ERROR_COLOR),
            MathTex(r"H_{\text{bad}}\mathbf{x} = [1.0,\; -10^{-9}]^\top \;\;\text{(FAILS!)}", font_size=16, color=ERROR_COLOR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        bad_card = make_hud_panel(bad_card_content, padding=0.22, min_width=5.4, stroke_color=ERROR_COLOR).move_to(
            LEFT * 3.15 + UP * 0.25
        )

        good_card_content = VGroup(
            Text("Stable Sign: v = x + sgn(x_1)||x|| e_1", font_size=16, color=I_HAT_COLOR, weight=BOLD),
            MathTex(r"\sigma = +\operatorname{sgn}(x_1) = +1 \;\text{(adds magnitudes)}", font_size=15, color=TEXT_COLOR),
            MathTex(r"\mathbf{v}_{\text{good}} = [1 + 1,\; 10^{-9}]^\top = [2.0,\; 10^{-9}]^\top", font_size=15, color=I_HAT_COLOR),
            MathTex(r"H_{\text{good}}\mathbf{x} = [-1.0,\; 0.0]^\top \;\;(|x_2|<10^{-15})", font_size=16, color=I_HAT_COLOR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        good_card = make_hud_panel(good_card_content, padding=0.22, min_width=5.4, stroke_color=I_HAT_COLOR).move_to(
            RIGHT * 3.15 + UP * 0.25
        )

        giv_guard = make_hud_panel(
            MathTex(
                r"\text{Givens Guard: } x_i=x_j=0 \implies r=\operatorname{hypot}(0,0)=0 \implies G=I_n \text{ (no NaN)}",
                font_size=15,
                color=LINE_COLOR,
            ),
            stroke_color=LINE_COLOR,
        ).to_edge(DOWN, buff=0.45)

        self.play(
            FadeIn(title_panel),
            FadeIn(ur_panel),
            FadeIn(bad_card),
            FadeIn(good_card),
            FadeIn(giv_guard),
            run_time=1.0,
        )
        self.wait(2.2)
        self.play(FadeOut(VGroup(bad_card, good_card, giv_guard)), run_time=0.6)

        # Beat 2: Repeated Power Drift Q^1000 (n = 0 -> 1000)
        orbit_plane = self.create_orbit_plane(LEFT * 3.25 + DOWN * 0.12)
        orbit_orig = self.create_origin_anchor(orbit_plane)
        unit_circ = self.create_ghost_circle(orbit_plane, radius=1.0, color=I_HAT_COLOR, stroke_width=2.2)

        axes_norm = Axes(
            x_range=[0, 1000, 250],
            y_range=[0, 5, 1],
            x_length=4.8,
            y_length=3.3,
            axis_config={"color": TEXT_COLOR, "stroke_width": 1.6, "include_tip": True},
        ).move_to(RIGHT * 3.1 + DOWN * 0.08)
        ax_x_lbl = MathTex(r"n \text{ (iterations)}", font_size=14, color=MUTED_TEXT).next_to(axes_norm.x_axis, DOWN, buff=0.12)
        ax_y_lbl = MathTex(r"\|M^n \mathbf{x}_0\|_2", font_size=14, color=MUTED_TEXT).next_to(axes_norm.y_axis, UP, buff=0.10)

        n_tracker = ValueTracker(0.0)
        theta_step = 0.08

        def spiral_pt(scale_r: float, step_k: float) -> np.ndarray:
            r = scale_r ** step_k
            ang = step_k * theta_step
            return np.array([r * math.cos(ang), r * math.sin(ang)])

        trace_exact = always_redraw(
            lambda: ParametricFunction(
                lambda k: orbit_plane.c2p(*spiral_pt(1.0, k)),
                t_range=[0.0, max(n_tracker.get_value(), 2.0), 2.5],
                color=I_HAT_COLOR,
                stroke_width=2.4,
            )
        )
        trace_grow = always_redraw(
            lambda: ParametricFunction(
                lambda k: orbit_plane.c2p(*spiral_pt(1.0012, k)),
                t_range=[0.0, max(n_tracker.get_value(), 2.0), 2.5],
                color=ERROR_COLOR,
                stroke_width=2.2,
            )
        )
        trace_shrink = always_redraw(
            lambda: ParametricFunction(
                lambda k: orbit_plane.c2p(*spiral_pt(0.998, k)),
                t_range=[0.0, max(n_tracker.get_value(), 2.0), 2.5],
                color=MUTED_TEXT,
                stroke_width=2.0,
            )
        )

        graph_exact = always_redraw(
            lambda: axes_norm.plot(lambda k: 1.0, x_range=[0.0, max(n_tracker.get_value(), 5.0), 5.0], color=I_HAT_COLOR, stroke_width=2.6)
        )
        graph_grow = always_redraw(
            lambda: axes_norm.plot(lambda k: 1.0012**k, x_range=[0.0, max(n_tracker.get_value(), 5.0), 5.0], color=ERROR_COLOR, stroke_width=2.4)
        )
        graph_shrink = always_redraw(
            lambda: axes_norm.plot(lambda k: 0.998**k, x_range=[0.0, max(n_tracker.get_value(), 5.0), 5.0], color=MUTED_TEXT, stroke_width=2.2)
        )

        n_num = DecimalNumber(0, num_decimal_places=0, font_size=16, color=TEXT_COLOR, edge_to_fix=LEFT).set_z_index(12)
        n_num.add_updater(lambda m: m.set_value(n_tracker.get_value()).set_z_index(12))
        grow_num = DecimalNumber(1.00, num_decimal_places=2, font_size=16, color=ERROR_COLOR, edge_to_fix=LEFT).set_z_index(12)
        grow_num.add_updater(lambda m: m.set_value(1.0012 ** n_tracker.get_value()).set_z_index(12))

        dl_drift = make_hud_panel(
            VGroup(
                VGroup(MathTex(r"n =", font_size=16, color=TEXT_COLOR), n_num).arrange(RIGHT, buff=0.08),
                MathTex(r"\|Q^{1000\top}Q^{1000} - I\|_F \approx 1.5\times 10^{-13}", font_size=14, color=I_HAT_COLOR),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.08),
            min_width=4.6,
            stroke_color=I_HAT_COLOR,
        ).to_corner(DL, buff=0.25)

        dr_drift = make_hud_panel(
            VGroup(
                VGroup(MathTex(r"\|(1.0012 R)^n \mathbf{x}_0\|_2 =", font_size=15, color=TEXT_COLOR), grow_num).arrange(RIGHT, buff=0.08),
                MathTex(r"\|(0.998 R)^{1000}\|_2 \approx 0.14 \;\text{(collapse)}", font_size=14, color=MUTED_TEXT),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.08),
            min_width=4.6,
            stroke_color=ERROR_COLOR,
        ).to_corner(DR, buff=0.25)

        ur_drift = make_hud_panel(
            MathTex(r"|\lambda_i(Q)| = 1 \implies \text{No exponential blowup or decay}", font_size=15, color=I_HAT_COLOR),
            stroke_color=I_HAT_COLOR,
        ).to_corner(UR, buff=0.25)

        self.play(
            FadeOut(ur_panel),
            Create(orbit_plane),
            FadeIn(orbit_orig),
            Create(unit_circ),
            Create(axes_norm),
            FadeIn(ax_x_lbl),
            FadeIn(ax_y_lbl),
            FadeIn(dl_drift),
            FadeIn(dr_drift),
            run_time=0.6,
        )
        ur_panel = ur_drift
        self.play(FadeIn(ur_panel), run_time=0.35)
        self.add(trace_exact, trace_grow, trace_shrink, graph_exact, graph_grow, graph_shrink)
        self.play(n_tracker.animate.set_value(float(Q_DRIFT_N)), run_time=3.0, rate_func=linear)
        self.wait(0.8)

        for mob in [trace_exact, trace_grow, trace_shrink, graph_exact, graph_grow, graph_shrink, n_num, grow_num]:
            mob.clear_updaters()

        beat2_objs = VGroup(
            orbit_plane, orbit_orig, unit_circ, axes_norm, ax_x_lbl, ax_y_lbl,
            trace_exact, trace_grow, trace_shrink, graph_exact, graph_grow, graph_shrink,
            dl_drift, dr_drift,
        )
        self.play(FadeOut(beat2_objs), run_time=0.6)

        # Beat 3: Givens vs. Householder 4x4 Sparsity Grid & Final Summary Card
        def make_4x4_grid(highlight_cells: dict[tuple[int, int], str], title_str: str, subtitle_str: str, col: str) -> VGroup:
            cells = VGroup()
            for r in range(4):
                for c in range(4):
                    fill_c = highlight_cells.get((r, c), PANEL_BG)
                    op = 0.65 if (r, c) in highlight_cells else 0.85
                    sq = Square(side_length=0.48, fill_color=fill_c, fill_opacity=op, stroke_color=PANEL_BORDER, stroke_width=1.4)
                    sq.move_to(np.array([(c - 1.5) * 0.52, (1.5 - r) * 0.52, 0.0]))
                    cells.add(sq)
            t = Text(title_str, font_size=15, color=col, weight=BOLD)
            s = Text(subtitle_str, font_size=12, color=TEXT_COLOR)
            return VGroup(t, cells, s).arrange(DOWN, buff=0.14)

        giv_cells = {(2, j): LINE_COLOR for j in range(4)} | {(3, j): LINE_COLOR for j in range(4)}
        giv_cells[(3, 0)] = TARGET_COLOR
        hh_cells = {(r, c): SUBSPACE_COLOR for r in range(4) for c in range(4)}
        hh_cells[(1, 0)] = TARGET_COLOR
        hh_cells[(2, 0)] = TARGET_COLOR
        hh_cells[(3, 0)] = TARGET_COLOR

        grid_l = make_hud_panel(
            make_4x4_grid(giv_cells, "Givens G(2, 3, theta)", "Modifies 2 rows -> Sparse / Banded O(n)", LINE_COLOR),
            padding=0.20,
            stroke_color=LINE_COLOR,
        ).move_to(LEFT * 3.0 + UP * 0.1)
        grid_r = make_hud_panel(
            make_4x4_grid(hh_cells, "Householder H_1", "Dense rank-1 update -> 2x Fewer FLOPs for QR", SUBSPACE_COLOR),
            padding=0.20,
            stroke_color=SUBSPACE_COLOR,
        ).move_to(RIGHT * 3.0 + UP * 0.1)

        self.play(FadeIn(grid_l), FadeIn(grid_r), run_time=0.8)
        self.wait(1.2)
        self.play(FadeOut(VGroup(grid_l, grid_r)), run_time=0.5)

        summary_content = VGroup(
            Text("Sprint 8 Summary: Orthogonal Matrices & Rotations", font_size=18, color=TARGET_COLOR, weight=BOLD),
            MathTex(r"1.\;\;Q^\top Q = I_n \iff Q^{-1} = Q^\top,\;\; \|Q\mathbf{x}\|_2 = \|\mathbf{x}\|_2,\;\; \kappa_2(Q) = 1", font_size=15, color=TEXT_COLOR),
            MathTex(r"2.\;\;SO(n)\;(\det=+1) \text{ vs } O(n)\setminus SO(n)\;(\det=-1);\;\; H_2 H_1 = R(2\Delta\alpha)", font_size=15, color=I_HAT_COLOR),
            MathTex(r"3.\;\;\text{Givens } G(i,j,\theta)\;\text{(sparse 1-entry zero) vs Householder } H\;\text{(dense subcol)}", font_size=15, color=LINE_COLOR),
            MathTex(r"4.\;\;\mathbf{v} = \mathbf{x} + \operatorname{sgn}(x_1)\|\mathbf{x}\|_2\mathbf{e}_1 \text{ prevents cancellation};\;\; |\lambda_i(Q)| = 1", font_size=15, color=VEC_COLOR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        summary_panel = make_hud_panel(summary_content, padding=0.26, stroke_color=TARGET_COLOR).move_to(ORIGIN).set_z_index(20)

        self.play(FadeIn(summary_panel), run_time=0.8)
        self.wait(2.4)

        cleanup = VGroup(title_panel, ur_panel, summary_panel)
        if is_standalone:
            self.play(FadeOut(cleanup), run_time=0.7)
            return None
        return cleanup


# -----------------------------------------------------------------------------
# Modular Executable Scene Classes
# -----------------------------------------------------------------------------
class Act1SpatialPuzzle(BaseOrthogonalRotationsScene):
    """Standalone executable scene for Act 1."""

    def construct(self):
        self.run_act1(is_standalone=True)


class Act2DynamicDeformation(BaseOrthogonalRotationsScene):
    """Standalone executable scene for Act 2."""

    def construct(self):
        self.run_act2(is_standalone=True)


class Act3AlgebraicMorph(BaseOrthogonalRotationsScene):
    """Standalone executable scene for Act 3."""

    def construct(self):
        self.run_act3(is_standalone=True)


class Act4DegenerateStressTest(BaseOrthogonalRotationsScene):
    """Standalone executable scene for Act 4."""

    def construct(self):
        self.run_act4(is_standalone=True)


class OrthogonalRotationsScene(BaseOrthogonalRotationsScene):
    """Master unified scene seamlessly orchestrating all 4 acts."""

    def construct(self):
        g1 = self.run_act1(is_standalone=False)
        self.play(FadeOut(g1), run_time=0.7)

        g2 = self.run_act2(is_standalone=False)
        self.play(FadeOut(g2), run_time=0.7)

        g3 = self.run_act3(is_standalone=False)
        self.play(FadeOut(g3), run_time=0.7)

        g4 = self.run_act4(is_standalone=False)
        self.play(FadeOut(g4), run_time=0.7)
