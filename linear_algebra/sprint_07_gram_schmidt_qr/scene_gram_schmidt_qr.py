"""
Sprint 7: Gram-Schmidt Orthonormalization & QR Decomposition in 3D (ManimCE v0.21.0+)
Curriculum: Linear Algebra & Theoretical CS / Machine Learning Foundations
References: MML §§3.5, 3.8.2–3.8.3; VMLS §§5.3–5.4, 10.4, 11.3, 11.5, 12.1–12.3; Kreyszig §20.9

This production script implements the 4-act pedagogical narrative in 3D:
- Act 1: Concrete Spatial Puzzle (Skewed 3D Basis {a1, a2, a3}, Parallelepiped & Target b)
- Act 2: Dynamic Deformation (Sequential Orthogonal Stripping onto Unit Sphere S^2 -> Q)
- Act 3: Algebraic Morph (Tip-to-Tail A = QR, Expansion b = Q(Q^T b), Back-Sub & Volume)
- Act 4: Degenerate Stress Test (Coplanar Collapse R_33 = 0 & CGS vs MGS vs Householder Drift)

Render Commands:
    Preview Low-Res:   .venv/bin/manim -pql linear_algebra/sprint_07_gram_schmidt_qr/scene_gram_schmidt_qr.py GramSchmidtQRScene
    Headless Low-Res:  .venv/bin/manim -ql linear_algebra/sprint_07_gram_schmidt_qr/scene_gram_schmidt_qr.py GramSchmidtQRScene
    Production Hi-Res: .venv/bin/manim -pqh linear_algebra/sprint_07_gram_schmidt_qr/scene_gram_schmidt_qr.py GramSchmidtQRScene
    Headless Hi-Res:   .venv/bin/manim -qh linear_algebra/sprint_07_gram_schmidt_qr/scene_gram_schmidt_qr.py GramSchmidtQRScene
    Single Act Test:   .venv/bin/manim -ql linear_algebra/sprint_07_gram_schmidt_qr/scene_gram_schmidt_qr.py Act2DynamicDeformation
"""

from __future__ import annotations

import numpy as np
from manim import (
    BLUE,
    BOLD,
    DEGREES,
    DL,
    DOWN,
    DR,
    LEFT,
    PI,
    RIGHT,
    TAU,
    UL,
    UR,
    Arrow3D,
    Create,
    DashedLine,
    DecimalNumber,
    Dot3D,
    FadeIn,
    FadeOut,
    Line3D,
    MathTex,
    Mobject,
    ParametricFunction,
    Polygon,
    ReplacementTransform,
    RoundedRectangle,
    Surface,
    Text,
    ThreeDAxes,
    ThreeDScene,
    ValueTracker,
    VGroup,
    VMobject,
    always_redraw,
    smooth,
)

# -----------------------------------------------------------------------------
# Rule 1: 3Blue1Brown Semantic Color Palette
# -----------------------------------------------------------------------------
BG_COLOR = "#1C1C24"          # Dark Slate - High-contrast viewport background
PANEL_BG = "#16161E"          # Obsidian   - HUD panel background container
PANEL_BORDER = "#414868"      # Slate-Teal - HUD panel container border
TEXT_COLOR = "#ECEFF4"        # Snow       - Primary typography
MUTED_TEXT = "#7AA2F7"        # Light Blue - Secondary badges & annotations
I_HAT_COLOR = "#83C167"       # Green C    - Column 1 / Basis 1 (a1, q1)
J_HAT_COLOR = "#FC6255"       # Red C      - Column 2 / Basis 2 (a2, q2)
K_HAT_COLOR = "#58C4DD"       # Blue C     - Column 3 / Basis 3 (a3, q3)
SUBSPACE_COLOR = "#2A9D8F"    # Teal       - Subspace span(q1, q2) & 3D right angles
VEC_COLOR = "#F7D96F"         # Yellow     - Projection shadows & stripped residuals
TARGET_COLOR = "#F9E498"      # Gold       - Target vector b = [1, -2, 3]^T
ERROR_COLOR = "#F7768E"       # Coral Red  - Degenerate collapse & CGS drift alert
ORIGIN_COLOR = "#FFFFFF"      # White      - Origin anchor (0, 0, 0)

PHI_DEFAULT = 70 * DEGREES
THETA_DEFAULT = -45 * DEGREES
AMBIENT_ROTATION_RATE = 0.06

# -----------------------------------------------------------------------------
# Mathematical Constants & Assertions (Rule 6: Exact NumPy Precision)
# -----------------------------------------------------------------------------
A_MAT = np.array([
    [2.0, 0.0, -1.0],
    [1.0, 2.0, -1.0],
    [2.0, 2.0, 3.0],
])

Q_MAT = np.array([
    [2.0, -2.0, -1.0],
    [1.0, 2.0, -2.0],
    [2.0, 1.0, 2.0],
]) / 3.0

R_MAT = np.array([
    [3.0, 2.0, 1.0],
    [0.0, 2.0, 1.0],
    [0.0, 0.0, 3.0],
])

A1_COORDS = A_MAT[:, 0].copy()
A2_COORDS = A_MAT[:, 1].copy()
A3_COORDS = A_MAT[:, 2].copy()

Q1_COORDS = Q_MAT[:, 0].copy()
Q2_COORDS = Q_MAT[:, 1].copy()
Q3_COORDS = Q_MAT[:, 2].copy()

U1_VEC = np.array([2.0, 1.0, 2.0])
P21_VEC = 2.0 * Q1_COORDS
U2_VEC = np.array([-4.0 / 3.0, 4.0 / 3.0, 2.0 / 3.0])
P12_3_VEC = np.array([0.0, 1.0, 1.0])
U3_VEC = np.array([-1.0, -2.0, 2.0])

B_TARGET = np.array([1.0, -2.0, 3.0])
QTB_VEC = np.array([2.0, -1.0, 3.0])
X_TRUE = np.array([1.0, -1.0, 1.0])
A3_DEG = np.array([-2.0, 1.0, 0.0])

# Pre-flight mathematical consistency verification
assert np.allclose(Q_MAT.T @ Q_MAT, np.eye(3), atol=1e-12), "Q must be orthonormal"
assert np.allclose(Q_MAT @ R_MAT, A_MAT, atol=1e-12), "A must equal Q @ R"
assert np.isclose(np.linalg.det(Q_MAT), 1.0, atol=1e-12), "det(Q) must be +1"
assert np.isclose(np.linalg.det(A_MAT), 18.0, atol=1e-12), "det(A) must be 18"
assert np.allclose(U1_VEC, A1_COORDS, atol=1e-12), "u1 must equal a1"
assert np.allclose(Q1_COORDS, U1_VEC / 3.0, atol=1e-12), "q1 must be u1 / 3"
assert np.allclose(U2_VEC, A2_COORDS - P21_VEC, atol=1e-12), "u2 = a2 - 2*q1"
assert np.isclose(np.dot(U2_VEC, Q1_COORDS), 0.0, atol=1e-12), "u2 perp q1"
assert np.allclose(Q2_COORDS, U2_VEC / 2.0, atol=1e-12), "q2 must be u2 / 2"
assert np.allclose(P12_3_VEC, 1.0 * Q1_COORDS + 1.0 * Q2_COORDS, atol=1e-12), "p12,3 = q1 + q2"
assert np.allclose(U3_VEC, A3_COORDS - P12_3_VEC, atol=1e-12), "u3 = a3 - p12,3"
assert np.isclose(np.dot(U3_VEC, Q1_COORDS), 0.0, atol=1e-12), "u3 perp q1"
assert np.isclose(np.dot(U3_VEC, Q2_COORDS), 0.0, atol=1e-12), "u3 perp q2"
assert np.allclose(Q3_COORDS, U3_VEC / 3.0, atol=1e-12), "q3 must be u3 / 3"
assert np.allclose(Q_MAT.T @ B_TARGET, QTB_VEC, atol=1e-12), "Q^T b must equal [2, -1, 3]"
assert np.allclose(A_MAT @ X_TRUE, B_TARGET, atol=1e-12), "A @ x_true must equal b"
assert np.allclose(R_MAT @ X_TRUE, QTB_VEC, atol=1e-12), "R @ x_true must equal Q^T b"
assert np.allclose(A3_DEG, A2_COORDS - A1_COORDS, atol=1e-12), "a3_deg = a2 - a1"
assert np.allclose(A3_DEG, -1.0 * Q1_COORDS + 2.0 * Q2_COORDS, atol=1e-12), "a3_deg in span(q1, q2)"


# -----------------------------------------------------------------------------
# Rule 3 & Rule 5: HUD Panel Factory & 3D Geometric Helpers
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
    """Wrap any 2D HUD mobject in a dark rounded rectangle with high z-index."""
    width = max(content.width + 2 * padding, min_width)
    bg = RoundedRectangle(
        corner_radius=corner_radius,
        width=width,
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


def create_right_angle_3d(
    axes: ThreeDAxes,
    origin_coords: np.ndarray,
    u_dir: np.ndarray,
    v_dir: np.ndarray,
    size: float = 0.34,
    color: str = SUBSPACE_COLOR,
) -> VGroup:
    """Build a true 3D right-angle marker using Polygon + VMobject (never 2D RightAngle)."""
    u_hat = u_dir / np.linalg.norm(u_dir)
    v_hat = v_dir / np.linalg.norm(v_dir)

    p0 = axes.c2p(*origin_coords)
    p1 = axes.c2p(*(origin_coords + size * u_hat))
    p2 = axes.c2p(*(origin_coords + size * (u_hat + v_hat)))
    p3 = axes.c2p(*(origin_coords + size * v_hat))

    fill_poly = Polygon(
        p0, p1, p2, p3,
        color=color,
        fill_color=color,
        fill_opacity=0.35,
        stroke_width=0.0,
    )
    elbow = VMobject(color=color, stroke_width=2.2)
    elbow.set_points_as_corners([p1, p2, p3])
    return VGroup(fill_poly, elbow)


def safe_arrow_3d(
    axes: ThreeDAxes,
    start_coords: np.ndarray,
    end_coords: np.ndarray,
    color: str,
    resolution: int = 8,
    thickness: float = 0.022,
    base_radius: float = 0.07,
    height: float = 0.24,
) -> Mobject:
    """Create a lean-resolution Arrow3D, collapsing gracefully to Dot3D near zero length."""
    diff = np.asarray(end_coords, dtype=float) - np.asarray(start_coords, dtype=float)
    if np.linalg.norm(diff) < 0.10:
        return Dot3D(point=axes.c2p(*end_coords), radius=0.075, color=color, resolution=(6, 6))
    return Arrow3D(
        start=axes.c2p(*start_coords),
        end=axes.c2p(*end_coords),
        color=color,
        resolution=resolution,
        thickness=thickness,
        base_radius=base_radius,
        height=height,
    )


def create_parallelepiped_wireframe(
    axes: ThreeDAxes,
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    color: str = "#4C566A",
) -> VGroup:
    """Create the 12 edges of the parallelepiped spanned by {v1, v2, v3}."""
    pts = [
        np.zeros(3),
        v1,
        v1 + v2,
        v2,
        v3,
        v1 + v3,
        v1 + v2 + v3,
        v2 + v3,
    ]
    edge_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    lines = VGroup()
    for i, j in edge_pairs:
        lines.add(
            Line3D(
                start=axes.c2p(*pts[i]),
                end=axes.c2p(*pts[j]),
                color=color,
                thickness=0.012,
                resolution=6,
            )
        )
    return lines


def create_unit_sphere_rings(axes: ThreeDAxes, color: str = "#5E81AC") -> VGroup:
    """Build 3 orthogonal great-circle wireframe rings of unit sphere S^2 for fast 1080p60 render."""
    ring_xy = ParametricFunction(
        lambda t: axes.c2p(np.cos(t), np.sin(t), 0.0),
        t_range=[0.0, TAU],
        color=color,
        stroke_width=1.4,
        stroke_opacity=0.65,
    )
    ring_xz = ParametricFunction(
        lambda t: axes.c2p(np.cos(t), 0.0, np.sin(t)),
        t_range=[0.0, TAU],
        color=color,
        stroke_width=1.4,
        stroke_opacity=0.55,
    )
    ring_yz = ParametricFunction(
        lambda t: axes.c2p(0.0, np.cos(t), np.sin(t)),
        t_range=[0.0, TAU],
        color=color,
        stroke_width=1.4,
        stroke_opacity=0.55,
    )
    return VGroup(ring_xy, ring_xz, ring_yz)


def create_unit_sphere_surface(axes: ThreeDAxes) -> Surface:
    """Create a brief low-res translucent Surface shell for S^2 reveal."""
    return Surface(
        lambda u, v: axes.c2p(
            np.sin(u) * np.cos(v),
            np.sin(u) * np.sin(v),
            np.cos(u),
        ),
        u_range=[0.0, PI],
        v_range=[0.0, TAU],
        resolution=(12, 14),
        fill_color=BLUE,
        fill_opacity=0.07,
        stroke_width=0.0,
    )


def create_subspace_plane(
    axes: ThreeDAxes,
    q1: np.ndarray,
    q2: np.ndarray,
    extent: float = 2.6,
    color: str = SUBSPACE_COLOR,
) -> Surface:
    """Build the 2D planar sheet span(q1, q2) in 3D space."""
    return Surface(
        lambda u, v: axes.c2p(*(u * q1 + v * q2)),
        u_range=[-extent, extent],
        v_range=[-extent, extent],
        resolution=(10, 10),
        fill_color=color,
        fill_opacity=0.28,
        stroke_color=color,
        stroke_width=0.6,
    )


# -----------------------------------------------------------------------------
# Base Scene Class (ThreeDScene Subclass with Isotropic Axes & HUD Pinning)
# -----------------------------------------------------------------------------
class BaseGramSchmidtQRScene(ThreeDScene):
    """Master base class providing isotropic 3D axes, HUD pinning, and 4-act methods."""

    def setup(self):
        super().setup()
        self.camera.background_color = BG_COLOR
        self.set_camera_orientation(phi=PHI_DEFAULT, theta=THETA_DEFAULT)

    def create_axes_3d(self) -> ThreeDAxes:
        """Rule 2: Isotropic 5.6x5.6x5.6 3D Axes on [-4, 4] so S^2 & 90-deg stay unwarped."""
        return ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[-4, 4, 1],
            x_length=5.6,
            y_length=5.6,
            z_length=5.6,
            axis_config={
                "color": "#4C566A",
                "stroke_width": 1.5,
                "include_ticks": True,
                "tick_size": 0.06,
            },
        )

    def pin_hud(self, panel: Mobject, corner=UL, buff: float = 0.25):
        """Rule 3: Pin 2D HUD panel to camera viewport."""
        panel.to_corner(corner, buff=buff)
        self.add_fixed_in_frame_mobjects(panel)

    def unpin_hud(self, *panels: Mobject):
        """Remove panels from BOTH fixed_in_frame_mobjects and scene mobjects (ManimCE 0.21 fix)."""
        self.remove_fixed_in_frame_mobjects(*panels)
        self.remove(*panels)

    # -------------------------------------------------------------------------
    # Act 1: Concrete Spatial Puzzle (Skewed 3D Basis & Coupling)
    # -------------------------------------------------------------------------
    def run_act1(self):
        self.stop_ambient_camera_rotation()
        self.set_camera_orientation(phi=PHI_DEFAULT, theta=THETA_DEFAULT)

        axes = self.create_axes_3d()
        origin_dot = Dot3D(point=axes.c2p(0, 0, 0), radius=0.08, color=ORIGIN_COLOR, resolution=(6, 6))

        title_text = VGroup(
            Text("Sprint 7: Gram-Schmidt & QR Decomposition", font_size=20, color=TEXT_COLOR, weight=BOLD),
            Text("Act 1: Skewed 3D Basis Puzzle", font_size=14, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        title_panel = make_hud_panel(title_text)
        self.pin_hud(title_panel, corner=UL)

        matrix_math = MathTex(
            r"A = \begin{bmatrix} 2 & 0 & -1 \\ 1 & 2 & -1 \\ 2 & 2 & 3 \end{bmatrix}",
            font_size=22,
            color=TEXT_COLOR,
        )
        matrix_panel = make_hud_panel(matrix_math)
        self.pin_hud(matrix_panel, corner=UR)

        self.play(Create(axes), FadeIn(origin_dot), run_time=0.9)

        arrow_a1 = safe_arrow_3d(axes, np.zeros(3), A1_COORDS, color=I_HAT_COLOR)
        arrow_a2 = safe_arrow_3d(axes, np.zeros(3), A2_COORDS, color=J_HAT_COLOR)
        arrow_a3 = safe_arrow_3d(axes, np.zeros(3), A3_COORDS, color=K_HAT_COLOR)
        box_wire = create_parallelepiped_wireframe(axes, A1_COORDS, A2_COORDS, A3_COORDS)

        self.play(FadeIn(arrow_a1), FadeIn(arrow_a2), FadeIn(arrow_a3), run_time=1.0)
        self.play(Create(box_wire), run_time=1.0)

        target_dot = Dot3D(point=axes.c2p(*B_TARGET), radius=0.09, color=TARGET_COLOR, resolution=(6, 6))
        target_arrow = safe_arrow_3d(axes, np.zeros(3), B_TARGET, color=TARGET_COLOR)

        coupling_text = VGroup(
            MathTex(r"\mathbf{a}_1^\top \mathbf{a}_2 = 6,\;\mathbf{a}_1^\top \mathbf{a}_3 = 3,\;\mathbf{a}_2^\top \mathbf{a}_3 = 4 \neq 0", font_size=18, color=VEC_COLOR),
            MathTex(r"\text{Target } \mathbf{b} = [1, -2, 3]^\top \quad (\det A = 18)", font_size=17, color=TARGET_COLOR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        coupling_panel = make_hud_panel(coupling_text)
        self.pin_hud(coupling_panel, corner=DL)

        self.play(FadeIn(target_dot), FadeIn(target_arrow), run_time=0.8)

        # Short active parallax orbit so depth is unmistakable without wasting static wait time
        self.begin_ambient_camera_rotation(rate=AMBIENT_ROTATION_RATE)
        self.wait(2.5)
        self.stop_ambient_camera_rotation()

        puzzle_text = VGroup(
            Text("Skewed frame couples coordinates.", font_size=15, color=TEXT_COLOR, weight=BOLD),
            Text("Strip shadows sequentially -> orthonormal Q?", font_size=14, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.05)
        puzzle_panel = make_hud_panel(puzzle_text)
        self.pin_hud(puzzle_panel, corner=DR)
        self.wait(1.6)

        self.unpin_hud(title_panel, matrix_panel, coupling_panel, puzzle_panel)
        self.play(
            FadeOut(arrow_a1), FadeOut(arrow_a2), FadeOut(arrow_a3),
            FadeOut(box_wire), FadeOut(target_dot), FadeOut(target_arrow),
            FadeOut(axes), FadeOut(origin_dot),
            run_time=0.7,
        )

    # -------------------------------------------------------------------------
    # Act 2: Dynamic Deformation (Sequential 3D Stripping onto Unit Sphere S^2)
    # -------------------------------------------------------------------------
    def run_act2(self):
        self.stop_ambient_camera_rotation()
        self.set_camera_orientation(phi=PHI_DEFAULT, theta=THETA_DEFAULT)

        axes = self.create_axes_3d()
        origin_dot = Dot3D(point=axes.c2p(0, 0, 0), radius=0.08, color=ORIGIN_COLOR, resolution=(6, 6))
        sphere_rings = create_unit_sphere_rings(axes)
        sphere_shell = create_unit_sphere_surface(axes)

        title_text = VGroup(
            Text("Act 2: Sequential Orthogonal Stripping", font_size=20, color=TEXT_COLOR, weight=BOLD),
            Text("Project, subtract shadow, normalize onto S^2", font_size=14, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        title_panel = make_hud_panel(title_text)
        self.pin_hud(title_panel, corner=UL)

        step1_math = VGroup(
            MathTex(r"\mathbf{u}_1 = \mathbf{a}_1 = [2, 1, 2]^\top", font_size=18, color=I_HAT_COLOR),
            MathTex(r"R_{11} = \|\mathbf{u}_1\|_2 = 3 \implies \mathbf{q}_1 = \frac{1}{3}\mathbf{u}_1", font_size=18, color=TEXT_COLOR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        step_panel = make_hud_panel(step1_math)
        self.pin_hud(step_panel, corner=UR)

        self.play(Create(axes), FadeIn(origin_dot), Create(sphere_rings), FadeIn(sphere_shell), run_time=1.0)

        # Step 1: a1 -> q1
        arrow_a1 = safe_arrow_3d(axes, np.zeros(3), A1_COORDS, color=I_HAT_COLOR)
        arrow_q1 = safe_arrow_3d(axes, np.zeros(3), Q1_COORDS, color=I_HAT_COLOR)
        self.play(FadeIn(arrow_a1), run_time=0.6)
        self.play(FadeOut(sphere_shell), ReplacementTransform(arrow_a1, arrow_q1), run_time=1.0)
        self.wait(0.6)

        # Step 2: Strip projection 2*q1 from a2 -> u2 -> q2
        step2_math = VGroup(
            MathTex(r"R_{12} = \mathbf{q}_1^\top \mathbf{a}_2 = 2", font_size=18, color=VEC_COLOR),
            MathTex(r"\mathbf{u}_2 = \mathbf{a}_2 - 2\mathbf{q}_1,\quad R_{22} = \|\mathbf{u}_2\|_2 = 2", font_size=18, color=J_HAT_COLOR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        new_step_panel = make_hud_panel(step2_math)
        new_step_panel.to_corner(UR, buff=0.25)
        self.unpin_hud(step_panel)
        step_panel = new_step_panel
        self.pin_hud(step_panel, corner=UR)

        arrow_a2_ghost = safe_arrow_3d(axes, np.zeros(3), A2_COORDS, color=J_HAT_COLOR)
        shadow_21 = safe_arrow_3d(axes, np.zeros(3), P21_VEC, color=VEC_COLOR)
        drop_21 = DashedLine(
            axes.c2p(*A2_COORDS),
            axes.c2p(*P21_VEC),
            dash_length=0.10,
            color=VEC_COLOR,
            stroke_width=2.0,
        )
        self.play(FadeIn(arrow_a2_ghost), FadeIn(shadow_21), Create(drop_21), run_time=0.9)

        # Live ValueTracker stripping v2(t) = a2 - t * (2*q1)
        t2 = ValueTracker(0.0)
        sweep_v2 = always_redraw(
            lambda: safe_arrow_3d(
                axes,
                np.zeros(3),
                (1.0 - t2.get_value()) * A2_COORDS + t2.get_value() * U2_VEC,
                color=J_HAT_COLOR,
            )
        )

        dot2_label = MathTex(r"\mathbf{q}_1^\top \mathbf{v}_2(t) =", font_size=20, color=TEXT_COLOR)
        dot2_num = DecimalNumber(
            2.00,
            num_decimal_places=2,
            include_sign=True,
            mob_class=Text,
            font_size=20,
            color=VEC_COLOR,
        )
        dot2_row = VGroup(dot2_label, dot2_num).arrange(RIGHT, buff=0.12)
        dot2_panel = make_hud_panel(dot2_row, min_width=3.2)
        self.pin_hud(dot2_panel, corner=DR)

        def _update_dot2(mob):
            v_cur = (1.0 - t2.get_value()) * A2_COORDS + t2.get_value() * U2_VEC
            mob.set_value(float(np.dot(Q1_COORDS, v_cur)))
            mob.next_to(dot2_label, RIGHT, buff=0.12)
            self.add_fixed_in_frame_mobjects(mob)

        dot2_num.add_updater(_update_dot2)
        self.add(sweep_v2)

        self.play(FadeOut(arrow_a2_ghost), t2.animate.set_value(1.0), run_time=2.2, rate_func=smooth)
        dot2_num.clear_updaters()
        sweep_v2.clear_updaters()

        elbow_12 = create_right_angle_3d(axes, np.zeros(3), Q1_COORDS, U2_VEC, size=0.34, color=SUBSPACE_COLOR)
        arrow_q2 = safe_arrow_3d(axes, np.zeros(3), Q2_COORDS, color=J_HAT_COLOR)
        self.play(Create(elbow_12), FadeOut(shadow_21), FadeOut(drop_21), run_time=0.7)
        self.play(ReplacementTransform(sweep_v2, arrow_q2), run_time=0.9)
        self.unpin_hud(dot2_panel)

        # Step 3: Subspace plane span(q1, q2) & strip p12,3 from a3 -> u3 -> q3
        step3_math = VGroup(
            MathTex(r"R_{13} = 1,\; R_{23} = 1 \implies \mathbf{p}_{12,3} = \mathbf{q}_1 + \mathbf{q}_2", font_size=17, color=VEC_COLOR),
            MathTex(r"\mathbf{u}_3 = \mathbf{a}_3 - \mathbf{p}_{12,3},\quad R_{33} = \|\mathbf{u}_3\|_2 = 3", font_size=17, color=K_HAT_COLOR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        new_step_panel3 = make_hud_panel(step3_math)
        new_step_panel3.to_corner(UR, buff=0.25)
        self.unpin_hud(step_panel)
        step_panel = new_step_panel3
        self.pin_hud(step_panel, corner=UR)

        plane_s2 = create_subspace_plane(axes, Q1_COORDS, Q2_COORDS, extent=2.4)
        arrow_a3_ghost = safe_arrow_3d(axes, np.zeros(3), A3_COORDS, color=K_HAT_COLOR)
        shadow_12_3 = safe_arrow_3d(axes, np.zeros(3), P12_3_VEC, color=VEC_COLOR)
        drop_12_3 = DashedLine(
            axes.c2p(*A3_COORDS),
            axes.c2p(*P12_3_VEC),
            dash_length=0.10,
            color=VEC_COLOR,
            stroke_width=2.0,
        )
        self.play(Create(plane_s2), FadeIn(arrow_a3_ghost), FadeIn(shadow_12_3), Create(drop_12_3), run_time=1.1)

        t3 = ValueTracker(0.0)
        sweep_v3 = always_redraw(
            lambda: safe_arrow_3d(
                axes,
                np.zeros(3),
                (1.0 - t3.get_value()) * A3_COORDS + t3.get_value() * U3_VEC,
                color=K_HAT_COLOR,
            )
        )

        dot3_l1 = MathTex(r"\mathbf{q}_1^\top \mathbf{v}_3 =", font_size=18, color=TEXT_COLOR)
        dot3_n1 = DecimalNumber(1.00, num_decimal_places=2, include_sign=True, mob_class=Text, font_size=18, color=VEC_COLOR)
        dot3_l2 = MathTex(r"\mathbf{q}_2^\top \mathbf{v}_3 =", font_size=18, color=TEXT_COLOR)
        dot3_n2 = DecimalNumber(1.00, num_decimal_places=2, include_sign=True, mob_class=Text, font_size=18, color=VEC_COLOR)
        row1 = VGroup(dot3_l1, dot3_n1).arrange(RIGHT, buff=0.10)
        row2 = VGroup(dot3_l2, dot3_n2).arrange(RIGHT, buff=0.10)
        dot3_col = VGroup(row1, row2).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        dot3_panel = make_hud_panel(dot3_col, min_width=3.2)
        self.pin_hud(dot3_panel, corner=DR)

        def _update_dot3_n1(mob):
            v_cur = (1.0 - t3.get_value()) * A3_COORDS + t3.get_value() * U3_VEC
            mob.set_value(float(np.dot(Q1_COORDS, v_cur)))
            mob.next_to(dot3_l1, RIGHT, buff=0.10)
            self.add_fixed_in_frame_mobjects(mob)

        def _update_dot3_n2(mob):
            v_cur = (1.0 - t3.get_value()) * A3_COORDS + t3.get_value() * U3_VEC
            mob.set_value(float(np.dot(Q2_COORDS, v_cur)))
            mob.next_to(dot3_l2, RIGHT, buff=0.10)
            self.add_fixed_in_frame_mobjects(mob)

        dot3_n1.add_updater(_update_dot3_n1)
        dot3_n2.add_updater(_update_dot3_n2)
        self.add(sweep_v3)

        self.play(FadeOut(arrow_a3_ghost), t3.animate.set_value(1.0), run_time=2.2, rate_func=smooth)
        dot3_n1.clear_updaters()
        dot3_n2.clear_updaters()
        sweep_v3.clear_updaters()

        elbow_13 = create_right_angle_3d(axes, np.zeros(3), Q1_COORDS, U3_VEC, size=0.34, color=SUBSPACE_COLOR)
        elbow_23 = create_right_angle_3d(axes, np.zeros(3), Q2_COORDS, U3_VEC, size=0.34, color=SUBSPACE_COLOR)
        arrow_q3 = safe_arrow_3d(axes, np.zeros(3), Q3_COORDS, color=K_HAT_COLOR)
        self.play(Create(elbow_13), Create(elbow_23), FadeOut(shadow_12_3), FadeOut(drop_12_3), run_time=0.7)
        self.play(ReplacementTransform(sweep_v3, arrow_q3), FadeOut(plane_s2), run_time=1.0)

        self.unpin_hud(dot3_panel)
        inv_badge = make_hud_panel(
            MathTex(r"Q^\top Q = I_3,\quad \|\mathbf{q}_i\|_2 = 1,\quad \det(Q) = +1", font_size=18, color=SUBSPACE_COLOR)
        )
        self.pin_hud(inv_badge, corner=DL)

        self.begin_ambient_camera_rotation(rate=AMBIENT_ROTATION_RATE)
        self.wait(2.5)
        self.stop_ambient_camera_rotation()

        self.unpin_hud(title_panel, step_panel, inv_badge)
        self.play(
            FadeOut(arrow_q1), FadeOut(arrow_q2), FadeOut(arrow_q3),
            FadeOut(elbow_12), FadeOut(elbow_13), FadeOut(elbow_23),
            FadeOut(sphere_rings), FadeOut(axes), FadeOut(origin_dot),
            run_time=0.7,
        )

    # -------------------------------------------------------------------------
    # Act 3: Algebraic Morph (A = QR, Expansion b = Q(Q^T b), Back-Sub & Vol)
    # -------------------------------------------------------------------------
    def run_act3(self):
        self.stop_ambient_camera_rotation()
        self.set_camera_orientation(phi=PHI_DEFAULT, theta=THETA_DEFAULT)

        axes = self.create_axes_3d()
        origin_dot = Dot3D(point=axes.c2p(0, 0, 0), radius=0.08, color=ORIGIN_COLOR, resolution=(6, 6))
        sphere_rings = create_unit_sphere_rings(axes)

        title_text = VGroup(
            Text("Act 3: Upper-Triangular R & Inversion-Free Solve", font_size=20, color=TEXT_COLOR, weight=BOLD),
            Text("A = QR | b = Q(Q^T b) | R x = Q^T b", font_size=14, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        title_panel = make_hud_panel(title_text)
        self.pin_hud(title_panel, corner=UL)

        arrow_q1 = safe_arrow_3d(axes, np.zeros(3), Q1_COORDS, color=I_HAT_COLOR)
        arrow_q2 = safe_arrow_3d(axes, np.zeros(3), Q2_COORDS, color=J_HAT_COLOR)
        arrow_q3 = safe_arrow_3d(axes, np.zeros(3), Q3_COORDS, color=K_HAT_COLOR)

        self.play(
            Create(axes), FadeIn(origin_dot), Create(sphere_rings),
            FadeIn(arrow_q1), FadeIn(arrow_q2), FadeIn(arrow_q3),
            run_time=0.9,
        )

        # Tip-to-tail reconstruction of a3 = 1*q1 + 1*q2 + 3*q3
        seg_1 = safe_arrow_3d(axes, np.zeros(3), 1.0 * Q1_COORDS, color=I_HAT_COLOR)
        seg_2 = safe_arrow_3d(axes, 1.0 * Q1_COORDS, 1.0 * Q1_COORDS + 1.0 * Q2_COORDS, color=J_HAT_COLOR)
        seg_3 = safe_arrow_3d(axes, 1.0 * Q1_COORDS + 1.0 * Q2_COORDS, A3_COORDS, color=K_HAT_COLOR)
        arrow_a3 = safe_arrow_3d(axes, np.zeros(3), A3_COORDS, color=VEC_COLOR)

        qr_card = VGroup(
            MathTex(r"A = Q R \quad (R_{ij} = \mathbf{q}_i^\top \mathbf{a}_j,\; R_{ii} > 0\text{ unique})", font_size=17, color=TEXT_COLOR),
            MathTex(
                r"R = \begin{bmatrix} 3 & 2 & 1 \\ 0 & 2 & 1 \\ 0 & 0 & 3 \end{bmatrix},\quad R_{ij}=0\;(i>j)",
                font_size=19,
                color=MUTED_TEXT,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        qr_panel = make_hud_panel(qr_card)
        self.pin_hud(qr_panel, corner=UR)

        self.play(FadeIn(seg_1), run_time=0.6)
        self.play(FadeIn(seg_2), run_time=0.6)
        self.play(FadeIn(seg_3), run_time=0.6)
        self.play(FadeIn(arrow_a3), run_time=0.7)
        self.wait(1.2)

        # Orthonormal expansion b = Q(Q^T b) and back-substitution R x = Q^T b
        self.play(FadeOut(seg_1), FadeOut(seg_2), FadeOut(seg_3), FadeOut(arrow_a3), run_time=0.5)

        b_dot = Dot3D(point=axes.c2p(*B_TARGET), radius=0.09, color=TARGET_COLOR, resolution=(6, 6))
        b_arrow = safe_arrow_3d(axes, np.zeros(3), B_TARGET, color=TARGET_COLOR)
        b_s1 = safe_arrow_3d(axes, np.zeros(3), 2.0 * Q1_COORDS, color=I_HAT_COLOR)
        b_s2 = safe_arrow_3d(axes, 2.0 * Q1_COORDS, 2.0 * Q1_COORDS - 1.0 * Q2_COORDS, color=J_HAT_COLOR)
        b_s3 = safe_arrow_3d(axes, 2.0 * Q1_COORDS - 1.0 * Q2_COORDS, B_TARGET, color=K_HAT_COLOR)

        exp_card = VGroup(
            MathTex(r"\mathbf{b} = Q(Q^\top \mathbf{b}),\quad Q^\top \mathbf{b} = [2, -1, 3]^\top", font_size=18, color=TARGET_COLOR),
            Text("Zero (A^T A)^-1 inversion needed!", font_size=14, color=SUBSPACE_COLOR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        exp_panel = make_hud_panel(exp_card)
        self.pin_hud(exp_panel, corner=DL)

        self.play(FadeIn(b_s1), FadeIn(b_s2), FadeIn(b_s3), FadeIn(b_dot), FadeIn(b_arrow), run_time=1.1)

        solve_card = VGroup(
            MathTex(r"R\hat{\mathbf{x}} = Q^\top\mathbf{b} \implies \hat{\mathbf{x}} = [1, -1, 1]^\top", font_size=18, color=TEXT_COLOR),
            MathTex(r"|\det A| = R_{11} R_{22} R_{33} = 3 \cdot 2 \cdot 3 = 18", font_size=18, color=VEC_COLOR),
            MathTex(r"A^\top A = R^\top R \quad (\text{Cholesky Gram Link})", font_size=16, color=MUTED_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        solve_panel = make_hud_panel(solve_card)
        self.pin_hud(solve_panel, corner=DR)

        # Brief 2s orbital sweep then static hold for fast caching
        self.begin_ambient_camera_rotation(rate=AMBIENT_ROTATION_RATE)
        self.wait(2.0)
        self.stop_ambient_camera_rotation()
        self.wait(2.0)

        self.unpin_hud(title_panel, qr_panel, exp_panel, solve_panel)
        self.play(
            FadeOut(arrow_q1), FadeOut(arrow_q2), FadeOut(arrow_q3),
            FadeOut(b_s1), FadeOut(b_s2), FadeOut(b_s3), FadeOut(b_dot), FadeOut(b_arrow),
            FadeOut(sphere_rings), FadeOut(axes), FadeOut(origin_dot),
            run_time=0.7,
        )

    # -------------------------------------------------------------------------
    # Act 4: Degenerate Stress Test (Rank-Deficient Collapse & CGS vs MGS Drift)
    # -------------------------------------------------------------------------
    def run_act4(self):
        self.stop_ambient_camera_rotation()
        self.set_camera_orientation(phi=PHI_DEFAULT, theta=THETA_DEFAULT)

        axes = self.create_axes_3d()
        origin_dot = Dot3D(point=axes.c2p(0, 0, 0), radius=0.08, color=ORIGIN_COLOR, resolution=(6, 6))
        plane_s2 = create_subspace_plane(axes, Q1_COORDS, Q2_COORDS, extent=2.4)

        title_text = VGroup(
            Text("Act 4: Linear Dependence & Numerical Drift", font_size=20, color=TEXT_COLOR, weight=BOLD),
            Text("Early Termination (R_33 = 0) & CGS vs MGS vs HH", font_size=14, color=ERROR_COLOR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        title_panel = make_hud_panel(title_text)
        self.pin_hud(title_panel, corner=UL)

        arrow_q1 = safe_arrow_3d(axes, np.zeros(3), Q1_COORDS, color=I_HAT_COLOR)
        arrow_q2 = safe_arrow_3d(axes, np.zeros(3), Q2_COORDS, color=J_HAT_COLOR)
        self.play(Create(axes), FadeIn(origin_dot), Create(plane_s2), FadeIn(arrow_q1), FadeIn(arrow_q2), run_time=0.9)

        deg_card = VGroup(
            MathTex(r"\mathbf{a}_3^{\text{deg}} = \mathbf{a}_2 - \mathbf{a}_1 = [-2, 1, 0]^\top", font_size=18, color=ERROR_COLOR),
            MathTex(r"\mathbf{a}_3^{\text{deg}} = -1\mathbf{q}_1 + 2\mathbf{q}_2 \in \operatorname{span}(\mathbf{q}_1, \mathbf{q}_2)", font_size=17, color=TEXT_COLOR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        deg_panel = make_hud_panel(deg_card)
        self.pin_hud(deg_panel, corner=UR)

        t_deg = ValueTracker(0.0)
        sweep_deg = always_redraw(
            lambda: safe_arrow_3d(
                axes,
                np.zeros(3),
                (1.0 - t_deg.get_value()) * A3_DEG,
                color=ERROR_COLOR,
            )
        )
        self.add(sweep_deg)
        self.wait(0.6)
        self.play(t_deg.animate.set_value(1.0), run_time=2.0, rate_func=smooth)
        sweep_deg.clear_updaters()

        term_card = VGroup(
            MathTex(r"\mathbf{u}_3 = \mathbf{0} \implies R_{33} = 0 \quad (\text{rank} = 2 < 3)", font_size=18, color=ERROR_COLOR),
            Text("Early termination prevents 0/0; Thin QR = Q_2 R_{2x3}", font_size=14, color=TEXT_COLOR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        term_panel = make_hud_panel(term_card)
        self.pin_hud(term_panel, corner=DL)
        self.wait(1.2)

        # Pinned Lauchli (epsilon=1e-8) stability verdict card comparing CGS vs MGS vs Householder
        drift_card = VGroup(
            Text("Lauchli 4x3 (eps=1e-8, kappa=1.73e8)", font_size=15, color=MUTED_TEXT, weight=BOLD),
            MathTex(r"\text{CGS: } \|Q^\top Q - I\|_F = 7.07 \times 10^{-1}\;(\mathcal{O}(\epsilon \kappa^2))", font_size=16, color=ERROR_COLOR),
            MathTex(r"\text{MGS: } \|Q^\top Q - I\|_F = 1.16 \times 10^{-8}\;(\mathcal{O}(\epsilon \kappa))", font_size=16, color=VEC_COLOR),
            MathTex(r"\text{Householder: } \|Q^\top Q - I\|_F = 2.69 \times 10^{-16}\;(\mathcal{O}(\epsilon))", font_size=16, color=I_HAT_COLOR),
            Text("Normal Eq A^T A singular -> QR stays accurate!", font_size=13, color=SUBSPACE_COLOR),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        drift_panel = make_hud_panel(drift_card)
        self.pin_hud(drift_panel, corner=DR)

        # Static hold caches frames cleanly
        self.wait(3.0)

        self.unpin_hud(title_panel, deg_panel, term_panel, drift_panel)
        self.play(
            FadeOut(sweep_deg), FadeOut(plane_s2),
            FadeOut(arrow_q1), FadeOut(arrow_q2),
            FadeOut(axes), FadeOut(origin_dot),
            run_time=0.7,
        )


# -----------------------------------------------------------------------------
# Individual Act Executable Scenes
# -----------------------------------------------------------------------------
class Act1SpatialPuzzle(BaseGramSchmidtQRScene):
    """Executes Act 1: Skewed 3D Basis & Coupling Puzzle."""

    def construct(self):
        self.run_act1()


class Act2DynamicDeformation(BaseGramSchmidtQRScene):
    """Executes Act 2: Sequential 3D Gram-Schmidt Stripping onto S^2."""

    def construct(self):
        self.run_act2()


class Act3AlgebraicMorph(BaseGramSchmidtQRScene):
    """Executes Act 3: Upper-Triangular R, Orthonormal Expansion & Back-Substitution."""

    def construct(self):
        self.run_act3()


class Act4DegenerateStressTest(BaseGramSchmidtQRScene):
    """Executes Act 4: Linear Dependence R_33 = 0 & CGS vs MGS Stability Drift."""

    def construct(self):
        self.run_act4()


# -----------------------------------------------------------------------------
# Master Composite Production Scene
# -----------------------------------------------------------------------------
class GramSchmidtQRScene(BaseGramSchmidtQRScene):
    """Master production scene rendering the complete 4-Act Sprint 7 module."""

    def construct(self):
        self.run_act1()
        self.run_act2()
        self.run_act3()
        self.run_act4()
