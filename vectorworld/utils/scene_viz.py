import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib
from matplotlib.path import Path as MplPath
from matplotlib.patches import (
    FancyBboxPatch, Polygon, Wedge
)
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.transforms as transforms
from scipy import ndimage

# Optional: scipy for spline smoothing
try:
    from scipy.interpolate import splprep, splev
    HAS_SCIPY_SPLINE = True
except ImportError:
    HAS_SCIPY_SPLINE = False


# ============================================================================
# Academic Color Palette - Publication Ready, High Contrast
# (Updated: ego=deep red, veh=blue/cyan family, cyclist=purple, ped=orange)
# ============================================================================

ACADEMIC_PALETTE = {
    # === Background & Road ===
    "bg": "#ffffff",                  # pure white
    "road_surface": "#e6fffa",        # very light cyan (drivable)
    "road_surface_alpha": 1.0,

    # Offroad is now the OUTER region (gray fill), background is the gap (white)
    "offroad": "#9ca3af",             # gray-400
    "offroad_alpha": 0.88,

    # === Road Elements ===
    "road_edge": "#ea580c",           # orange-600
    "road_edge_alpha": 0.98,
    "centerline": "#475569",          # slate-600
    "centerline_glow": "#94a3b8",     # slate-400

    # === Traffic Lights ===
    "lane_green": "#16a34a",          # green-600
    "lane_red": "#dc2626",            # red-600

    # === Ego Vehicle (deep academic red) ===
    "ego_fill": "#b91c1c",            # red-700
    "ego_fill_alpha": 0.96,
    "ego_edge": "#7f1d1d",            # red-900
    "ego_glow": "#ef4444",            # red-500
    "ego_glow_alpha": 0.22,

    # === NPC Vehicles (blue family) ===
    "veh_fill": "#60a5fa",            # blue-400
    "veh_fill_alpha": 0.86,
    "veh_edge": "#1d4ed8",            # blue-700

    # === Pedestrians (orange family) ===
    "ped_fill": "#fdba74",            # orange-300
    "ped_fill_alpha": 0.92,
    "ped_edge": "#c2410c",            # orange-700

    # === Cyclists (purple family) ===
    "cyc_fill": "#c4b5fd",            # violet-300
    "cyc_fill_alpha": 0.88,
    "cyc_edge": "#6d28d9",            # violet-700

    # === Other agents ===
    "other_fill": "#a7f3d0",          # emerald-200
    "other_fill_alpha": 0.82,
    "other_edge": "#059669",          # emerald-600

    # === Trajectory beads (fixed alpha, color gradient only) ===
    "traj_alpha": 0.88,               # Fixed alpha for all beads
    "traj_bead_size": 22,
    "traj_line_width": 2.0,
    "traj_line_alpha": 0.55,          # Fixed line alpha

    # === Safety Halo ===
    "safety_normal": "#0ea5e9",       # sky-500
    "safety_normal_alpha": 0.12,
    "safety_warning": "#ef4444",      # red-500
    "safety_warning_alpha": 0.33,

    # === Route ===
    "route_line": "#f59e0b",          # amber-500
    "route_dot": "#d97706",           # amber-600
    "route_alpha": 0.90,

    # === Motion Code Visualization ===
    "motion_pred": "#8b5cf6",         # violet-500
    "motion_pred_alpha": 0.85,
    "motion_gt": "#f97316",           # orange-500
    "motion_gt_alpha": 0.80,

    # === Agent ID text ===
    "id_text_ego": "#ffffff",         # white for contrast on red
    "id_text_npc": "#0f172a",         # slate-900
    "id_font_scale": 2.8,
}

# Backward compatibility aliases
UNIFIED_COLORS = {
    "bg": ACADEMIC_PALETTE["bg"],
    "offroad": ACADEMIC_PALETTE["offroad"],
    "offroad_alpha": ACADEMIC_PALETTE["offroad_alpha"],
    "road_surface": ACADEMIC_PALETTE["road_surface"],
    "road_edge": ACADEMIC_PALETTE["road_edge"],
    "road_edge_alpha": ACADEMIC_PALETTE["road_edge_alpha"],
    "road_edge_low": "#9ca3af",
    "road_edge_low_alpha": 0.75,
    "centerline": ACADEMIC_PALETTE["centerline"],
    "centerline_glow": ACADEMIC_PALETTE["centerline_glow"],
    "centerline_glow_alpha": 0.25,
    "lane_green": ACADEMIC_PALETTE["lane_green"],
    "lane_green_glow": "#22c55e",
    "lane_red": ACADEMIC_PALETTE["lane_red"],
    "lane_red_glow": "#ef4444",
    "veh": ACADEMIC_PALETTE["veh_fill"],
    "veh_alpha": ACADEMIC_PALETTE["veh_fill_alpha"],
    "ped": ACADEMIC_PALETTE["ped_fill"],
    "ped_alpha": ACADEMIC_PALETTE["ped_fill_alpha"],
    "cyc": ACADEMIC_PALETTE["cyc_fill"],
    "cyc_alpha": ACADEMIC_PALETTE["cyc_fill_alpha"],
    "other": ACADEMIC_PALETTE["other_fill"],
    "other_alpha": ACADEMIC_PALETTE["other_fill_alpha"],
    "ego_fill": ACADEMIC_PALETTE["ego_fill"],
    "ego_fill_alpha": ACADEMIC_PALETTE["ego_fill_alpha"],
    "ego_edge": ACADEMIC_PALETTE["ego_edge"],
    "ego_edge_alpha": 1.0,
    "agent_edge": ACADEMIC_PALETTE["veh_edge"],
    "agent_edge_alpha": 0.90,
    "trajectory": "#ef4444",
    "trajectory_alpha": 0.85,
    "route": ACADEMIC_PALETTE["route_line"],
    "route_alpha": ACADEMIC_PALETTE["route_alpha"],
    "route_dot": ACADEMIC_PALETTE["route_dot"],
    "motion_pred": ACADEMIC_PALETTE["motion_pred"],
    "motion_pred_alpha": ACADEMIC_PALETTE["motion_pred_alpha"],
    "motion_gt": ACADEMIC_PALETTE["motion_gt"],
    "motion_gt_alpha": ACADEMIC_PALETTE["motion_gt_alpha"],
}

COLORS_2D = {
    "bg": UNIFIED_COLORS["bg"],
    "offroad": UNIFIED_COLORS["offroad"],
    "road_surface": UNIFIED_COLORS["road_surface"],
    "road_edge": UNIFIED_COLORS["road_edge"],
    "centerline": UNIFIED_COLORS["centerline"],
    "veh": UNIFIED_COLORS["veh"],
    "ped": UNIFIED_COLORS["ped"],
    "cyc": UNIFIED_COLORS["cyc"],
    "other": UNIFIED_COLORS["other"],
    "ego": UNIFIED_COLORS["ego_fill"],
    "motion_pred": UNIFIED_COLORS["motion_pred"],
    "motion_gt": UNIFIED_COLORS["motion_gt"],
    "route": UNIFIED_COLORS["route"],
    "trajectory": UNIFIED_COLORS["trajectory"],
}

COLORS_SIM = COLORS_2D.copy()
COLORS_SIM.update({
    "ego_fill": UNIFIED_COLORS["ego_fill"],
    "ego_edge": UNIFIED_COLORS["ego_edge"],
    "agent_edge": UNIFIED_COLORS["agent_edge"],
    "centerline_glow": UNIFIED_COLORS["centerline_glow"],
})


# ============================================================================
# Constants
# ============================================================================

DEFAULT_LANE_WIDTH = 4.2
MAX_TRAJECTORY_LENGTH_M = 8.0
DEFAULT_SAFETY_RADIUS = 6.0
TRAJECTORY_NUM_BEADS = 12

# Three-layer control:
# - Road: road_mask
# - Background (white): dilate(road_mask, gap) - road_mask
# - Offroad (gray): outside dilated region
DEFAULT_BG_GAP_M = 3.0

# Per-agent-type speed limits (m/s) for normalization
AGENT_TYPE_MAX_SPEEDS = {
    0: 25.0,
    1: 25.0,
    2: 2.0,
    3: 8.0,
    4: 5.0,
}
DEFAULT_MAX_SPEED = 15.0


# ============================================================================
# Utility Functions
# ============================================================================

def _safe_float(v, default: float = 0.0) -> float:
    try:
        fv = float(v)
        return fv if np.isfinite(fv) else default
    except (TypeError, ValueError):
        return default


def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n


def _smooth_curve_spline(pts: np.ndarray, smoothness: float = 0.5, num_output: int = 60) -> np.ndarray:
    """Smooth curve using B-spline interpolation."""
    if not HAS_SCIPY_SPLINE or pts.shape[0] < 4:
        return pts
    try:
        diffs = np.diff(pts, axis=0)
        mask = np.linalg.norm(diffs, axis=1) > 1e-4
        mask = np.concatenate([[True], mask])
        pts_clean = pts[mask]
        if pts_clean.shape[0] < 4:
            return pts

        k = min(3, pts_clean.shape[0] - 1)
        tck, _ = splprep([pts_clean[:, 0], pts_clean[:, 1]], s=smoothness, k=k)
        u_new = np.linspace(0.0, 1.0, num_output)
        x_new, y_new = splev(u_new, tck)
        return np.stack([x_new, y_new], axis=-1)
    except Exception:
        return pts


def _smooth_curve(pts: np.ndarray, smoothness: float = 1.0, num_output: int = 100) -> np.ndarray:
    return _smooth_curve_spline(pts, smoothness, num_output)


def _build_lane_strip(centerline: np.ndarray, half_width: float):
    pts = np.asarray(centerline, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
        return None, None
    pts = pts[:, :2]
    seg = pts[1:] - pts[:-1]
    t = _safe_normalize(seg)
    n_seg = np.stack([-t[..., 1], t[..., 0]], axis=-1)
    n = np.zeros_like(pts)
    if len(n_seg) == 1:
        n[0] = n_seg[0]
        n[-1] = n_seg[0]
    else:
        n[1:-1] = _safe_normalize(n_seg[:-1] + n_seg[1:])
        n[0] = n_seg[0]
        n[-1] = n_seg[-1]
    left = pts + n * half_width
    right = pts - n * half_width
    return left, right


def _hex_to_rgb01(h: str) -> np.ndarray:
    rgba = np.array(mcolors.to_rgba(h), dtype=np.float32)
    return rgba[:3]


def _make_gradient_rgba_fixed_alpha(hex_colors: list, n: int, alpha: float) -> list:
    """
    Build an n-length RGBA gradient from multiple hex stops with FIXED alpha.
    No transparency gradient - only color gradient for academic publication quality.
    """
    if n <= 0:
        return []
    if len(hex_colors) == 1:
        rgb = _hex_to_rgb01(hex_colors[0])
        return [(float(rgb[0]), float(rgb[1]), float(rgb[2]), float(alpha)) for _ in range(n)]

    stops = np.stack([_hex_to_rgb01(c) for c in hex_colors], axis=0)  # (K,3)
    K = stops.shape[0]
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)

    # piecewise linear interpolation across K-1 segments
    seg = np.clip((t * (K - 1)).astype(np.int32), 0, K - 2)
    local_t = (t * (K - 1)) - seg.astype(np.float32)

    rgb = (1.0 - local_t)[:, None] * stops[seg] + local_t[:, None] * stops[seg + 1]
    return [(float(rgb[i, 0]), float(rgb[i, 1]), float(rgb[i, 2]), float(alpha)) for i in range(n)]


# ============================================================================
# Road Mask Computation (Three-layer: Road / Background gap / Offroad outer)
# ============================================================================

def _compute_road_mask_grid(
    road_points: np.ndarray,
    lanes_mask: np.ndarray | None,
    lane_width: float,
    xlim: tuple | None,
    ylim: tuple | None,
    resolution: float = 0.25,
    padding: float = 5.0,
    morph_iterations: int = 2,
    bg_gap_m: float = DEFAULT_BG_GAP_M,
):
    """
    Returns:
        xs, ys: grid coordinates
        road_mask: drivable area
        offroad_outer_mask: outer region (gray)
        buffered_mask: dilated(road_mask, gap)  (road + white-gap region)
    """
    roads = np.asarray(road_points, dtype=np.float64)
    if roads.size == 0 or roads.ndim != 3 or roads.shape[2] < 2:
        return None, None, None, None, None

    if lanes_mask is not None:
        m = np.asarray(lanes_mask).astype(bool)
        if m.shape == roads.shape[:2]:
            roads = np.where(m[..., None], roads, np.nan)

    pts_all = roads.reshape(-1, 2)
    pts_valid = pts_all[~np.isnan(pts_all).any(axis=1)]
    if pts_valid.shape[0] == 0:
        return None, None, None, None, None

    if xlim is None:
        x_min = float(pts_valid[:, 0].min() - padding)
        x_max = float(pts_valid[:, 0].max() + padding)
    else:
        x_min, x_max = float(xlim[0]), float(xlim[1])

    if ylim is None:
        y_min = float(pts_valid[:, 1].min() - padding)
        y_max = float(pts_valid[:, 1].max() + padding)
    else:
        y_min, y_max = float(ylim[0]), float(ylim[1])

    if not all(np.isfinite([x_min, x_max, y_min, y_max])):
        return None, None, None, None, None

    res = max(0.12, float(resolution))
    nx = max(64, int(math.ceil((x_max - x_min) / res)) + 1)
    ny = max(64, int(math.ceil((y_max - y_min) / res)) + 1)
    xs = np.linspace(x_min, x_max, nx, dtype=np.float64)
    ys = np.linspace(y_min, y_max, ny, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    grid_pts = np.stack([X.ravel(), Y.ravel()], axis=-1)

    road_mask_flat = np.zeros(grid_pts.shape[0], dtype=bool)
    half_w = float(lane_width) * 0.5
    n_lanes = roads.shape[0]

    # Union of lane strips
    for i in range(n_lanes):
        cl = roads[i, :, :2]
        valid = ~np.isnan(cl[:, 0])
        cl = cl[valid]
        if cl.shape[0] < 2:
            continue
        left, right = _build_lane_strip(cl, half_w)
        if left is None:
            continue
        poly = np.vstack([left, right[::-1]])
        path = MplPath(poly)
        inside = path.contains_points(grid_pts, radius=0.05)
        road_mask_flat |= inside

    road_mask = road_mask_flat.reshape(ny, nx)

    # Morphological cleanup for "small but complete"
    struct = ndimage.generate_binary_structure(2, 1)
    if morph_iterations > 0:
        road_mask = ndimage.binary_closing(road_mask, structure=struct, iterations=int(morph_iterations))
    road_mask = ndimage.binary_fill_holes(road_mask)

    # Background gap (white) is ensured by dilating road and NOT painting that ring.
    gap_px = int(max(1, round(float(bg_gap_m) / res)))
    struct_dilate = ndimage.generate_binary_structure(2, 2)
    buffered_mask = ndimage.binary_dilation(road_mask, structure=struct_dilate, iterations=gap_px)

    # Offroad is OUTER region beyond buffer (gray)
    offroad_outer_mask = (~buffered_mask)

    return xs, ys, road_mask, offroad_outer_mask, buffered_mask


def compute_road_mask(
    road_points: np.ndarray,
    lanes_mask: np.ndarray | None = None,
    lane_width: float = DEFAULT_LANE_WIDTH,
    xlim: tuple | None = None,
    ylim: tuple | None = None,
    resolution: float = 0.4,
    padding: float = 4.0,
):
    """Compute a simple offroad mask for external use (kept for backward compatibility)."""
    xs, ys, road_mask, _, _ = _compute_road_mask_grid(
        road_points,
        lanes_mask=lanes_mask,
        lane_width=lane_width,
        xlim=xlim,
        ylim=ylim,
        resolution=resolution,
        padding=padding,
        morph_iterations=2,
        bg_gap_m=DEFAULT_BG_GAP_M,
    )
    if road_mask is None:
        return xs, ys, None
    return xs, ys, (~road_mask).astype(np.float32)


def _extract_open_road_edges(
    road_mask: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    min_pixel_length: int = 8,
    smooth_factor: float = 2.0,
) -> list:
    """Extract and smooth open road edge curves from road_mask boundary."""
    H, W = road_mask.shape
    x_min, x_max = xs[0], xs[-1]
    y_min, y_max = ys[0], ys[-1]
    dx = (x_max - x_min) / (W - 1) if W > 1 else 1.0
    dy = (y_max - y_min) / (H - 1) if H > 1 else 1.0

    edge_mask = np.zeros_like(road_mask, dtype=bool)
    for i in range(H):
        for j in range(W):
            if not road_mask[i, j]:
                continue
            for ni in (i - 1, i, i + 1):
                for nj in (j - 1, j, j + 1):
                    if ni < 0 or ni >= H or nj < 0 or nj >= W:
                        edge_mask[i, j] = True
                        break
                    if not road_mask[ni, nj]:
                        edge_mask[i, j] = True
                        break
                if edge_mask[i, j]:
                    break

    edge_pixels = {(i, j) for i in range(H) for j in range(W) if edge_mask[i, j]}

    def on_canvas_boundary(i, j):
        return i <= 1 or i >= H - 2 or j <= 1 or j >= W - 2

    boundary_starts = [p for p in edge_pixels if on_canvas_boundary(*p)]
    curves_pixel = []
    visited_global = set()

    def trace_from(start_i, start_j, visited_local):
        path = [(start_i, start_j)]
        visited_local.add((start_i, start_j))
        cur = (start_i, start_j)
        while True:
            ci, cj = cur
            neighbors = [
                (ci - 1, cj), (ci + 1, cj), (ci, cj - 1), (ci, cj + 1),
                (ci - 1, cj - 1), (ci - 1, cj + 1), (ci + 1, cj - 1), (ci + 1, cj + 1),
            ]
            found = False
            for ni, nj in neighbors:
                if (ni, nj) in edge_pixels and (ni, nj) not in visited_local:
                    path.append((ni, nj))
                    visited_local.add((ni, nj))
                    cur = (ni, nj)
                    found = True
                    break
            if not found:
                break
            if on_canvas_boundary(cur[0], cur[1]) and len(path) > 1:
                break
        return path

    for si, sj in boundary_starts:
        if (si, sj) in visited_global:
            continue
        visited_local = set()
        path = trace_from(si, sj, visited_local)
        if len(path) >= min_pixel_length:
            curves_pixel.append(path)
            visited_global.update(path)

    curves_world = []
    for path in curves_pixel:
        coords = np.array([[x_min + pj * dx, y_min + pi * dy] for (pi, pj) in path], dtype=np.float64)
        if len(coords) >= 4 and smooth_factor > 0:
            coords = _smooth_curve(coords, smoothness=smooth_factor, num_output=max(len(coords) * 2, 80))
        curves_world.append(coords)

    return curves_world


# ============================================================================
# Road Renderer (Three-layer with NO seam artifacts)
# Key fix: Use interpolation="nearest" to prevent edge color bleeding
# ============================================================================

class CenterlineRoadRenderer2D:
    def __init__(self, lane_width: float = DEFAULT_LANE_WIDTH, grid_resolution: float = 0.25):
        self.lane_width = float(lane_width)
        self.grid_resolution = float(grid_resolution)

    def draw(
        self,
        ax,
        road_points: np.ndarray,
        lane_types: np.ndarray | None = None,
        facecolor: str = ACADEMIC_PALETTE["road_surface"],
        offroad_color: str = ACADEMIC_PALETTE["offroad"],
        edgecolor: str = ACADEMIC_PALETTE["road_edge"],
        centerline_color: str = ACADEMIC_PALETTE["centerline"],
        centerline_style=(0, (5, 4)),
        edge_width: float = 4.0,
        centerline_width: float = 1.6,
        alpha_fill: float = 1.0,
        z_base: float = 1.0,
        xlim: tuple | None = None,
        ylim: tuple | None = None,
        grid_resolution: float | None = None,
        use_low_road_edge: bool = False,
        bg_gap_m: float = DEFAULT_BG_GAP_M,
    ):
        roads = np.asarray(road_points, dtype=np.float64)
        if roads.size == 0 or roads.ndim != 3 or roads.shape[2] < 2:
            return

        res = grid_resolution if grid_resolution is not None else self.grid_resolution
        xs, ys, road_mask, offroad_outer_mask, buffered_mask = _compute_road_mask_grid(
            roads,
            lanes_mask=None,
            lane_width=self.lane_width,
            xlim=xlim,
            ylim=ylim,
            resolution=res,
            morph_iterations=2,
            bg_gap_m=bg_gap_m,
        )
        if xs is None or road_mask is None or offroad_outer_mask is None:
            return

        # ====================================================================
        # KEY FIX: Three-layer rendering with explicit gap fill
        # Gap region now filled with WHITE (not transparent) to prevent
        # bilinear interpolation from bleeding gray into the road edge
        # ====================================================================
        if alpha_fill > 0.0:
            offroad_alpha = float(ACADEMIC_PALETTE.get("offroad_alpha", 0.88)) * float(alpha_fill)
            road_alpha = float(ACADEMIC_PALETTE.get("road_surface_alpha", 1.0)) * float(alpha_fill)

            H, W = road_mask.shape
            rgba = np.zeros((H, W, 4), dtype=np.float32)

            road_rgba = mcolors.to_rgba(facecolor)
            off_rgba = mcolors.to_rgba(offroad_color)
            white_rgba = mcolors.to_rgba("#ffffff")

            # Gap region = buffered but not road (white background)
            gap_mask = buffered_mask & (~road_mask)

            # Layer 1: Offroad outer (gray)
            rgba[offroad_outer_mask, 0] = off_rgba[0]
            rgba[offroad_outer_mask, 1] = off_rgba[1]
            rgba[offroad_outer_mask, 2] = off_rgba[2]
            rgba[offroad_outer_mask, 3] = offroad_alpha

            # Layer 2: Gap region (white, opaque - KEY FIX)
            rgba[gap_mask, 0] = white_rgba[0]
            rgba[gap_mask, 1] = white_rgba[1]
            rgba[gap_mask, 2] = white_rgba[2]
            rgba[gap_mask, 3] = 1.0  # Fully opaque white

            # Layer 3: Road surface (cyan)
            rgba[road_mask, 0] = road_rgba[0]
            rgba[road_mask, 1] = road_rgba[1]
            rgba[road_mask, 2] = road_rgba[2]
            rgba[road_mask, 3] = road_alpha

            # Use "nearest" interpolation to prevent edge color bleeding
            ax.imshow(
                rgba,
                origin="lower",
                extent=(xs[0], xs[-1], ys[0], ys[-1]),
                interpolation="nearest",  # KEY FIX: prevents gray seam lines
                zorder=z_base,
                aspect="auto",
            )

        # Road edges (boundary of road_mask) - drawn above fills
        edge_curves = _extract_open_road_edges(
            road_mask, xs, ys,
            min_pixel_length=6,
            smooth_factor=2.0,
        )

        if use_low_road_edge:
            edge_c = UNIFIED_COLORS.get("road_edge_low", edgecolor)
            edge_alpha = UNIFIED_COLORS.get("road_edge_low_alpha", 0.75)
        else:
            edge_c = edgecolor
            edge_alpha = float(ACADEMIC_PALETTE.get("road_edge_alpha", 0.98))

        road_edge_zorder = z_base + 15.0
        for curve in edge_curves:
            if len(curve) < 2:
                continue
            ax.plot(
                curve[:, 0],
                curve[:, 1],
                color=edge_c,
                linewidth=edge_width,
                solid_capstyle="round",
                solid_joinstyle="round",
                alpha=edge_alpha,
                zorder=road_edge_zorder,
            )

        # Centerlines
        n_lanes = roads.shape[0]
        lane_types_idx = None
        if lane_types is not None:
            lt = np.asarray(lane_types)
            if lt.ndim == 2:
                lane_types_idx = np.argmax(lt, axis=1)
            else:
                lane_types_idx = lt.astype(int)

        centerline_zorder = z_base + 3.0
        for i in range(n_lanes):
            cl = roads[i, :, :2]
            valid = np.isfinite(cl[:, 0]) & np.isfinite(cl[:, 1])
            cl = cl[valid]
            if cl.shape[0] < 2:
                continue

            cl_color = centerline_color
            cl_style = centerline_style
            cl_lw = centerline_width
            cl_zorder = centerline_zorder

            if lane_types_idx is not None and i < len(lane_types_idx):
                lt_i = int(lane_types_idx[i])
                if lt_i == 1:
                    cl_color = ACADEMIC_PALETTE["lane_green"]
                    cl_style = "solid"
                    cl_lw = centerline_width * 1.8
                    cl_zorder = centerline_zorder + 0.5
                elif lt_i == 2:
                    cl_color = ACADEMIC_PALETTE["lane_red"]
                    cl_style = "solid"
                    cl_lw = centerline_width * 1.8
                    cl_zorder = centerline_zorder + 0.5

            ax.plot(
                cl[:, 0],
                cl[:, 1],
                color=cl_color,
                linewidth=cl_lw,
                linestyle=cl_style,
                solid_capstyle="round",
                zorder=cl_zorder,
            )


# ============================================================================
# Agent Parsing
# ============================================================================

def _infer_ego_pose(states: np.ndarray):
    arr = np.asarray(states, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return 0.0, 0.0, 0.0
    ego = arr[-1]
    if ego.shape[0] >= 8:
        x = _safe_float(ego[0], 0.0)
        y = _safe_float(ego[1], 0.0)
        yaw = _safe_float(ego[4], 0.0)
    elif ego.shape[0] >= 7:
        x = _safe_float(ego[0], 0.0)
        y = _safe_float(ego[1], 0.0)
        cos_t = _safe_float(ego[3], 1.0)
        sin_t = _safe_float(ego[4], 0.0)
        yaw = math.atan2(sin_t, cos_t)
    else:
        x = _safe_float(ego[0], 0.0)
        y = _safe_float(ego[1], 0.0)
        yaw = 0.0
    return x, y, yaw


def _ego_view_bounds(states: np.ndarray, front=50.0, back=18.0, side=28.0):
    x0, y0, yaw = _infer_ego_pose(states)
    x_min = x0 - side
    x_max = x0 + side
    y_min = y0 - back
    y_max = y0 + front
    return x_min, x_max, y_min, y_max, yaw


def _parse_agent_state(s: np.ndarray, agent_type_idx: int = 0):
    s = np.asarray(s, dtype=float).reshape(-1)
    if s.size == 0 or np.all(np.isnan(s)):
        return None

    if s.shape[0] >= 8:
        x = _safe_float(s[0], 0.0)
        y = _safe_float(s[1], 0.0)
        vx = _safe_float(s[2], 0.0)
        vy = _safe_float(s[3], 0.0)
        speed_raw = math.hypot(vx, vy)
        theta = _safe_float(s[4], 0.0)
        length = _safe_float(s[5], 4.5)
        width = _safe_float(s[6], 2.0)
        exists = _safe_float(s[7], 0.0) > 0.0
    elif s.shape[0] >= 7:
        x = _safe_float(s[0], 0.0)
        y = _safe_float(s[1], 0.0)
        speed_raw = max(0.0, _safe_float(s[2], 0.0))
        cos_t = _safe_float(s[3], 1.0)
        sin_t = _safe_float(s[4], 0.0)
        theta = math.atan2(sin_t, cos_t)
        length = _safe_float(s[5], 4.5)
        width = _safe_float(s[6], 2.0)
        exists = True
    else:
        return None

    if not exists:
        return None
    if not (np.isfinite(x) and np.isfinite(y)):
        return None

    length = length if (np.isfinite(length) and length > 0) else 4.5
    width = width if (np.isfinite(width) and width > 0) else 2.0

    max_speed = AGENT_TYPE_MAX_SPEEDS.get(agent_type_idx, DEFAULT_MAX_SPEED)
    speed_norm = float(np.clip(speed_raw / max_speed, 0.0, 1.0))

    return {
        "x": x,
        "y": y,
        "speed": speed_raw,
        "speed_norm": speed_norm,
        "theta": theta,
        "length": length,
        "width": width,
        "agent_type": int(agent_type_idx),
    }


def _get_agent_colors(agent_type_idx: int, is_ego: bool):
    if is_ego:
        return (
            ACADEMIC_PALETTE["ego_fill"],
            ACADEMIC_PALETTE["ego_edge"],
            ACADEMIC_PALETTE["ego_fill_alpha"],
        )

    if agent_type_idx in (0, 1):
        return (
            ACADEMIC_PALETTE["veh_fill"],
            ACADEMIC_PALETTE["veh_edge"],
            ACADEMIC_PALETTE["veh_fill_alpha"],
        )
    elif agent_type_idx == 2:
        return (
            ACADEMIC_PALETTE["ped_fill"],
            ACADEMIC_PALETTE["ped_edge"],
            ACADEMIC_PALETTE["ped_fill_alpha"],
        )
    elif agent_type_idx == 3:
        return (
            ACADEMIC_PALETTE["cyc_fill"],
            ACADEMIC_PALETTE["cyc_edge"],
            ACADEMIC_PALETTE["cyc_fill_alpha"],
        )
    else:
        return (
            ACADEMIC_PALETTE["other_fill"],
            ACADEMIC_PALETTE["other_edge"],
            ACADEMIC_PALETTE["other_fill_alpha"],
        )


# ============================================================================
# Agent Shape Drawing
# ============================================================================

def _draw_vehicle_rounded(ax, parsed, fill_color, edge_color, alpha, linewidth, zorder, is_ego=False):
    if parsed is None:
        return
    x, y = parsed["x"], parsed["y"]
    length = parsed["length"]
    width = parsed["width"]
    theta = parsed["theta"]

    corner_radius = min(width, length) * 0.18
    rect = FancyBboxPatch(
        (x - width / 2.0, y - length / 2.0),
        width,
        length,
        boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=corner_radius),
        facecolor=fill_color,
        edgecolor=edge_color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    rotation = transforms.Affine2D().rotate_deg_around(x, y, np.degrees(theta) - 90.0)
    rect.set_transform(rotation + ax.transData)
    ax.add_patch(rect)

    # Heading chevron
    chevron_size = length * 0.22
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    front_center = np.array([x + cos_t * length * 0.35, y + sin_t * length * 0.35])
    left_vec = np.array([-sin_t, cos_t])

    p1 = front_center - left_vec * chevron_size * 0.5
    p2 = front_center + np.array([cos_t, sin_t]) * chevron_size * 0.6
    p3 = front_center + left_vec * chevron_size * 0.5

    chevron = Polygon(
        [p1, p2, p3],
        closed=True,
        facecolor=edge_color,
        edgecolor="none",
        alpha=alpha * 0.85,
        zorder=zorder + 0.1,
    )
    ax.add_patch(chevron)

    # Ego glow
    if is_ego:
        glow_rect = FancyBboxPatch(
            (x - width / 2.0 - 0.3, y - length / 2.0 - 0.3),
            width + 0.6,
            length + 0.6,
            boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=corner_radius + 0.2),
            facecolor="none",
            edgecolor=ACADEMIC_PALETTE["ego_glow"],
            linewidth=linewidth * 2.5,
            alpha=ACADEMIC_PALETTE["ego_glow_alpha"],
            zorder=zorder - 0.1,
        )
        glow_rect.set_transform(rotation + ax.transData)
        ax.add_patch(glow_rect)


def _draw_pedestrian_square(ax, parsed, fill_color, edge_color, alpha, linewidth, zorder):
    if parsed is None:
        return
    x, y = parsed["x"], parsed["y"]
    theta = parsed["theta"]

    size = min(max(parsed["width"], parsed["length"]) * 0.5, 1.5)
    size = max(size, 0.8)
    half = size / 2.0
    corner_radius = size * 0.12

    rect = FancyBboxPatch(
        (x - half, y - half),
        size,
        size,
        boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=corner_radius),
        facecolor=fill_color,
        edgecolor=edge_color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    rotation = transforms.Affine2D().rotate_deg_around(x, y, np.degrees(theta) - 90.0)
    rect.set_transform(rotation + ax.transData)
    ax.add_patch(rect)

    arrow_len = size * 0.4
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    tip = np.array([x + cos_t * (half + arrow_len * 0.3), y + sin_t * (half + arrow_len * 0.3)])
    left_vec = np.array([-sin_t, cos_t])
    base_center = np.array([x + cos_t * half * 0.3, y + sin_t * half * 0.3])

    p1 = base_center - left_vec * arrow_len * 0.3
    p2 = tip
    p3 = base_center + left_vec * arrow_len * 0.3

    arrow = Polygon(
        [p1, p2, p3],
        closed=True,
        facecolor=edge_color,
        edgecolor="none",
        alpha=alpha * 0.85,
        zorder=zorder + 0.1,
    )
    ax.add_patch(arrow)


def _draw_cyclist_rectangle(ax, parsed, fill_color, edge_color, alpha, linewidth, zorder):
    if parsed is None:
        return
    x, y = parsed["x"], parsed["y"]
    theta = parsed["theta"]

    rect_length = max(parsed["length"] * 0.6, 2.2)
    rect_width = max(parsed["width"] * 0.5, 0.9)
    if rect_length < rect_width * 1.5:
        rect_length = rect_width * 1.8

    corner_radius = min(rect_width, rect_length) * 0.15
    rect = FancyBboxPatch(
        (x - rect_width / 2.0, y - rect_length / 2.0),
        rect_width,
        rect_length,
        boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=corner_radius),
        facecolor=fill_color,
        edgecolor=edge_color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    rotation = transforms.Affine2D().rotate_deg_around(x, y, np.degrees(theta) - 90.0)
    rect.set_transform(rotation + ax.transData)
    ax.add_patch(rect)

    head_len = rect_length * 0.45
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    hx = x + cos_t * head_len
    hy = y + sin_t * head_len

    ax.plot(
        [x, hx], [y, hy],
        color=edge_color,
        linewidth=linewidth * 1.0,
        solid_capstyle="round",
        alpha=alpha * 0.9,
        zorder=zorder + 0.1,
    )
    ax.scatter(
        [hx], [hy],
        s=linewidth * 8,
        c=[edge_color],
        alpha=alpha * 0.9,
        zorder=zorder + 0.15,
        edgecolors="none",
    )


def _draw_agent_id(ax, parsed, agent_id, is_ego: bool, zorder: float):
    """
    Draw agent ID label.
    
    Changes:
    - Ego now displays "Ego" instead of "E"
    - Text is drawn at the highest zorder to ensure visibility
    - Ego text uses white color for contrast on red background
    """
    if parsed is None:
        return
    x, y = parsed["x"], parsed["y"]
    length = parsed.get("length", 4.5)
    width = parsed.get("width", 2.0)

    if is_ego:
        label = "Ego"  # Changed from "E" to "Ego"
        text_color = ACADEMIC_PALETTE["id_text_ego"]  # White for contrast
        fontweight = "bold"
    else:
        label = str(agent_id)
        text_color = ACADEMIC_PALETTE["id_text_npc"]
        fontweight = "bold"

    min_dim = min(length, width)
    # Slightly smaller font for "Ego" to fit
    if is_ego:
        fontsize = max(7, min(12, min_dim * ACADEMIC_PALETTE["id_font_scale"] * 0.75))
    else:
        fontsize = max(9, min(16, min_dim * ACADEMIC_PALETTE["id_font_scale"]))

    # Ensure text is at the topmost layer (zorder + 500 for absolute top)
    text_zorder = zorder + 500

    ax.text(
        x, y, label,
        fontsize=fontsize,
        fontweight=fontweight,
        fontfamily="sans-serif",
        ha="center",
        va="center",
        color=text_color,
        zorder=text_zorder,
    )


def _draw_agent_shape(ax, parsed, agent_type_idx, is_ego, linewidth, zorder):
    if parsed is None:
        return

    fill_c, edge_c, alpha = _get_agent_colors(agent_type_idx, is_ego)

    if agent_type_idx == 2:
        _draw_pedestrian_square(ax, parsed, fill_c, edge_c, alpha, linewidth, zorder)
    elif agent_type_idx == 3:
        _draw_cyclist_rectangle(ax, parsed, fill_c, edge_c, alpha, linewidth, zorder)
    else:
        _draw_vehicle_rounded(ax, parsed, fill_c, edge_c, alpha, linewidth, zorder, is_ego)


# ============================================================================
# Trajectory Visualization - High Contrast Academic Color Gradients
# KEY: Fixed alpha, color-only gradient for publication quality
# ============================================================================

# Academic color gradient stops (higher contrast, more distinct)
_TRAJ_GRADIENT_STOPS = {
    # Ego: Deep academic red gradient (maroon -> crimson -> scarlet)
    "ego": ["#7f1d1d", "#991b1b", "#b91c1c", "#dc2626", "#ef4444"],
    
    # Vehicle: Blue -> Cyan gradient (navy -> blue -> sky -> cyan)
    "veh": ["#1e3a8a", "#1d4ed8", "#3b82f6", "#0ea5e9", "#06b6d4"],
    
    # Cyclist: Purple gradient (deep violet -> purple -> lavender)
    "cyc": ["#4c1d95", "#6d28d9", "#7c3aed", "#8b5cf6", "#a78bfa"],
    
    # Pedestrian: Orange gradient (brown-orange -> orange -> amber)
    "ped": ["#7c2d12", "#9a3412", "#c2410c", "#ea580c", "#f97316"],
    
    # Other: Emerald gradient
    "other": ["#064e3b", "#047857", "#059669", "#10b981", "#34d399"],
}


def _traj_key(agent_type_idx: int, is_ego: bool) -> str:
    if is_ego:
        return "ego"
    if agent_type_idx in (0, 1):
        return "veh"
    if agent_type_idx == 3:
        return "cyc"
    if agent_type_idx == 2:
        return "ped"
    return "other"


def _draw_trajectory_granular(
    ax,
    parsed,
    is_ego: bool,
    agent_type_idx: int,
    max_length_m: float = MAX_TRAJECTORY_LENGTH_M,
    dt: float = 0.1,
    num_beads: int = TRAJECTORY_NUM_BEADS,
    zorder: float = 50,
):
    """
    Draw trajectory with academic color gradient (fixed alpha, color-only gradient).
    
    NOTE: In closed-loop simulation, we only have current state (x, y, vx, vy, theta).
    True future trajectory is unknown as it's generated step-by-step by the behaviour model.
    This function uses heuristic extrapolation based on current velocity and heading.
    
    Design:
    - Ego: Deep red academic gradient
    - Vehicles: Blue -> Cyan gradient
    - Cyclists: Purple gradient
    - Pedestrians: Orange gradient
    - Fixed alpha for all beads (no transparency gradient)
    """
    if parsed is None:
        return

    x, y = parsed["x"], parsed["y"]
    speed = float(parsed.get("speed", 0.0))
    theta = parsed["theta"]

    if not np.isfinite(speed) or speed < 0.2:
        return
    if not np.isfinite(theta):
        theta = 0.0

    # Horizon based on speed
    horizon_s = min(max_length_m / max(speed, 0.5), 5.0)
    n_pts = max(8, num_beads * 4)
    ts = np.linspace(0.0, horizon_s, n_pts)

    # Heuristic curvature - slightly larger for visual effect
    # This is a visual approximation since we don't have true future trajectory
    curvature = 0.02 * np.sin(theta * 2 + x * 0.1)

    future_x = np.zeros(n_pts)
    future_y = np.zeros(n_pts)
    for i, t in enumerate(ts):
        angle_t = theta + curvature * t
        future_x[i] = x + speed * math.cos(angle_t) * t
        future_y[i] = y + speed * math.sin(angle_t) * t

    future_pts = np.stack([future_x, future_y], axis=-1)
    if len(future_pts) >= 4:
        future_pts = _smooth_curve_spline(future_pts, smoothness=0.2, num_output=num_beads * 5)
    if not np.all(np.isfinite(future_pts)):
        return

    diffs = np.diff(future_pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumulative_dist = np.zeros(len(future_pts))
    cumulative_dist[1:] = np.cumsum(seg_lengths)
    total_length = cumulative_dist[-1]
    if total_length < 0.5:
        return

    # Get color gradient stops for this agent type
    key = _traj_key(int(agent_type_idx), bool(is_ego))
    stops = _TRAJ_GRADIENT_STOPS.get(key, _TRAJ_GRADIENT_STOPS["veh"])
    
    # Fixed alpha values from palette - NO transparency gradient
    bead_alpha = float(ACADEMIC_PALETTE["traj_alpha"])
    line_alpha = float(ACADEMIC_PALETTE["traj_line_alpha"])

    # Draw connecting line with color gradient (FIXED alpha - no transparency gradient)
    line_n = len(future_pts) - 1
    if line_n > 0:
        line_colors = _make_gradient_rgba_fixed_alpha(stops, line_n, line_alpha)
        segments = [[future_pts[i], future_pts[i + 1]] for i in range(line_n)]
        lc = LineCollection(
            segments,
            colors=line_colors,
            linewidths=float(ACADEMIC_PALETTE["traj_line_width"]),
            capstyle="round",
            joinstyle="round",
            zorder=zorder - 0.5,
        )
        ax.add_collection(lc)

    # Bead positions along trajectory
    bead_distances = np.linspace(0, total_length * 0.95, num_beads)
    bead_x = np.interp(bead_distances, cumulative_dist, future_pts[:, 0])
    bead_y = np.interp(bead_distances, cumulative_dist, future_pts[:, 1])

    # Color gradient for beads (FIXED alpha - no transparency gradient)
    bead_colors = _make_gradient_rgba_fixed_alpha(stops, num_beads, bead_alpha)
    bead_size = float(ACADEMIC_PALETTE["traj_bead_size"])
    bead_sizes = np.full(num_beads, bead_size)

    ax.scatter(
        bead_x,
        bead_y,
        s=bead_sizes,
        c=bead_colors,
        edgecolors="none",
        zorder=zorder,
        marker="o",
    )


# ============================================================================
# Safety Halo + Route + Legend
# ============================================================================

def _draw_safety_halo(
    ax,
    ego_parsed,
    all_agent_parsed: list,
    safety_radius: float = DEFAULT_SAFETY_RADIUS,
    zorder: float = 45,
):
    if ego_parsed is None:
        return
    ex, ey = ego_parsed["x"], ego_parsed["y"]

    intrusion = False
    for p in all_agent_parsed:
        if p is None:
            continue
        dist = math.hypot(p["x"] - ex, p["y"] - ey)
        if dist < safety_radius:
            intrusion = True
            break

    if intrusion:
        halo_color = ACADEMIC_PALETTE["safety_warning"]
        halo_alpha = ACADEMIC_PALETTE["safety_warning_alpha"]
    else:
        halo_color = ACADEMIC_PALETTE["safety_normal"]
        halo_alpha = ACADEMIC_PALETTE["safety_normal_alpha"]

    inner_radius = max(ego_parsed["length"], ego_parsed["width"]) * 0.6
    outer_radius = safety_radius

    ring = Wedge(
        (ex, ey),
        outer_radius,
        0, 360,
        width=outer_radius - inner_radius,
        facecolor=halo_color,
        edgecolor="none",
        alpha=halo_alpha,
        zorder=zorder,
    )
    ax.add_patch(ring)


def _draw_route_dots(
    ax,
    route,
    dot_spacing: float = 2.5,
    dot_size: float = 18,
    linewidth: float = 2.0,
    zorder: float = 10,
):
    if route is None or len(route) < 2:
        return

    r = np.asarray(route, dtype=float)
    if r.ndim != 2 or r.shape[1] < 2:
        return

    ax.plot(
        r[:, 0],
        r[:, 1],
        color=ACADEMIC_PALETTE["route_line"],
        linewidth=linewidth,
        linestyle=(0, (6, 3)),
        alpha=ACADEMIC_PALETTE["route_alpha"],
        solid_capstyle="round",
        zorder=zorder,
    )

    cumulative_dist = np.zeros(len(r))
    for i in range(1, len(r)):
        cumulative_dist[i] = cumulative_dist[i - 1] + np.linalg.norm(r[i] - r[i - 1])

    total_length = cumulative_dist[-1]
    if total_length < dot_spacing:
        return

    num_dots = int(total_length / dot_spacing)
    dot_distances = np.linspace(dot_spacing * 0.5, total_length - dot_spacing * 0.5, num_dots)

    dot_x = np.interp(dot_distances, cumulative_dist, r[:, 0])
    dot_y = np.interp(dot_distances, cumulative_dist, r[:, 1])

    ax.scatter(
        dot_x,
        dot_y,
        color=ACADEMIC_PALETTE["route_dot"],
        s=dot_size,
        alpha=ACADEMIC_PALETTE["route_alpha"] * 0.95,
        zorder=zorder + 0.1,
        edgecolors="none",
        marker="o",
    )


def _draw_legend(ax, elements: list, location="lower right", fontsize=7):
    handles = []
    for elem in elements:
        if elem["type"] == "patch":
            handles.append(mpatches.Patch(
                facecolor=elem.get("facecolor", elem.get("color", "#000000")),
                edgecolor=elem.get("edgecolor", "none"),
                label=elem["label"],
                alpha=elem.get("alpha", 1.0),
            ))
        elif elem["type"] == "line":
            handles.append(mlines.Line2D(
                [], [],
                color=elem.get("color", "#000000"),
                linewidth=elem.get("linewidth", 2.0),
                linestyle=elem.get("linestyle", "solid"),
                label=elem["label"],
                alpha=elem.get("alpha", 1.0),
            ))
        elif elem["type"] == "circle":
            handles.append(mlines.Line2D(
                [], [],
                marker="o",
                color="w",
                markerfacecolor=elem.get("color", "#000000"),
                markersize=elem.get("markersize", 6),
                label=elem["label"],
                linestyle="None",
            ))
        elif elem["type"] == "square":
            handles.append(mlines.Line2D(
                [], [],
                marker="s",
                color="w",
                markerfacecolor=elem.get("color", "#000000"),
                markersize=elem.get("markersize", 6),
                label=elem["label"],
                linestyle="None",
            ))

    if handles:
        ax.legend(
            handles=handles,
            loc=location,
            frameon=True,
            framealpha=0.92,
            facecolor="#ffffff",
            edgecolor="#d1d5db",
            fontsize=fontsize,
            handlelength=1.5,
            handleheight=0.8,
            labelspacing=0.4,
        )


# ============================================================================
# Main Rendering Function
# ============================================================================
def render_simulation_frame_2d(
    agent_states,
    agent_types,
    route,
    lanes,
    lanes_mask,
    t,
    name,
    movie_path: str = "video_frames",
    lightweight: bool = False,
    lane_width: float = DEFAULT_LANE_WIDTH,
    show_legend: bool = True,
    show_safety_halo: bool = True,
    safety_radius: float = DEFAULT_SAFETY_RADIUS,
    agent_ids: np.ndarray | list | None = None,
):
    """
    Render simulation frame with academic styling.

    == sync, corrected by elderman ==
    
    1. Fixed image size:
       - figsize=(8.0, 7.0) with dpi to produce definitive pixel dimensions
       - dpi=150-> 1200x1050 pixels
       - dpi=100->800x 700 pixels
       - All frames are set in the same setting, ensuring the same size.
    
    2. Fixed view range:
       - Fixed area centred on ego: first 50 m, then 18 m, about 28 m
       - The field of view is fixed at 56m x 68m
       - To ensure that the landscape is consistent
    
    3. Uncertainty conservation:
       - Use bbox_inches=Noe to avoid automatic cropping
       - Use pad_inches=0.0 to avoid extra margins
       - Make sure the output size is exactly the same as the figsize*dpi.
    
    4. Libx264 compatibility:
       - Figsize ensures the pixel size is even.
       - Extra use of ffmpeg's pad filter as a double guarantee
    
    =================================================
    """
    png_dir = os.path.join(movie_path, str(name))
    os.makedirs(png_dir, exist_ok=True)

    # Fixed DPI Settings
    dpi = 100 if lightweight else 150

    # Fixed figsize to ensure even numbers of pixels (libx264 requirements)
    # dpi = 150: 8.0*150 = 1200, 7.0*150 = 1050 (all even)
    # dpi = 100: 8.0*100 = 800, 7.0*100 = 700 (all even)
    fig = plt.figure(figsize=(8.0, 7.0), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])

    bg = ACADEMIC_PALETTE["bg"]
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    states = np.asarray(agent_states, dtype=float)
    types = np.asarray(agent_types) if agent_types is not None else None
    lanes_arr = np.asarray(lanes, dtype=float) if lanes is not None else None
    lanes_mask_arr = np.asarray(lanes_mask).astype(bool) if lanes_mask is not None else None

    # Calculate a fixed view boundary (centric with ego)
    x_min, x_max, y_min, y_max, _ = _ego_view_bounds(states)

    # Agent type indices
    if types is not None:
        t_arr = np.asarray(types)
        if t_arr.ndim == 2:
            types_idx = np.argmax(t_arr, axis=1)
        else:
            types_idx = t_arr.astype(int)
    else:
        types_idx = np.zeros(states.shape[0], dtype=int)

    # Parse all agents
    num_agents = states.shape[0]
    all_parsed = []
    for a_idx in range(num_agents):
        parsed = _parse_agent_state(states[a_idx], agent_type_idx=int(types_idx[a_idx]))
        all_parsed.append(parsed)

    ego_parsed = all_parsed[-1] if num_agents > 0 else None
    npc_parsed = all_parsed[:-1] if num_agents > 1 else []

    # Draw road
    if lanes_arr is not None and lanes_arr.size > 0:
        n_lanes, n_pts, _ = lanes_arr.shape
        if lanes_mask_arr is None:
            lanes_mask_arr = np.ones((n_lanes, n_pts), dtype=bool)
        lanes_masked = np.where(lanes_mask_arr[..., None], lanes_arr, np.nan)

        road_renderer = CenterlineRoadRenderer2D(lane_width=lane_width, grid_resolution=0.25)
        road_renderer.draw(
            ax,
            lanes_masked,
            lane_types=None,
            facecolor=ACADEMIC_PALETTE["road_surface"],
            offroad_color=ACADEMIC_PALETTE["offroad"],
            edgecolor=ACADEMIC_PALETTE["road_edge"],
            centerline_color=ACADEMIC_PALETTE["centerline"],
            centerline_style=(0, (5, 4)),
            edge_width=4.0,
            centerline_width=1.6,
            alpha_fill=1.0,
            z_base=1.0,
            xlim=(x_min, x_max),
            ylim=(y_min, y_max),
            bg_gap_m=DEFAULT_BG_GAP_M,
        )

    # Route
    _draw_route_dots(
        ax,
        route,
        dot_spacing=3.0,
        dot_size=15,
        linewidth=2.0,
        zorder=10,
    )

    # Safety halo
    if show_safety_halo and ego_parsed is not None:
        _draw_safety_halo(
            ax,
            ego_parsed,
            npc_parsed,
            safety_radius=safety_radius,
            zorder=45,
        )

    # Agent ids
    if agent_ids is not None:
        agent_ids = list(agent_ids)
    else:
        agent_ids = list(range(1, num_agents)) + [-1]

    # Draw NPC then ego
    draw_order = list(range(num_agents - 1)) + [num_agents - 1]

    for a_idx in draw_order:
        is_ego = a_idx == num_agents - 1
        t_idx = int(types_idx[a_idx])
        parsed = all_parsed[a_idx]
        if parsed is None:
            continue

        z_agent = 100 + a_idx if not is_ego else 200
        lw = 1.4 if not is_ego else 1.65

        _draw_trajectory_granular(
            ax,
            parsed,
            is_ego=is_ego,
            agent_type_idx=t_idx,
            max_length_m=MAX_TRAJECTORY_LENGTH_M,
            dt=0.1,
            num_beads=TRAJECTORY_NUM_BEADS,
            zorder=z_agent - 5,
        )

        _draw_agent_shape(ax, parsed, t_idx, is_ego, lw, z_agent)

        aid = agent_ids[a_idx] if a_idx < len(agent_ids) else a_idx
        _draw_agent_id(ax, parsed, aid, is_ego, z_agent)

    # Set a fixed view boundary
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # Legend
    if show_legend:
        legend_elements = [
            {"type": "patch", "color": ACADEMIC_PALETTE["offroad"], "label": "Offroad", "alpha": ACADEMIC_PALETTE["offroad_alpha"]},
            {"type": "patch", "color": ACADEMIC_PALETTE["road_surface"], "label": "Road", "edgecolor": "#d1d5db"},
            {"type": "line", "color": ACADEMIC_PALETTE["road_edge"], "linewidth": 2.5, "label": "Road Edge"},
            {"type": "patch", "color": ACADEMIC_PALETTE["ego_fill"], "label": "Ego", "alpha": ACADEMIC_PALETTE["ego_fill_alpha"]},
            {"type": "patch", "color": ACADEMIC_PALETTE["veh_fill"], "label": "Vehicle", "alpha": ACADEMIC_PALETTE["veh_fill_alpha"]},
            {"type": "square", "color": ACADEMIC_PALETTE["ped_fill"], "label": "Pedestrian", "markersize": 6},
            {"type": "patch", "color": ACADEMIC_PALETTE["cyc_fill"], "label": "Cyclist", "alpha": ACADEMIC_PALETTE["cyc_fill_alpha"]},
            {"type": "line", "color": ACADEMIC_PALETTE["route_line"], "linewidth": 2.0, "linestyle": (0, (6, 3)), "label": "Route"},
        ]
        if show_safety_halo:
            legend_elements.append(
                {"type": "patch", "color": ACADEMIC_PALETTE["safety_normal"], "label": "Safety Zone", "alpha": 0.28}
            )
        _draw_legend(ax, legend_elements, location="lower right", fontsize=6)

    # Determined saving: use fixed dimensions without any cropping or adjustment
    outfile = os.path.join(png_dir, f"frame_{int(t):04d}.png")
    fig.savefig(
        outfile,
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        bbox_inches=None,  # Do not usetight to ensure size certainty
        pad_inches=0.0,    # No extra margin
    )
    plt.close(fig)


def render_simulation_frame_3d(
    agent_states,
    agent_types,
    route,
    lanes,
    lanes_mask,
    t,
    name,
    movie_path: str = "video_frames",
    lightweight: bool = False,
    agent_ids: np.ndarray | list | None = None,
):
    """Alias for 2D rendering."""
    render_simulation_frame_2d(
        agent_states=agent_states,
        agent_types=agent_types,
        route=route,
        lanes=lanes,
        lanes_mask=lanes_mask,
        t=t,
        name=name,
        movie_path=movie_path,
        lightweight=lightweight,
        lane_width=DEFAULT_LANE_WIDTH,
        agent_ids=agent_ids,
    )