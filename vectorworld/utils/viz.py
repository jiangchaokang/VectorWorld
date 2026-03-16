import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib import cm
import shlex

from configs.config import LANE_CONNECTION_TYPES_WAYMO, LANE_CONNECTION_TYPES_NUPLAN
from vectorworld.utils.geometry import *
from vectorworld.utils.data_helpers import unnormalize_motion_code
from vectorworld.utils.scene_viz import (
    CenterlineRoadRenderer2D,
    render_simulation_frame_2d,
    COLORS_2D,
    ACADEMIC_PALETTE,
    DEFAULT_LANE_WIDTH,
    _safe_float,
    _smooth_curve_spline,
    _parse_agent_state,
    _draw_agent_shape,
    _draw_agent_id,
    _draw_route_dots,
    _draw_legend,
)

import subprocess
import shutil
from pathlib import Path

from PIL import Image

import imageio_ffmpeg


def plot_k_disks_vocabulary(V: np.ndarray, png_path: str, dpi: int = 1000):
    """Plot k-disks vocabulary points."""
    arr = np.asarray(V, dtype=float)
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.shape[1] > 2:
        arr = arr[:, :2]
    fig, ax = plt.subplots(figsize=(3, 3), dpi=dpi)
    bg = ACADEMIC_PALETTE["bg"]
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.axhline(0.0, color="#e5e7eb", linewidth=0.6)
    ax.axvline(0.0, color="#e5e7eb", linewidth=0.6)
    ax.scatter(arr[:, 0], arr[:, 1], s=6, c="#2563eb", alpha=0.9, edgecolors="none")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.0,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)


def _validate_motion_vec(motion_vec: np.ndarray) -> np.ndarray | None:
    """Validate and sanitize motion vector."""
    if motion_vec is None:
        return None
    motion_vec = np.asarray(motion_vec, dtype=float).reshape(-1)
    d = motion_vec.shape[0]
    if d < 2 or d % 2 != 0:
        return None
    if not np.all(np.isfinite(motion_vec)):
        motion_vec = np.nan_to_num(motion_vec, nan=0.0, posinf=0.0, neginf=0.0)
    k = d // 2
    pts_body = motion_vec.reshape(k, 2)
    if np.allclose(pts_body, 0.0, atol=1e-2):
        return None
    if np.any(np.abs(pts_body) > 100.0):
        return None
    return motion_vec


def _draw_motion_trajectory_granular(
    ax,
    agent_state: np.ndarray,
    motion_vec: np.ndarray,
    color: str,
    linewidth: float,
    alpha: float,
    zorder: float,
    motion_max_displacement: float = 12.0,
    num_beads: int = 10,
):
    """Draw motion code trajectory as uniform granular beads in world coordinates."""
    motion_vec = _validate_motion_vec(motion_vec)
    if motion_vec is None:
        return

    d = motion_vec.shape[0]
    k = d // 2
    pts_body = motion_vec.reshape(k, 2)

    x0 = _safe_float(agent_state[0], 0.0)
    y0 = _safe_float(agent_state[1], 0.0)
    cos_theta = _safe_float(agent_state[3], 1.0)
    sin_theta = _safe_float(agent_state[4], 0.0)
    theta = np.arctan2(sin_theta, cos_theta)

    r = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=float
    )
    pts_world = pts_body @ r.T + np.array([x0, y0], dtype=float)[None, :]

    if not np.all(np.isfinite(pts_world)):
        return

    if len(pts_world) >= 4:
        pts_world = _smooth_curve_spline(pts_world, smoothness=0.3, num_output=max(len(pts_world) * 3, 30))

    if len(pts_world) < 2:
        return

    diffs = np.diff(pts_world, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumulative_dist = np.zeros(len(pts_world))
    cumulative_dist[1:] = np.cumsum(seg_lengths)
    total_length = cumulative_dist[-1]

    if total_length < 0.3:
        return

    # Connecting line with FIXED alpha (no transparency gradient)
    line_alpha = alpha * 0.4
    segments = []
    line_colors = []
    n = len(pts_world)
    for i in range(n - 1):
        segments.append([pts_world[i], pts_world[i + 1]])
        rgba = list(mcolors.to_rgba(color))
        rgba[3] = line_alpha
        line_colors.append(rgba)

    lc = LineCollection(
        segments,
        colors=line_colors,
        linewidths=linewidth * 0.6,
        capstyle="round",
        joinstyle="round",
        zorder=zorder - 0.5,
    )
    ax.add_collection(lc)

    # Uniform beads with FIXED alpha
    num_beads = min(num_beads, len(pts_world))
    bead_distances = np.linspace(0, total_length * 0.95, num_beads)

    bead_x = np.interp(bead_distances, cumulative_dist, pts_world[:, 0])
    bead_y = np.interp(bead_distances, cumulative_dist, pts_world[:, 1])

    bead_size = 16
    bead_sizes = np.full(num_beads, bead_size)
    bead_colors = [list(mcolors.to_rgba(color))[:3] + [alpha] for _ in range(num_beads)]

    ax.scatter(
        bead_x,
        bead_y,
        s=bead_sizes,
        c=bead_colors,
        edgecolors="none",
        zorder=zorder,
        marker="o",
    )


def _draw_motion_history_boxes(
    ax,
    agent_state: np.ndarray,
    motion_vec: np.ndarray,
    color: str,
    linewidth: float,
    alpha_min: float,
    alpha_max: float,
    zorder: float,
    max_boxes: int | None = None,
):
    """Draw historical bbox ghosts along motion trajectory."""
    motion_vec = _validate_motion_vec(motion_vec)
    if motion_vec is None:
        return

    d = motion_vec.shape[0]
    k = d // 2
    pts_body = motion_vec.reshape(k, 2)

    x0 = _safe_float(agent_state[0], 0.0)
    y0 = _safe_float(agent_state[1], 0.0)
    cos_theta = _safe_float(agent_state[3], 1.0)
    sin_theta = _safe_float(agent_state[4], 0.0)
    theta = np.arctan2(sin_theta, cos_theta)

    r = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=float
    )
    pts_world = pts_body @ r.T + np.array([x0, y0], dtype=float)[None, :]

    if k <= 1:
        return
    pts_hist = pts_world[:-1]
    h = pts_hist.shape[0]
    if h == 0:
        return
    if max_boxes is not None and max_boxes > 0 and h > max_boxes:
        idxs = np.linspace(0, h - 1, num=max_boxes, dtype=int)
        pts_hist = pts_hist[idxs]
        h = pts_hist.shape[0]

    length = _safe_float(agent_state[5], 4.5)
    width = _safe_float(agent_state[6], 2.0)
    alphas = np.linspace(alpha_min, alpha_max, num=h)

    corner_radius = min(width, length) * 0.15

    for (px, py), a in zip(pts_hist, alphas):
        if not (np.isfinite(px) and np.isfinite(py)):
            continue
        bbox_x_min = px - width / 2.0
        bbox_y_min = py - length / 2.0
        rect = mpatches.FancyBboxPatch(
            (bbox_x_min, bbox_y_min),
            width,
            length,
            ec=color,
            fc=color,
            linewidth=linewidth,
            alpha=a,
            boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=corner_radius),
            linestyle=(0, (2, 2)),
            zorder=zorder,
        )
        rotation = transforms.Affine2D().rotate_deg_around(
            px, py, np.degrees(theta) - 90.0
        ) + ax.transData
        rect.set_transform(rotation)
        ax.add_patch(rect)

def plot_scene(
    agent_states,
    road_points,
    agent_types,
    lane_types,
    name,
    save_dir,
    return_fig=False,
    tile_occupancy=None,
    adaptive_limits=False,
    route=None,
    agent_motion_gt=None,
    agent_motion_pred=None,
    motion_max_displacement=6.0,
    show_legend=True,
    gt_agent_states=None,
    gt_road_points=None,
    gt_agent_types=None,
    gt_lane_types=None,
    overlay_gt=False,
    motion_viz_mode="boxes",
    # ---------------- NEW (backward compatible) ----------------
    agent_id_policy: str = "auto",   # "auto" | "all" | "ego_only" | "none"
    legend_policy: str = "auto",     # "auto" | "always" | "never"
):
    """Plot a single scene with academic styling.

    Key fix for large stitched scenes:
    - When adaptive_limits=True and the stitched map grows, xlim/ylim expands.
      In Matplotlib, text/markers are in screen-space units and do NOT scale with xlim/ylim.
      So we must either scale them by the same scale_factor or auto-hide them.

    This function keeps previous behavior by default but improves aesthetics:
    - Agent IDs: auto policy (hide or ego-only when zoomed out)
    - Route dots: dot_size scaled with scale_factor (area ~ 1/scale_factor^2)
    - Legend: auto hide when zoomed out
    """
    colors = COLORS_2D
    agent_states = np.asarray(agent_states)
    road_points = np.asarray(road_points)
    if agent_types is not None:
        agent_types = np.asarray(agent_types)
    if lane_types is not None:
        lane_types = np.asarray(lane_types)
    if gt_agent_states is not None:
        gt_agent_states = np.asarray(gt_agent_states)
    if gt_road_points is not None:
        gt_road_points = np.asarray(gt_road_points)
    if gt_agent_types is not None:
        gt_agent_types = np.asarray(gt_agent_types)
    if gt_lane_types is not None:
        gt_lane_types = np.asarray(gt_lane_types)

    def _to_type_idx(arr):
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.ndim == 2:
            return np.argmax(arr, axis=1)
        return arr.astype(int)

    agent_types_pred_idx = _to_type_idx(agent_types)
    lane_types_idx = _to_type_idx(lane_types)
    agent_types_gt_idx = _to_type_idx(gt_agent_types)
    gt_lane_types_idx = _to_type_idx(gt_lane_types)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    fig.patch.set_facecolor(ACADEMIC_PALETTE["bg"])
    ax.set_facecolor(ACADEMIC_PALETTE["bg"])

    # ---------------- View limits ----------------
    if adaptive_limits and tile_occupancy is not None and len(tile_occupancy) > 0:
        x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf
        for tile_corners in tile_occupancy:
            x_min = min(x_min, tile_corners[:, 0].min())
            x_max = max(x_max, tile_corners[:, 0].max())
            y_min = min(y_min, tile_corners[:, 1].min())
            y_max = max(y_max, tile_corners[:, 1].max())
        if not np.isfinite([x_min, x_max, y_min, y_max]).all():
            x_max, x_min = 32.0, -32.0
            y_max, y_min = 32.0, -32.0
    else:
        x_max, x_min = 32.0, -32.0
        y_max, y_min = 32.0, -32.0

    x_range = float(x_max - x_min)
    y_range = float(y_max - y_min)
    base_range = 64.0
    scale_factor = max(x_range, y_range) / base_range if max(x_range, y_range) > 0 else 1.0
    scale_factor = max(1e-6, float(scale_factor))

    # Linewidth scaling (existing logic, keep)
    bbox_linewidth = 0.6 / scale_factor
    route_linewidth = 2.0 / scale_factor
    traj_linewidth = 1.6 / scale_factor
    road_edge_width = 3.0 / scale_factor
    centerline_width = 1.4 / scale_factor

    # ---------------- NEW: UI scaling ----------------
    # ui_scale controls screen-space elements (font sizes, scatter marker sizes)
    ui_scale = 1.0 / max(1.0, scale_factor)

    # Agent ID policy
    pol = str(agent_id_policy or "auto").lower()
    if pol == "auto":
        if scale_factor <= 1.5:
            pol = "all"
        elif scale_factor <= 3.0:
            pol = "ego_only"
        else:
            pol = "none"
    if pol not in {"all", "ego_only", "none"}:
        pol = "none"

    # Legend policy
    leg_pol = str(legend_policy or "auto").lower()
    if leg_pol == "auto":
        show_legend_eff = bool(show_legend) and (scale_factor <= 2.0)
    elif leg_pol == "always":
        show_legend_eff = bool(show_legend)
    else:
        show_legend_eff = False

    # Route dot scaling: scatter size is points^2
    base_dot_size = 12.0
    dot_size = base_dot_size * (ui_scale ** 2)
    dot_size = float(np.clip(dot_size, 0.5, base_dot_size))

    # ID fontsize scaling
    base_id_fs = 10.0
    id_fs = base_id_fs * ui_scale
    id_fs = float(np.clip(id_fs, 3.0, 12.0))

    def _draw_agent_id_scaled(ax, parsed, agent_id, is_ego: bool, zorder: float):
        if parsed is None:
            return
        if pol == "none":
            return
        if (pol == "ego_only") and (not is_ego):
            return

        # extra guard: when zoomed out a lot, even ego label can clutter
        if (not is_ego) and (scale_factor > 2.5):
            return

        x = float(parsed["x"])
        y = float(parsed["y"])
        label = "Ego" if is_ego else str(agent_id)

        txt_color = ACADEMIC_PALETTE["id_text_ego"] if is_ego else ACADEMIC_PALETTE["id_text_npc"]
        box_fc = "#111827" if is_ego else "#ffffff"
        box_alpha = 0.35 if is_ego else 0.55

        ax.text(
            x,
            y,
            label,
            fontsize=(id_fs * 0.95 if is_ego else id_fs),
            fontweight="bold",
            fontfamily="sans-serif",
            ha="center",
            va="center",
            color=txt_color,
            zorder=zorder + 500,
            bbox=dict(
                boxstyle="round,pad=0.12,rounding_size=0.15",
                facecolor=box_fc,
                edgecolor="none",
                alpha=box_alpha,
            ),
        )

    # ---------------- Lanes ----------------
    road_renderer = CenterlineRoadRenderer2D(lane_width=DEFAULT_LANE_WIDTH, grid_resolution=0.25)

    def _draw_lanes(pts, ltypes_idx_arr, is_gt: bool):
        if pts is None or len(pts) == 0:
            return

        if is_gt:
            road_renderer.draw(
                ax,
                pts,
                lane_types=None,
                facecolor=ACADEMIC_PALETTE["bg"],
                offroad_color=ACADEMIC_PALETTE["offroad"],
                edgecolor="#94a3b8",
                centerline_color="#94a3b8",
                centerline_style=(0, (3, 3)),
                edge_width=road_edge_width * 0.6,
                centerline_width=centerline_width * 0.7,
                alpha_fill=0.0,
                z_base=1.1,
                xlim=(x_min, x_max),
                ylim=(y_min, y_max),
                use_low_road_edge=True,
            )
        else:
            road_renderer.draw(
                ax,
                pts,
                lane_types=ltypes_idx_arr,
                facecolor=ACADEMIC_PALETTE["road_surface"],
                offroad_color=ACADEMIC_PALETTE["offroad"],
                edgecolor=ACADEMIC_PALETTE["road_edge"],
                centerline_color=ACADEMIC_PALETTE["centerline"],
                centerline_style=(0, (5, 4)),
                edge_width=road_edge_width,
                centerline_width=centerline_width,
                alpha_fill=1.0,
                z_base=1.0,
                xlim=(x_min, x_max),
                ylim=(y_min, y_max),
            )

            if ltypes_idx_arr is not None:
                n_lanes = pts.shape[0]
                for i in range(n_lanes):
                    lt = int(ltypes_idx_arr[i]) if i < len(ltypes_idx_arr) else 0
                    if lt == 0:
                        continue

                    cl = pts[i, :, :2]
                    valid = np.isfinite(cl[:, 0]) & np.isfinite(cl[:, 1])
                    cl = cl[valid]
                    if cl.shape[0] < 2:
                        continue

                    cl_color = ACADEMIC_PALETTE["lane_green"] if lt == 1 else ACADEMIC_PALETTE["lane_red"]
                    ax.plot(
                        cl[:, 0],
                        cl[:, 1],
                        color=cl_color,
                        linewidth=centerline_width * 1.8,
                        linestyle="solid",
                        solid_capstyle="round",
                        zorder=2.8,
                        alpha=0.95,
                    )

    _draw_lanes(road_points, lane_types_idx, is_gt=False)
    if overlay_gt and gt_road_points is not None:
        _draw_lanes(gt_road_points, gt_lane_types_idx, is_gt=True)

    # ---------------- Route ----------------
    if route is not None:
        _draw_route_dots(
            ax,
            route,
            dot_spacing=4.0,
            dot_size=dot_size,
            linewidth=route_linewidth,
            zorder=5.0,
        )

    # ---------------- Optional GT agent overlay ----------------
    if overlay_gt and gt_agent_states is not None:
        na_gt = len(gt_agent_states)
        for a in range(na_gt):
            length = _safe_float(gt_agent_states[a, 5], 4.5)
            width = _safe_float(gt_agent_states[a, 6], 2.0)
            x = _safe_float(gt_agent_states[a, 0], 0.0)
            y = _safe_float(gt_agent_states[a, 1], 0.0)
            bbox_x_min = x - width / 2.0
            bbox_y_min = y - length / 2.0
            corner_radius = min(width, length) * 0.15
            rect_gt = mpatches.FancyBboxPatch(
                (bbox_x_min, bbox_y_min),
                width,
                length,
                ec="#64748b",
                fc="none",
                linewidth=bbox_linewidth * 0.9,
                alpha=0.85,
                boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=corner_radius),
                linestyle=(0, (3, 3)),
                zorder=3.0,
            )
            cos_theta = _safe_float(gt_agent_states[a, 3], 1.0)
            sin_theta = _safe_float(gt_agent_states[a, 4], 0.0)
            theta = np.arctan2(sin_theta, cos_theta)
            rotation = transforms.Affine2D().rotate_deg_around(
                x, y, np.degrees(theta) - 90.0,
            ) + ax.transData
            rect_gt.set_transform(rotation)
            ax.add_patch(rect_gt)

    # ---------------- Agents ----------------
    na = len(agent_states)
    for a in range(na):
        t_idx = int(agent_types_pred_idx[a]) if agent_types_pred_idx is not None else 0
        is_ego = a == 0

        parsed = _parse_agent_state(agent_states[a], agent_type_idx=t_idx)
        if parsed is None:
            continue

        z_agent = 4.0 + a * 0.01
        _draw_agent_shape(ax, parsed, t_idx, is_ego, bbox_linewidth, z_agent)
        _draw_agent_id_scaled(ax, parsed, a, is_ego, z_agent)

    # ---------------- Motion code (optional) ----------------
    motion_gt_np = agent_motion_gt
    motion_pred_np = agent_motion_pred
    if motion_gt_np is not None or motion_pred_np is not None:
        mode = (motion_viz_mode or "boxes").lower()
        if mode not in {"curve", "boxes", "both"}:
            mode = "boxes"
        max_boxes = 6

        for a in range(na):
            base_state_gt = (
                gt_agent_states[a]
                if (gt_agent_states is not None and a < len(gt_agent_states))
                else agent_states[a]
            )
            base_state_pred = agent_states[a]

            if motion_gt_np is not None and a < len(motion_gt_np):
                if mode in {"curve", "both"}:
                    _draw_motion_trajectory_granular(
                        ax,
                        base_state_gt,
                        motion_gt_np[a],
                        color=ACADEMIC_PALETTE["motion_gt"],
                        linewidth=traj_linewidth * 1.0,
                        alpha=ACADEMIC_PALETTE["motion_gt_alpha"],
                        zorder=3.4,
                        motion_max_displacement=motion_max_displacement,
                        num_beads=8,
                    )
                if mode in {"boxes", "both"}:
                    _draw_motion_history_boxes(
                        ax,
                        base_state_gt,
                        motion_gt_np[a],
                        color=ACADEMIC_PALETTE["motion_gt"],
                        linewidth=bbox_linewidth * 0.8,
                        alpha_min=0.12,
                        alpha_max=0.50,
                        zorder=3.45,
                        max_boxes=max_boxes,
                    )

            if motion_pred_np is not None and a < len(motion_pred_np):
                if mode in {"curve", "both"}:
                    _draw_motion_trajectory_granular(
                        ax,
                        base_state_pred,
                        motion_pred_np[a],
                        color=ACADEMIC_PALETTE["motion_pred"],
                        linewidth=traj_linewidth * 0.95,
                        alpha=ACADEMIC_PALETTE["motion_pred_alpha"],
                        zorder=3.6,
                        motion_max_displacement=motion_max_displacement,
                        num_beads=8,
                    )
                if mode in {"boxes", "both"}:
                    _draw_motion_history_boxes(
                        ax,
                        base_state_pred,
                        motion_pred_np[a],
                        color=ACADEMIC_PALETTE["motion_pred"],
                        linewidth=bbox_linewidth * 0.8,
                        alpha_min=0.12,
                        alpha_max=0.55,
                        zorder=3.65,
                        max_boxes=max_boxes,
                    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # ---------------- Legend ----------------
    if show_legend_eff:
        legend_elements = [
            {"type": "patch", "color": ACADEMIC_PALETTE["offroad"], "label": "Offroad", "alpha": ACADEMIC_PALETTE["offroad_alpha"]},
            {"type": "patch", "color": ACADEMIC_PALETTE["road_surface"], "label": "Road", "edgecolor": "#d1d5db"},
            {"type": "line", "color": ACADEMIC_PALETTE["road_edge"], "linewidth": 2.5, "label": "Road Edge"},
            {"type": "patch", "color": ACADEMIC_PALETTE["ego_fill"], "label": "Ego", "alpha": ACADEMIC_PALETTE["ego_fill_alpha"]},
            {"type": "patch", "color": ACADEMIC_PALETTE["veh_fill"], "label": "Vehicle", "alpha": ACADEMIC_PALETTE["veh_fill_alpha"]},
            {"type": "square", "color": ACADEMIC_PALETTE["ped_fill"], "label": "Pedestrian", "markersize": 6},
            {"type": "patch", "color": ACADEMIC_PALETTE["cyc_fill"], "label": "Cyclist", "alpha": ACADEMIC_PALETTE["cyc_fill_alpha"]},
        ]
        if lane_types_idx is not None:
            if 1 in lane_types_idx:
                legend_elements.append(
                    {"type": "line", "color": ACADEMIC_PALETTE["lane_green"], "linewidth": 2.0, "label": "Green Light"}
                )
            if 2 in lane_types_idx:
                legend_elements.append(
                    {"type": "line", "color": ACADEMIC_PALETTE["lane_red"], "linewidth": 2.0, "label": "Red Light"}
                )
        if route is not None:
            legend_elements.append(
                {"type": "line", "color": ACADEMIC_PALETTE["route_line"], "linewidth": 2.0, "linestyle": (0, (6, 3)), "label": "Route"}
            )
        if overlay_gt and gt_agent_states is not None:
            legend_elements.append(
                {"type": "patch", "facecolor": "none", "edgecolor": "#64748b", "label": "GT Agents"}
            )
        if motion_pred_np is not None:
            legend_elements.append(
                {"type": "line", "color": ACADEMIC_PALETTE["motion_pred"], "linewidth": 1.5, "label": "Motion (Pred)"}
            )
        if motion_gt_np is not None:
            legend_elements.append(
                {"type": "line", "color": ACADEMIC_PALETTE["motion_gt"], "linewidth": 1.5, "label": "Motion (GT)"}
            )

        # scale legend font a bit when zoomed out
        leg_fs = int(max(4, round(6.0 * ui_scale)))
        _draw_legend(ax, legend_elements, location="lower left", fontsize=leg_fs)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if return_fig:
        return fig
    else:
        plt.margins(0.0)
        ax.margins(0.0)
        fig.savefig(
            os.path.join(save_dir, name),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close(fig)
        return None

def plot_lane_graph(
    road_points,
    lane_conn,
    edge_index_lane_to_lane,
    lane_conn_type,
    name,
    save_dir,
    return_fig=False,
):
    """Plot lane graph with connections."""
    fig, ax = plt.subplots()
    fig.patch.set_facecolor(ACADEMIC_PALETTE["bg"])
    ax.set_facecolor(ACADEMIC_PALETTE["bg"])

    for i in range(len(road_points)):
        lane = road_points[i, :, :2]
        ax.plot(lane[:, 0], lane[:, 1], color="#374151", linewidth=1.5)
        label_idx = len(lane) // 2
        ax.annotate(
            str(i),
            (lane[label_idx, 0], lane[label_idx, 1]),
            zorder=5,
            fontsize=5,
        )

    for j in range(lane_conn.shape[0]):
        if lane_conn[j, lane_conn_type] == 1:
            src_idx = edge_index_lane_to_lane[0, j]
            dest_idx = edge_index_lane_to_lane[1, j]
            lane_src = road_points[src_idx, :, :2]
            lane_dest = road_points[dest_idx, :, :2]
            src_pos = lane_src[10, :2]
            dest_pos = lane_dest[10, :2]

            if lane_conn.shape[1] == 6:
                edge_color = "#a855f7"
                if lane_conn[j, 2] == 1:
                    edge_color = "#ef4444"
                elif lane_conn[j, 3] == 1:
                    edge_color = "#22c55e"
                elif lane_conn[j, 4] == 1:
                    edge_color = "#3b82f6"
            else:
                edge_color = "#ef4444" if lane_conn[j, 1] == 1 else "#22c55e"

            ax.arrow(
                src_pos[0],
                src_pos[1],
                dest_pos[0] - src_pos[0],
                dest_pos[1] - src_pos[1],
                length_includes_head=True,
                head_width=1.0,
                head_length=1.0,
                zorder=10,
                color=edge_color,
            )

    ax.set_aspect("equal", adjustable="box")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if return_fig:
        return fig
    else:
        fig.savefig(os.path.join(save_dir, name), dpi=300)
        plt.close(fig)
        return None


def visualize_batch(
    num_samples,
    agent_samples,
    lane_samples,
    agent_types,
    lane_types,
    lane_conn_samples,
    data,
    save_dir,
    epoch,
    batch_idx,
    log_to_tb=False,
    visualize_lane_graph=False,
    agent_motion_gt=None,
    agent_motion_pred=None,
    motion_max_displacement=6.0,
    overlay_gt=False,
    gt_agent_states=None,
    gt_lane_states=None,
    gt_agent_types=None,
    gt_lane_types=None,
    motion_viz_mode="boxes",
):
    """Visualize a batch of scenes during training/evaluation."""
    if lane_conn_samples.shape[-1] == 4:
        lane_connection_types = LANE_CONNECTION_TYPES_NUPLAN
    else:
        lane_connection_types = LANE_CONNECTION_TYPES_WAYMO

    agent_samples = agent_samples.detach().cpu().numpy()
    lane_samples = lane_samples.detach().cpu().numpy()
    agent_types_np = agent_types.detach().cpu().numpy()
    lane_types_np = lane_types.detach().cpu().numpy() if lane_types is not None else None
    lane_conn_samples = lane_conn_samples.detach().cpu().numpy()

    lane_batch = data["lane"].batch
    lane_row = data["lane", "to", "lane"].edge_index[0]
    lane_conn_batch = lane_batch[lane_row]
    edge_index_l2l = data["lane", "to", "lane"].edge_index
    lane_conn_batch = lane_conn_batch.cpu().numpy()
    agent_batch = data["agent"].batch.cpu().numpy()
    lane_batch = data["lane"].batch.cpu().numpy()

    motion_gt_np = None
    motion_pred_np = None
    if agent_motion_gt is not None:
        motion_gt_np = agent_motion_gt.detach().cpu().numpy()
        motion_gt_np = unnormalize_motion_code(motion_gt_np, motion_max_displacement)
    if agent_motion_pred is not None:
        motion_pred_np = agent_motion_pred.detach().cpu().numpy()
        motion_pred_np = unnormalize_motion_code(motion_pred_np, motion_max_displacement)

    figures = {} if log_to_tb else None

    for i in range(num_samples):
        scene_i_agents = agent_samples[agent_batch == i]
        scene_i_lanes = lane_samples[lane_batch == i]
        scene_i_agent_types = agent_types_np[agent_batch == i]
        scene_i_lane_types = (
            lane_types_np[lane_batch == i] if lane_types_np is not None else None
        )

        if overlay_gt and gt_agent_states is not None and gt_lane_states is not None:
            scene_i_agents_gt = gt_agent_states[agent_batch == i]
            scene_i_lanes_gt = gt_lane_states[lane_batch == i]
            scene_i_agent_types_gt = (
                gt_agent_types[agent_batch == i]
                if gt_agent_types is not None
                else scene_i_agent_types
            )
            scene_i_lane_types_gt = (
                gt_lane_types[lane_batch == i]
                if gt_lane_types is not None
                else scene_i_lane_types
            )
        else:
            scene_i_agents_gt = None
            scene_i_lanes_gt = None
            scene_i_agent_types_gt = None
            scene_i_lane_types_gt = None

        scene_i_motion_gt = (
            motion_gt_np[agent_batch == i] if motion_gt_np is not None else None
        )
        scene_i_motion_pred = (
            motion_pred_np[agent_batch == i] if motion_pred_np is not None else None
        )

        tag = f"epoch_{epoch}_batch_{batch_idx}_sample_{i}"
        fig = plot_scene(
            scene_i_agents,
            scene_i_lanes,
            scene_i_agent_types,
            scene_i_lane_types,
            name=f"{tag}.png",
            save_dir=save_dir,
            return_fig=log_to_tb,
            agent_motion_gt=scene_i_motion_gt,
            agent_motion_pred=scene_i_motion_pred,
            motion_max_displacement=motion_max_displacement,
            show_legend=True,
            gt_agent_states=scene_i_agents_gt,
            gt_road_points=scene_i_lanes_gt,
            gt_agent_types=scene_i_agent_types_gt,
            gt_lane_types=scene_i_lane_types_gt,
            overlay_gt=overlay_gt,
            motion_viz_mode=motion_viz_mode,
        )
        if log_to_tb and fig is not None:
            figures[f"scene_plot/{tag}"] = fig

        if visualize_lane_graph:
            scene_i_lane_conns = lane_conn_samples[lane_conn_batch == i]
            shift = np.where(lane_batch == i)[0].min()
            edge_index_i_l2l = (
                edge_index_l2l[:, lane_conn_batch == i].cpu().numpy() - shift
            )

            if lane_conn_samples.shape[-1] == 4:
                edge_type_list = [
                    lane_connection_types["pred"],
                    lane_connection_types["succ"],
                ]
            else:
                edge_type_list = [
                    lane_connection_types["pred"],
                    lane_connection_types["succ"],
                    lane_connection_types["left"],
                    lane_connection_types["right"],
                ]

            for typ in edge_type_list:
                tag_g = f"{tag}_lanegraph_{typ}"
                fig_g = plot_lane_graph(
                    scene_i_lanes,
                    scene_i_lane_conns,
                    edge_index_i_l2l,
                    typ,
                    name=f"{tag_g}.png",
                    save_dir=save_dir,
                    return_fig=log_to_tb,
                )
                if log_to_tb and fig_g is not None:
                    figures[f"lane_graph/{tag_g}"] = fig_g

    return figures


def render_state(
    agent_states,
    agent_types,
    route,
    lanes,
    lanes_mask,
    t,
    name,
    movie_path="video_frames",
    lightweight=False,
    agent_ids=None,
):
    """Render a single simulation frame (called by Simulator)."""
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
        show_legend=True,
        show_safety_halo=True,
        safety_radius=8.0,
        agent_ids=agent_ids,
    )


def _resolve_ffmpeg_exe() -> str | None:
    """Resolve ffmpeg executable path."""
    if imageio_ffmpeg is not None:
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    return shutil.which("ffmpeg")


def generate_video(name, output_dir, delete_images=False, fps: int = 5):
    """
    Generate MP4 video from frame images using ffmpeg concat demuxer.

    == sync, corrected by elderman == @elder_man
    
    Issue 1: Low number of video frames
    - Reason: Concat demuxer did not specify a pattern for each frame, resulting in ffmpeg using default behaviour
    - Fix: specify duringation = 1/ fps per frame
    
    Question 0: No need for padding
    - Reason: The scene is fixed. All frames should be identical in size.
    - Repair: remove the padding detection logic and report the error directly if the dimensions are inconsistent
    
    Question 3: Remove try-except
    - Let the error come to light. It's easy to debug.
    
    =================================================

    Args:
        Name: Video Name (also a subdirectorate name to store frame images)
        Output_dir: Output directory
        delete_images: Whether to delete frame images
        fps: Output video frame rate

    Output:
        <output_dir>/<name>.mp4
    """
    image_folder = os.path.join(output_dir, str(name))
    if not os.path.isdir(image_folder):
        print(f"[generate_video] No folder {image_folder}")
        return

    # Retrieve all frame files (sort to ensure correct order)
    image_filenames = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])
    if len(image_filenames) == 0:
        print(f"[generate_video] No images found in {image_folder}")
        return

    image_paths = [os.path.join(image_folder, f) for f in image_filenames]
    num_frames = len(image_paths)
    print(f"[generate_video] Processing {num_frames} frames for video '{name}'")

    # Resolve ffmpeg path
    ffmpeg_exe = _resolve_ffmpeg_exe()
    if ffmpeg_exe is None:
        raise RuntimeError(
            "[generate_video] ffmpeg not found. Install ffmpeg or imageio-ffmpeg."
        )

    # Validate frame size consistency (scenario range fixed, no need for padding)
    # If the size doesn't match, it means there's a problem with the rendering.
    if Image is not None and num_frames > 1:
        first_size = None
        for i, p in enumerate(image_paths):
            with Image.open(p) as im:
                size = im.size
                if first_size is None:
                    first_size = size
                elif size != first_size:
                    raise RuntimeError(
                        f"[generate_video] Frame size mismatch at frame {i}: "
                        f"expected {first_size}, got {size}. "
                        f"Scene bounds should be fixed - check render_simulation_frame_2d."
                    )
        print(f"[generate_video] All frames have consistent size: {first_size}")

    # Calculate the duration of each frame
    frame_duration = 1.0 / float(fps)

    # Create Concat List File (Key Fix: Designation for each frame)
    # Concat demuxer requires list files to be in the same directory as images, using relative paths
    list_path = os.path.join(image_folder, "_frames.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for i, fname in enumerate(image_filenames):
            # File command uses a path relative to the list file (i.e. file name only)
            f.write(f"file {shlex.quote(fname)}\n")
            # Critical restoration: assignment of Duration for each frame
            f.write(f"duration {frame_duration:.6f}\n")
        # The last frame requires reassigning the file (concat demuxer special requirements)
        # Otherwise, the last frame might not last for the right time.
        if image_filenames:
            f.write(f"file {shlex.quote(image_filenames[-1])}\n")

    out_path = os.path.join(output_dir, f"{name}.mp4")

    # Ensure output size is even (libx264 requirement)
    vf_filter = "pad=ceil(iw/2)*2:ceil(ih/2)*2"

    cmd = [
        ffmpeg_exe,
        "-y",                    # Overwrite Output File
        "-loglevel", "warning",  # Show warning level log
        "-f", "concat",          # Use concat demuxer
        "-safe", "0",            # Allow unsafe file paths
        "-i", list_path,         # Enter List File
        "-vf", vf_filter,        # Video filter: ensure even size
        "-c:v", "libx264",       # H.264Encoding
        "-pix_fmt", "yuv420p",   # Pixels format (best compatibility)
        "-movflags", "+faststart",  # Metadata prefix, easy flow
        "-crf", "18",            # Quality factor (18 = visual)
        "-preset", "veryfast",   # Encoding Speed Preset
        out_path,
    ]

    print(f"[generate_video] Running ffmpeg command")
    proc_result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Clear List File
    os.remove(list_path)

    if proc_result.returncode != 0:
        stderr = proc_result.stderr.decode("utf-8", errors="ignore") if proc_result.stderr else ""
        raise RuntimeError(f"[generate_video] ffmpeg failed for {name}. stderr:\n{stderr}")

    print(f"[generate_video] Video saved: {out_path} ({num_frames} frames at {fps} fps)")

    # Clear Frame Images
    if delete_images:
        for p in image_paths:
            os.remove(p)
        # Try to delete empty directory
        Path(image_folder).rmdir()