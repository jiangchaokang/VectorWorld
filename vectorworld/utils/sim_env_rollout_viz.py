# utils/sim_env_rollout_viz.py
#
# Rollout video visualizer for simulation_environments generation.
# - Main view: fixed-size local window (always readable)
# - Side column: sliding window of last K inpainted steps (thumbnails)
# - Agent motion replay: use agent_motion (body-frame polyline) to animate agents
#
# This module is intentionally self-contained and only depends on existing viz primitives.

from __future__ import annotations

import os
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from utils.scene_viz import (
    CenterlineRoadRenderer2D,
    ACADEMIC_PALETTE,
    DEFAULT_LANE_WIDTH,
    _parse_agent_state,
    _draw_agent_shape,
    _draw_route_dots,
)

from utils.viz import generate_video


# =============================================================================
# Small utilities
# =============================================================================
def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _polyline_length(route_xy: Optional[np.ndarray]) -> float:
    if route_xy is None:
        return 0.0
    r = np.asarray(route_xy, dtype=np.float32)
    if r.ndim != 2 or r.shape[0] < 2:
        return 0.0
    d = np.diff(r[:, :2], axis=0)
    return float(np.linalg.norm(d, axis=1).sum())


def _tile_center(env: Dict[str, Any]) -> np.ndarray:
    """Use last tile occupancy center if available; fallback to route end; else origin."""
    occ = env.get("tile_occupancy", None)
    if isinstance(occ, list) and len(occ) > 0:
        last = np.asarray(occ[-1], dtype=np.float32)
        if last.ndim == 2 and last.shape[0] >= 3 and last.shape[1] >= 2 and np.isfinite(last).all():
            return last[:, :2].mean(axis=0)
    route = env.get("route", None)
    if route is not None:
        r = np.asarray(route, dtype=np.float32)
        if r.ndim == 2 and r.shape[0] > 0:
            return r[-1, :2].copy()
    return np.zeros((2,), dtype=np.float32)


def _axis_px_size(fig: plt.Figure, ax: plt.Axes) -> Tuple[int, int]:
    """Return (w_px, h_px) of an axes in the given figure."""
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    w = int(round(bbox.width))
    h = int(round(bbox.height))
    w = max(16, w)
    h = max(16, h)
    return w, h


def _fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    """Convert a Matplotlib figure to uint8 RGB array."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return buf.reshape(h, w, 3)


def _draw_hud(
    ax: plt.Axes,
    text: str,
    *,
    fontsize: int = 16,
    loc: str = "tl",
) -> None:
    """HUD text with translucent rounded box (no extra deps)."""
    if loc == "tl":
        x, y, ha, va = 0.015, 0.985, "left", "top"
    elif loc == "tr":
        x, y, ha, va = 0.985, 0.985, "right", "top"
    elif loc == "bl":
        x, y, ha, va = 0.015, 0.015, "left", "bottom"
    else:
        x, y, ha, va = 0.985, 0.015, "right", "bottom"

    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontfamily="DejaVu Sans Mono",
        fontweight="bold",
        color="#f9fafb",
        zorder=2000,
        bbox=dict(
            boxstyle="round,pad=0.35,rounding_size=0.15",
            facecolor="#111827",
            edgecolor="none",
            alpha=0.60,
        ),
    )


# =============================================================================
# Motion replay helpers (adapted from tools/denoise_viz.py)
# =============================================================================
def _validate_motion_vec_np(motion_vec: Optional[np.ndarray], atol: float = 1e-2) -> Optional[np.ndarray]:
    if motion_vec is None:
        return None
    v = np.asarray(motion_vec, dtype=np.float32).reshape(-1)
    if v.size < 2 or (v.size % 2) != 0:
        return None
    if not np.all(np.isfinite(v)):
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if np.allclose(v, 0.0, atol=atol):
        return None
    if float(np.max(np.abs(v))) > 200.0:
        return None
    return v


def _agent_heading_from_state(agent_state: np.ndarray) -> float:
    cos_t = float(agent_state[3]) if agent_state.shape[0] > 3 else 1.0
    sin_t = float(agent_state[4]) if agent_state.shape[0] > 4 else 0.0
    n = math.hypot(cos_t, sin_t)
    if n < 1e-6:
        cos_t, sin_t = 1.0, 0.0
    else:
        cos_t /= n
        sin_t /= n
    return math.atan2(sin_t, cos_t)


def _rotmat(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def _motion_body_points(motion_vec: np.ndarray) -> Optional[np.ndarray]:
    v = _validate_motion_vec_np(motion_vec)
    if v is None:
        return None
    k = v.size // 2
    pts = v.reshape(k, 2).astype(np.float32)
    # enforce endpoint at (0,0)
    pts = pts - pts[-1][None, :]
    pts[-1] = np.array([0.0, 0.0], dtype=np.float32)
    return pts


def _body_to_world(agent_state: np.ndarray, pts_body: np.ndarray) -> np.ndarray:
    x0 = float(agent_state[0])
    y0 = float(agent_state[1])
    theta = _agent_heading_from_state(agent_state)
    R = _rotmat(theta)
    pts_world = pts_body @ R.T + np.array([x0, y0], dtype=np.float32)[None, :]
    pts_world[-1] = np.array([x0, y0], dtype=np.float32)
    return pts_world.astype(np.float32)


def _polyline_prepare(pts: np.ndarray, eps: float = 1e-4) -> Tuple[np.ndarray, np.ndarray, float]:
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return pts, np.array([0.0], dtype=np.float32), 0.0
    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    if pts.shape[0] < 2:
        return pts, np.array([0.0], dtype=np.float32), 0.0
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], diffs > eps])
    pts = pts[keep]
    if pts.shape[0] < 2:
        return pts, np.array([0.0], dtype=np.float32), 0.0
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1).astype(np.float32)
    cum = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)
    total = float(cum[-1])
    return pts, cum, total


def _polyline_sample(pts: np.ndarray, cum: np.ndarray, total: float, u: float) -> np.ndarray:
    if pts.shape[0] == 0:
        return np.zeros((2,), dtype=np.float32)
    if pts.shape[0] == 1 or total <= 1e-6:
        return pts[-1].astype(np.float32)
    u = float(np.clip(u, 0.0, 1.0))
    target = u * total
    j = int(np.searchsorted(cum, target, side="right") - 1)
    j = int(np.clip(j, 0, pts.shape[0] - 2))
    t0 = float(cum[j])
    t1 = float(cum[j + 1])
    if t1 <= t0 + 1e-9:
        return pts[j].astype(np.float32)
    a = (target - t0) / (t1 - t0)
    p = (1.0 - a) * pts[j] + a * pts[j + 1]
    return p.astype(np.float32)


def _polyline_heading(pts: np.ndarray, cum: np.ndarray, total: float, u: float, du: float = 0.03) -> Optional[float]:
    if pts.shape[0] < 2 or total <= 1e-6:
        return None
    u0 = float(np.clip(u - du, 0.0, 1.0))
    u1 = float(np.clip(u + du, 0.0, 1.0))
    p0 = _polyline_sample(pts, cum, total, u0)
    p1 = _polyline_sample(pts, cum, total, u1)
    v = p1 - p0
    n = float(np.linalg.norm(v))
    if n <= 1e-5:
        return None
    return math.atan2(float(v[1]), float(v[0]))


def _world_to_body_points(pos_xy: np.ndarray, theta: float, pts_world: np.ndarray) -> np.ndarray:
    R = _rotmat(theta)
    body = (pts_world - pos_xy[None, :]) @ R  # row-vector convention => rotate by -theta
    body = body.astype(np.float32)
    body[-1] = np.array([0.0, 0.0], dtype=np.float32)
    return body


def _make_replay_u_list(K: int, interp: int, hold_first: int, hold_last: int, max_frames: Optional[int]) -> List[float]:
    base_frames = max(2, int((K - 1) * max(1, int(interp)) + 1))
    u_main = np.linspace(0.0, 1.0, num=base_frames, dtype=np.float32).tolist()
    u_list: List[float] = []
    u_list.extend([0.0] * int(max(0, hold_first)))
    u_list.extend(u_main)
    u_list.extend([1.0] * int(max(0, hold_last)))
    if max_frames is not None and int(max_frames) > 0 and len(u_list) > int(max_frames):
        u_list = u_list[: int(max_frames)]
    return u_list


# =============================================================================
# Scene drawing (for video)
# =============================================================================
def _to_type_idx(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim == 2:
        return np.argmax(a, axis=1).astype(np.int32)
    return a.astype(np.int32)


def _map_agent_type_to_scene_viz_idx(
    agent_type_idx: np.ndarray,
    *,
    dataset_name: str,
) -> np.ndarray:
    """
    scene_viz expects 5-class indices:
      0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: other

    Our models often use 3-class indices:
      waymo:  0 veh, 1 ped, 2 cyc
      nuplan: 0 veh, 1 ped, 2 static_object

    Map 3-class -> 5-class for correct shapes/colors.
    """
    idx = np.asarray(agent_type_idx, dtype=np.int32)
    if idx.size == 0:
        return idx

    max_idx = int(idx.max())
    if max_idx <= 2:
        if str(dataset_name).lower() == "nuplan":
            # 0 veh -> 1, 1 ped -> 2, 2 static -> 4(other)
            lut = np.array([1, 2, 4], dtype=np.int32)
        else:
            # waymo: 0 veh -> 1, 1 ped -> 2, 2 cyc -> 3
            lut = np.array([1, 2, 3], dtype=np.int32)
        idx = lut[np.clip(idx, 0, 2)]
    return idx


def _crop_lanes(
    road_points: np.ndarray,
    lane_types: Optional[np.ndarray],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    margin: float = 6.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    rp = np.asarray(road_points, dtype=np.float32)
    if rp.ndim != 3 or rp.shape[0] == 0:
        return rp, lane_types

    x0, x1 = float(xlim[0]) - margin, float(xlim[1]) + margin
    y0, y1 = float(ylim[0]) - margin, float(ylim[1]) + margin

    # Robust min/max (ignore NaN)
    mn = np.nanmin(rp[:, :, :2], axis=1)
    mx = np.nanmax(rp[:, :, :2], axis=1)

    keep = (mx[:, 0] >= x0) & (mn[:, 0] <= x1) & (mx[:, 1] >= y0) & (mn[:, 1] <= y1)
    rp2 = rp[keep]
    lt2 = None
    if lane_types is not None:
        lt = np.asarray(lane_types)
        lt2 = lt[keep]
    return rp2, lt2


def _crop_agents_keep_ego(
    agent_states: np.ndarray,
    agent_types: Optional[np.ndarray],
    agent_motion: Optional[np.ndarray],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    margin: float = 10.0,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    a = np.asarray(agent_states, dtype=np.float32)
    if a.ndim != 2 or a.shape[0] == 0:
        return a, agent_types, agent_motion

    x0, x1 = float(xlim[0]) - margin, float(xlim[1]) + margin
    y0, y1 = float(ylim[0]) - margin, float(ylim[1]) + margin

    xy = a[:, :2]
    keep = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)

    # Always keep ego at index 0 to preserve semantics
    keep[0] = True

    a2 = a[keep]
    t2 = None
    m2 = None
    if agent_types is not None:
        t2 = np.asarray(agent_types)[keep]
    if agent_motion is not None:
        m2 = np.asarray(agent_motion)[keep]
    return a2, t2, m2


def _draw_motion_curve(
    ax: plt.Axes,
    agent_state: np.ndarray,
    motion_vec: np.ndarray,
    *,
    color: str,
    linewidth: float,
    alpha: float,
    zorder: float,
    num_beads: int = 8,
    bead_size: float = 18.0,  # points^2
) -> None:
    mv = _validate_motion_vec_np(motion_vec)
    if mv is None:
        return

    k = mv.size // 2
    pts_body = mv.reshape(k, 2).astype(np.float32)

    x0 = _safe_float(agent_state[0], 0.0)
    y0 = _safe_float(agent_state[1], 0.0)
    theta = _agent_heading_from_state(agent_state)
    R = _rotmat(theta)
    pts_world = pts_body @ R.T + np.array([x0, y0], dtype=np.float32)[None, :]

    if not np.all(np.isfinite(pts_world)) or pts_world.shape[0] < 2:
        return

    # Simple polyline with per-seg alpha (fixed)
    segs = [[pts_world[i], pts_world[i + 1]] for i in range(len(pts_world) - 1)]
    rgba = list(mcolors.to_rgba(color))
    rgba[3] = float(alpha) * 0.55
    lc = LineCollection(
        segs,
        colors=[rgba] * len(segs),
        linewidths=float(linewidth),
        capstyle="round",
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_collection(lc)

    # beads (uniform)
    nb = int(max(2, min(num_beads, len(pts_world))))
    idxs = np.linspace(0, len(pts_world) - 1, num=nb, dtype=int)
    beads = pts_world[idxs]
    bead_rgba = list(mcolors.to_rgba(color))
    bead_rgba[3] = float(alpha)
    ax.scatter(
        beads[:, 0],
        beads[:, 1],
        s=float(bead_size),
        c=[bead_rgba],
        edgecolors="none",
        zorder=zorder + 0.1,
    )


def _draw_tile_bounds(
    ax: plt.Axes,
    tile_occupancy: Optional[List[np.ndarray]],
    *,
    highlight_last: bool = True,
    color: str = "#6b7280",
    alpha: float = 0.25,
    lw: float = 1.2,
    zorder: float = 6.0,
) -> None:
    if tile_occupancy is None or len(tile_occupancy) == 0:
        return

    for i, corners in enumerate(tile_occupancy):
        c = np.asarray(corners, dtype=np.float32)
        if c.ndim != 2 or c.shape[0] < 3:
            continue
        col = color
        a = alpha
        w = lw
        if highlight_last and i == (len(tile_occupancy) - 1):
            col = ACADEMIC_PALETTE["route_line"]  # amber
            a = 0.55
            w = lw * 1.8

        xs = np.concatenate([c[:, 0], c[:1, 0]])
        ys = np.concatenate([c[:, 1], c[:1, 1]])
        ax.plot(xs, ys, color=col, alpha=a, linewidth=w, zorder=zorder)


def draw_scene_on_axis(
    ax: plt.Axes,
    *,
    env: Dict[str, Any],
    dataset_name: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    agent_states_override: Optional[np.ndarray] = None,
    agent_motion_override: Optional[np.ndarray] = None,
    motion_viz_mode: str = "curve",
    show_agent_ids: str = "ego_only",  # "none" | "ego_only" | "all"
    show_route: bool = True,
    show_tiles: bool = True,
    road_grid_res: float = 0.35,
) -> None:
    ax.set_facecolor(ACADEMIC_PALETTE["bg"])

    agent_states = env.get("agent_states", None)
    road_points = env.get("road_points", None)
    agent_types = env.get("agent_types", None)
    lane_types = env.get("lane_types", None)
    route = env.get("route", None)
    tile_occupancy = env.get("tile_occupancy", None)
    agent_motion = env.get("agent_motion", None)

    if agent_states_override is not None:
        agent_states = agent_states_override
    if agent_motion_override is not None:
        agent_motion = agent_motion_override

    agent_states = np.asarray(agent_states, dtype=np.float32) if agent_states is not None else np.zeros((0, 7), dtype=np.float32)
    road_points = np.asarray(road_points, dtype=np.float32) if road_points is not None else np.zeros((0, 20, 2), dtype=np.float32)

    # crop (performance)
    road_points_c, lane_types_c = _crop_lanes(road_points, lane_types, xlim, ylim, margin=6.0)
    agent_states_c, agent_types_c, agent_motion_c = _crop_agents_keep_ego(
        agent_states, agent_types, agent_motion, xlim, ylim, margin=12.0
    )

    # types
    agent_type_idx = _to_type_idx(agent_types_c)
    if agent_type_idx is None:
        agent_type_idx = np.zeros((agent_states_c.shape[0],), dtype=np.int32)
    agent_type_idx = _map_agent_type_to_scene_viz_idx(agent_type_idx, dataset_name=dataset_name)

    lane_type_idx = _to_type_idx(lane_types_c)

    # Road
    road_renderer = CenterlineRoadRenderer2D(lane_width=DEFAULT_LANE_WIDTH, grid_resolution=float(road_grid_res))
    if road_points_c.shape[0] > 0:
        road_renderer.draw(
            ax,
            road_points_c,
            lane_types=lane_type_idx,
            facecolor=ACADEMIC_PALETTE["road_surface"],
            offroad_color=ACADEMIC_PALETTE["offroad"],
            edgecolor=ACADEMIC_PALETTE["road_edge"],
            centerline_color=ACADEMIC_PALETTE["centerline"],
            centerline_style=(0, (5, 4)),
            edge_width=3.2,
            centerline_width=1.5,
            alpha_fill=1.0,
            z_base=1.0,
            xlim=xlim,
            ylim=ylim,
        )

    # Tiles
    if show_tiles:
        _draw_tile_bounds(ax, tile_occupancy, highlight_last=True)

    # Route
    if show_route and route is not None:
        _draw_route_dots(
            ax,
            route,
            dot_spacing=3.0,
            dot_size=14,      # local fixed view => ok
            linewidth=2.2,
            zorder=5.5,
        )

    # Agents + motion
    motion_mode = str(motion_viz_mode or "curve").lower()
    if motion_mode not in {"curve", "boxes", "both"}:
        motion_mode = "curve"

    na = int(agent_states_c.shape[0])
    is_ego_mask = np.zeros((na,), dtype=bool)
    if na > 0:
        is_ego_mask[0] = True

    # Draw order: NPC then ego (ego on top)
    order = list(range(1, na)) + ([0] if na > 0 else [])

    # Motion curves first (below agents)
    if agent_motion_c is not None and na > 0:
        mv = np.asarray(agent_motion_c, dtype=np.float32)
        if mv.ndim == 2 and mv.shape[0] == na:
            for a in order:
                col = ACADEMIC_PALETTE["motion_pred"]
                a_alpha = 0.65 if (a == 0) else 0.45
                _draw_motion_curve(
                    ax,
                    agent_states_c[a],
                    mv[a],
                    color=col,
                    linewidth=2.0 if (a == 0) else 1.6,
                    alpha=a_alpha,
                    zorder=7.0,
                    num_beads=7,
                    bead_size=20.0 if (a == 0) else 14.0,
                )

    # Agent shapes + optional ids
    for rank, a in enumerate(order):
        t_idx = int(agent_type_idx[a]) if a < len(agent_type_idx) else 1
        is_ego = bool(a == 0)

        parsed = _parse_agent_state(agent_states_c[a], agent_type_idx=t_idx)
        if parsed is None:
            continue

        z_agent = 20.0 + rank * 0.01
        lw = 1.5 if not is_ego else 1.8
        _draw_agent_shape(ax, parsed, t_idx, is_ego, lw, z_agent)

        if show_agent_ids in {"all", "ego_only"}:
            if (show_agent_ids == "ego_only") and (not is_ego):
                continue
            label = "Ego" if is_ego else str(a)
            txt_col = ACADEMIC_PALETTE["id_text_ego"] if is_ego else ACADEMIC_PALETTE["id_text_npc"]
            ax.text(
                float(parsed["x"]),
                float(parsed["y"]),
                label,
                ha="center",
                va="center",
                fontsize=10 if is_ego else 9,
                fontweight="bold",
                color=txt_col,
                zorder=z_agent + 50.0,
                bbox=dict(
                    boxstyle="round,pad=0.12,rounding_size=0.15",
                    facecolor="#111827" if is_ego else "#ffffff",
                    edgecolor="none",
                    alpha=0.35 if is_ego else 0.55,
                ),
            )

    ax.set_xlim(float(xlim[0]), float(xlim[1]))
    ax.set_ylim(float(ylim[0]), float(ylim[1]))
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


# =============================================================================
# Rollout video renderer
# =============================================================================
def render_sim_env_rollout_video(
    *,
    snapshots: List[Dict[str, Any]],
    output_dir: str,
    dataset_name: str,
    run_name: str = "",
    ldm_type: str = "",
    fps: int = 12,
    filmstrip_k: int = 5,
    view_size_m: float = 80.0,
    pan_frames: int = 6,
    replay_interp: int = 3,
    replay_hold_first: int = 2,
    replay_hold_last: int = 4,
    max_replay_frames_per_step: Optional[int] = 28,
    motion_viz_mode: str = "curve",
    show_agent_ids: str = "ego_only",
    road_grid_res: float = 0.35,
    dpi: int = 120,
    delete_frames: bool = False,
) -> str:
    """
    Create a demo-quality MP4 that visualizes the inpainting rollout.

    Output:
      - frames: <output_dir>/rollout/frame_XXXX.png
      - mp4   : <output_dir>/rollout.mp4
      - meta  : <output_dir>/rollout_meta.json
    """
    os.makedirs(output_dir, exist_ok=True)
    frame_root = os.path.join(output_dir, "rollout")
    os.makedirs(frame_root, exist_ok=True)

    if snapshots is None or len(snapshots) == 0:
        raise ValueError("snapshots is empty")

    # Layout: K rows, 2 cols (left main spans all rows)
    K = int(max(1, filmstrip_k))
    fig = plt.figure(figsize=(16, 9), dpi=int(dpi))
    fig.patch.set_facecolor(ACADEMIC_PALETTE["bg"])

    gs = fig.add_gridspec(
        nrows=K,
        ncols=2,
        width_ratios=[3.2, 1.0],
        wspace=0.02,
        hspace=0.02,
    )
    ax_main = fig.add_subplot(gs[:, 0])
    thumb_axes = [fig.add_subplot(gs[i, 1]) for i in range(K)]
    for ax in [ax_main] + thumb_axes:
        ax.set_facecolor(ACADEMIC_PALETTE["bg"])
        ax.axis("off")

    # Determine thumbnail pixel size for caching
    thumb_w_px, thumb_h_px = _axis_px_size(fig, thumb_axes[0])

    # Precompute per-step centers + route lengths
    centers = [_tile_center(env) for env in snapshots]
    route_lens = [_polyline_length(env.get("route", None)) for env in snapshots]

    # Pre-render thumbnails (static)
    thumbs: List[np.ndarray] = []
    for step_idx, env in enumerate(snapshots):
        cx, cy = float(centers[step_idx][0]), float(centers[step_idx][1])
        half = float(view_size_m) * 0.5
        xlim = (cx - half, cx + half)
        ylim = (cy - half, cy + half)

        f_th = plt.figure(figsize=(thumb_w_px / dpi, thumb_h_px / dpi), dpi=int(dpi))
        f_th.patch.set_facecolor(ACADEMIC_PALETTE["bg"])
        ax_th = f_th.add_axes([0.0, 0.0, 1.0, 1.0])
        ax_th.axis("off")
        draw_scene_on_axis(
            ax_th,
            env=env,
            dataset_name=dataset_name,
            xlim=xlim,
            ylim=ylim,
            motion_viz_mode=motion_viz_mode,
            show_agent_ids="none",
            show_route=True,
            show_tiles=True,
            road_grid_res=road_grid_res,
        )
        _draw_hud(ax_th, f"step {step_idx:02d}", fontsize=12, loc="tl")
        img = _fig_to_rgb(f_th)
        plt.close(f_th)
        thumbs.append(img)

    # Helper: update side filmstrip once per step
    def _update_filmstrip(step_idx: int) -> None:
        start = max(0, step_idx - K + 1)
        idxs = list(range(start, step_idx + 1))
        # pad at top (oldest)
        pad = K - len(idxs)
        show = [None] * pad + idxs

        for slot, ax in enumerate(thumb_axes):
            ax.clear()
            ax.axis("off")
            ax.set_facecolor(ACADEMIC_PALETTE["bg"])

            si = show[slot]
            if si is None:
                ax.text(
                    0.5,
                    0.5,
                    " ",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#9ca3af",
                )
                continue

            ax.imshow(thumbs[si])
            # highlight current step
            edge = ACADEMIC_PALETTE["route_line"] if (si == step_idx) else "#cbd5e1"
            lw = 3.0 if (si == step_idx) else 1.5
            rect = Rectangle(
                (0.0, 0.0),
                1.0,
                1.0,
                transform=ax.transAxes,
                fill=False,
                linewidth=lw,
                edgecolor=edge,
                zorder=10.0,
            )
            ax.add_patch(rect)

            ax.text(
                0.02,
                0.98,
                f"{si:02d}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=12,
                fontweight="bold",
                color="#111827",
                zorder=11.0,
                bbox=dict(
                    boxstyle="round,pad=0.22,rounding_size=0.12",
                    facecolor="#ffffff",
                    edgecolor="none",
                    alpha=0.78,
                ),
            )

    # Precompute replay traj info per step (for speed)
    traj_infos: List[List[Optional[Tuple[np.ndarray, np.ndarray, float]]]] = []
    K_motion_per_step: List[int] = []

    for env in snapshots:
        agent_states = np.asarray(env.get("agent_states", np.zeros((0, 7))), dtype=np.float32)
        agent_motion = env.get("agent_motion", None)
        if agent_motion is None:
            traj_infos.append([None] * int(agent_states.shape[0]))
            K_motion_per_step.append(0)
            continue

        mv = np.asarray(agent_motion, dtype=np.float32)
        if mv.ndim != 2 or mv.shape[0] != agent_states.shape[0] or (mv.shape[1] % 2) != 0:
            traj_infos.append([None] * int(agent_states.shape[0]))
            K_motion_per_step.append(0)
            continue

        Kp = int(mv.shape[1] // 2)
        K_motion_per_step.append(Kp)

        info_step: List[Optional[Tuple[np.ndarray, np.ndarray, float]]] = []
        for a in range(agent_states.shape[0]):
            pts_body = _motion_body_points(mv[a])
            if pts_body is None or pts_body.shape[0] < 2:
                info_step.append(None)
                continue
            pts_world = _body_to_world(agent_states[a], pts_body)
            pts_world, cum, total = _polyline_prepare(pts_world)
            if pts_world.shape[0] < 2 or total <= 1e-4:
                info_step.append(None)
                continue
            info_step.append((pts_world, cum, total))
        traj_infos.append(info_step)

    # Render frames
    meta = {
        "dataset_name": dataset_name,
        "run_name": run_name,
        "ldm_type": ldm_type,
        "fps": int(fps),
        "filmstrip_k": int(K),
        "view_size_m": float(view_size_m),
        "pan_frames": int(pan_frames),
        "replay_interp": int(replay_interp),
        "replay_hold_first": int(replay_hold_first),
        "replay_hold_last": int(replay_hold_last),
        "max_replay_frames_per_step": max_replay_frames_per_step,
        "motion_viz_mode": motion_viz_mode,
        "show_agent_ids": show_agent_ids,
        "road_grid_res": float(road_grid_res),
        "dpi": int(dpi),
        "num_steps": int(len(snapshots)),
    }
    with open(os.path.join(output_dir, "rollout_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    frame_idx = 0

    for step_idx, env in enumerate(snapshots):
        _update_filmstrip(step_idx)

        # Pan from previous center to current center (camera motion)
        c_prev = centers[step_idx - 1] if step_idx > 0 else centers[step_idx]
        c_cur = centers[step_idx]
        n_pan = int(pan_frames) if step_idx > 0 else 0
        for p in range(n_pan):
            t = float(p + 1) / float(max(1, n_pan))
            c = (1.0 - t) * c_prev + t * c_cur
            cx, cy = float(c[0]), float(c[1])

            half = float(view_size_m) * 0.5
            xlim = (cx - half, cx + half)
            ylim = (cy - half, cy + half)

            ax_main.clear()
            ax_main.axis("off")
            draw_scene_on_axis(
                ax_main,
                env=env,
                dataset_name=dataset_name,
                xlim=xlim,
                ylim=ylim,
                motion_viz_mode=motion_viz_mode,
                show_agent_ids=show_agent_ids,
                show_route=True,
                show_tiles=True,
                road_grid_res=road_grid_res,
            )
            hud = (
                f"{run_name}  |  {dataset_name}  |  {ldm_type}\n"
                f"step {step_idx+1:02d}/{len(snapshots):02d}   pan {p+1:02d}/{n_pan:02d}\n"
                f"route_len={route_lens[step_idx]:.1f}m   tiles={len(env.get('tile_occupancy', []))}"
            )
            _draw_hud(ax_main, hud, fontsize=16, loc="tl")

            out_path = os.path.join(frame_root, f"frame_{frame_idx:05d}.png")
            fig.savefig(
                out_path,
                dpi=int(dpi),
                bbox_inches=None,
                pad_inches=0.0,
                facecolor=fig.get_facecolor(),
                edgecolor="none",
            )
            frame_idx += 1

        # Replay within this step (agents move along their history)
        agent_states_final = np.asarray(env.get("agent_states", np.zeros((0, 7))), dtype=np.float32)
        agent_motion_final = env.get("agent_motion", None)
        Kp = int(K_motion_per_step[step_idx])

        if agent_motion_final is None or Kp <= 1:
            u_list = [1.0] * max(1, int(replay_hold_last) + 1)
        else:
            u_list = _make_replay_u_list(
                K=Kp,
                interp=int(replay_interp),
                hold_first=int(replay_hold_first),
                hold_last=int(replay_hold_last),
                max_frames=max_replay_frames_per_step,
            )

        info_step = traj_infos[step_idx]
        cx, cy = float(centers[step_idx][0]), float(centers[step_idx][1])
        half = float(view_size_m) * 0.5
        xlim = (cx - half, cx + half)
        ylim = (cy - half, cy + half)

        mv_final = None
        if agent_motion_final is not None:
            mv_final = np.asarray(agent_motion_final, dtype=np.float32)
            if mv_final.ndim != 2 or mv_final.shape[0] != agent_states_final.shape[0]:
                mv_final = None

        for r_idx, u in enumerate(u_list):
            agents_t = agent_states_final.copy()
            motion_t = None
            if mv_final is not None:
                motion_t = np.zeros_like(mv_final, dtype=np.float32)

            # animate
            for a in range(agents_t.shape[0]):
                info = info_step[a] if a < len(info_step) else None
                if info is None:
                    continue

                pts_world, cum, total = info
                pos = _polyline_sample(pts_world, cum, total, float(u))

                theta_final = _agent_heading_from_state(agent_states_final[a])
                theta = theta_final
                theta_est = _polyline_heading(pts_world, cum, total, float(u), du=0.03)
                if theta_est is not None:
                    theta = float(theta_est)

                agents_t[a, 0] = float(pos[0])
                agents_t[a, 1] = float(pos[1])
                agents_t[a, 3] = float(math.cos(theta))
                agents_t[a, 4] = float(math.sin(theta))

                if motion_t is not None:
                    # trail points in world sampled over [0,u]
                    if float(u) <= 1e-6:
                        trail_world = np.repeat(pos[None, :], repeats=Kp, axis=0).astype(np.float32)
                    else:
                        us = np.linspace(0.0, float(u), num=Kp, dtype=np.float32)
                        trail_world = np.stack(
                            [_polyline_sample(pts_world, cum, total, float(ui)) for ui in us],
                            axis=0,
                        ).astype(np.float32)
                        trail_world[-1] = pos

                    body_trail = _world_to_body_points(pos_xy=pos, theta=theta, pts_world=trail_world)
                    motion_t[a] = body_trail.reshape(-1)

            ax_main.clear()
            ax_main.axis("off")
            draw_scene_on_axis(
                ax_main,
                env=env,
                dataset_name=dataset_name,
                xlim=xlim,
                ylim=ylim,
                agent_states_override=agents_t,
                agent_motion_override=motion_t if motion_t is not None else None,
                motion_viz_mode=motion_viz_mode,
                show_agent_ids=show_agent_ids,
                show_route=True,
                show_tiles=True,
                road_grid_res=road_grid_res,
            )
            hud = (
                f"{run_name}  |  {dataset_name}  |  {ldm_type}\n"
                f"step {step_idx+1:02d}/{len(snapshots):02d}   replay {r_idx+1:02d}/{len(u_list):02d}   u={float(u):.2f}\n"
                f"route_len={route_lens[step_idx]:.1f}m   tiles={len(env.get('tile_occupancy', []))}"
            )
            _draw_hud(ax_main, hud, fontsize=16, loc="tl")

            out_path = os.path.join(frame_root, f"frame_{frame_idx:05d}.png")
            fig.savefig(
                out_path,
                dpi=int(dpi),
                bbox_inches=None,
                pad_inches=0.0,
                facecolor=fig.get_facecolor(),
                edgecolor="none",
            )
            frame_idx += 1

    plt.close(fig)

    # Build mp4
    generate_video(name="rollout", output_dir=output_dir, delete_images=bool(delete_frames), fps=int(fps))
    return os.path.join(output_dir, "rollout.mp4")