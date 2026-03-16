import numpy as np
import torch
import networkx as nx
from vectorworld.utils.pyg_helpers import get_edge_index_complete_graph
from configs.config import NUPLAN_VEHICLE, UNIFIED_FORMAT_INDICES, NON_PARTITIONED
from scipy.spatial import distance
from vectorworld.utils.lane_graph_helpers import find_lane_groups, find_lane_group_id, resample_polyline, resample_lanes
from vectorworld.utils.sledge_helpers import calculate_progress, interpolate_path, coords_in_frame, find_consecutive_true_indices
from tqdm import tqdm
from torch_geometric.data import Batch
from vectorworld.utils.data_container import ScenarioDreamerData
from vectorworld.utils.pyg_helpers import get_edge_index_bipartite
from vectorworld.utils.data_helpers import normalize_scene
from typing import Any, Dict, List, Tuple, Optional


# ==============================================================================
# Motion Reconstruction Metrics (AE)
# ==============================================================================

def compute_ae_motion_reconstruction_metrics(
    motion_gt: np.ndarray,
    motion_pred: np.ndarray,
    motion_max_displacement: float = 12.0,
    static_eps_norm: float = 0.03,
) -> Dict[str, float]:
    """Compute AE motion code reconstruction metrics."""
    from vectorworld.utils.data_helpers import unnormalize_motion_code as _unnorm_motion

    motion_gt = np.asarray(motion_gt, dtype=np.float32)
    motion_pred = np.asarray(motion_pred, dtype=np.float32)
    assert motion_gt.shape == motion_pred.shape

    N, D = motion_gt.shape
    K = D // 2

    diff_norm = motion_pred - motion_gt
    motion_coord_l1 = float(np.mean(np.abs(diff_norm)))
    motion_coord_l2 = float(np.sqrt(np.mean(diff_norm ** 2)))

    motion_gt_phys = np.asarray(_unnorm_motion(motion_gt, motion_max_displacement), dtype=np.float32)
    motion_pred_phys = np.asarray(_unnorm_motion(motion_pred, motion_max_displacement), dtype=np.float32)

    gt_pts = motion_gt_phys.reshape(N, K, 2)
    pred_pts = motion_pred_phys.reshape(N, K, 2)

    gt_seg_len = np.linalg.norm(gt_pts[:, 1:] - gt_pts[:, :-1], axis=-1)
    pred_seg_len = np.linalg.norm(pred_pts[:, 1:] - pred_pts[:, :-1], axis=-1)

    path_len_gt = np.sum(gt_seg_len, axis=-1)
    path_len_pred = np.sum(pred_seg_len, axis=-1)

    disp_gt = np.linalg.norm(gt_pts, axis=-1).max(axis=-1)
    disp_pred = np.linalg.norm(pred_pts, axis=-1).max(axis=-1)

    motion_path_l1 = float(np.mean(np.abs(path_len_pred - path_len_gt)))
    motion_disp_l1 = float(np.mean(np.abs(disp_pred - disp_gt)))

    static_eps_m = float(static_eps_norm) * float(motion_max_displacement)
    nonzero_gt_mask = path_len_gt > static_eps_m
    nonzero_pred_mask = path_len_pred > static_eps_m

    motion_nonzero_frac_gt = float(np.mean(nonzero_gt_mask.astype(np.float32)))
    motion_nonzero_frac_pred = float(np.mean(nonzero_pred_mask.astype(np.float32)))

    dyn_mask = nonzero_gt_mask
    if np.any(dyn_mask):
        denom = np.maximum(path_len_gt[dyn_mask], static_eps_m)
        rel_err = np.abs(path_len_pred[dyn_mask] - path_len_gt[dyn_mask]) / denom
        motion_path_rel_l1 = float(np.mean(rel_err))
    else:
        motion_path_rel_l1 = np.nan

    static_mask = ~nonzero_gt_mask
    motion_static_jitter = float(np.mean(path_len_pred[static_mask])) if np.any(static_mask) else 0.0

    return {
        "motion_coord_l1": motion_coord_l1,
        "motion_coord_l2": motion_coord_l2,
        "motion_path_l1": motion_path_l1,
        "motion_path_rel_l1": motion_path_rel_l1,
        "motion_disp_l1": motion_disp_l1,
        "motion_nonzero_frac_gt": motion_nonzero_frac_gt,
        "motion_nonzero_frac_pred": motion_nonzero_frac_pred,
        "motion_static_jitter": motion_static_jitter,
    }


# ==============================================================================
# Frechet Distance (general purpose)
# ==============================================================================

def compute_frechet_distance(X1: np.ndarray, X2: np.ndarray, apply_sqrt: bool = True, eps: float = 1e-6) -> float:
    """Compute Frechet distance between two sets of features."""
    from scipy.linalg import sqrtm

    X1 = np.asarray(X1, dtype=np.float64)
    X2 = np.asarray(X2, dtype=np.float64)
    if X1.ndim == 1:
        X1 = X1.reshape(-1, 1)
    if X2.ndim == 1:
        X2 = X2.reshape(-1, 1)

    mu1, mu2 = X1.mean(axis=0), X2.mean(axis=0)

    sigma1 = np.cov(X1, rowvar=False) if X1.shape[0] >= 2 else np.eye(X1.shape[1], dtype=np.float64)
    sigma2 = np.cov(X2, rowvar=False) if X2.shape[0] >= 2 else np.eye(X2.shape[1], dtype=np.float64)

    if sigma1.ndim == 0:
        sigma1 = np.array([[float(sigma1)]], dtype=np.float64)
    if sigma2.ndim == 0:
        sigma2 = np.array([[float(sigma2)]], dtype=np.float64)

    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps

    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fd = float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    fd = max(fd, 0.0)
    return float(np.sqrt(fd)) if apply_sqrt else fd


# ==============================================================================
# JSD Helper
# ==============================================================================

def jsd(sim: np.ndarray, gt: np.ndarray, clip_min: float, clip_max: float, bin_size: float) -> float:
    """Jensen-Shannon divergence between generated and real distributions."""
    gt = np.clip(gt, clip_min, clip_max)
    sim = np.clip(sim, clip_min, clip_max)
    bin_edges = np.arange(clip_min, clip_max + bin_size, bin_size)
    P = np.histogram(sim, bins=bin_edges)[0] / max(len(sim), 1)
    Q = np.histogram(gt, bins=bin_edges)[0] / max(len(gt), 1)
    return distance.jensenshannon(P, Q) ** 2


# ==============================================================================
# Lane Graph Helpers
# ==============================================================================

def compute_vehicle_circles(xy_position, heading, length, width):
    """Compute circle centroids and radii for vehicle collision detection."""
    num_circles = 5
    radius = width / 2
    relative_x_positions = np.linspace(-length / 2 + radius, length / 2 - radius, num_circles)
    dx = np.cos(heading) * relative_x_positions
    dy = np.sin(heading) * relative_x_positions
    centroids = np.column_stack((xy_position[0] + dx, xy_position[1] + dy))
    return centroids, np.array([radius]).repeat(num_circles)


def compute_collision_rate(samples: List[Dict]) -> float:
    """Compute collision rate for vehicles in samples."""
    num_vehicles_all = 0
    num_vehicles_in_collision_all = 0

    for data in tqdm(samples, desc="Computing collision rate"):
        vehicles = data["vehicles"]
        if len(vehicles) <= 1:
            num_vehicles_all += len(vehicles)
            continue

        centroids_all = []
        for vehicle in vehicles:
            heading = np.arctan2(
                vehicle[UNIFIED_FORMAT_INDICES["sin_heading"]],
                vehicle[UNIFIED_FORMAT_INDICES["cos_heading"]],
            )
            centroids, _ = compute_vehicle_circles(
                vehicle[: UNIFIED_FORMAT_INDICES["pos_y"] + 1],
                heading,
                vehicle[UNIFIED_FORMAT_INDICES["length"]],
                vehicle[UNIFIED_FORMAT_INDICES["width"]],
            )
            centroids_all.append(centroids)
        centroids_all = np.array(centroids_all)

        for j in range(len(vehicles)):
            is_in_collision = False
            for k in range(len(vehicles)):
                if j == k:
                    continue
                thresh = (vehicles[j, 6] + vehicles[k, 6]) / np.sqrt(3.8)
                dist = np.linalg.norm(centroids_all[j, :, None] - centroids_all[k, None, :], axis=-1)
                if (dist < thresh).any():
                    is_in_collision = True
                    break
            if is_in_collision:
                num_vehicles_in_collision_all += 1
        num_vehicles_all += len(vehicles)

    return num_vehicles_in_collision_all / max(num_vehicles_all, 1)


def get_compact_lane_graph(G, lanes, num_points_per_lane: int = 20):
    """Merge connected lanes and resample to fixed points."""
    lanes_dict = {l: lane for l, lane in enumerate(lanes)}
    pre_pairs = {l: [] for l in range(len(lanes))}
    suc_pairs = {l: [] for l in range(len(lanes))}

    for edge in G.edges():
        pre_pairs[edge[1]].append(edge[0])
        suc_pairs[edge[0]].append(edge[1])

    lane_groups = find_lane_groups(pre_pairs, suc_pairs)

    compact_lanes = {}
    compact_suc_pairs = {}
    for lane_group_id in lane_groups:
        compact_lane = []
        compact_suc_pair = []
        for i, lane_id in enumerate(lane_groups[lane_group_id]):
            if i == 0:
                compact_lane.append(lanes_dict[lane_id])
            else:
                compact_lane.append(lanes_dict[lane_id][1:])
            if i == len(lane_groups[lane_group_id]) - 1:
                for suc_lane_id in suc_pairs[lane_id]:
                    compact_suc_pair.append(find_lane_group_id(suc_lane_id, lane_groups))
        compact_lane = np.concatenate(compact_lane, axis=0)
        compact_lanes[lane_group_id] = compact_lane
        compact_suc_pairs[lane_group_id] = compact_suc_pair

    idx_to_new_idx = {lane_id: new_idx for new_idx, lane_id in enumerate(compact_lanes.keys())}

    compact_suc_pairs_reindexed = {}
    compact_lanes_all = []
    for lane_id in compact_lanes.keys():
        compact_suc_pairs_reindexed[idx_to_new_idx[lane_id]] = [idx_to_new_idx[idx] for idx in compact_suc_pairs[lane_id]]
        lane_pts = compact_lanes[lane_id]
        if len(lane_pts) != num_points_per_lane:
            lane_pts = resample_polyline(lane_pts, num_points=num_points_per_lane)
        compact_lanes_all.append(lane_pts[None, :, :])

    compact_lanes_all = np.concatenate(compact_lanes_all, axis=0)

    num_lanes = len(compact_lanes_all)
    A = np.zeros((num_lanes, num_lanes))
    for lid, suc_list in compact_suc_pairs_reindexed.items():
        for suc_lid in suc_list:
            A[lid, suc_lid] = 1
    compact_G = nx.DiGraph(incoming_graph_data=A)

    return compact_G, compact_lanes_all


def _get_sledge_lane_graph_nuplan(data: Dict[str, Any]) -> Tuple[nx.DiGraph, np.ndarray]:
    """SLEDGE preprocessing for nuPlan GT lane graphs."""
    frame = (64, 64)
    pixel_size = 0.25
    lines = data["lines"]
    A = data["G"]["states"]
    G = nx.DiGraph(incoming_graph_data=A)

    lines_in_frame = []
    indices_to_remove = []
    for i, (line_states, line_mask) in enumerate(zip(lines["states"], lines["mask"])):
        line_in_mask = line_states[line_mask]
        if len(line_in_mask) < 2:
            indices_to_remove.append(i)
            continue

        path_progress = calculate_progress(line_in_mask)
        path_length = path_progress[-1]
        states_se2_array = line_in_mask.copy()
        states_se2_array[:, 2] = np.unwrap(states_se2_array[:, 2], axis=0)

        distances = np.arange(0, path_length + pixel_size, pixel_size)
        line = interpolate_path(distances, path_length, path_progress, states_se2_array, as_array=True)

        frame_mask = coords_in_frame(line[..., :2], frame)
        indices_segments = find_consecutive_true_indices(frame_mask)
        line_segments = [line[seg] for seg in indices_segments if len(line[seg]) >= 3]

        if line_segments:
            lines_in_frame.append(line_segments)
        else:
            indices_to_remove.append(i)

    mapping = {}
    new_count = 0
    for idx in range(len(G)):
        if idx not in indices_to_remove:
            mapping[idx] = new_count
            new_count += 1

    for idx in indices_to_remove:
        G.remove_node(idx)
    G = nx.relabel_nodes(G, mapping)

    inv_mapping = {v: k for k, v in mapping.items()}

    edges_to_remove = []
    for edge in G.edges():
        src_states = data["lines"]["states"][inv_mapping[edge[0]]]
        dst_states = data["lines"]["states"][inv_mapping[edge[1]]]
        src_mask = data["lines"]["mask"][inv_mapping[edge[0]]]
        dst_mask = data["lines"]["mask"][inv_mapping[edge[1]]]
        src = src_states[src_mask][-1, :2]
        dst = dst_states[dst_mask][0, :2]
        if np.abs(src[0]) > 32 or np.abs(src[1]) > 32 or np.abs(dst[0]) > 32 or np.abs(dst[1]) > 32:
            edges_to_remove.append(edge)

    for edge in edges_to_remove:
        G.remove_edge(edge[0], edge[1])

    final_lines_dict = {}
    num_lines_before_splitting = len(lines_in_frame)
    new_lines_count = 0
    edges_to_add = []

    for i, line_segments in enumerate(lines_in_frame):
        final_lines_dict[i] = line_segments[0]
        if len(line_segments) > 1:
            for j, seg in enumerate(line_segments[1:]):
                new_idx = num_lines_before_splitting + new_lines_count
                final_lines_dict[new_idx] = seg
                G.add_node(new_idx)
                if j == len(line_segments) - 2:
                    for suc_node in list(G.successors(i)):
                        G.remove_edge(i, suc_node)
                        edges_to_add.append((new_idx, suc_node))
                new_lines_count += 1

    for edge in edges_to_add:
        G.add_edge(edge[0], edge[1])

    final_lines_list = [final_lines_dict[i] for i in range(len(G))]

    vector_states = np.zeros((len(final_lines_list), 20, 2), dtype=np.float32)
    for line_idx, line in enumerate(final_lines_list):
        path_progress = calculate_progress(line)
        path_length = path_progress[-1]
        states_se2_array = line.copy()
        states_se2_array[:, 2] = np.unwrap(states_se2_array[:, 2], axis=0)
        distances = np.linspace(0, path_length, num=20, endpoint=True)
        vector_states[line_idx] = interpolate_path(distances, path_length, path_progress, states_se2_array, as_array=True)[..., :2]

    return G, vector_states


def get_networkx_lane_graph_without_traffic_lights(data: Dict) -> Tuple[nx.DiGraph, np.ndarray]:
    """Get lane graph excluding traffic lights (for nuPlan generated data)."""
    num_lanes = data["num_lanes"]
    l2l_edge_index = get_edge_index_complete_graph(num_lanes)
    lane_conn = data["road_connection_types"]

    SUCC_IDX = 1
    is_succ = lane_conn[:, SUCC_IDX] == 1
    edges = l2l_edge_index[:, is_succ].transpose(1, 0)

    lane_types = np.argmax(data["lane_types"], axis=1)
    is_centerline = lane_types == 0
    traffic_lights = np.where(lane_types != 0)[0]

    edges_filtered = [edge[None, :] for edge in edges if edge[0].item() not in traffic_lights and edge[1].item() not in traffic_lights]
    edges = np.concatenate(edges_filtered, axis=0) if edges_filtered else np.array([]).reshape(0, 2)

    lanes = data["road_points"]
    centerlines = lanes[is_centerline]

    idx_to_new_idx = {}
    centerline_count = 0
    for i in range(num_lanes):
        if lane_types[i] == 0:
            idx_to_new_idx[i] = centerline_count
            centerline_count += 1

    num_centerlines = len(centerlines)
    A = np.zeros((num_centerlines, num_centerlines))
    for edge in edges:
        A[idx_to_new_idx[edge[0].item()], idx_to_new_idx[edge[1].item()]] = 1

    return nx.DiGraph(incoming_graph_data=A), centerlines


def get_networkx_lane_graph(data: Dict) -> Tuple[nx.DiGraph, np.ndarray]:
    """Get lane graph from ScenarioDreamer format data."""
    num_lanes = data["num_lanes"]
    l2l_edge_index = get_edge_index_complete_graph(num_lanes)
    lane_conn = data["road_connection_types"]

    SUCC_IDX = 1
    is_succ = lane_conn[:, SUCC_IDX] == 1
    edges_succ = l2l_edge_index[:, is_succ].transpose(1, 0)

    centerlines = data["road_points"]
    num_centerlines = len(centerlines)
    A = np.zeros((num_centerlines, num_centerlines))
    for edge in edges_succ:
        A[edge[0].item(), edge[1].item()] = 1

    return nx.DiGraph(incoming_graph_data=A), centerlines


def convert_data_to_unified_format(data: Dict, dataset_name: str) -> Dict[str, Any]:
    """Convert scene data to unified format for metrics computation."""
    assert data.get("lg_type", NON_PARTITIONED) == NON_PARTITIONED

    if dataset_name in ("waymo", "waymo_gt"):
        G_succ, centerlines = get_networkx_lane_graph(data)
    elif dataset_name == "nuplan":
        G_succ, centerlines = get_networkx_lane_graph_without_traffic_lights(data)
    elif dataset_name == "nuplan_gt":
        G_succ, centerlines = _get_sledge_lane_graph_nuplan(data)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    agents = data["agent_states"]
    agent_types = np.argmax(data["agent_types"], axis=1)
    vehicles = agents[agent_types == NUPLAN_VEHICLE]

    compact_G, compact_centerlines = get_compact_lane_graph(G_succ, centerlines)

    return {
        "G": compact_G,
        "lanes": compact_centerlines,
        "vehicles": vehicles,
    }


# ==============================================================================
# Lane Metrics
# ==============================================================================

def get_lane_length(positions: np.ndarray) -> float:
    """Compute total length of a lane polyline."""
    diffs = np.diff(positions, axis=0)
    return float(np.sqrt(np.sum(diffs ** 2, axis=1)).sum())


def compute_route_length(samples: List[Dict]) -> Tuple[float, float]:
    """Compute route length statistics."""
    path_lengths_all = []
    for data in tqdm(samples, desc="Computing route lengths"):
        G = data["G"]
        lanes = data["lanes"]
        if len(lanes) == 0:
            continue

        ego_lane_index = int(np.argmin(np.min(np.linalg.norm(lanes, axis=-1), axis=1)))
        start_idx = int(np.argmin(np.linalg.norm(lanes[ego_lane_index], axis=-1)))

        try:
            paths = dict(nx.single_source_shortest_path(G, source=ego_lane_index))
        except Exception:
            paths = {ego_lane_index: [ego_lane_index]}

        best = 0.0
        for path in paths.values():
            total = 0.0
            for k, lid in enumerate(path):
                if k == 0:
                    total += get_lane_length(lanes[lid, start_idx:])
                else:
                    total += get_lane_length(lanes[lid])
            best = max(best, total)
        path_lengths_all.append(best)

    arr = np.array(path_lengths_all)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def compute_endpoint_dist(samples: List[Dict]) -> Tuple[float, float]:
    """Compute endpoint distance statistics."""
    endpoint_distances = []
    for data in tqdm(samples, desc="Computing endpoint distances"):
        G = data["G"]
        lanes = data["lanes"]
        for src, dst in G.edges():
            endpoint_distances.append(float(np.linalg.norm(lanes[src, -1] - lanes[dst, 0])))

    arr = np.array(endpoint_distances) if endpoint_distances else np.array([0.0])
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def get_keypoint_G(G: nx.DiGraph, lanes: np.ndarray) -> nx.DiGraph:
    """Build keypoint graph from lane graph."""
    keypoint_G = nx.DiGraph()
    for lane in G.nodes():
        start_kp = f"kp_start_{lane}"
        end_kp = f"kp_end_{lane}"
        length = get_lane_length(lanes[lane])
        keypoint_G.add_edge(start_kp, end_kp, length=length)

    equivalent_key_points = {}
    counter = 0
    for edge in G.edges():
        kp_1 = f"kp_end_{edge[0]}"
        kp_2 = f"kp_start_{edge[1]}"

        found_in_dict = False
        for kp in equivalent_key_points:
            if kp_1 in equivalent_key_points[kp] or kp_2 in equivalent_key_points[kp]:
                equivalent_key_points[kp].add(kp_1)
                equivalent_key_points[kp].add(kp_2)
                found_in_dict = True
                break

        if not found_in_dict:
            new_kp = f"kp_{counter}"
            counter += 1
            equivalent_key_points[new_kp] = {kp_1, kp_2}

    inv_map = {old_kp: new_kp for new_kp, old_kps in equivalent_key_points.items() for old_kp in old_kps}
    mapping = {node: inv_map.get(node, node) for node in keypoint_G.nodes()}

    return nx.relabel_nodes(keypoint_G, mapping, copy=True)


def get_num_keypoints(G: nx.DiGraph) -> np.ndarray:
    return np.array([len(G)])


def get_degree_keypoints(G: nx.DiGraph) -> np.ndarray:
    return np.array([d for _, d in G.degree()])


def urban_planning_reach_and_convenience(G_edges: nx.DiGraph) -> Tuple[np.ndarray, np.ndarray]:
    """Compute reach and convenience metrics."""
    reach = []
    convenience = []

    for source in G_edges.nodes():
        lengths_unweighted = dict(nx.single_source_shortest_path_length(G_edges, source))
        reach.append(len(lengths_unweighted) - 1)

        lengths_weighted = dict(nx.single_source_dijkstra_path_length(G_edges, source, weight="length"))
        for target, dist in lengths_weighted.items():
            if source != target:
                convenience.append(dist)

    return np.array(reach, dtype=np.float32), np.array(convenience, dtype=np.float32)


def compute_urban_planning_metrics(samples: List[Dict], gt_samples: List[Dict]) -> Tuple[float, float, float, float]:
    """Compute Frechet distances for urban planning metrics."""
    degree_gen, num_kp_gen, reach_gen, conv_gen = [], [], [], []
    degree_real, num_kp_real, reach_real, conv_real = [], [], [], []

    for data_gen, data_real in tqdm(zip(samples, gt_samples), total=len(samples), desc="Computing urban planning metrics"):
        for data, deg_list, kp_list, reach_list, conv_list in [
            (data_gen, degree_gen, num_kp_gen, reach_gen, conv_gen),
            (data_real, degree_real, num_kp_real, reach_real, conv_real),
        ]:
            keyG = get_keypoint_G(data["G"], data["lanes"])
            deg_list.append(get_degree_keypoints(keyG))
            kp_list.append(get_num_keypoints(keyG))
            r, c = urban_planning_reach_and_convenience(keyG)
            reach_list.append(r)
            conv_list.append(c)

    return (
        compute_frechet_distance(np.concatenate(degree_gen), np.concatenate(degree_real)) * 10,
        compute_frechet_distance(np.concatenate(num_kp_gen), np.concatenate(num_kp_real)),
        compute_frechet_distance(np.concatenate(reach_gen), np.concatenate(reach_real)),
        compute_frechet_distance(np.concatenate(conv_gen), np.concatenate(conv_real)) * 10,
    )


# ==============================================================================
# Agent Metrics
# ==============================================================================

def get_onroad_vehicles(vehicles: np.ndarray, lanes: np.ndarray, tol: float = 1.5) -> np.ndarray:
    """Filter vehicles that are on-road."""
    lanes_flat = lanes.reshape(-1, 2)
    dists = np.linalg.norm(vehicles[:, np.newaxis, :2] - lanes_flat[np.newaxis, :, :], axis=-1).min(axis=1)
    return vehicles[dists <= tol]


def get_nearest_dists(vehicles: np.ndarray) -> np.ndarray:
    """Get nearest distance to other vehicles."""
    dists = np.linalg.norm(vehicles[:, np.newaxis, :2] - vehicles[np.newaxis, :, :2], axis=-1)
    np.fill_diagonal(dists, 1e6)
    return dists.min(axis=1)


def get_lateral_devs(vehicles: np.ndarray, lanes: np.ndarray) -> np.ndarray:
    """Compute lateral deviation from lanes."""
    diffs = vehicles[:, np.newaxis, np.newaxis, :2] - lanes[np.newaxis, :, :, :]
    return np.sqrt(np.min(np.sum(diffs ** 2, axis=-1), axis=(1, 2)))


def get_angular_devs(vehicles: np.ndarray, lanes: np.ndarray) -> np.ndarray:
    """Compute angular deviation from lane direction."""
    agent_headings = np.arctan2(
        vehicles[:, UNIFIED_FORMAT_INDICES["sin_heading"]],
        vehicles[:, UNIFIED_FORMAT_INDICES["cos_heading"]],
    )
    direction_vectors = lanes[:, 1:, :] - lanes[:, :-1, :]
    centerline_headings = np.arctan2(direction_vectors[..., 1], direction_vectors[..., 0])

    diffs = vehicles[:, np.newaxis, np.newaxis, :2] - lanes[np.newaxis, :, :, :]
    dists_squared = np.sum(diffs ** 2, axis=-1)
    nearest_flat = np.argmin(dists_squared.reshape(dists_squared.shape[0], -1), axis=-1)
    nearest_cl = nearest_flat // dists_squared.shape[2]
    nearest_pt = np.clip(nearest_flat % dists_squared.shape[2], 1, dists_squared.shape[-1] - 1)
    nearest_headings = centerline_headings[nearest_cl, nearest_pt - 1]

    angular_dev = np.arctan2(
        np.sin(agent_headings - nearest_headings),
        np.cos(agent_headings - nearest_headings),
    )
    return np.degrees(angular_dev)


def compute_jsd_metrics(samples: List[Dict], gt_samples: List[Dict]) -> Tuple[float, ...]:
    """Compute JSD metrics for agent attributes."""
    data_gen = {"nd": [], "lat": [], "ang": [], "len": [], "wid": [], "spd": []}
    data_real = {"nd": [], "lat": [], "ang": [], "len": [], "wid": [], "spd": []}

    for s_gen, s_real in tqdm(zip(samples, gt_samples), total=len(samples), desc="Computing agent JSD"):
        for s, d in [(s_gen, data_gen), (s_real, data_real)]:
            v = s["vehicles"]
            lanes = resample_lanes(s["lanes"], num_points=100)
            onroad = get_onroad_vehicles(v, lanes)

            if len(v) > 1:
                d["nd"].append(get_nearest_dists(v))
            if len(onroad) > 0:
                d["lat"].append(get_lateral_devs(onroad, lanes))
                d["ang"].append(get_angular_devs(onroad, lanes))
            d["len"].append(v[:, UNIFIED_FORMAT_INDICES["length"]])
            d["wid"].append(v[:, UNIFIED_FORMAT_INDICES["width"]])
            d["spd"].append(v[:, UNIFIED_FORMAT_INDICES["speed"]])

    def cat(lst):
        return np.concatenate(lst, axis=0) if lst else np.array([0.0])

    return (
        jsd(cat(data_gen["nd"]), cat(data_real["nd"]), 0, 50, 1) * 10,
        jsd(cat(data_gen["lat"]), cat(data_real["lat"]), 0, 1.5, 0.1) * 10,
        jsd(cat(data_gen["ang"]), cat(data_real["ang"]), -200, 200, 5) * 100,
        jsd(cat(data_gen["len"]), cat(data_real["len"]), 0, 25, 0.1) * 100,
        jsd(cat(data_gen["wid"]), cat(data_real["wid"]), 0, 5, 0.1) * 100,
        jsd(cat(data_gen["spd"]), cat(data_real["spd"]), 0, 50, 1) * 100,
    )


def compute_lane_metrics(samples: List[Dict], gt_samples: List[Dict]) -> Dict[str, float]:
    """Compute all lane-related metrics."""
    route_mean, route_std = compute_route_length(samples)
    endpoint_mean, endpoint_std = compute_endpoint_dist(samples)
    fd_conn, fd_dens, fd_reach, fd_conv = compute_urban_planning_metrics(samples, gt_samples)

    return {
        "route_length_mean": route_mean,
        "route_length_std": route_std,
        "endpoint_dist_mean": endpoint_mean,
        "endpoint_dist_std": endpoint_std,
        "frechet_connectivity": fd_conn,
        "frechet_density": fd_dens,
        "frechet_reach": fd_reach,
        "frechet_convenience": fd_conv,
    }


def compute_agent_metrics(samples: List[Dict], gt_samples: List[Dict]) -> Dict[str, float]:
    """Compute all agent-related metrics."""
    nd_jsd, lat_jsd, ang_jsd, len_jsd, wid_jsd, spd_jsd = compute_jsd_metrics(samples, gt_samples)
    collision = compute_collision_rate(samples)

    return {
        "nearest_dist_jsd": nd_jsd,
        "lat_dev_jsd": lat_jsd,
        "ang_dev_jsd": ang_jsd,
        "length_jsd": len_jsd,
        "width_jsd": wid_jsd,
        "speed_jsd": spd_jsd,
        "collision_rate": collision * 100,
    }


# ==============================================================================
# NEW: Physically-Meaningful Diversity Metrics
# ==============================================================================

def compute_geometric_diversity_metrics(
    samples: List[Dict],
    gt_samples: List[Dict],
) -> Dict[str, float]:
    """
    Compute physically-meaningful diversity metrics.
    
    Unlike embedding-based metrics, these:
    1. Have clear physical interpretation
    2. Compare to GT distribution via JSD (measures both diversity and alignment)
    3. Are model-agnostic and robust
    
    Returns:
        dict with diversity metrics in physical space
    """
    # Lane topology diversity
    num_kp_gen, num_kp_gt = [], []
    degrees_gen, degrees_gt = [], []
    route_lengths_gen, route_lengths_gt = [], []
    
    # Agent diversity
    agent_counts_gen, agent_counts_gt = [], []
    speeds_gen, speeds_gt = [], []
    static_fracs_gen, static_fracs_gt = [], []
    
    STATIC_SPEED_THRESH = 0.5  # m/s

    for s_gen, s_gt in zip(samples, gt_samples):
        # Lane topology
        keyG_gen = get_keypoint_G(s_gen["G"], s_gen["lanes"])
        keyG_gt = get_keypoint_G(s_gt["G"], s_gt["lanes"])
        
        num_kp_gen.append(len(keyG_gen))
        num_kp_gt.append(len(keyG_gt))
        
        degrees_gen.extend([d for _, d in keyG_gen.degree()])
        degrees_gt.extend([d for _, d in keyG_gt.degree()])
        
        # Route length per scene
        lanes_gen = s_gen["lanes"]
        if len(lanes_gen) > 0:
            ego_idx = int(np.argmin(np.min(np.linalg.norm(lanes_gen, axis=-1), axis=1)))
            start_idx = int(np.argmin(np.linalg.norm(lanes_gen[ego_idx], axis=-1)))
            try:
                paths = dict(nx.single_source_shortest_path(s_gen["G"], source=ego_idx))
            except Exception:
                paths = {ego_idx: [ego_idx]}
            best = 0.0
            for path in paths.values():
                total = sum(get_lane_length(lanes_gen[lid, (start_idx if k == 0 else 0):]) for k, lid in enumerate(path))
                best = max(best, total)
            route_lengths_gen.append(best)
        
        lanes_gt = s_gt["lanes"]
        if len(lanes_gt) > 0:
            ego_idx = int(np.argmin(np.min(np.linalg.norm(lanes_gt, axis=-1), axis=1)))
            start_idx = int(np.argmin(np.linalg.norm(lanes_gt[ego_idx], axis=-1)))
            try:
                paths = dict(nx.single_source_shortest_path(s_gt["G"], source=ego_idx))
            except Exception:
                paths = {ego_idx: [ego_idx]}
            best = 0.0
            for path in paths.values():
                total = sum(get_lane_length(lanes_gt[lid, (start_idx if k == 0 else 0):]) for k, lid in enumerate(path))
                best = max(best, total)
            route_lengths_gt.append(best)
        
        # Agent diversity
        v_gen = s_gen["vehicles"]
        v_gt = s_gt["vehicles"]
        
        agent_counts_gen.append(len(v_gen))
        agent_counts_gt.append(len(v_gt))
        
        if len(v_gen) > 0:
            speeds_gen.extend(v_gen[:, UNIFIED_FORMAT_INDICES["speed"]].tolist())
            static_frac = float(np.mean(v_gen[:, UNIFIED_FORMAT_INDICES["speed"]] < STATIC_SPEED_THRESH))
            static_fracs_gen.append(static_frac)
        
        if len(v_gt) > 0:
            speeds_gt.extend(v_gt[:, UNIFIED_FORMAT_INDICES["speed"]].tolist())
            static_frac = float(np.mean(v_gt[:, UNIFIED_FORMAT_INDICES["speed"]] < STATIC_SPEED_THRESH))
            static_fracs_gt.append(static_frac)

    metrics = {}

    # Lane topology diversity (JSD → lower is better for alignment, but also shows diversity)
    if degrees_gen and degrees_gt:
        metrics["lane_degree_jsd"] = jsd(np.array(degrees_gen), np.array(degrees_gt), 0, 10, 1)
    else:
        metrics["lane_degree_jsd"] = 0.0
    
    if num_kp_gen and num_kp_gt:
        metrics["lane_num_keypoints_jsd"] = jsd(np.array(num_kp_gen), np.array(num_kp_gt), 0, 50, 1)
    else:
        metrics["lane_num_keypoints_jsd"] = 0.0
    
    # Route length diversity (std → higher shows more variation)
    if route_lengths_gen:
        metrics["lane_route_length_std"] = float(np.std(route_lengths_gen))
        metrics["lane_route_length_mean"] = float(np.mean(route_lengths_gen))
    else:
        metrics["lane_route_length_std"] = 0.0
        metrics["lane_route_length_mean"] = 0.0
    
    # Agent count diversity
    if agent_counts_gen and agent_counts_gt:
        metrics["agent_count_jsd"] = jsd(np.array(agent_counts_gen), np.array(agent_counts_gt), 0, 30, 1)
        metrics["agent_count_std"] = float(np.std(agent_counts_gen))
    else:
        metrics["agent_count_jsd"] = 0.0
        metrics["agent_count_std"] = 0.0
    
    # Speed distribution diversity
    if speeds_gen and speeds_gt:
        metrics["speed_distribution_jsd"] = jsd(np.array(speeds_gen), np.array(speeds_gt), 0, 30, 1)
    else:
        metrics["speed_distribution_jsd"] = 0.0
    
    # Static/dynamic ratio
    if static_fracs_gen and static_fracs_gt:
        metrics["static_frac_jsd"] = jsd(np.array(static_fracs_gen), np.array(static_fracs_gt), 0, 1, 0.1)
        metrics["static_frac_mean_gen"] = float(np.mean(static_fracs_gen))
        metrics["static_frac_mean_gt"] = float(np.mean(static_fracs_gt))
    else:
        metrics["static_frac_jsd"] = 0.0
        metrics["static_frac_mean_gen"] = 0.0
        metrics["static_frac_mean_gt"] = 0.0

    return metrics


def compute_motion_diversity_metrics(
    samples: List[Dict],
    gt_samples: List[Dict],
    motion_max_displacement: float = 12.0,
    static_eps_m: float = 0.36,
) -> Dict[str, float]:
    """
    Compute motion code diversity metrics.
    
    Requires samples to have 'agent_motion' field (physical coordinates).
    """
    from vectorworld.utils.data_helpers import unnormalize_motion_code
    
    path_lengths_gen, path_lengths_gt = [], []
    displacements_gen, displacements_gt = [], []
    nonzero_fracs_gen, nonzero_fracs_gt = [], []

    for s_gen, s_gt in zip(samples, gt_samples):
        motion_gen = s_gen.get("agent_motion", None)
        motion_gt = s_gt.get("agent_motion", None)
        
        for motion, pl_list, disp_list, nz_list in [
            (motion_gen, path_lengths_gen, displacements_gen, nonzero_fracs_gen),
            (motion_gt, path_lengths_gt, displacements_gt, nonzero_fracs_gt),
        ]:
            if motion is None or len(motion) == 0:
                continue
            
            motion = np.asarray(motion, dtype=np.float32)
            if motion.ndim != 2 or motion.shape[1] == 0:
                continue
            
            K = motion.shape[1] // 2
            pts = motion.reshape(-1, K, 2)
            
            # Path length
            seg_len = np.linalg.norm(pts[:, 1:] - pts[:, :-1], axis=-1)
            path_len = np.sum(seg_len, axis=-1)
            pl_list.extend(path_len.tolist())
            
            # Max displacement from origin
            disp = np.linalg.norm(pts, axis=-1).max(axis=-1)
            disp_list.extend(disp.tolist())
            
            # Non-zero fraction
            nz_frac = float(np.mean(path_len > static_eps_m))
            nz_list.append(nz_frac)

    metrics = {}
    
    if path_lengths_gen and path_lengths_gt:
        metrics["motion_path_length_jsd"] = jsd(np.array(path_lengths_gen), np.array(path_lengths_gt), 0, 15, 0.5)
        metrics["motion_path_length_std"] = float(np.std(path_lengths_gen))
    else:
        metrics["motion_path_length_jsd"] = 0.0
        metrics["motion_path_length_std"] = 0.0
    
    if displacements_gen and displacements_gt:
        metrics["motion_displacement_jsd"] = jsd(np.array(displacements_gen), np.array(displacements_gt), 0, 15, 0.5)
    else:
        metrics["motion_displacement_jsd"] = 0.0
    
    if nonzero_fracs_gen and nonzero_fracs_gt:
        metrics["motion_nonzero_frac_diff"] = abs(float(np.mean(nonzero_fracs_gen)) - float(np.mean(nonzero_fracs_gt)))
        metrics["motion_nonzero_frac_gen"] = float(np.mean(nonzero_fracs_gen))
        metrics["motion_nonzero_frac_gt"] = float(np.mean(nonzero_fracs_gt))
    else:
        metrics["motion_nonzero_frac_diff"] = 0.0
        metrics["motion_nonzero_frac_gen"] = 0.0
        metrics["motion_nonzero_frac_gt"] = 0.0

    return metrics


# ==============================================================================
# FD Helpers: Running Mean/Cov and Batch Building
# ==============================================================================

class RunningMeanCov:
    """Online mean/cov accumulator for embeddings."""
    def __init__(self, dim: int):
        self.dim = int(dim)
        self.count = 0
        self.sum = np.zeros((self.dim,), dtype=np.float64)
        self.sumsq = np.zeros((self.dim, self.dim), dtype=np.float64)

    def update(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"Expected (N, {self.dim}), got {X.shape}")
        self.count += X.shape[0]
        self.sum += X.sum(axis=0)
        self.sumsq += X.T @ X

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.count <= 1:
            mu = self.sum / max(1, self.count)
            return mu.astype(np.float64), np.eye(self.dim, dtype=np.float64)
        mu = self.sum / self.count
        cov = (self.sumsq - self.count * np.outer(mu, mu)) / (self.count - 1)
        return mu.astype(np.float64), cov.astype(np.float64)


def _convert_sledge_to_scenariodreamer(
    sledge_data: Dict[str, Any],
    num_points_per_lane: int = 20,
) -> Dict[str, Any]:
    """
    Convert SLEDGE format (NuPlan GT) to ScenarioDreamer format for AE batch building.
    
    SLEDGE format has:
        lines, vehicles, pedestrians, static_objects, green_lights, red_lights, 
        ego, G, agent_states, agent_types, lg_type
    
    ScenarioDreamer format needs:
        num_agents, num_lanes, agent_states, agent_types, road_points, 
        road_connection_types, lane_types, lg_type, map_id
    """
    # Extract lane graph from SLEDGE format
    G_sledge, lane_points = _get_sledge_lane_graph_nuplan(sledge_data)
    
    num_lanes = len(lane_points)
    num_agents = len(sledge_data.get("agent_states", []))
    
    # Build road_connection_types from graph edges
    # Complete graph edge format for ScenarioDreamer
    l2l_edge_index = get_edge_index_complete_graph(num_lanes)
    num_edges = l2l_edge_index.shape[1]
    
    # NuPlan connection types: {"none": 0, "pred": 1, "succ": 2, "self": 3}
    num_conn_types = 4
    road_conn = np.zeros((num_edges, num_conn_types), dtype=np.float32)
    road_conn[:, 0] = 1  # default to "none"
    
    # Set self-loops
    for i in range(num_lanes):
        for e in range(num_edges):
            if l2l_edge_index[0, e] == i and l2l_edge_index[1, e] == i:
                road_conn[e, 0] = 0
                road_conn[e, 3] = 1  # self
    
    # Set successor edges from graph
    for src, dst in G_sledge.edges():
        for e in range(num_edges):
            if l2l_edge_index[0, e] == src and l2l_edge_index[1, e] == dst:
                road_conn[e, 0] = 0
                road_conn[e, 2] = 1  # succ
            elif l2l_edge_index[0, e] == dst and l2l_edge_index[1, e] == src:
                road_conn[e, 0] = 0
                road_conn[e, 1] = 1  # pred
    
    # Lane types: all centerlines (type 0)
    lane_types = np.zeros((num_lanes, 3), dtype=np.float32)
    lane_types[:, 0] = 1  # lane type
    
    return {
        "num_agents": num_agents,
        "num_lanes": num_lanes,
        "agent_states": sledge_data.get("agent_states", np.zeros((0, 7), dtype=np.float32)),
        "agent_types": sledge_data.get("agent_types", np.zeros((0, 3), dtype=np.float32)),
        "road_points": lane_points,
        "road_connection_types": road_conn,
        "lane_types": lane_types,
        "lg_type": sledge_data.get("lg_type", 0),
        "map_id": sledge_data.get("map_id", 0),
    }


def _is_sledge_format(scene: Dict[str, Any]) -> bool:
    """Check if scene is in SLEDGE format (NuPlan GT)."""
    sledge_keys = {"lines", "G", "vehicles"}
    sd_keys = {"num_agents", "num_lanes", "road_points", "road_connection_types"}
    
    has_sledge = all(k in scene for k in sledge_keys)
    has_sd = all(k in scene for k in sd_keys)
    
    return has_sledge and not has_sd


def build_ae_batch_for_lane_embeddings(
    scene_dicts: List[Dict[str, Any]],
    dataset_cfg: Any,
    dataset_name: str,
    device: str = "cuda",
) -> Batch:
    """
    Build a PyG Batch for AE encoder to extract lane embeddings.
    
    Supports two input formats:
    1. ScenarioDreamer format (LDM generated or Waymo GT)
    2. SLEDGE format (NuPlan GT) - automatically converted
    """
    data_list = []

    motion_cfg = getattr(dataset_cfg, "motion", None)
    motion_enabled = motion_cfg is not None and getattr(motion_cfg, "enabled", False)
    motion_dim = int(getattr(motion_cfg, "dim", 0)) if motion_enabled else 0

    for s in scene_dicts:
        # Auto-detect and convert SLEDGE format
        if _is_sledge_format(s):
            s = _convert_sledge_to_scenariodreamer(s)
        
        # Validate required keys
        required = ["num_agents", "num_lanes", "agent_states", "agent_types", "road_points", "road_connection_types"]
        missing = [k for k in required if k not in s]
        if missing:
            raise KeyError(f"[build_ae_batch_for_lane_embeddings] scene missing keys: {missing}")

        num_agents = int(s["num_agents"])
        num_lanes = int(s["num_lanes"])

        agent_states = np.asarray(s["agent_states"], dtype=np.float32).copy()
        road_points = np.asarray(s["road_points"], dtype=np.float32).copy()

        # Normalize to [-1,1]
        agent_states_norm, road_points_norm = normalize_scene(
            agent_states,
            road_points,
            fov=float(dataset_cfg.fov),
            min_speed=float(dataset_cfg.min_speed),
            max_speed=float(dataset_cfg.max_speed),
            min_length=float(dataset_cfg.min_length),
            max_length=float(dataset_cfg.max_length),
            min_width=float(dataset_cfg.min_width),
            max_width=float(dataset_cfg.max_width),
            min_lane_x=float(dataset_cfg.min_lane_x),
            max_lane_x=float(dataset_cfg.max_lane_x),
            min_lane_y=float(dataset_cfg.min_lane_y),
            max_lane_y=float(dataset_cfg.max_lane_y),
        )

        d = ScenarioDreamerData()
        d["lg_type"] = int(s.get("lg_type", 0))
        d["map_id"] = int(s.get("map_id", 0))
        d["num_lanes"] = num_lanes
        d["num_agents"] = num_agents
        d["num_lanes_after_origin"] = 0

        d["agent"].x = torch.from_numpy(agent_states_norm).float()
        d["agent"].type = torch.from_numpy(np.asarray(s["agent_types"], dtype=np.float32)).float()

        if motion_dim > 0:
            d["agent"].motion = torch.zeros((num_agents, motion_dim), dtype=torch.float32)

        d["lane"].x = torch.from_numpy(road_points_norm).float()
        if dataset_name == "nuplan" and "lane_types" in s:
            d["lane"].type = torch.from_numpy(np.asarray(s["lane_types"], dtype=np.float32)).float()

        # Edges
        d["lane", "to", "lane"].edge_index = get_edge_index_complete_graph(num_lanes)
        d["agent", "to", "agent"].edge_index = get_edge_index_complete_graph(num_agents)
        d["lane", "to", "agent"].edge_index = get_edge_index_bipartite(num_lanes, num_agents)
        d["lane", "to", "lane"].type = torch.from_numpy(np.asarray(s["road_connection_types"], dtype=np.float32)).float()

        # Encoder masks (all True for eval)
        d["lane", "to", "lane"].encoder_mask = torch.ones(d["lane", "to", "lane"].edge_index.shape[1], dtype=torch.bool)
        d["agent", "to", "agent"].encoder_mask = torch.ones(d["agent", "to", "agent"].edge_index.shape[1], dtype=torch.bool)
        d["lane", "to", "agent"].encoder_mask = torch.ones(d["lane", "to", "agent"].edge_index.shape[1], dtype=torch.bool)

        # Partition masks (all False for non-partitioned)
        d["lane"].partition_mask = torch.zeros((num_lanes,), dtype=torch.bool)
        d["agent"].partition_mask = torch.zeros((num_agents,), dtype=torch.bool)

        data_list.append(d)

    return Batch.from_data_list(data_list).to(device)


def compute_embedding_diversity_metrics(embeddings: np.ndarray, max_pairs: int = 2_000_000) -> Dict[str, float]:
    """Simple diversity metrics on embedding space (kept for backward compatibility)."""
    X = np.asarray(embeddings, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        return {"cov_trace": 0.0, "pairwise_l2_mean": 0.0, "pairwise_l2_std": 0.0}

    cov = np.cov(X, rowvar=False)
    cov_trace = float(np.trace(cov))

    N = X.shape[0]
    rng = np.random.default_rng(0)
    num_pairs = min(max_pairs, N * (N - 1) // 2)
    i = rng.integers(0, N, size=num_pairs, endpoint=False)
    j = rng.integers(0, N, size=num_pairs, endpoint=False)
    mask = i != j
    d = np.linalg.norm(X[i[mask]] - X[j[mask]], axis=1)

    return {
        "cov_trace": cov_trace,
        "pairwise_l2_mean": float(d.mean()),
        "pairwise_l2_std": float(d.std(ddof=1)) if d.size > 1 else 0.0,
    }

def compute_sim_agent_jsd_metrics(
        metrics_dict,
        gt_lin_speeds,
        sim_lin_speeds,
        gt_ang_speeds,
        sim_ang_speeds,
        gt_accels,
        sim_accels,
        gt_dist_near_veh,
        sim_dist_near_veh):
    """ Computes the JSD sim agent metrics computed on the simulated and ground truth data."""
    
    # lin speed jsd 
    lin_speeds_gt = np.concatenate(gt_lin_speeds, axis=0)
    lin_speeds_sim = np.concatenate(sim_lin_speeds, axis=0)
    lin_speeds_gt = np.clip(lin_speeds_gt, 0, 30)
    lin_speeds_sim = np.clip(lin_speeds_sim, 0, 30)
    bin_edges = np.arange(201) * 0.5 * (100 / 30)
    P_lin_speeds_sim = np.histogram(lin_speeds_sim, bins=bin_edges)[0] / len(lin_speeds_sim)
    Q_lin_speeds_sim = np.histogram(lin_speeds_gt, bins=bin_edges)[0] / len(lin_speeds_gt)
    metrics_dict['lin_speed_jsd'] = distance.jensenshannon(P_lin_speeds_sim, Q_lin_speeds_sim) ** 2
    
    # ang speed jsd
    ang_speeds_gt = np.concatenate(gt_ang_speeds, axis=0)
    ang_speeds_sim = np.concatenate(sim_ang_speeds, axis=0)
    ang_speeds_gt = np.clip(ang_speeds_gt, -50, 50)
    ang_speeds_sim = np.clip(ang_speeds_sim, -50, 50)
    bin_edges = np.arange(201) * 0.5 - 50 
    P_ang_speeds_sim = np.histogram(ang_speeds_sim, bins=bin_edges)[0] / len(ang_speeds_sim)
    Q_ang_speeds_sim = np.histogram(ang_speeds_gt, bins=bin_edges)[0] / len(ang_speeds_gt)
    metrics_dict['ang_speed_jsd'] = distance.jensenshannon(P_ang_speeds_sim, Q_ang_speeds_sim) ** 2

    # accel jsd
    accels_gt = np.concatenate(gt_accels, axis=0)
    accels_sim = np.concatenate(sim_accels, axis=0)
    accels_gt = np.clip(accels_gt, -10, 10)
    accels_sim = np.clip(accels_sim, -10, 10)
    bin_edges = np.arange(201) * 0.1 - 10
    P_accels_sim = np.histogram(accels_sim, bins=bin_edges)[0] / len(accels_sim)
    Q_accels_sim = np.histogram(accels_gt, bins=bin_edges)[0] / len(accels_gt)
    metrics_dict['accel_jsd'] = distance.jensenshannon(P_accels_sim, Q_accels_sim) ** 2

    # nearest dist jsd
    nearest_dists_gt = np.concatenate(gt_dist_near_veh, axis=0)
    nearest_dists_sim = np.concatenate(sim_dist_near_veh, axis=0)
    nearest_dists_gt = np.clip(nearest_dists_gt, 0, 40)
    nearest_dists_sim = np.clip(nearest_dists_sim, 0, 40)
    bin_edges = np.arange(201) * 0.5 * (100 / 40)
    P_nearest_dists_sim = np.histogram(nearest_dists_sim, bins=bin_edges)[0] / len(nearest_dists_sim)
    Q_nearest_dists_sim = np.histogram(nearest_dists_gt, bins=bin_edges)[0] / len(nearest_dists_gt)
    metrics_dict['nearest_dist_jsd'] = distance.jensenshannon(P_nearest_dists_sim, Q_nearest_dists_sim) ** 2

    return metrics_dict