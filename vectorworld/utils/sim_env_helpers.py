import os
import pickle
import numpy as np
import networkx as nx
import random
import torch
import copy

from configs.config import LANE_CONNECTION_TYPES_WAYMO, LANE_CONNECTION_TYPES_NUPLAN
from vectorworld.utils.metrics_helpers import (
    get_networkx_lane_graph,
    get_networkx_lane_graph_without_traffic_lights,
    get_lane_length,
    get_compact_lane_graph,
)
from vectorworld.utils.geometry import normalize_angle
from vectorworld.utils.lane_graph_helpers import (
    resample_polyline,
    resample_polyline_every,
    resample_lanes,
    estimate_heading,
)
from vectorworld.utils.collision_helpers import batched_collision_checker, is_colliding
from vectorworld.utils.pyg_helpers import get_edge_index_complete_graph
from vectorworld.utils.viz import plot_scene


def postprocess_sim_env(
    pre_env,
    route_length=500,
    dataset="waymo",
    heading_tolerance=np.pi / 6,
    length_tolerance=15,
    num_points_per_lane=50,
    offroad_threshold=2.5,
):
    """Postprocess complete simulation environment to be compatible with simulator.

    Main steps
    --------
    1. A dense sampling of roote and cropping to a specified length;
    Delete:
       - Collision agent;
       - offload agents;
       - heading away from the nearest line above the threshold;
       - A car of an unusual size;
    3. Reorder the agent into [N,1,8], with the last one being ego;
    4. Compression of lane graph, resample to fixed points;
    5. Calculate the line index sequence corresponding to root;
    If pre_env exists`agent_motion`, is updated simultaneously with the filter / reload and writes post_env['agent_motion'].

    Variance from original
    ----------
    - for the roote length ** less than **route_length, no more direct failure by the assert,
      Instead, we cut the road_length to the available length (online simulations can tolerate a slightly shorter route).
      The behaviour is identical to that of the original when the offline creation of the root is long enough.
    """
    if dataset == "nuplan":
        LANE_CONNECTION_TYPES = LANE_CONNECTION_TYPES_NUPLAN
    else:
        LANE_CONNECTION_TYPES = LANE_CONNECTION_TYPES_WAYMO

    # simulator-compatible environment
    post_env = {}

    # 1) Route Processing: According to 1m spacing resample
    ego_route = resample_polyline_every(pre_env["route"], every=1.0)
    # If root is too short, no more assert, but crop to the available length
    if len(ego_route) < route_length:
        route_length_clipped = len(ego_route)
    else:
        route_length_clipped = route_length
    ego_route = ego_route[:route_length_clipped]
    post_env["route"] = ego_route

    # Record if there is a motion
    has_motion = "agent_motion" in pre_env
    if has_motion:
        agent_motion = pre_env["agent_motion"]
    else:
        agent_motion = None

    # 2) Deleting Colliding / offload / heading abnormal / dimensions
    agents = pre_env["agent_states"]  # (N,7)
    agent_types_onehot = pre_env["agent_types"]  # (N,3) one-hot over {veh,ped,cyc}
    agent_types_int = np.argmax(agent_types_onehot, axis=1)  # 0=veh,1=ped,2=cyc
    num_agents = pre_env["num_agents"]

    agents_to_remove = []
    agent_ids_to_process = np.arange(num_agents)[1:]  # Skip index 0 as an ego candidate (following roll to the end)

    while len(agent_ids_to_process) > 0:
        focal_agent_id = agent_ids_to_process[:1]

        # --- Collision detection (focal vs Other +ego(0)) ---
        if len(agent_ids_to_process) > 1:
            other_agent_ids = np.append(agent_ids_to_process[1:], [0])
            focal_agent = copy.deepcopy(agents[focal_agent_id])
            other_agents = copy.deepcopy(agents[other_agent_ids])

            # Test collisions with heading instead of (cos, sin)
            focal_agent[:, 4] = np.arctan2(
                agents[focal_agent_id, 4], agents[focal_agent_id, 3]
            )
            other_agents[:, 4] = np.arctan2(
                agents[other_agent_ids, 4], agents[other_agent_ids, 3]
            )
            collisions = batched_collision_checker(
                focal_agent[:, None, [0, 1, 4, 5, 6]],
                other_agents[:, None, [0, 1, 4, 5, 6]],
            )[:, 0].astype(bool)

            if np.any(collisions):
                agents_to_remove.append(focal_agent_id[0])

        # ---offroad / heading / size filter ---
        if agent_types_int[focal_agent_id[0]] == 0:
            offroad = (
                np.linalg.norm(
                    agents[focal_agent_id, :2] - pre_env["road_points"].reshape(-1, 2),
                    axis=-1,
                ).min()
                > offroad_threshold
            )

            aligned = True
            if not offroad:
                aligned = False
                lanes_near_agent_mask = (
                    np.linalg.norm(
                        pre_env["road_points"] - agents[focal_agent_id, None, :2],
                        axis=-1,
                    ).min(-1)
                    <= offroad_threshold
                )
                lanes_near_agent = pre_env["road_points"][lanes_near_agent_mask]
                for lane in lanes_near_agent:
                    closest_idx = np.argmin(
                        np.linalg.norm(lane - agents[focal_agent_id, :2], axis=-1)
                    )
                    if closest_idx == lane.shape[0] - 1:
                        closest_idx -= 1
                    next_idx = closest_idx + 1
                    diff = lane[next_idx] - lane[closest_idx]
                    lane_heading = np.arctan2(diff[1], diff[0])
                    agent_heading = np.arctan2(
                        agents[focal_agent_id, 4], agents[focal_agent_id, 3]
                    )
                    heading_diff = np.abs(
                        normalize_angle(agent_heading - lane_heading)
                    )[0]
                    if heading_diff < heading_tolerance:
                        aligned = True
                        break

            too_big = agents[focal_agent_id[0], 5] > length_tolerance

            if offroad or too_big or not aligned:
                agents_to_remove.append(focal_agent_id[0])

        agent_ids_to_process = agent_ids_to_process[1:]

    # Update pre_env from agents_to_remove
    if len(agents_to_remove) > 0:
        valid_agent_ids = np.setdiff1d(np.arange(num_agents), agents_to_remove)
        pre_env["agent_states"] = agents[valid_agent_ids]
        pre_env["agent_types"] = agent_types_int[valid_agent_ids]
        if has_motion and agent_motion is not None:
            pre_env["agent_motion"] = agent_motion[valid_agent_ids]
    else:
        pre_env["agent_types"] = agent_types_int

    # NPPlan: Remove state object
    if dataset == "nuplan":
        dynamic_agent_mask = pre_env["agent_types"] < 2
        pre_env["agent_states"] = pre_env["agent_states"][dynamic_agent_mask]
        pre_env["agent_types"] = pre_env["agent_types"][dynamic_agent_mask]
        if has_motion and "agent_motion" in pre_env:
            pre_env["agent_motion"] = pre_env["agent_motion"][dynamic_agent_mask]

    # 3. Reformats-> (N, 1,8), last one is ego
    agents_states_now = pre_env["agent_states"]  # (N,7)
    N_final = agents_states_now.shape[0]

    agents_full = np.zeros((N_final, 8), dtype=float)
    agents_full[:, :2] = agents_states_now[:, :2]
    agents_full[:, 2] = agents_states_now[:, 2] * agents_states_now[:, 3]  # vx
    agents_full[:, 3] = agents_states_now[:, 2] * agents_states_now[:, 4]  # vy
    agents_full[:, 4] = np.arctan2(
        agents_states_now[:, 4], agents_states_now[:, 3]
    )  # heading
    agents_full[:, 5:7] = agents_states_now[:, 5:]
    agents_full[:, 7] = 1.0  # existence
    agents_full = agents_full[:, None, :]  # (N,1,8)

    agent_types_int_now = pre_env["agent_types"]
    agent_types_onehot5 = np.eye(5)[agent_types_int_now]
    # Harmonize to [unset, vehicle, pedestrian, cyclist, other]
    agent_types_onehot5 = agent_types_onehot5[:, [4, 0, 1, 2, 3]]

    # ego set the current index0 (consistent with original):roll(-1)Make original index0 the last
    post_env["agents"] = np.roll(agents_full, -1, axis=0)
    post_env["agent_types"] = np.roll(agent_types_onehot5, -1, axis=0)
    post_env["num_agents"] = len(post_env["agents"])

    # Synchronize roll agent_motion
    if has_motion and "agent_motion" in pre_env:
        motion_now = pre_env["agent_motion"]  # (N, Dm)
        post_env["agent_motion"] = np.roll(motion_now, -1, axis=0)

    # 4) lane coat compression & resample
    num_lanes = pre_env["num_lanes"]
    l2l_edge_index = get_edge_index_complete_graph(num_lanes)
    lane_conn = pre_env["road_connection_types"]  # (E,6 or 4)

    is_succ = lane_conn[:, LANE_CONNECTION_TYPES["pred"]] == 1
    edges_succ = l2l_edge_index[:, is_succ].transpose(1, 0)

    centerlines = pre_env["road_points"]
    num_centerlines = len(centerlines)
    A_succ = np.zeros((num_centerlines, num_centerlines))
    for edge in edges_succ:
        A_succ[edge[0].item(), edge[1].item()] = 1

    G_succ = nx.DiGraph(incoming_graph_data=A_succ)
    compact_G, compact_centerlines = get_compact_lane_graph(
        G_succ, centerlines, num_points_per_lane=num_points_per_lane
    )

    route_lane_indices, valid = get_route_lane_indices(
        copy.deepcopy(compact_centerlines),
        copy.deepcopy(compact_G),
        copy.deepcopy(post_env["route"]),
    )
    if valid:
        post_env["route_lane_indices"] = route_lane_indices
    else:
        post_env["route_lane_indices"] = None

    post_env["lanes"] = compact_centerlines
    post_env["lane_graph"] = compact_G
    post_env["num_lanes"] = len(post_env["lanes"])

    return post_env


def get_route_lane_indices(
    lanes, G, route, upsample_points=10000, dist_threshold=3.25
):
    """Get the lane indices that correspond to the route."""
    valid = True
    lanes = resample_lanes(lanes, num_points=upsample_points)

    start_lane_id = np.argmin(np.linalg.norm(lanes, axis=-1).min(1))
    route_end = route[-1]
    end_lane_id = np.argmin(
        np.linalg.norm(lanes - route_end, axis=-1).min(1)
    )

    if start_lane_id == end_lane_id:
        paths = [[start_lane_id]]
    else:
        paths = list(nx.all_simple_paths(G, start_lane_id, end_lane_id))
        if len(paths) == 0:
            valid = False
            return None, valid

    start_lane_point = np.linalg.norm(
        lanes[start_lane_id], axis=-1
    ).argmin()
    end_lane_point = np.linalg.norm(
        lanes[end_lane_id] - route_end, axis=-1
    ).argmin()

    # find path that best fits the route
    best_path = None
    best_path_error = float("inf")
    best_path_points_on_route = None
    route_upsampled = resample_polyline(
        np.concatenate(
            [lanes[start_lane_id, start_lane_point][None, ...], route], axis=0
        ),
        num_points=upsample_points,
    )
    for path in paths:
        path_points_on_route = []
        for i, lane_id in enumerate(path):
            lane = lanes[lane_id]
            if i == 0 and i == len(path) - 1:
                # single-lane path
                lane_points = lane[start_lane_point : end_lane_point + 1]
            elif i == 0:
                lane_points = lane[start_lane_point:]
            elif i == len(path) - 1:
                lane_points = lane[:end_lane_point]
            else:
                lane_points = lane

            path_points_on_route.append(lane_points)
        path_points_on_route = np.concatenate(path_points_on_route, axis=0)
        path_points_on_route = resample_polyline(
            path_points_on_route, num_points=upsample_points
        )
        # compute error to route
        path_error = np.linalg.norm(
            path_points_on_route - route_upsampled, axis=-1
        ).max()
        if path_error < best_path_error:
            best_path_error = path_error
            best_path = path
            best_path_points_on_route = path_points_on_route

    if best_path_error > dist_threshold:
        valid = False
    return best_path, valid


def clean_up_scene(
    data,
    dataset,
    mode="initial_scene",
    endpoint_threshold=1,
    offroad_threshold=2.5,
):
    """I'm not going to be able to do this, but I'm not going to be able to do it.

    Important: If Exists in Data`agent_motion`(shape: (Na, Dm)),
    Synchronizes the deletion of the corresponding movement line when deleting an agent, thus maintaining alignment.
    """
    lanes = data["road_points"]
    road_connection_types = data["road_connection_types"]
    if dataset == "nuplan":
        lane_types = data["lane_types"]
        LANE_CONNECTION_TYPES = LANE_CONNECTION_TYPES_NUPLAN
    else:
        LANE_CONNECTION_TYPES = LANE_CONNECTION_TYPES_WAYMO
    num_lanes = data["num_lanes"]

    if dataset == "waymo":
        G, _ = get_networkx_lane_graph(data)
    else:
        G, _ = get_networkx_lane_graph_without_traffic_lights(data)

    if mode == "inpainting":
        lanes_before_partition_mask = data["lane_mask"].astype(bool)
        lane_ids_after_partition = np.arange(num_lanes)[~lanes_before_partition_mask]
        cond_lane_ids = data["lane_ids"].astype(int)

    # 1) weighting lane
    node_groups = {}
    for node in G.nodes:
        if mode == "inpainting" and (node not in lane_ids_after_partition):
            continue
        predecessors = set(G.predecessors(node))
        successors = set(G.successors(node))
        key = (frozenset(predecessors), frozenset(successors))
        node_groups.setdefault(key, []).append(node)

    lanes_to_remove = []
    for key, lane_ids in node_groups.items():
        lane_ids_to_process = list(lane_ids)
        while len(lane_ids_to_process) > 1:
            lane_id = np.array(lane_ids_to_process[:1])
            other = np.array(lane_ids_to_process[1:])

            start_lane_id = lanes[lane_id, 0]
            start_other = lanes[other, 0]
            end_lane_id = lanes[lane_id, -1]
            end_other = lanes[other, -1]

            start_close = (
                np.linalg.norm(start_lane_id - start_other, axis=-1)
                < endpoint_threshold
            )
            end_close = (
                np.linalg.norm(end_lane_id - end_other, axis=-1)
                < endpoint_threshold
            )
            lanes_close = start_close & end_close

            if dataset == "nuplan":
                other_lane_ids = other[lanes_close]
                if other_lane_ids.size == 0:
                    both_lanes_or_both_traffic_light_segments = False
                elif np.argmax(lane_types[lane_id.item()]) == 0:
                    both_lanes_or_both_traffic_light_segments = np.any(
                        np.argmax(lane_types[other_lane_ids], axis=-1) == 0
                    )
                else:
                    both_lanes_or_both_traffic_light_segments = np.any(
                        np.argmax(lane_types[other_lane_ids], axis=-1) != 0
                    )
            else:
                both_lanes_or_both_traffic_light_segments = True

            if np.any(lanes_close) and both_lanes_or_both_traffic_light_segments:
                if (not _near_border(lanes[lane_ids_to_process[0], -1])) and (
                    not _near_border(lanes[lane_ids_to_process[0], 0])
                ):
                    lanes_to_remove.append(lane_ids_to_process[0])

            lane_ids_to_process.pop(0)

    lanes_to_remove = np.array(lanes_to_remove)
    if len(lanes_to_remove) > 0:
        lane_graph_adj_pre = road_connection_types[
            :, LANE_CONNECTION_TYPES["pred"]
        ].reshape(num_lanes, num_lanes)
        lane_graph_adj_pre = np.delete(lane_graph_adj_pre, lanes_to_remove, axis=0)
        lane_graph_adj_pre = np.delete(lane_graph_adj_pre, lanes_to_remove, axis=1)

        lane_graph_adj_suc = road_connection_types[
            :, LANE_CONNECTION_TYPES["succ"]
        ].reshape(num_lanes, num_lanes)
        lane_graph_adj_suc = np.delete(lane_graph_adj_suc, lanes_to_remove, axis=0)
        lane_graph_adj_suc = np.delete(lane_graph_adj_suc, lanes_to_remove, axis=1)

        if dataset == "waymo":
            lane_graph_adj_left = road_connection_types[
                :, LANE_CONNECTION_TYPES["left"]
            ].reshape(num_lanes, num_lanes)
            lane_graph_adj_left = np.delete(lane_graph_adj_left, lanes_to_remove, axis=0)
            lane_graph_adj_left = np.delete(lane_graph_adj_left, lanes_to_remove, axis=1)

            lane_graph_adj_right = road_connection_types[
                :, LANE_CONNECTION_TYPES["right"]
            ].reshape(num_lanes, num_lanes)
            lane_graph_adj_right = np.delete(
                lane_graph_adj_right, lanes_to_remove, axis=0
            )
            lane_graph_adj_right = np.delete(
                lane_graph_adj_right, lanes_to_remove, axis=1
            )

        lane_graph_pre = lane_graph_adj_pre.reshape(-1)
        lane_graph_suc = lane_graph_adj_suc.reshape(-1)
        lane_graph_self = torch.eye(len(lane_graph_adj_suc)).reshape(-1)
        if dataset == "waymo":
            lane_graph_left = lane_graph_adj_left.reshape(-1)
            lane_graph_right = lane_graph_adj_right.reshape(-1)

        new_road_connection_types = np.zeros(len(lane_graph_suc)).astype(int)
        new_road_connection_types[lane_graph_pre == 1] = LANE_CONNECTION_TYPES["pred"]
        new_road_connection_types[lane_graph_suc == 1] = LANE_CONNECTION_TYPES["succ"]
        new_road_connection_types[lane_graph_self == 1] = LANE_CONNECTION_TYPES["self"]
        if dataset == "waymo":
            new_road_connection_types[lane_graph_left == 1] = LANE_CONNECTION_TYPES[
                "left"
            ]
            new_road_connection_types[lane_graph_right == 1] = LANE_CONNECTION_TYPES[
                "right"
            ]
        new_road_connection_types = np.eye(6 if dataset == "waymo" else 4)[
            new_road_connection_types.astype(int)
        ]

        valid_lane_ids = np.setdiff1d(np.arange(num_lanes), lanes_to_remove)
        data["road_points"] = lanes[valid_lane_ids]
        data["num_lanes"] = len(data["road_points"])
        data["road_connection_types"] = new_road_connection_types
        if dataset == "nuplan":
            data["lane_types"] = lane_types[valid_lane_ids]
        if mode == "inpainting":
            data["lane_mask"] = lanes_before_partition_mask[valid_lane_ids]
            data["lane_ids"] = cond_lane_ids[valid_lane_ids]

    # 2) delete overlaps / offload agents, and update them sync_motion
    agents = data["agent_states"]
    agent_types = data["agent_types"]
    if mode == "inpainting":
        agents_before_partition_mask = data["agent_mask"].astype(bool)
    num_agents = data["num_agents"]
    agents_to_remove = [] if mode == "initial_scene" else [0]
    agent_ids_to_process = np.arange(num_agents)[1:]

    has_motion = "agent_motion" in data
    if has_motion:
        motion_all = data["agent_motion"]

    while len(agent_ids_to_process) > 0:
        focal_agent_id = agent_ids_to_process[:1]

        if len(agent_ids_to_process) > 1:
            other_agent_ids = agent_ids_to_process[1:]
            if mode == "initial_scene":
                other_agent_ids = np.append(other_agent_ids, [0])

            focal_agent = copy.deepcopy(agents[focal_agent_id])
            other_agents = copy.deepcopy(agents[other_agent_ids])
            focal_agent[:, 4] = np.arctan2(
                agents[focal_agent_id, 4], agents[focal_agent_id, 3]
            )
            other_agents[:, 4] = np.arctan2(
                agents[other_agent_ids, 4], agents[other_agent_ids, 3]
            )

            collisions = batched_collision_checker(
                focal_agent[:, None, [0, 1, 4, 5, 6]],
                other_agents[:, None, [0, 1, 4, 5, 6]],
            )[:, 0].astype(bool)

            if np.any(collisions):
                agents_to_remove.append(focal_agent_id[0])

        # offroad vehicle
        if np.argmax(agent_types[focal_agent_id[0]]) == 0:
            offroad = (
                np.linalg.norm(
                    agents[focal_agent_id, :2] - data["road_points"].reshape(-1, 2),
                    axis=-1,
                ).min()
                > offroad_threshold
            )
            if offroad:
                agents_to_remove.append(focal_agent_id[0])

        agent_ids_to_process = agent_ids_to_process[1:]

    if len(agents_to_remove) > 0:
        valid_agent_ids = np.setdiff1d(np.arange(num_agents), agents_to_remove)
        data["agent_states"] = agents[valid_agent_ids]
        data["agent_types"] = agent_types[valid_agent_ids]
        data["num_agents"] = len(data["agent_states"])
        if has_motion:
            data["agent_motion"] = motion_all[valid_agent_ids]
        if mode == "inpainting":
            data["agent_mask"] = agents_before_partition_mask[valid_agent_ids]

    return data


def check_scene_validity(data, dataset):
    """Check if the generated scene is valid based on several criteria."""
    if dataset == "waymo":
        G, lanes = get_networkx_lane_graph(data)
    else:
        G, lanes = get_networkx_lane_graph_without_traffic_lights(data)

    # filter 1: lane endpoints must either connect or be near border
    passed_filter1 = True
    for lane_id, lane in enumerate(lanes):
        if G.out_degree(lane_id) == 0:
            if not _near_border(lane[-1]):
                passed_filter1 = False
                break
        if G.in_degree(lane_id) == 0:
            if not _near_border(lane[0]):
                passed_filter1 = False
                break

    # filter 2: ego near a lane
    closest_lane_id = np.linalg.norm(lanes, axis=-1).min(1).argmin()
    closest_dist = np.linalg.norm(lanes[closest_lane_id], axis=-1).min()
    passed_filter2 = closest_dist <= 2.5

    # filter 3: continuity between connected lanes
    passed_filter3 = True
    for edge in G.edges():
        pre_endpoint = lanes[edge[0]][-1]
        suc_startpoint = lanes[edge[1]][0]
        if np.linalg.norm(pre_endpoint - suc_startpoint) > 1.5:
            passed_filter3 = False

    return passed_filter1 and passed_filter2 and passed_filter3


def check_scene_validity_inpainting(data, dataset, heading_tolerance=np.pi / 3):
    """Check validity for inpainted scenes."""
    if dataset == "waymo":
        G, lanes = get_networkx_lane_graph(data)
        lane_before_partition_mask = data["lane_mask"].astype(bool)
    else:
        G, lanes = get_networkx_lane_graph_without_traffic_lights(data)
        lane_before_partition_mask = data["lane_mask"].astype(bool)[
            np.argmax(data["lane_types"], axis=1) == 0
        ]

    passed_filter1 = True
    lane_ids_with_endpoints_near_partition = []
    lane_ids_with_startpoints_near_partition = []
    heading_offset = np.pi / 2 if dataset == "waymo" else 0
    for lane_id, lane in enumerate(lanes):
        first_heading, last_heading = estimate_heading(lane)
        if _near_partition(lane[-1], dataset) and lane_before_partition_mask[lane_id] and np.abs(
            normalize_angle(last_heading - heading_offset)
        ) < heading_tolerance:
            lane_ids_with_endpoints_near_partition.append(lane_id)
        elif _near_partition(lane[0], dataset) and lane_before_partition_mask[
            lane_id
        ] and np.abs(normalize_angle(first_heading + heading_offset)) < heading_tolerance:
            lane_ids_with_startpoints_near_partition.append(lane_id)

    for lane_id in lane_ids_with_endpoints_near_partition:
        if G.out_degree(lane_id) == 0:
            passed_filter1 = False
            break
    for lane_id in lane_ids_with_startpoints_near_partition:
        if G.in_degree(lane_id) == 0:
            passed_filter1 = False
            break

    passed_filter2 = True
    inpainted_lanes = lanes[~lane_before_partition_mask]
    inpainted_lane_ids = np.arange(len(lanes))[~lane_before_partition_mask]
    for lane_id, lane in zip(inpainted_lane_ids, inpainted_lanes):
        if G.out_degree(lane_id) == 0:
            if not _near_border(lane[-1]):
                passed_filter2 = False
                break
        if G.in_degree(lane_id) == 0:
            if not _near_border(lane[0]):
                passed_filter2 = False
                break

    passed_filter3 = True
    for edge in G.edges():
        tolerance = 1.5
        if (edge[0] in lane_ids_with_endpoints_near_partition) or (
            edge[1] in lane_ids_with_startpoints_near_partition
        ):
            tolerance = 2.5
        pre_endpoint = lanes[edge[0]][-1]
        suc_startpoint = lanes[edge[1]][0]
        if np.linalg.norm(pre_endpoint - suc_startpoint) > tolerance:
            passed_filter3 = False

    return passed_filter1, passed_filter2, passed_filter3


def sample_route(d, dataset, heading_tolerance=np.pi / 3, num_points_in_route=1000):
    """Sample a valid route for the ego vehicle in the scene."""
    if dataset == "waymo":
        G, lanes = get_networkx_lane_graph(d)
    else:
        G, lanes = get_networkx_lane_graph_without_traffic_lights(d)

    start_lane = np.linalg.norm(lanes, axis=-1).min(1).argmin()
    start_idx = np.argmin(np.linalg.norm(lanes[start_lane], axis=-1))
    if start_idx == lanes.shape[1] - 1:
        diff = lanes[start_lane][start_idx] - lanes[start_lane][start_idx - 1]
    else:
        diff = lanes[start_lane][start_idx + 1] - lanes[start_lane][start_idx]
    lane_heading = np.arctan2(diff[1], diff[0])

    offset = np.pi / 2 if dataset == "waymo" else 0
    if np.abs(normalize_angle(lane_heading - offset)) >= heading_tolerance:
        return None, False

    paths = [[start_lane]]
    for target in G.nodes:
        if target != start_lane:
            paths.extend(nx.all_simple_paths(G, start_lane, target))

    valid_paths = []
    for path in paths:
        last_lane = path[-1]
        if not (_near_border(lanes[last_lane, -1]) and _valid_route_end(last_lane, lanes[last_lane])):
            continue
        valid_paths.append(path)

    if len(valid_paths) == 0:
        return None, False

    random.shuffle(valid_paths)
    route_as_lane_ids = valid_paths[0]

    route_lanes = []
    for i, lane_id in enumerate(route_as_lane_ids):
        lane = lanes[lane_id]
        if i == 0 and len(route_as_lane_ids) == 1:
            end_idx = 20
            start_idx = np.argmin(np.linalg.norm(lane, axis=-1))
            if start_idx == end_idx:
                end_idx += 1
            route_lanes.append(lane[start_idx:end_idx])
        elif i == 0:
            start_idx = np.argmin(np.linalg.norm(lane, axis=-1))
            route_lanes.append(lane[start_idx:])
        else:
            route_lanes.append(lane)

    route_lanes = np.concatenate(route_lanes, axis=0)
    assert len(route_lanes) > 0
    route_lanes = resample_polyline(route_lanes, num_points=num_points_in_route)
    return route_lanes, True


def get_default_route_center_yaw(dataset):
    """ Get default route center and yaw for the dataset if no valid route is found."""
    if dataset == "waymo":
        return np.array([0, 32]), np.pi / 2
    else:
        return np.array([32, 0]), 0


def generate_simulation_environments(model, cfg, save_dir):
    """Generate simulation environments using the trained Scenario Dreamer LDMModel.

    Enhancements (best-paper demo viz):
    1) Fix large stitched-map readability by relying on improved plot_scene() scaling.
    2) Collect rollout snapshots per environment and render a demo-quality MP4:
       - main fixed-size local view
       - sliding window thumbnails (K=5)
       - agent motion replay (agent_motion history makes agents move)
    """
    partial_samples_dir = os.path.join(save_dir, "partial_sim_envs")
    complete_samples_dir = os.path.join(save_dir, "complete_sim_envs")

    os.makedirs(partial_samples_dir, exist_ok=True)
    os.makedirs(complete_samples_dir, exist_ok=True)

    viz_root = getattr(getattr(cfg.eval, "sim_envs", {}), "viz_dir", None)
    if viz_root is None or str(viz_root) == "":
        viz_root = os.path.join(save_dir, f"viz_sim_envs_{cfg.dataset_name}")
    os.makedirs(viz_root, exist_ok=True)

    print(complete_samples_dir)

    assert len(os.listdir(partial_samples_dir)) == 0, "Partial samples directory must be empty before generation."
    assert len(os.listdir(complete_samples_dir)) == 0, "Complete samples directory must be empty before generation."

    # ----------------------------
    # Video viz knobs (safe defaults; no YAML edits required)
    # ----------------------------
    # We intentionally avoid OmegaConf.select here to keep this function standalone and robust.
    # If you want to override, you can add hydra keys:
    #   ldm.eval.sim_envs.viz_video.enabled=true
    #   ldm.eval.sim_envs.viz_video.fps=12
    #   ldm.eval.sim_envs.viz_video.filmstrip_k=5
    #   ldm.eval.sim_envs.viz_video.view_size_m=80
    #   ldm.eval.sim_envs.viz_video.pan_frames=6
    #   ldm.eval.sim_envs.viz_video.replay_interp=3
    #   ldm.eval.sim_envs.viz_video.hold_first=2
    #   ldm.eval.sim_envs.viz_video.hold_last=4
    #   ldm.eval.sim_envs.viz_video.dpi=120
    #   ldm.eval.sim_envs.viz_video.motion_viz_mode=curve
    #   ldm.eval.sim_envs.viz_video.show_agent_ids=ego_only
    sim_envs_cfg = getattr(cfg.eval, "sim_envs", None)
    viz_video_cfg = getattr(sim_envs_cfg, "viz_video", None) if sim_envs_cfg is not None else None

    # Enable video by default when cfg.eval.visualize is True
    video_enabled = bool(getattr(cfg.eval, "visualize", False))
    if viz_video_cfg is not None:
        try:
            video_enabled = bool(getattr(viz_video_cfg, "enabled", video_enabled))
        except Exception:
            pass

    fps = int(getattr(viz_video_cfg, "fps", 12)) if viz_video_cfg is not None else 12
    filmstrip_k = int(getattr(viz_video_cfg, "filmstrip_k", 5)) if viz_video_cfg is not None else 5
    view_size_m = float(getattr(viz_video_cfg, "view_size_m", float(cfg.dataset.fov) * 1.25)) if viz_video_cfg is not None else float(cfg.dataset.fov) * 1.25
    pan_frames = int(getattr(viz_video_cfg, "pan_frames", 6)) if viz_video_cfg is not None else 6
    replay_interp = int(getattr(viz_video_cfg, "replay_interp", 3)) if viz_video_cfg is not None else 3
    hold_first = int(getattr(viz_video_cfg, "hold_first", 2)) if viz_video_cfg is not None else 2
    hold_last = int(getattr(viz_video_cfg, "hold_last", 4)) if viz_video_cfg is not None else 4
    dpi = int(getattr(viz_video_cfg, "dpi", 120)) if viz_video_cfg is not None else 120
    motion_viz_mode = str(getattr(viz_video_cfg, "motion_viz_mode", "curve")) if viz_video_cfg is not None else "curve"
    show_agent_ids = str(getattr(viz_video_cfg, "show_agent_ids", "ego_only")) if viz_video_cfg is not None else "ego_only"
    road_grid_res = float(getattr(viz_video_cfg, "road_grid_res", 0.35)) if viz_video_cfg is not None else 0.35

    # Collect rollout snapshots for video
    viz_traces = {}  # sample_id -> list[snapshot_env_dict]

    def _freeze_env_for_viz(env_dict):
        """Store only the fields needed for video; deep-copy arrays to avoid mutation issues."""
        out = {}
        for k in [
            "agent_states",
            "road_points",
            "agent_types",
            "lane_types",
            "route",
            "tile_occupancy",
            "agent_motion",
            "route_completed",
        ]:
            if k not in env_dict:
                continue
            v = env_dict[k]
            if k == "tile_occupancy":
                if isinstance(v, list):
                    out[k] = [np.asarray(x).copy() for x in v]
                else:
                    out[k] = []
            elif isinstance(v, np.ndarray):
                out[k] = v.copy()
            else:
                # route can be list; types can be list; keep as numpy when possible
                try:
                    out[k] = np.asarray(v).copy()
                except Exception:
                    out[k] = v
        return out

    def _append_trace(sample_id: str, env_dict: dict):
        if not video_enabled:
            return
        if sample_id not in viz_traces:
            viz_traces[sample_id] = []
        viz_traces[sample_id].append(_freeze_env_for_viz(env_dict))

    max_num_samples = int(cfg.eval.num_samples * cfg.eval.sim_envs.overhead_factor)
    print(f"Generating {cfg.eval.num_samples} simulation environments...")
    print(
        f"To account for degenerate samples, we will generate {max_num_samples} samples (overhead_factor={cfg.eval.sim_envs.overhead_factor})."
    )

    it = 0
    while len(os.listdir(complete_samples_dir)) < cfg.eval.num_samples:
        if it == 0:
            mode = "initial_scene"
            num_iters = 1
            num_samples = max_num_samples
        else:
            mode = "inpainting"
            num_iters = cfg.eval.sim_envs.num_inpainting_candidates
            num_samples = len(os.listdir(partial_samples_dir))
            if num_samples == 0:
                print("No partial samples available for inpainting. Ending generation.")
                break

        print(f"Iteration {it}: generating in mode {mode}...")
        candidate_next_samples = {}
        num_failed_check_1, num_failed_check_2, num_failed_check_3 = 0, 0, 0
        num_failed_found_route = 0
        num_failed_overlapping_tiles = 0

        for iter_idx in range(num_iters):
            samples = model.generate(
                mode=mode,
                num_samples=num_samples,
                batch_size=cfg.eval.batch_size,
                cache_samples=False,
                visualize=False,
                conditioning_path=partial_samples_dir if mode == "inpainting" else None,
                cache_dir=None,
                viz_dir=None,
                return_samples=True,
                nocturne_compatible_only=False
                if cfg.dataset_name == "nuplan"
                else cfg.eval.sim_envs.nocturne_compatible_only,
            )

            for sample_id in samples:
                if mode == "initial_scene":
                    valid = check_scene_validity(samples[sample_id], cfg.dataset_name)
                    route, found_route = sample_route(samples[sample_id], cfg.dataset_name)
                    if found_route:
                        route_completed = get_lane_length(route) >= cfg.eval.sim_envs.route_length

                        tile_corners = np.array(
                            [
                                [cfg.dataset.fov / 2, cfg.dataset.fov / 2],
                                [-cfg.dataset.fov / 2, cfg.dataset.fov / 2],
                                [-cfg.dataset.fov / 2, -cfg.dataset.fov / 2],
                                [cfg.dataset.fov / 2, -cfg.dataset.fov / 2],
                            ]
                        )
                        tile_occupancy = [tile_corners]

                        if valid:
                            data = clean_up_scene(samples[sample_id], cfg.dataset_name, mode)
                            data["route"] = route
                            data["route_completed"] = route_completed
                            data["tile_occupancy"] = tile_occupancy

                            # --- static png (debug/overview) ---
                            if cfg.eval.visualize:
                                agent_motion_pred = data.get("agent_motion", None)
                                plot_scene(
                                    data["agent_states"],
                                    data["road_points"],
                                    np.argmax(data["agent_types"], axis=1),
                                    np.argmax(data["lane_types"], axis=1)
                                    if cfg.dataset_name == "nuplan"
                                    else None,
                                    f"{it}_{sample_id}_{'PARTIAL' if not data['route_completed'] else 'COMPLETE'}.png",
                                    viz_root,
                                    return_fig=False,
                                    tile_occupancy=None,
                                    adaptive_limits=False,
                                    route=data["route"],
                                    agent_motion_pred=agent_motion_pred,
                                )

                            # --- record snapshot for video ---
                            _append_trace(sample_id, data)

                            filename = f"{sample_id}.pkl"
                            write_dir = complete_samples_dir if data["route_completed"] else partial_samples_dir
                            with open(os.path.join(write_dir, filename), "wb") as f_out:
                                pickle.dump(data, f_out)

                else:
                    check_1, check_2, check_3 = check_scene_validity_inpainting(samples[sample_id], cfg.dataset_name)
                    valid = check_1 and check_2 and check_3
                    route, found_route = sample_route(samples[sample_id], cfg.dataset_name)

                    num_failed_check_1 += int(not check_1)
                    num_failed_check_2 += int(not check_2)
                    num_failed_check_3 += int(not check_3)
                    num_failed_found_route += int(not found_route)

                    valid = valid and found_route
                    if valid:
                        with open(os.path.join(partial_samples_dir, f"{sample_id}.pkl"), "rb") as f_in:
                            current_env = pickle.load(f_in)

                        existing_route = current_env["route"]
                        tile_corners = np.array(
                            [
                                [cfg.dataset.fov / 2, cfg.dataset.fov / 2],
                                [-cfg.dataset.fov / 2, cfg.dataset.fov / 2],
                                [-cfg.dataset.fov / 2, -cfg.dataset.fov / 2],
                                [cfg.dataset.fov / 2, -cfg.dataset.fov / 2],
                            ]
                        )
                        _, last_heading = estimate_heading(existing_route)
                        transform_dict = {"center": existing_route[-1], "yaw": last_heading}
                        transformed_tile_corners = _transform_corners(tile_corners, transform_dict, cfg.dataset_name)
                        overlapping_tiles = _check_overlapping_tiles(transformed_tile_corners, current_env["tile_occupancy"])
                        num_failed_overlapping_tiles += int(overlapping_tiles)

                        if not overlapping_tiles:
                            data = clean_up_scene(samples[sample_id], cfg.dataset_name, mode)
                            data["route"] = route
                            data["tile_occupancy"] = [transformed_tile_corners]

                            if sample_id not in candidate_next_samples:
                                candidate_next_samples[sample_id] = []
                            candidate_next_samples[sample_id].append(data)

        if it > 0:
            print(f"Number of failed validity check 1: {num_failed_check_1}")
            print(f"Number of failed validity check 2: {num_failed_check_2}")
            print(f"Number of failed validity check 3: {num_failed_check_3}")
            print(f"Number of failed to find route: {num_failed_found_route}")
            print(f"Number of failed overlapping tiles: {num_failed_overlapping_tiles}")

            new_envs = {}
            for sample_id in candidate_next_samples:
                with open(os.path.join(partial_samples_dir, f"{sample_id}.pkl"), "rb") as f_in:
                    current_env = pickle.load(f_in)

                sampled_candidate = _sample_candidate(candidate_next_samples[sample_id], cfg.dataset_name)
                new_env = _extend_simulation_environment(
                    current_env,
                    sampled_candidate,
                    cfg.eval.sim_envs.route_length,
                    cfg.dataset_name,
                )
                new_envs[sample_id] = new_env

            # clear partial dir
            for filename in os.listdir(partial_samples_dir):
                os.remove(os.path.join(partial_samples_dir, filename))

            # write new envs
            for sample_id in new_envs:
                if len(os.listdir(complete_samples_dir)) >= cfg.eval.num_samples:
                    break

                filename = f"{sample_id}.pkl"
                write_dir = complete_samples_dir if new_envs[sample_id]["route_completed"] else partial_samples_dir
                with open(os.path.join(write_dir, filename), "wb") as f_out:
                    pickle.dump(new_envs[sample_id], f_out)

                # record snapshot
                _append_trace(sample_id, new_envs[sample_id])

            # static png (overview)
            if cfg.eval.visualize:
                for sample_id in new_envs:
                    data_env = new_envs[sample_id]
                    agent_motion_pred = data_env.get("agent_motion", None)
                    plot_scene(
                        data_env["agent_states"],
                        data_env["road_points"],
                        np.argmax(data_env["agent_types"], axis=1),
                        np.argmax(data_env["lane_types"], axis=1)
                        if cfg.dataset_name == "nuplan"
                        else None,
                        f"{it}_{sample_id}_{'PARTIAL' if not data_env['route_completed'] else 'COMPLETE'}.png",
                        viz_root,
                        return_fig=False,
                        tile_occupancy=data_env["tile_occupancy"],
                        adaptive_limits=True,
                        route=data_env["route"],
                        agent_motion_pred=agent_motion_pred,
                    )

        it += 1

    print(f"Generation completed. Generated {len(os.listdir(complete_samples_dir))} simulation environments.")

    # ------------------------------------------------------------------
    # NEW: Export web-friendly typed-array bundles for GitHub Pages viewer
    # ------------------------------------------------------------------
    sim_envs_cfg = getattr(cfg.eval, "sim_envs", None)
    web_cfg = getattr(sim_envs_cfg, "web_export", None) if sim_envs_cfg is not None else None
    web_enabled = bool(getattr(web_cfg, "enabled", False)) if web_cfg is not None else False

    if web_enabled:
        from vectorworld.utils.web_scene_export import export_sim_env_pkl_to_web_dir

        out_dir = str(getattr(web_cfg, "out_dir", os.path.join(save_dir, "web_sim_envs")))
        lane_width_m = float(getattr(web_cfg, "lane_width_m", 4.2))
        route_every_m = float(getattr(web_cfg, "route_every_m", 1.0))
        drop_self_edges = bool(getattr(web_cfg, "drop_self_edges", True))

        export_sim_env_pkl_to_web_dir(
            pkl_dir=complete_samples_dir,
            out_dir=out_dir,
            dataset=str(cfg.dataset_name),
            run_name=str(cfg.eval.run_name),
            lane_width_m=lane_width_m,
            route_every_m=route_every_m,
            drop_self_edges=drop_self_edges,
        )
        print(f"[web_export] done. web bundles saved to: {out_dir}")

    # ----------------------------
    # Final: render rollout videos for complete envs
    # ----------------------------
    if video_enabled and len(viz_traces) > 0:
        try:
            from vectorworld.utils.sim_env_rollout_viz import render_sim_env_rollout_video
        except Exception as e:
            print(f"[sim_env_viz] Failed to import sim_env_rollout_viz.py: {e}")
            return

        complete_files = sorted([f for f in os.listdir(complete_samples_dir) if f.endswith(".pkl")])
        # cap to requested num_samples
        complete_files = complete_files[: int(cfg.eval.num_samples)]

        video_root = os.path.join(viz_root, "rollout_videos")
        os.makedirs(video_root, exist_ok=True)

        ldm_type = str(getattr(getattr(cfg, "model", {}), "ldm_type", "ldm"))
        run_name = str(getattr(getattr(cfg, "eval", {}), "run_name", ""))

        rendered = 0
        for fn in complete_files:
            sid = os.path.splitext(fn)[0]
            if sid not in viz_traces or len(viz_traces[sid]) == 0:
                continue

            out_dir = os.path.join(video_root, sid)
            os.makedirs(out_dir, exist_ok=True)

            try:
                mp4_path = render_sim_env_rollout_video(
                    snapshots=viz_traces[sid],
                    output_dir=out_dir,
                    dataset_name=str(cfg.dataset_name),
                    run_name=run_name,
                    ldm_type=ldm_type,
                    fps=fps,
                    filmstrip_k=filmstrip_k,
                    view_size_m=view_size_m,
                    pan_frames=pan_frames,
                    replay_interp=replay_interp,
                    replay_hold_first=hold_first,
                    replay_hold_last=hold_last,
                    motion_viz_mode=motion_viz_mode,
                    show_agent_ids=show_agent_ids,
                    road_grid_res=road_grid_res,
                    dpi=dpi,
                    delete_frames=False,
                )
                print(f"[sim_env_viz] rollout video saved: {mp4_path}")
                rendered += 1
            except Exception as e:
                print(f"[sim_env_viz] Failed to render video for {sid}: {e}")

        print(f"[sim_env_viz] rendered {rendered}/{len(complete_files)} rollout videos to: {video_root}")

def _transform_scene(
    agents,
    lanes,
    route,
    transform_dict,
    dataset,
    agent_motion=None,
):
    """Transform agents, lanes, and route from local tile frame to global frame.

    Note:
    -Agents /lanes / root will do it according to transform_dictSE(2)(b) Conversion;
    - Agent_motion, if provided, is considered as **coordinate system perlyline**, not rotated with tile (copy only).
    """
    yaw_offset = np.pi / 2 if dataset == "waymo" else 0
    yaw = transform_dict["yaw"]
    angle_of_rotation = normalize_angle(yaw - yaw_offset)
    translation = transform_dict["center"]

    rotation_matrix = np.array(
        [
            [np.cos(angle_of_rotation), -np.sin(angle_of_rotation)],
            [np.sin(angle_of_rotation), np.cos(angle_of_rotation)],
        ]
    )

    new_agents = np.zeros_like(agents)
    new_agents[:, :2] = np.dot(agents[:, :2], rotation_matrix.T) + translation
    cos_theta = agents[:, 3]
    sin_theta = agents[:, 4]
    theta = np.arctan2(sin_theta, cos_theta)
    new_theta = normalize_angle(theta + angle_of_rotation)
    new_agents[:, 2] = agents[:, 2]
    new_agents[:, 3] = np.cos(new_theta)
    new_agents[:, 4] = np.sin(new_theta)
    new_agents[:, 5:] = agents[:, 5:]

    new_lanes = np.dot(lanes.reshape(-1, 2), rotation_matrix.T) + translation
    new_lanes = new_lanes.reshape(lanes.shape)

    new_route = np.dot(route, rotation_matrix.T) + translation

    new_motion = None
    if agent_motion is not None:
        new_motion = np.asarray(agent_motion).copy()

    return new_agents, new_lanes, new_route, new_motion


def _extend_simulation_environment(current_env, new_tile, target_route_length, dataset):
    """Extend the current simulation environment with the new inpainted tile.

    - Yeah, new_tile after-partitionAt the same time, agents`agent_states`and`agent_motion`;
    -If the current_env / new_tile contains`agent_motion`, and receive the global environment`agent_motion`Here we go.
    """
    existing_route = current_env["route"]
    _, last_heading = estimate_heading(existing_route)
    transform_dict = {"center": existing_route[-1], "yaw": last_heading}

    after_partition_agent_mask = ~new_tile["agent_mask"].astype(bool)
    after_partition_lane_mask = ~new_tile["lane_mask"].astype(bool)

    new_tile_agents = new_tile["agent_states"][after_partition_agent_mask]
    new_tile_lanes = new_tile["road_points"][after_partition_lane_mask]
    new_tile_route = new_tile["route"]
    if "agent_motion" in new_tile:
        new_tile_motion = new_tile["agent_motion"][after_partition_agent_mask]
    else:
        new_tile_motion = None

    new_agents, new_lanes, new_route, new_motion = _transform_scene(
        new_tile_agents,
        new_tile_lanes,
        new_tile_route,
        transform_dict,
        dataset,
        agent_motion=new_tile_motion,
    )

    G_current_env = get_networkx_lane_graph(current_env)[0]
    old_num_agents = current_env["num_agents"]

    current_env["agent_states"] = np.concatenate(
        [current_env["agent_states"], new_agents], axis=0
    )
    current_env["road_points"] = np.concatenate(
        [current_env["road_points"], new_lanes], axis=0
    )

    after_partition_lane_ids_new_tile = np.arange(len(new_tile["road_points"]))[
        after_partition_lane_mask
    ]
    before_partition_lane_ids_new_tile = np.arange(len(new_tile["road_points"]))[
        new_tile["lane_mask"].astype(bool)
    ]

    num_new_lanes = len(new_lanes)
    for i in range(num_new_lanes):
        G_current_env.add_node(current_env["num_lanes"] + i)

    new_tile_id_to_current_env_id = {}
    for lane_id in range(len(before_partition_lane_ids_new_tile)):
        new_tile_id_to_current_env_id[lane_id] = int(new_tile["lane_ids"][lane_id])
    for i, lane_id in enumerate(after_partition_lane_ids_new_tile):
        new_tile_id_to_current_env_id[lane_id] = current_env["num_lanes"] + i

    road_connection_types_map_current_env = {}
    road_connection_types_current_env = current_env["road_connection_types"]
    l2l_edge_index_current_env = get_edge_index_complete_graph(
        current_env["num_lanes"]
    ).transpose(1, 0)
    for i, edge in enumerate(l2l_edge_index_current_env):
        road_connection_types_map_current_env[
            (edge[0].item(), edge[1].item())
        ] = np.argmax(road_connection_types_current_env[i])

    road_connection_types_map_new_tile = {}
    road_connection_types_new_tile = new_tile["road_connection_types"]
    l2l_edge_index_new_tile = get_edge_index_complete_graph(
        len(new_tile["road_points"])
    ).transpose(1, 0)
    for i, edge in enumerate(l2l_edge_index_new_tile):
        road_connection_types_map_new_tile[
            (
                new_tile_id_to_current_env_id[edge[0].item()],
                new_tile_id_to_current_env_id[edge[1].item()],
            )
        ] = np.argmax(road_connection_types_new_tile[i])

    num_current_env_lanes = current_env["num_lanes"]
    num_augmented_env_lanes = num_current_env_lanes + num_new_lanes
    l2l_edge_index_augmented_env = get_edge_index_complete_graph(
        num_augmented_env_lanes
    ).transpose(1, 0)
    new_road_connection_types = np.zeros(len(l2l_edge_index_augmented_env))

    for i, edge in enumerate(l2l_edge_index_augmented_env):
        src = edge[0].item()
        dst = edge[1].item()

        if (src, dst) in road_connection_types_map_current_env:
            new_road_connection_types[i] = road_connection_types_map_current_env[
                (src, dst)
            ]
        elif (src, dst) in road_connection_types_map_new_tile:
            new_road_connection_types[i] = road_connection_types_map_new_tile[
                (src, dst)
            ]
        else:
            continue

    current_env["road_connection_types"] = np.eye(6 if dataset == "waymo" else 4)[
        new_road_connection_types.astype(int)
    ]
    current_env["agent_types"] = np.concatenate(
        [current_env["agent_types"], new_tile["agent_types"][after_partition_agent_mask]],
        axis=0,
    )
    if dataset == "nuplan":
        current_env["lane_types"] = np.concatenate(
            [current_env["lane_types"], new_tile["lane_types"][after_partition_lane_mask]],
            axis=0,
        )

    current_env["route"] = np.concatenate([existing_route, new_route], axis=0)

    # Update Agent_motion
    if new_motion is not None:
        if "agent_motion" in current_env:
            current_env["agent_motion"] = np.concatenate(
                [current_env["agent_motion"], new_motion], axis=0
            )
        else:
            zeros_old = np.zeros(
                (old_num_agents, new_motion.shape[1]), dtype=new_motion.dtype
            )
            current_env["agent_motion"] = np.concatenate(
                [zeros_old, new_motion], axis=0
            )

    current_env["num_agents"] = len(current_env["agent_states"])
    current_env["num_lanes"] = len(current_env["road_points"])
    current_env["route_completed"] = (
        get_lane_length(current_env["route"]) >= target_route_length
    )
    current_env["tile_occupancy"].extend(new_tile["tile_occupancy"])

    return current_env


def _sample_candidate(candidates, dataset):
    """Sample one candidate extension (prefer those containing vehicles)."""
    candidates_with_vehicles = []
    for candidate in candidates:
        num_vehicles = (candidate["agent_types"] == 0).astype(int).sum()
        if num_vehicles > 0:
            candidates_with_vehicles.append(candidate)
    if len(candidates_with_vehicles) == 0:
        sampled_candidate = random.sample(candidates, 1)[0]
    else:
        sampled_candidate = random.sample(candidates_with_vehicles, 1)[0]
    return sampled_candidate


def _near_border(pos, fov=64, threshold=1):
    """Check if position is near border of FOV."""
    if np.abs(np.abs(pos[0]) - fov / 2) < threshold or np.abs(
        np.abs(pos[1]) - fov / 2
    ) < threshold:
        return True
    return False


def _near_partition(pos, dataset, threshold=2.5):
    """Check if position is near partition (y=0 or x=0)."""
    IDX = 1 if dataset == "waymo" else 0
    return np.abs(pos[IDX]) < threshold


def _valid_route_end(lane_id, lane, fov=64, border_threshold=1, heading_threshold=5 * np.pi / 180):
    """Check if route end is valid (heading aligned with border)."""
    _, last_heading = estimate_heading(lane)
    last_pos = lane[-1]
    if np.abs(last_pos[1] - fov / 2) < border_threshold:
        target_angle = np.pi / 2
    elif np.abs(last_pos[1] + fov / 2) < border_threshold:
        target_angle = -np.pi / 2
    elif np.abs(last_pos[0] - fov / 2) < border_threshold:
        target_angle = 0.0
    else:
        target_angle = np.pi
    differences = last_heading - target_angle
    normalized_differences = normalize_angle(differences)
    closest_difference = np.abs(normalized_differences)
    return closest_difference <= heading_threshold


def _transform_corners(corners, transform_dict, dataset):
    """Apply rotation and translation to corners."""
    yaw = transform_dict["yaw"]
    angle_offset = np.pi / 2 if dataset == "waymo" else 0
    angle_of_rotation = normalize_angle(yaw - angle_offset)
    translation = transform_dict["center"]
    rotation_matrix = np.array(
        [
            [np.cos(angle_of_rotation), -np.sin(angle_of_rotation)],
            [np.sin(angle_of_rotation), np.cos(angle_of_rotation)],
        ]
    )
    rotated_corners = np.dot(corners, rotation_matrix.T)
    return rotated_corners + translation


def _check_overlapping_tiles(new_tile_corners, existing_tiles, ignore_last_n=3):
    """Check if new tile overlaps with existing tiles."""
    overlapping = False
    for corners in existing_tiles[:-ignore_last_n]:
        if is_colliding(new_tile_corners, corners):
            overlapping = True
            break
    return overlapping


# ======================================================================
# Memory layout scene generation: parallelLDM + tile-stitching + postprocess
# ======================================================================
def generate_simulation_environments_in_memory(
    model,
    cfg_ldm,
    num_envs: int,
    dataset: str = "waymo",
    route_length: int | None = None,
):
    """Generates a number of post_env scenes in memory (long-route tile)-stitch) editions.

    Design elements
    --------
    1. To give priority to generating a complete set of root_len roote_lenth (complete_envs_pre);
    If there is no complete scene after multiple rounds of indigence_scene + inputing,
       But the longer partial_env, the second thing to do, uses the partial_env,
       And rely on the roote_legth_clipped inside the postprocess_sim_env;
    Not even partial_env (LDM(a) The return to the empty list in extreme cases of total degradation;
    4. Online scene generation ** No longer abandoned scenes due to lack of access to roote_lane_indices**
       It's keeping these post_env, allowing IDMPolicy to search in the simulation phase for line path based on root point dynamics.
    """
    if route_length is None:
        route_length = int(cfg_ldm.eval.sim_envs.route_length)

    overhead_factor = int(cfg_ldm.eval.sim_envs.overhead_factor)
    num_inpainting_candidates = int(cfg_ldm.eval.sim_envs.num_inpainting_candidates)
    nocturne_compatible_only = (
        False
        if dataset == "nuplan"
        else bool(cfg_ldm.eval.sim_envs.nocturne_compatible_only)
    )

    fov = float(cfg_ldm.dataset.fov)

    # Sample at most num_envs* overhead_factor
    max_num_samples = max(int(num_envs * overhead_factor), num_envs)
    complete_envs_pre: list[dict] = []         # Full pre_env list of already root
    partial_envs: dict[str, dict] = {}         # The sample_id-> not yet full of roote_legth pre-env

    it = 0
    max_outer_iters = max(20, num_envs * 3)

    while len(complete_envs_pre) < num_envs and it < max_outer_iters:
        if it == 0:
            mode = "initial_scene"
            num_iters = 1
            num_samples = max_num_samples
        else:
            mode = "inpainting"
            num_iters = num_inpainting_candidates
            num_samples = len(partial_envs)
            if num_samples == 0:
                # No public_env scalable, early end
                break

        # --- single wheel (intical_scene or inputing) ---
        candidate_next_samples: dict[str, list[dict]] = {}

        for _ in range(num_iters):
            if mode == "initial_scene":
                samples = model.generate(
                    mode="initial_scene",
                    num_samples=num_samples,
                    batch_size=cfg_ldm.eval.batch_size,
                    cache_samples=False,
                    visualize=False,
                    conditioning_path=None,
                    cache_dir=None,
                    viz_dir=None,
                    return_samples=True,
                    nocturne_compatible_only=nocturne_compatible_only,
                )
            else:
                if not partial_envs:
                    break
                # Intenting on current partial_envs
                cond_scenes = dict(partial_envs)  # sample_id -> pre_env
                samples = model.generate(
                    mode="inpainting",
                    num_samples=len(cond_scenes),
                    batch_size=cfg_ldm.eval.batch_size,
                    cache_samples=False,
                    visualize=False,
                    conditioning_path=None,
                    cache_dir=None,
                    viz_dir=None,
                    return_samples=True,
                    nocturne_compatible_only=nocturne_compatible_only,
                    conditioning_scenes=cond_scenes,
                )

            for sample_id, sample_env in samples.items():
                if mode == "initial_scene":
                    # 1) Fundamental Geometric Legitimacy
                    valid = check_scene_validity(sample_env, dataset)
                    # 2) Can we get a route?
                    route, found_route = sample_route(sample_env, dataset)
                    if not found_route:
                        continue
                    route_completed = get_lane_length(route) >= route_length

                    tile_corners = np.array(
                        [
                            [fov / 2, fov / 2],
                            [-fov / 2, fov / 2],
                            [-fov / 2, -fov / 2],
                            [fov / 2, -fov / 2],
                        ]
                    )
                    tile_occupancy = [tile_corners]

                    if valid:
                        data_env = clean_up_scene(sample_env, dataset, mode)
                        data_env["route"] = route
                        data_env["route_completed"] = route_completed
                        data_env["tile_occupancy"] = tile_occupancy

                        if route_completed:
                            complete_envs_pre.append(data_env)
                            if len(complete_envs_pre) >= num_envs:
                                break
                        else:
                            partial_envs[sample_id] = data_env
                else:
                    # ---inputing mode: trying to extend the partial_envs
                    check_1, check_2, check_3 = check_scene_validity_inpainting(
                        sample_env, dataset
                    )
                    valid = check_1 and check_2 and check_3
                    route, found_route = sample_route(sample_env, dataset)
                    if not (valid and found_route):
                        continue

                    current_env = partial_envs.get(sample_id, None)
                    if current_env is None:
                        continue
                    existing_route = current_env["route"]

                    tile_corners = np.array(
                        [
                            [fov / 2, fov / 2],
                            [-fov / 2, fov / 2],
                            [-fov / 2, -fov / 2],
                            [fov / 2, -fov / 2],
                        ]
                    )
                    _, last_heading = estimate_heading(existing_route)
                    transform_dict = {
                        "center": existing_route[-1],
                        "yaw": last_heading,
                    }
                    transformed_tile_corners = _transform_corners(
                        tile_corners, transform_dict, dataset
                    )
                    overlapping_tiles = _check_overlapping_tiles(
                        transformed_tile_corners, current_env["tile_occupancy"]
                    )
                    if overlapping_tiles:
                        continue

                    data_env = clean_up_scene(sample_env, dataset, mode)
                    data_env["route"] = route
                    data_env["tile_occupancy"] = [transformed_tile_corners]

                    candidate_next_samples.setdefault(sample_id, []).append(data_env)

            if mode == "initial_scene" and len(complete_envs_pre) >= num_envs:
                break

        if mode == "initial_scene":
            if len(complete_envs_pre) >= num_envs:
                break
        else:
            # If this round fails to extend effectively any partial_env,
            # Do not empty partial_envs, jump out of while, back with partial_envs.
            if not candidate_next_samples:
                break

            # Select one of the best from the candidate_next_samples for each party, and extend
            new_envs_pre: dict[str, dict] = {}
            for sample_id, cand_list in candidate_next_samples.items():
                current_env = partial_envs.get(sample_id, None)
                if current_env is None:
                    continue
                sampled_cand = _sample_candidate(cand_list, dataset)
                new_env = _extend_simulation_environment(
                    current_env, sampled_cand, route_length, dataset
                )
                new_envs_pre[sample_id] = new_env

            # Update partial_envs/ complete_envs_pre
            partial_envs = {}
            for sid, env_pre in new_envs_pre.items():
                if env_pre.get("route_completed", False):
                    complete_envs_pre.append(env_pre)
                    if len(complete_envs_pre) >= num_envs:
                        break
                else:
                    partial_envs[sid] = env_pre

        it += 1

    # If, after many rounds, there is no complete scene, but there is partial_envs, the next step is to step back.
    if len(complete_envs_pre) == 0 and partial_envs:
        for _, env_pre in list(partial_envs.items())[:num_envs]:
            complete_envs_pre.append(env_pre)

    # - > Post_env (uniform postprocess_sim_env)
    # Online generation is no longer discarded because of the failure of roote_lane_indices;
    # root_lane_indices only asIDMOne optimized branch use.
    post_envs: list[dict] = []
    for pre_env in complete_envs_pre:
        post = postprocess_sim_env(
            pre_env, route_length=route_length, dataset=dataset
        )
        # Whether or not the root_lane_indices is None, keep the scene
        post_envs.append(post)
        if len(post_envs) >= num_envs:
            break

    return post_envs

def generate_single_sim_env(model, cfg_ldm, dataset: str = "waymo", route_length: int | None = None):
    """Generates a single post_env scene (online mode).

    If internal generation fails, it will be thrown out of RuntimeError.
    """
    envs = generate_simulation_environments_in_memory(
        model=model,
        cfg_ldm=cfg_ldm,
        num_envs=1,
        dataset=dataset,
        route_length=route_length,
    )
    if not envs:
        raise RuntimeError("Failed to generate a valid simulation environment.")
    return envs[0]