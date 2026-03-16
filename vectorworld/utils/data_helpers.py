import numpy as np
import torch
np.set_printoptions(suppress=True)
from vectorworld.utils.data_container import get_batches, get_features
from typing import Tuple, Any, Dict, List
from configs.config import PARTITIONED
import os
import pickle


def extract_raw_waymo_data(agents_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert the list-of-dict agent format from Waymo to flat arrays.

    Parameters
    ----------
    agents_data
        List where each element corresponds to a single agent and
        replicates Waymo's *per-time-step* trajectory dictionaries.

    Returns
    -------
    agent_data
        Array with shape ``(num_agents, T, 8)`` containing position
        ``(x, y)``, velocity ``(vx, vy)``, heading *(rad)*, length,
        width and existence mask for each time-step ``T``.
    agent_types
        One-hot encoded array of shape ``(num_agents, 5)`` for
        ``{"unset": 0, "vehicle": 1, "pedestrian": 2, "cyclist": 3, "other": 4}``.
    """
    
    # Get indices of non-parked cars and cars that exist for the entire episode
    agent_data = []
    agent_types = []

    for n in range(len(agents_data)):
        # Position ---------------------------------------------------
        ag_position = agents_data[n]['position']
        x_values = [entry['x'] for entry in ag_position]
        y_values = [entry['y'] for entry in ag_position]
        ag_position = np.column_stack((x_values, y_values))
        
        # Heading (unwrap to (‑pi, pi]) ------------------------------
        ag_heading = np.radians(np.array(agents_data[n]['heading']).reshape((-1, 1)))
        ag_heading = np.mod(ag_heading + np.pi, 2 * np.pi) - np.pi
        
        # Velocity ---------------------------------------------------
        ag_velocity = agents_data[n]['velocity']
        x_values = [entry['x'] for entry in ag_velocity]
        y_values = [entry['y'] for entry in ag_velocity]
        ag_velocity = np.column_stack((x_values, y_values))
        
        # Existence & size -----------------------------------------
        ag_existence = np.array(agents_data[n]['valid']).reshape((-1, 1))
        ag_length = np.ones((len(ag_position), 1)) * agents_data[n]['length']
        ag_width = np.ones((len(ag_position), 1)) * agents_data[n]['width']
        
        # Pack -------------------------------------------------------
        agent_type = get_object_type_onehot_waymo(agents_data[n]['type'])
        ag_state = np.concatenate((ag_position, ag_velocity, ag_heading, ag_length, ag_width, ag_existence), axis=-1)
        agent_data.append(ag_state)
        agent_types.append(agent_type)
    
    # convert to numpy array
    agent_data = np.array(agent_data)
    agent_types = np.array(agent_types)
    
    return agent_data, agent_types


def add_batch_dim(arr):
    return np.expand_dims(arr, axis=0)


def get_object_type_onehot_waymo(agent_type):
    """Return the one-hot NumPy vector encoding of an agent type."""
    agent_types = {"unset": 0, "vehicle": 1, "pedestrian": 2, "cyclist": 3, "other": 4}
    return np.eye(len(agent_types))[agent_types[agent_type]]


def get_lane_connection_type_onehot_waymo(lane_connection_type):
    """Return the one-hot NumPy vector encoding of a lane-connection type."""
    lane_connection_types = {"none": 0, "pred": 1, "succ": 2, "left": 3, "right": 4, "self": 5}
    return np.eye(len(lane_connection_types))[lane_connection_types[lane_connection_type]]


def get_lane_connection_type_onehot_nuplan(lane_connection_type):
    """Converts a lane connection type to a one-hot encoded vector."""
    lane_connection_types = {"none": 0, "pred": 1, "succ": 2, "self": 3}
    return np.eye(len(lane_connection_types))[lane_connection_types[lane_connection_type]]


def get_lane_type_onehot_nuplan(lane_type):
    """Converts a lane type to a one-hot encoded vector."""
    lane_types = {"lane": 0, "green_light": 1, "red_light": 2}
    return np.eye(len(lane_types))[lane_types[lane_type]]


def get_object_type_onehot_nuplan(agent_type):
    """Converts an agent type to a one-hot encoded vector."""
    agent_types = {"vehicle": 0, "pedestrian": 1, "static_object": 2}
    return np.eye(len(agent_types))[agent_types[agent_type]]


def reorder_indices(
        agent_mu: np.ndarray,
        agent_log_var: np.ndarray,
        lane_mu: np.ndarray,
        lane_log_var: np.ndarray,
        edge_index_lane_to_lane: np.ndarray,
        agent_states: np.ndarray,
        road_points: np.ndarray,
        lg_type: int,
        tolerance: float = 0.5 / 32,
        dataset: str = 'waymo'
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
    """Reorder agents and lanes to ensure deterministic ordering. This makes the positional
    encodings more meaningful.

    The routine performs a **hierarchical sort** on non-ego agents and on road
    lanes over the following metrics in the prescribed order: [min_y, min_x, max_y, max_x]

    A *tolerance* is applied so that small positional differences do not change the order.  After sorting, all latent
    representations, state tensors, and graph indices are permuted
    consistently.  The ego agent (index 0) is **never moved**.

    Parameters
    ----------
    agent_mu : np.ndarray
        Mean of the Gaussian latent variables for each agent with shape
        ``(N_agents, latent_dim)``.
    agent_log_var : np.ndarray
        Log-variance of the Gaussian latent variables for each agent with the
        same shape as *agent_mu*.
    lane_mu : np.ndarray
        Mean of the Gaussian latent variables for each lane with shape
        ``(N_lanes, latent_dim)``.
    lane_log_var : np.ndarray
        Log-variance of the Gaussian latent variables for each lane with the
        same shape as *lane_mu*.
    edge_index_lane_to_lane : np.ndarray
        Edge list of the lane-to-lane graph in COO format with shape
        ``(2, E)`` or ``(E, 2)``.  Indices are updated to reflect the new lane
        order.
    agent_states : np.ndarray
        Full state tensor for agents used to derive the sort keys.
    road_points : np.ndarray
        Sampled poly-line points for each lane with shape
        ``(N_lanes, N_points, 2)``.
    lg_type : int
        Scene layout type.  If ``lg_type == 1`` the function marks agents and
        lanes that are south of the horizontal partition (``y <= 0``).
        Otherwise no partitioning mask is applied.
    tolerance : float, optional
        Numerical tolerance (in the same units as coordinates) within which
        metric differences are considered equal.  Defaults to ``0.5 / 32``
        (≈0.0156).
    dataset : str, optional
        Either waymo or nuplan, which determines orientation of scene and therefore recursive ordering

    Returns
    -------
    Tuple[np.ndarray, ...]
        A 7-tuple containing:

        1. **agent_mu_sorted** - permuted *agent_mu* with ego agent first.
        2. **agent_log_var_sorted** - permuted *agent_log_var*.
        3. **lane_mu_sorted** - permuted *lane_mu*.
        4. **lane_log_var_sorted** - permuted *lane_log_var*.
        5. **edge_index_lane_to_lane_new** - updated edge indices.
        6. **agent_partition_mask** - boolean mask of shape ``(N_agents,)``
        indicating agents below the ``y=0`` partition when
        ``lg_type == 1``.
        7. **lane_partition_mask** - boolean mask of shape ``(N_lanes,)``
        indicating lanes below the partition when ``lg_type == 1``.

    Notes
    -----
    • The sorting of agents excludes the ego agent (index ``0``), which is
    re-inserted at the head of every returned tensor.

    • When *road_points* is empty (``shape[0] == 0``) the lane-related outputs
    are returned unchanged.
    """
    
    def hierarchical_sort(values, metrics, tolerance):
        """
        Recursively sorts indices based on a list of metrics and a tolerance.
        """
        indices = np.arange(len(values[metrics[0]]))
        
        def sort_recursive(indices, metric_idx):
            if len(indices) == 0:
                return indices  # Return empty array if no indices to sort
            if metric_idx >= len(metrics):
                return indices
            
            metric = metrics[metric_idx]
            values_metric = values[metric][indices]
            sorted_order = np.argsort(values_metric)
            indices = indices[sorted_order]
            values_metric_sorted = values_metric[sorted_order]
            
            # Group indices where the difference is less than tolerance
            groups = []
            current_group = [indices[0]]
            for i in range(1, len(indices)):
                diff = values_metric_sorted[i] - values_metric_sorted[i - 1]
                if diff < tolerance:
                    current_group.append(indices[i])
                else:
                    # Recursively sort the current group if needed
                    if len(current_group) > 1:
                        current_group = sort_recursive(np.array(current_group), metric_idx + 1).tolist()
                    groups.extend(current_group)
                    current_group = [indices[i]]
            # Handle the last group
            if len(current_group) > 1:
                current_group = sort_recursive(np.array(current_group), metric_idx + 1).tolist()
            groups.extend(current_group)
            return np.array(groups)
        
        return sort_recursive(indices, 0)
    
    if dataset == 'waymo':
        PARTITION_IDX = 1  # y-axis partition for Waymo 
    else:
        PARTITION_IDX = 0 # x-axis partition for Nuplan
    
    # Process Agents (ego is first index)
    non_ego_agent_mu = agent_mu[1:]
    non_ego_agent_log_var = agent_log_var[1:]
    non_ego_agent_states = agent_states[1:]
    
    if non_ego_agent_states.shape[0] > 0:
        # Calculate metrics for agents
        agent_min_y = non_ego_agent_states[:, 1]
        agent_min_x = non_ego_agent_states[:, 0]
        agent_max_y = non_ego_agent_states[:, 1]
        agent_max_x = non_ego_agent_states[:, 0]
        
        agent_values = {
            'min_y': agent_min_y,
            'min_x': agent_min_x,
            'max_y': agent_max_y,
            'max_x': agent_max_x
        }
        
        if dataset == 'waymo':
            agent_metrics = ['min_y', 'min_x', 'max_y', 'max_x']
        else:
            agent_metrics = ['min_x', 'min_y', 'max_x', 'max_y']
        perm = hierarchical_sort(agent_values, agent_metrics, tolerance)
        
        # Reorder non-ego agents
        non_ego_agent_mu = non_ego_agent_mu[perm]
        non_ego_agent_log_var = non_ego_agent_log_var[perm]
        non_ego_agent_states = non_ego_agent_states[perm]
    
    # Concatenate ego agent back
    agent_mu = np.concatenate([agent_mu[:1], non_ego_agent_mu], axis=0)
    agent_log_var = np.concatenate([agent_log_var[:1], non_ego_agent_log_var], axis=0)
    agent_states_sorted = np.concatenate([agent_states[:1], non_ego_agent_states], axis=0)

    # which agents are below the partition
    if lg_type == PARTITIONED:
        agent_partition_mask = agent_states_sorted[:, PARTITION_IDX] <= 0
    else:
        agent_partition_mask = np.zeros_like(agent_states_sorted[:, PARTITION_IDX] <= 0)
    
    if road_points.shape[0] > 0:
        lane_min_y = np.min(road_points[:, :, 1], axis=1)
        lane_min_x = np.min(road_points[:, :, 0], axis=1)
        lane_max_y = np.max(road_points[:, :, 1], axis=1)
        lane_max_x = np.max(road_points[:, :, 0], axis=1)
        
        lane_values = {
            'min_y': lane_min_y,
            'min_x': lane_min_x,
            'max_y': lane_max_y,
            'max_x': lane_max_x
        }
        
        if dataset == 'waymo':
            lane_metrics = ['min_y', 'min_x', 'max_y', 'max_x']
        else:
            lane_metrics = ['min_x', 'min_y', 'max_x', 'max_y']
        lane_perm = hierarchical_sort(lane_values, lane_metrics, tolerance)
        
        # Reorder lanes
        lane_mu = lane_mu[lane_perm]
        lane_log_var = lane_log_var[lane_perm]

        road_points_sorted = road_points[lane_perm]
        # which roads are below the partition
        if lg_type == PARTITIONED:
            lane_partition_mask = road_points_sorted[:, 9, PARTITION_IDX] <= 0
        else:
            lane_partition_mask = np.zeros_like(road_points_sorted[:, 9, PARTITION_IDX] <= 0)
        
        # Update edge indices
        old_index_positions = np.argsort(lane_perm)
        edge_index_lane_to_lane_new = old_index_positions[edge_index_lane_to_lane]
    else:
        edge_index_lane_to_lane_new = edge_index_lane_to_lane  # No change if no lanes
        # no lanes
        lane_partition_mask = road_points[:, 9, PARTITION_IDX] <= 0
    
    return agent_mu, agent_log_var, lane_mu, lane_log_var, edge_index_lane_to_lane_new, agent_partition_mask, lane_partition_mask


def modify_agent_states(agent_states):
    """Canonicalise velocity & heading for neural consumption. All remaining trailing columns (if any) are copied verbatim.

    Parameters
    ----------
    agent_states : np.ndarray
        Float32 array of shape ``(N, D)`` where columns ``2-4`` are
        ``vx``, ``vy``, and ``yaw`` respectively.

    Returns
    -------
    new_agent_states : np.ndarray
        Array with the *same* shape ``(N, D)`` where columns ``2-4``
        have been replaced by ``speed``, ``cosθ``, ``sinθ``.
    """
    new_agent_states = np.zeros_like(agent_states)
    new_agent_states[:, :2] = agent_states[:, :2]
    new_agent_states[:, 5:] = agent_states[:, 5:]
    new_agent_states[:, 2] = np.sqrt(agent_states[:, 2] ** 2 + agent_states[:, 3] ** 2)
    new_agent_states[:, 3] = np.cos(agent_states[:, 4])
    new_agent_states[:, 4] = np.sin(agent_states[:, 4])

    return new_agent_states


def normalize_scene(
        agent_states: np.ndarray,
        road_points: np.ndarray,
        fov: float,
        min_speed: float,
        max_speed: float,
        min_length: float,
        max_length: float,
        min_width: float,
        max_width: float,
        min_lane_x: float,
        max_lane_x: float,
        min_lane_y: float,
        max_lane_y: float
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Min-max normalise agent and lane features into **[-1, 1]**."""
    
    # pos_x
    agent_states[:, 0] = 2 * ((agent_states[:, 0] - (-1 * fov/2))
                            / fov) - 1
    # pos_y
    agent_states[:, 1] = 2 * ((agent_states[:, 1] - (-1 * fov/2))
                            / fov) - 1
    # speed
    agent_states[:, 2] = 2 * ((agent_states[:, 2] - (min_speed))
                            / (max_speed - min_speed)) - 1
    # length
    agent_states[:, 5] = 2 * ((agent_states[:, 5] - (min_length))
                            / (max_length - min_length)) - 1
    # width
    agent_states[:, 6] = 2 * ((agent_states[:, 6] - (min_width))
                            / (max_width - min_width)) - 1
    
    # road pos_x
    road_points[:, :, 0] = 2 * ((road_points[:, :, 0] - (min_lane_x))
                            / (max_lane_x - min_lane_x)) - 1
    road_points[:, :, 1] = 2 * ((road_points[:, :, 1] - (min_lane_y))
                            / (max_lane_y - min_lane_y)) - 1

    return agent_states, road_points

def normalize_scene_with_motion(
        agent_states: np.ndarray,
        road_points: np.ndarray,
        motion_raw: np.ndarray | None,
        motion_is_static: np.ndarray | None,
        fov: float,
        min_speed: float,
        max_speed: float,
        min_length: float,
        max_length: float,
        min_width: float,
        max_width: float,
        min_lane_x: float,
        max_lane_x: float,
        min_lane_y: float,
        max_lane_y: float,
        motion_max_displacement: float = 12.0,
        sentinel_value: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize agent & lane features *and* polyline motion code.

    -Agent & lane features: and: func:`normalize_scene`Convergence (aligned to]-1I'm sorry, I'm sorry.
    -Motion code (new 6-point view):

        1. Input`motion_raw`Shape as ``(Na, 2K)``, K=num_points;
           Each pair (x_i, y_i) is a historical trajectory point (m) under the **agent body coordinates**;
        2. Vertical x-direction assumes the physical domain to be ``[-D, 0]``, through

               x_norm = 2 * (x / D) + 1

           Will [T]-DMap To [,0]-11];
        3. Horizontal y assuming physical domain is ``[-D/2, D/2]``, through

               y_norm = clip(y / (D/2), -1, 1)

           Map To []-1One.

    - It's still.`motion_raw`Upstream direct to 0, corresponding to x_norm=1, y_norm=0,
      The semantic meaning is "historic trajectory coincides with the current point".

    Parameters
    ----------
    motion_raw:
        (Na, 2K) Not unicoded point coordinates (body coordinates, unit metres).
        Returns the empty position_code of the shape (Na, 0) if it is None.
    motion_is_static:
        The static boolean mask is used only for compatibility with the old interface; the new implementation is no longer dependent on Sentinel.

    Returns
    -------
    agent_states_norm : (Na, 7)
    road_points_norm  : (Nl, P, 2)
    motion_code_norm  : (Na, 2K)
        Normalized Polyline Point Coordinates, all fallen [-1One.
    """
    # Standard static geometry first, noormalize
    agent_states, road_points = normalize_scene(
        agent_states,
        road_points,
        fov=fov,
        min_speed=min_speed,
        max_speed=max_speed,
        min_length=min_length,
        max_length=max_length,
        min_width=min_width,
        max_width=max_width,
        min_lane_x=min_lane_x,
        max_lane_x=max_lane_x,
        min_lane_y=min_lane_y,
        max_lane_y=max_lane_y,
    )

    num_agents = agent_states.shape[0]

    # Returns empty position code when no movement information is available (dimensional 0)
    if motion_raw is None:
        motion_code = np.zeros((num_agents, 0), dtype=np.float32)
        return agent_states, road_points, motion_code

    motion_raw = np.asarray(motion_raw, dtype=np.float32)
    if motion_raw.ndim != 2 or motion_raw.shape[0] != num_agents:
        raise ValueError(
            f"motion_raw shape mismatch: expected (Na, D), got {motion_raw.shape}"
        )

    D = motion_raw.shape[1]
    if D == 0:
        motion_code = motion_raw.astype(np.float32)
        return agent_states, road_points, motion_code

    assert (
        D % 2 == 0
    ), f"Expected motion_raw last dim to be even (pairs of 2D points), got {D}"

    # Vertical range D_longitudinal, horizontal range D_lateral = D_longitudinal / 2
    D_long = float(motion_max_displacement)
    D_lat = D_long / 2.0

    xs = motion_raw[:, 0::2]  # (Na, K)
    ys = motion_raw[:, 1::2]

    # [-D_long, 0] → [-1,1]: x_norm = 2* (x / D_long) + 1
    x_norm = 2.0 * xs / D_long + 1.0
    # [-D_lat, D_lat] → [-1,1]
    y_norm = ys / D_lat

    motion_code = np.empty_like(motion_raw, dtype=np.float32)
    motion_code[:, 0::2] = np.clip(x_norm, -1.0, 1.0)
    motion_code[:, 1::2] = np.clip(y_norm, -1.0, 1.0)

    return agent_states, road_points, motion_code

def unnormalize_scene(
        agent_states: np.ndarray,
        road_points: np.ndarray,
        fov: float,
        min_speed: float,
        max_speed: float,
        min_length: float,
        max_length: float,
        min_width: float,
        max_width: float,
        min_lane_x: float,
        max_lane_x: float,
        min_lane_y: float,
        max_lane_y: float
    ) -> Tuple[np.ndarray, np.ndarray]:
    """ Unnormalize the agent states and lane points from a range of [-1, 1] to their original scale based on the dataset configuration."""
    # pos_x
    agent_states[:, 0] = ((torch.clip(agent_states[:, 0], -1, 1) + 1) / 2) * fov + (-1 * fov/2)
    # pos_y
    agent_states[:, 1] = ((torch.clip(agent_states[:, 1], -1, 1) + 1) / 2) * fov + (-1 * fov/2)
    # speed
    agent_states[:, 2] = ((torch.clip(agent_states[:, 2], -1, 1) + 1) / 2) * (max_speed - min_speed) + min_speed
    # cos_theta
    agent_states[:, 3] = torch.clip(agent_states[:, 3], -1, 1)
    # sin_theta
    agent_states[:, 4] = torch.clip(agent_states[:, 4], -1, 1)
    # length
    agent_states[:, 5] = ((torch.clip(agent_states[:, 5], -1, 1) + 1) / 2) * (max_length - min_length) + min_length
    # width
    agent_states[:, 6] = ((torch.clip(agent_states[:, 6], -1, 1) + 1) / 2) * (max_width - min_width) + min_width

    lower_clip = -1000
    upper_clip = 1000
    
    # lane pos_x
    road_points[:, :, 0] = ((torch.clip(road_points[:, :, 0], lower_clip, upper_clip) + 1) / 2) * (max_lane_x - min_lane_x) + min_lane_x
    # lane pos_y
    road_points[:, :, 1] = ((torch.clip(road_points[:, :, 1], lower_clip, upper_clip) + 1) / 2) * (max_lane_y - min_lane_y) + min_lane_y

    return agent_states, road_points

def unnormalize_motion_code(
        motion_code_norm: torch.Tensor | np.ndarray,
        motion_max_displacement: float = 12.0,
    ) -> torch.Tensor | np.ndarray:
    """Normalized polyline motion code from []-11. Maps the physical coordinates (m).

    New semantic
    -----
    motion_code_norm : (..., 2K)

        - Every pair of them (x_i_norm, y_norm) [whispers]-1- One. - Two.
        - Vertical physical domain is assumed to be ``x_i ∈ [-D, 0]``:
              x_i = ((x_i_norm + 1) / 2) * D - D
        - Horizontal physical domain assumed to be ``y_i ∈ [-D/2, D/2]``:
              y_i = y_i_norm * (D/2)

        Here.`D = motion_max_displacement``, for example, 12 meters.

    Back
    ----
    Motion_phys: Same shape as input
        Each (x_i, y_i) is the polyline point coordinates (metres) under the **agent body coordinates**.
        In visualization/ emulation environments, rotates them to the global coordinate system through the current (x, y, yaw).
    """
    D_long = float(motion_max_displacement)
    D_lat = D_long / 2.0

    if isinstance(motion_code_norm, np.ndarray):
        arr = motion_code_norm.astype(np.float32, copy=False)
        if arr.size == 0:
            return arr

        if arr.ndim != 2:
            raise ValueError(
                f"Expected 2D motion_code_norm for numpy, got shape={arr.shape}"
            )
        D = arr.shape[1]
        assert D % 2 == 0, f"Expected even last dim, got {D}"

        xs_norm = np.clip(arr[:, 0::2], -1.0, 1.0)
        ys_norm = np.clip(arr[:, 1::2], -1.0, 1.0)

        xs = ((xs_norm + 1.0) * 0.5) * D_long - D_long  # [-1,1]→[-D,0]
        ys = ys_norm * D_lat                             # [-1,1]→[-D/2,D/2]

        out = np.empty_like(arr, dtype=np.float32)
        out[:, 0::2] = xs
        out[:, 1::2] = ys
        return out

    else:
        # assume torch.Tensor
        arr = torch.clamp(motion_code_norm, -1.0, 1.0)
        if arr.numel() == 0:
            return arr
        if arr.dim() < 2:
            raise ValueError(
                f"Expected motion_code_norm with last dim 2K, got shape={arr.shape}"
            )
        D = arr.shape[-1]
        assert D % 2 == 0, f"Expected even last dim, got {D}"

        xs_norm = arr[..., 0::2]
        ys_norm = arr[..., 1::2]

        xs = ((xs_norm + 1.0) * 0.5) * D_long
        xs = xs - D_long
        ys = ys_norm * D_lat

        out = torch.empty_like(arr)
        out[..., 0::2] = xs
        out[..., 1::2] = ys
        return out


def randomize_indices(
    agent_states,
    agent_types,
    road_points,
    edge_index_lane_to_lane,
    lane_types=None):
    """Randomly permute non-ego agents and lane order during training.

    NOTE: Only the **7D static state** is assumed here.
    """
    non_ego_agent_states = agent_states[1:]
    non_ego_agent_types = agent_types[1:]

    num_non_ego_agents = len(non_ego_agent_states)
    perm = np.arange(num_non_ego_agents)
    np.random.shuffle(perm)
    non_ego_agent_states = non_ego_agent_states[perm]
    non_ego_agent_types = non_ego_agent_types[perm]

    agent_states = np.concatenate([agent_states[:1], non_ego_agent_states], axis=0)
    agent_types = np.concatenate([agent_types[:1], non_ego_agent_types], axis=0)

    lane_perm = np.arange(len(road_points))
    np.random.shuffle(lane_perm)
    road_points = road_points[lane_perm]
    if lane_types is not None:
        lane_types = lane_types[lane_perm]
    
    # edge_index may arrive as torch.Tensor or np.ndarray
    edge_index_lane_to_lane = np.asarray(edge_index_lane_to_lane)
    old_index_positions = np.argsort(lane_perm)
    edge_index_lane_to_lane_new = old_index_positions[edge_index_lane_to_lane]

    if lane_types is not None:
        return agent_states, agent_types, road_points, lane_types, edge_index_lane_to_lane_new
    else:
        return agent_states, agent_types, road_points, edge_index_lane_to_lane_new


def randomize_indices_with_motion(
    agent_states: np.ndarray,
    agent_types: np.ndarray,
    motion_code: np.ndarray,
    road_points: np.ndarray,
    edge_index_lane_to_lane: np.ndarray,
    lane_types: np.ndarray | None = None
):
    """Randomly permute non-ego agents and lanes *and* keep motion code aligned.

    This is identical to :func:`randomize_indices` but also shuffles the
    per-agent motion code tensor.

    Parameters
    ----------
    agent_states : (Na, 7)
        Static agent state vectors.
    agent_types : (Na, 3)
        One-hot agent types.
    motion_code : (Na, 4)
        Normalized motion code.
    road_points : (Nl, P, 2)
        Lane polylines.
    edge_index_lane_to_lane : (E, 2) or (2, E)
        Lane-to-lane connectivity (numpy or torch).
    lane_types : Optional[(Nl, C)]
        Lane-type one-hot vectors, if present.

    Returns
    -------
    agent_states, agent_types, motion_code, road_points, [lane_types], edge_index_lane_to_lane_new
    """
    non_ego_agent_states = agent_states[1:]
    non_ego_agent_types = agent_types[1:]
    non_ego_motion = motion_code[1:]

    num_non_ego_agents = len(non_ego_agent_states)
    perm = np.arange(num_non_ego_agents)
    np.random.shuffle(perm)
    non_ego_agent_states = non_ego_agent_states[perm]
    non_ego_agent_types = non_ego_agent_types[perm]
    non_ego_motion = non_ego_motion[perm]

    agent_states = np.concatenate([agent_states[:1], non_ego_agent_states], axis=0)
    agent_types = np.concatenate([agent_types[:1], non_ego_agent_types], axis=0)
    motion_code = np.concatenate([motion_code[:1], non_ego_motion], axis=0)

    lane_perm = np.arange(len(road_points))
    np.random.shuffle(lane_perm)
    road_points = road_points[lane_perm]
    if lane_types is not None:
        lane_types = lane_types[lane_perm]

    edge_index_lane_to_lane = np.asarray(edge_index_lane_to_lane)
    old_index_positions = np.argsort(lane_perm)
    edge_index_lane_to_lane_new = old_index_positions[edge_index_lane_to_lane]

    if lane_types is not None:
        return agent_states, agent_types, motion_code, road_points, lane_types, edge_index_lane_to_lane_new
    else:
        return agent_states, agent_types, motion_code, road_points, edge_index_lane_to_lane_new
    

# ---------------------------------------------------------------------------
# Latent normalization helpers
# ---------------------------------------------------------------------------
def _to_tensor_stats(mean, std, ref_latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replace (mean, std) with`ref_latents`Tensor with device / dtype and do the necessary shape check.

    Type of input supported:
    - float / int
    - list / tuple / numpy.ndarray
    - torch.Tensor

    Design elements
    --------
    1. Ifmean/stdis the metric broadcast directly to all dimensions;
    2. Ifmean/stdYes 1D vector Required length=latet_dim(ref_latents.shape[-1I'm sorry, but I don't know.
       Otherwise, it would be wrong to avoid silent broadcasting leading to potential numerical problems;
    3. No change in the original broadcasting semantics: all work on (N,D) or (N,1,D).
    """
    # Convert to Tensor, put on the same device / dtype as ref_latents
    mean_t = torch.as_tensor(mean, dtype=ref_latents.dtype, device=ref_latents.device)
    std_t = torch.as_tensor(std, dtype=ref_latents.dtype, device=ref_latents.device)

    # If the user accidentally transmits a high-dimensional array (e.g. (1, D)), this is flat 1D, semantic equivalent.
    if mean_t.ndim > 1:
        mean_t = mean_t.view(-1)
    if std_t.ndim > 1:
        std_t = std_t.view(-1)

    # Check whether the length corresponds to the flatt_dim if the vector by dimension
    if mean_t.ndim == 1:
        latent_dim = ref_latents.shape[-1]
        if mean_t.shape[0] != latent_dim or std_t.shape[0] != latent_dim:
            raise ValueError(
                f"[normalize_latents] Latent stats length mismatch: "
                f"got mean/std of length ({mean_t.shape[0]}, {std_t.shape[0]}) "
                f"but latent_dim = {latent_dim}. "
                "Please verify that latent_stats.pkl matches the "
                "`agent_latent_dim` / `lane_latent_dim` in the current model configuration, "
                "and re-run `cache_latent_stats` if necessary."
            )

    # Numerical protection: std at least 1e-6, avoid eliminating zeros
    std_t = torch.clamp(std_t, min=1e-6)
    return mean_t, std_t

def normalize_latents(
        agent_latents, 
        lane_latents,
        agent_latents_mean,
        agent_latents_std,
        lane_latents_mean,
        lane_latents_std):
    """Normalize agent & lane latents using cached mean/std(Supporting metrics or vector-by-dimensional vectors)

    Parameters`*_latents_mean/std`Could be float/ list/np.ndarray / torch.TensorI'm not sure what you're doing.
    This function automatically converts to the same level as latentdevice/dtypeTensor.
    """
    if agent_latents.numel() > 0:
        a_mean, a_std = _to_tensor_stats(agent_latents_mean, agent_latents_std, agent_latents)
        agent_latents = (agent_latents - a_mean) / a_std

    if lane_latents.numel() > 0:
        l_mean, l_std = _to_tensor_stats(lane_latents_mean, lane_latents_std, lane_latents)
        lane_latents = (lane_latents - l_mean) / l_std

    return agent_latents, lane_latents


def unnormalize_latents(
        agent_latents, 
        lane_latents,
        agent_latents_mean,
        agent_latents_std,
        lane_latents_mean,
        lane_latents_std):
    """Unnormalize agent & lane latents using cached mean/std(Supporting metrics or vector-by-dimensional vectors)"""
    if agent_latents.numel() > 0:
        a_mean, a_std = _to_tensor_stats(agent_latents_mean, agent_latents_std, agent_latents)
        agent_latents = agent_latents * a_std + a_mean

    if lane_latents.numel() > 0:
        l_mean, l_std = _to_tensor_stats(lane_latents_mean, lane_latents_std, lane_latents)
        lane_latents = lane_latents * l_std + l_mean

    return agent_latents, lane_latents


def reparameterize(mu, log_var):
    """ Reparameterization trick to sample from a Gaussian distribution
    Args:
        mu (torch.Tensor): Mean of the Gaussian distribution.
        log_var (torch.Tensor): Log variance of the Gaussian distribution.
    Returns:
        torch.Tensor: Sampled latent variable.
    """
    assert mu.shape == log_var.shape
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


def sample_latents(
        data, 
        agent_latents_mean,
        agent_latents_std,
        lane_latents_mean,
        lane_latents_std,
        normalize=True):
    """ Sample latents from the agent and lane data, and (optionally) normalize them."""
    agent_mu = data['agent'].x
    agent_log_var = data['agent'].log_var 
    agent_latents = reparameterize(agent_mu, agent_log_var)

    lane_mu = data['lane'].x 
    lane_log_var = data['lane'].log_var 
    lane_latents = reparameterize(lane_mu, lane_log_var)

    if normalize:
        agent_latents, lane_latents = normalize_latents(
            agent_latents, 
            lane_latents,
            agent_latents_mean,
            agent_latents_std,
            lane_latents_mean,
            lane_latents_std)
    
    return agent_latents, lane_latents


def convert_batch_to_scenarios(
        data,
        batch_size,
        batch_idx,
        cache_dir,
        conditioning_filenames=None,
        cache_samples=False,
        cache_lane_types=False,
        mode='initial_scene',
        output_ids=None,
    ):
    """ Converts batch output into individual scenarios. Optionally saves scenarios to disk.

    Add:
    ----
    1) If`data['agent']`Exists`.motion`Properties (Na, movement_dim), physical shift unit m),
       Export it to a pkl field`agent_motion`I don't know.
    2) If transferred`output_ids`(len=batch_size and mode!=intaining') is used as the output filename (not including extension).
       This allows the basename from which the sample is generated to be aligned to the HT filename for the event_set, so that debug can be matched to the subsequent paired indicator.

    Note:
    ----
    - It's still only exported downstream.`agent_states`for the first 7-dimensional static section;
    - `agent_motion`is optional.
    """
    if cache_samples and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    num_samples_in_batch = int(data.batch_size)

    if output_ids is not None and mode != "inpainting":
        if len(output_ids) != num_samples_in_batch:
            raise ValueError(
                f"[convert_batch_to_scenarios] output_ids length mismatch: "
                f"len(output_ids)={len(output_ids)} vs batch_size={num_samples_in_batch}"
            )

    agent_batch, lane_batch, lane_conn_batch = get_batches(data)
    x_agent, x_agent_states, x_agent_types, x_lane, x_lane_states, x_lane_types, x_lane_conn = get_features(data)

    if mode == 'inpainting':
        x_lane_mask = data['lane'].mask.float()
        x_agent_mask = data['agent'].mask.float()
        x_lane_ids = data['lane'].ids

    # motion (physical space)
    has_motion = hasattr(data['agent'], "motion")
    if has_motion:
        x_agent_motion = data['agent'].motion.detach().cpu().numpy()
    else:
        x_agent_motion = None

    # move to cpu
    lg_type = data['lg_type'].cpu().numpy()

    if 'map_id' in data:
        map_ids = data['map_id'].cpu().numpy()
    else:
        map_ids = np.zeros(num_samples_in_batch, dtype=np.int64)

    x_agent_states = x_agent_states.cpu().numpy()
    x_agent_types = x_agent_types.cpu().numpy()
    if cache_lane_types and x_lane_types is not None:
        x_lane_types = x_lane_types.cpu().numpy()
    x_lane_states = x_lane_states.cpu().numpy()
    x_lane_conn = x_lane_conn.cpu().numpy()
    agent_batch_np = agent_batch.cpu().numpy()
    lane_batch_np = lane_batch.cpu().numpy()
    lane_conn_batch_np = lane_conn_batch.cpu().numpy()
    if mode == 'inpainting':
        x_lane_mask = x_lane_mask.cpu().numpy()
        x_agent_mask = x_agent_mask.cpu().numpy()
        x_lane_ids = x_lane_ids.cpu().numpy()

    batch_of_scenarios = {}
    for i in range(num_samples_in_batch):
        map_id_i = map_ids[i]

        scene_i_agents_full = x_agent_states[agent_batch_np == i]
        scene_i_agents = scene_i_agents_full[:, :7]
        scene_i_lanes = x_lane_states[lane_batch_np == i]
        scene_i_agent_types = x_agent_types[agent_batch_np == i]
        if cache_lane_types and x_lane_types is not None:
            scene_i_lane_types = x_lane_types[lane_batch_np == i]
        scene_i_lane_conns = x_lane_conn[lane_conn_batch_np == i]
        lg_type_i = lg_type[i]

        if has_motion and x_agent_motion is not None:
            scene_i_agent_motion = x_agent_motion[agent_batch_np == i]
        else:
            scene_i_agent_motion = None

        if mode == 'inpainting':
            scene_i_lane_mask = x_lane_mask[lane_batch_np == i]
            scene_i_agent_mask = x_agent_mask[agent_batch_np == i]
            scene_i_lane_ids = x_lane_ids[lane_batch_np == i]

        data_dict = {
            'num_agents': len(scene_i_agents),
            'num_lanes': len(scene_i_lanes),
            'map_id': int(map_id_i),
            'lg_type': int(lg_type_i),
            'agent_states': scene_i_agents,
            'road_points': scene_i_lanes,
            'agent_types': scene_i_agent_types,
            'road_connection_types': scene_i_lane_conns
        }

        if cache_lane_types and x_lane_types is not None:
            data_dict['lane_types'] = scene_i_lane_types
        if mode == 'inpainting':
            data_dict['lane_mask'] = scene_i_lane_mask
            data_dict['agent_mask'] = scene_i_agent_mask
            data_dict['lane_ids'] = scene_i_lane_ids
        if scene_i_agent_motion is not None:
            data_dict['agent_motion'] = scene_i_agent_motion

        if mode == 'inpainting':
            scenario_id = conditioning_filenames[int(batch_idx * batch_size + i)]
        else:
            scenario_id = output_ids[i] if output_ids is not None else f"{i}_{batch_idx}"

        filename = f"{scenario_id}.pkl"
        batch_of_scenarios[scenario_id] = data_dict

        if cache_samples:
            with open(os.path.join(cache_dir, filename), 'wb') as f:
                pickle.dump(data_dict, f)

    return batch_of_scenarios