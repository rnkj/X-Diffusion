from typing import Any, Dict, List, Tuple
from collections import namedtuple
import os
import os

import cv2
import fpsample
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import numpy as np
import yaml
import torch
from pathlib import Path

CANONICAL_PUBLIC_EMBODIMENTS = ("robot", "human", "human_filtered")

ACTION_TYPE_ROBOT = "robot"
ACTION_TYPE_TRACKS = "tracks"
ACTION_TYPE_TRACKS_3D = "tracks_3d"
ACTION_TYPE_EE = "ee"
ACTION_TYPES = [ACTION_TYPE_ROBOT, ACTION_TYPE_TRACKS, ACTION_TYPE_TRACKS_3D, ACTION_TYPE_EE]

H5Batch = namedtuple("H5Batch", ["action", "state_cond", "label", "data_type", "images"])


def infer_public_embodiment(path: str | Path) -> str:
    path_obj = Path(path)
    for part in reversed(path_obj.parts):
        if part in CANONICAL_PUBLIC_EMBODIMENTS:
            return part
        if part.endswith("_imgs"):
            candidate = part[:-5]
            if candidate in CANONICAL_PUBLIC_EMBODIMENTS:
                return candidate
    raise ValueError(f"Could not infer canonical embodiment from path: {path_obj}")


def is_human_embodiment(embodiment: str) -> bool:
    return embodiment in {"human", "human_filtered"}


def canonical_label_for_embodiment(embodiment: str) -> int:
    return 0 if is_human_embodiment(embodiment) else 1


# from depth_anything_v2.dpt import DepthAnythingV2

TARGET_POINTCLOUD_RES = 10000

# Keypoints for wrist, two points on the thumb, and two points on the middle finger
# HAMER_KEYPOINTS = [0, 1, 3, 9, 11]


# =============== Align Correspondences Between Robot/Hand ===============
def create_point_to_idx_dict(mapping: List[int]) -> Dict[str, int]:
    return dict(zip(ALL_KEYS, mapping))


def validate_dictionaries(dicts: Tuple[Dict[str, int], ...]) -> None:
    # Check if all dictionaries have the same keys
    keys_sets = [set(d.keys()) for d in dicts]
    if len(set(map(frozenset, keys_sets))) != 1:
        raise ValueError("All dictionaries must have the same keys")

    # Check if LH and RH dictionaries have the same set of values
    if set(LH_POINT_TO_IDX.values()) != set(RH_POINT_TO_IDX.values()):
        raise ValueError("LH and RH dictionaries must have the same set of values")


ALL_KEYS = ["wrist", "wrist_low", "right1", "left1", "right2", "left2"]
KPT_TYPE_TO_KEYS = {
    "all": ["wrist", "right1", "left1", "right2", "left2"],
    "all_low": ["wrist_low", "right1", "left1", "right2", "left2"],
    "rh": ["wrist", "right1", "right2"],
    "lh": ["wrist", "left1", "left2"],
    "both3": ["wrist", "right2", "left2"],
    "both4": ["wrist", "right1", "left2", "right2"],
    "both3_low": ["wrist_low", "right2", "left2"],
}

# Assuming robot is left-handed
ROBOT_MAPPING: List[int] = [0, 0, 1, 2, 3, 4, 0]

# LH has the same mapping as the robot
LH_MAPPING: List[int] = [0, 5, 6, 2, 8, 4]

# RH flips mapping on thumb and middle finger
RH_MAPPING: List[int] = [0, 5, 2, 6, 4, 8]

ROBOT_POINT_TO_IDX = create_point_to_idx_dict(ROBOT_MAPPING)
LH_POINT_TO_IDX = create_point_to_idx_dict(LH_MAPPING)
RH_POINT_TO_IDX = create_point_to_idx_dict(RH_MAPPING)
HAMER_KEYPOINTS = LH_MAPPING.copy()


# =============== Helper Functions ===============


# def load_depth_model(encoder="vits"):
#     model_configs = {
#         "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
#         "vitb": {
#             "encoder": "vitb",
#             "features": 128,
#             "out_channels": [96, 192, 384, 768],
#         },
#         "vitl": {
#             "encoder": "vitl",
#             "features": 256,
#             "out_channels": [256, 512, 1024, 1024],
#         },
#         "vitg": {
#             "encoder": "vitg",
#             "features": 384,
#             "out_channels": [1536, 1536, 1536, 1536],
#         },
#     }

#     DEVICE = (
#         "cuda"
#         if torch.cuda.is_available()
#         else "mps"
#         if torch.backends.mps.is_available()
#         else "cpu"
#     )
#     model = DepthAnythingV2(**model_configs[encoder])
#     model.load_state_dict(
#         torch.load(
#             f"third_party/Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth",
#             map_location="cpu",
#         )
#     )
#     return model.to(DEVICE).eval()


def set_axis_limits(ax):
    """Set the limits and labels for the axes of a 3D plot."""
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([-0.2, 0.5])
    ax.set_zlim([0.0, 0.5])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

def make_delta_actions(action_data, is_ee_data=False):
    """
    Compute delta actions from consecutive timesteps.
    
    Args:
        action_data: Array of shape (T, D) where T is time and D is action dimension
        is_ee_data: If True, treat as ee data where first 3 dims are position, next 3 are euler angles
    """
    gripper = action_data[:, -1:]  
    action = action_data[:, :-1]  
    
    if is_ee_data:
        # For ee data: [pos_x, pos_y, pos_z, euler_x, euler_y, euler_z]
        # Position deltas are simple differences
        pos_delta = np.diff(action[:, :3], axis=0)
        
        # Euler angle deltas need proper rotation mathematics
        euler_angles = action[:, 3:6]  # Shape: (T, 3)
        
        rotations = R.from_euler('xyz', euler_angles)
        relative_rots = rotations[1:] * rotations[:-1].inv()
        euler_delta = relative_rots.as_euler('xyz')
                
        action_delta = np.concatenate([pos_delta, euler_delta], axis=1)
    else:
        # For keypoint data: simple differences
        action_delta = np.diff(action, axis=0)
    
    delta_actions = np.concatenate([action_delta, gripper[1:]], axis=1)
    return delta_actions

def make_dataset_from_episodes(
    episode_data_list,
    obs_horizon,
    pred_horizon,
    delta_actions,
    is_ee_data=False,
    image_data_list=None,
    camera_views=None,
) -> Dict[str, np.ndarray]:
    """
    Create dataset from a list of episodes, where each episode is a complete sequence.
    
    Args:
        episode_data_list: List of episode data dicts with keys 'action', 'state_cond', 'data_type'
        obs_horizon: Observation horizon
        pred_horizon: Prediction horizon
        delta_actions: Whether to compute delta actions
        is_ee_data: Whether data is end-effector data
        image_data_list: List of image data dicts (one per episode)
        camera_views: List of camera view names
        
    Returns:
        Dictionary with 'action', 'state_cond', 'data_type', and optionally image data
    """
    train_data = {
        "action": [],
        "state_cond": [],
        "data_type": [],
    }
    
    # Add image data keys if provided
    if image_data_list is not None and camera_views is not None:
        for camera_view in camera_views:
            train_data[f"images_{camera_view}"] = []
    
    for episode_idx, episode_data in enumerate(episode_data_list):
        action_data = episode_data['action']
        state_cond_data = episode_data['state_cond']
        data_type = episode_data['data_type']
        
        # Get image data for this episode if available
        episode_image_data = None
        if image_data_list is not None and episode_idx < len(image_data_list):
            episode_image_data = image_data_list[episode_idx]
        
        # Create episode ends for this single episode
        episode_length = len(action_data)
        current_episode_ends = [episode_length]
        
        # Use the original make_dataset function for this episode
        episode_train_data = make_dataset(
            action_data,
            state_cond_data,
            data_type,
            obs_horizon,
            pred_horizon,
            current_episode_ends,
            delta_actions,
            is_ee_data=is_ee_data,
            image_data=episode_image_data,
            camera_views=camera_views,
        )
        
        # Append data from this episode
        train_data["action"].extend(episode_train_data["action"])
        train_data["state_cond"].extend(episode_train_data["state_cond"])
        train_data["data_type"].extend(episode_train_data["data_type"])
        
        # Append image data if available
        if image_data_list is not None and camera_views is not None:
            for camera_view in camera_views:
                image_key = f"images_{camera_view}"
                if image_key in episode_train_data:
                    train_data[image_key].extend(episode_train_data[image_key])
    
    # Convert all lists to numpy arrays
    result = {
        "action": np.array(train_data["action"]),
        "state_cond": np.array(train_data["state_cond"]),
        "data_type": np.array(train_data["data_type"]),
    }
    
    # Add image data if it was processed
    if image_data_list is not None and camera_views is not None:
        for camera_view in camera_views:
            image_key = f"images_{camera_view}"
            if image_key in train_data and train_data[image_key]:
                result[image_key] = np.array(train_data[image_key])
    
    return result


def make_dataset(
    action_data,
    state_cond_data,
    data_key,
    obs_horizon,
    pred_horizon,
    current_episode_ends,
    delta_actions,
    is_ee_data=False,
    image_data=None,
    camera_views=None,
) -> Dict[str, np.ndarray]:
    train_data = {
        "action": [],
        "state_cond": [],
        "data_type": [],
    }
    
    # Add image data if provided
    if image_data is not None and camera_views is not None:
        for camera_view in camera_views:
            train_data[f"images_{camera_view}"] = []
    current_episode_ends = [0] + np.array(current_episode_ends).tolist()
    for episode_start, episode_end in zip(current_episode_ends[:-1], current_episode_ends[1:]):
        valid_starts_min = episode_start + 1 - obs_horizon 
        valid_starts_max = episode_end - pred_horizon - obs_horizon - 1
        for valid_start in range(valid_starts_min, valid_starts_max + 1):
            valid_end = valid_start + pred_horizon + obs_horizon - 1
            train_data["state_cond"].append(state_cond_data[valid_start:valid_start+obs_horizon])
            if delta_actions:
                actions = make_delta_actions(
                    action_data[valid_start+obs_horizon-1:valid_start+pred_horizon+obs_horizon],
                    is_ee_data=is_ee_data
                )
            else:
                actions = action_data[valid_start+obs_horizon   -1:valid_start+pred_horizon+obs_horizon    -1]
            train_data["action"].append(actions)
            train_data["data_type"].append(data_key)
            # Extract images for each camera view in the observation horizon
            if image_data is not None and camera_views is not None:
                for camera_view in camera_views:
                    if camera_view in image_data:
                        # Extract images for the observation horizon
                        obs_images = image_data[camera_view][valid_start:valid_start+obs_horizon]
                        # Ensure images are RGB 3-channel: (obs_horizon, H, W, 3) -> (obs_horizon, 3, H, W)
                        if obs_images.ndim == 4 and obs_images.shape[-1] == 3:
                            # If shape is (obs_horizon, H, W, 3), transpose to (obs_horizon, 3, H, W)
                            obs_images = np.transpose(obs_images, (0, 3, 1, 2))
                        elif obs_images.ndim == 3:
                            # If grayscale (obs_horizon, H, W), convert to RGB by repeating channels
                            obs_images = np.repeat(obs_images[:, np.newaxis, :, :], 3, axis=1)
                        
                        # Reshape to concatenate along channel dimension: (obs_horizon, 3, H, W) -> (obs_horizon*3, H, W)
                        obs_images = obs_images.reshape(obs_horizon * 3, obs_images.shape[2], obs_images.shape[3])
                        train_data[f"images_{camera_view}"].append(obs_images)
                    else:
                        # Create dummy images if camera view is missing (obs_horizon*3, H, W)
                        dummy_images = np.zeros((obs_horizon * 3, 96, 96), dtype=np.float32)
                        train_data[f"images_{camera_view}"].append(dummy_images)

    # Convert all data to numpy arrays
    result = {
        "action": np.array(train_data["action"]),
        "state_cond": np.array(train_data["state_cond"]),
        "data_type": np.array(train_data["data_type"]),
    }
    
    # Add image data if it was processed
    if image_data is not None and camera_views is not None:
        for camera_view in camera_views:
            result[f"images_{camera_view}"] = np.array(train_data[f"images_{camera_view}"])
    
    return result

def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
) -> np.ndarray:
    """
    Returns a list of [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx].

    - buffer_start_idx and buffer_end_idx define the range of data to sample from.
    - sample_start_idx and sample_end_idx consider the start and end points of this data with respect to a sample of sequence length. (See `sample_sequence` below for how padding is done.)

    The minimum start index is -pad_before, and the maximum start index is episode_length
        - sequence_length + pad_after.
    In practice, pad_before = obs_horizon - 1 and pad_after = action_horizon - 1. This
        means that the first obs_horizon - 1 points are used as padding before the start of
        the sequence, and the last action_horizon - 1 points are used as padding after the
        end of the sequence to meet the sequence length.
    """
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )
    indices = np.array(indices)
    return indices

def sample_sequence(
    train_data: Dict[str, np.ndarray],
    sequence_length: int,
    buffer_start_idx: int,
    buffer_end_idx: int,
    sample_start_idx: int,
    sample_end_idx: int,
) -> Dict[str, np.ndarray]:
    """
    For each key in train_data:
    1. Grab a sample of the data from buffer_start_idx to buffer_end_idx.
    2. If the sample is not the full sequence length, pad the start and end with the first and last values, respectively.

    Note it should always be the case that sample_end_idx - sample_start_idx == buffer_end_idx - buffer_start_idx.

    Example:
        Given:
            sequence_length = 8
            buffer_start_idx = 0
            buffer_end_idx = 5
            sample_start_idx = 1
            sample_end_idx = 6
        First grabs (input_arr[buffer_start_idx:buffer_end_idx]):
            sample = [v_0, v_1, v_2, v_3, v_4]
        Then pads (data[:sample_start_idx] = sample[0]):
            data = [v_0, ...]
        And also pads (data[sample_end_idx:] = sample[-1]):
            data = [..., v_4, v_4]
        Then finally fills in the sample:
            data = [v_0, v_0, v_1, v_2, v_3, v_4, v_4, v_4]
    """
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype
            )
            if sample_start_idx > 0:
                # pad start with first value
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                # pad end with last value
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


def downsample_pc(
    points: List[np.ndarray], colors: List[np.ndarray], num_points: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Downsample a point cloud using fast point cloud sampling.

    Args:
        points (List[np.ndarray]): List of point arrays.
        colors (List[np.ndarray]): List of color arrays corresponding to the points.
        num_points (int): The number of points to sample from each point cloud.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Tuple of lists containing the downsampled points and colors.
    """

    def downsample_with_fps(
        points: np.ndarray, colors: np.ndarray, num_points: int = 1024
    ) -> Tuple[np.ndarray, np.ndarray]:
        # fast point cloud sampling using fpsample
        sampled_indices = fpsample.bucket_fps_kdline_sampling(
            points, num_points, h=7, start_idx=0
        )
        return points[sampled_indices], colors[sampled_indices]

    is_single_point_cloud = False
    if len(points.shape) == 2:
        is_single_point_cloud = True
        points = points[None]
        colors = colors[None]

    points_subsampled, colors_subsampled = [], []
    for i in range(len(points)):
        p, c = downsample_with_fps(points[i], colors[i], num_points)
        points_subsampled.append(p)
        colors_subsampled.append(c)
    points_subsampled = np.stack(points_subsampled)
    colors_subsampled = np.stack(colors_subsampled)

    if is_single_point_cloud:
        points_subsampled = points_subsampled[0]
        colors_subsampled = colors_subsampled[0]
    return points_subsampled, colors_subsampled


def get_data_stats(data: np.ndarray) -> dict:
    """
    Calculate the minimum and maximum values for each feature in the dataset.

    Parameters:
    - data (np.ndarray): The input data array, where rows represent samples and columns represent features.

    Returns:
    - dict: A dictionary containing the minimum and maximum values for each feature across all samples.
    """
    # data = data.reshape(-1, data.shape[-1])
    stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
    return stats


def normalize_data(data: np.ndarray, stats: dict) -> np.ndarray:
    """
    Normalize the input data to the range [-1, 1] using the provided statistics.

    Parameters:
    - data (np.ndarray): The input data array to be normalized.
    - stats (dict): A dictionary containing the minimum ('min') and maximum ('max') values for each feature.

    Returns:
    - np.ndarray: The normalized data.
    """
    # Normalize to [0,1]
    ndata = (data - stats["min"]) / (stats["max"] - stats["min"] + 1e-6)
    # Normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata: np.ndarray, stats: dict) -> np.ndarray:
    """
    Convert data from the normalized range [-1, 1] back to its original scale using the provided statistics.

    Parameters:
    - ndata (np.ndarray): The normalized data array to be converted back.
    - stats (dict): A dictionary containing the minimum ('min') and maximum ('max') values for each feature.

    Returns:
    - np.ndarray: The data converted back to its original scale.
    """
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    return data


def add_padding_to_point_cloud(
    point_cloud: np.ndarray, rgb: np.ndarray, target_length: int
) -> np.ndarray:
    """
    This function adds padding to the point cloud to make it a square. To differentiate between the original points,
    the padding function makes a "special point" in the last index, in which it is assigned the value (-1, 1, original_length).
    By doing do, the original point cloud can be recoved by checking whether the last index is the special point, then
    grabbing the original_length value and removing the padding.


    Parameters:
    - point_cloud (np.ndarray): The point cloud to add padding to.
    - rgb (np.ndarray): The colors of the point cloud to add padding to.
    - target_value (int): The value to pad the point cloud with.

    Returns:
    - np.ndarray: The padded point cloud.
    """
    original_length = len(point_cloud)
    if original_length == target_length:
        return point_cloud, rgb
    padded_rgb = np.zeros((target_length, 3))
    padded_rgb[:original_length] = rgb
    padded_point_cloud = np.zeros((target_length, 3))
    padded_point_cloud[:original_length] = point_cloud
    padded_point_cloud[-1] = [-1, 1, original_length]
    return padded_point_cloud, padded_rgb


def remove_padding_from_point_cloud(
    point_cloud: np.ndarray, rgb: np.ndarray
) -> np.ndarray:
    """
    This function removes the padding from the point cloud. The padding is added by the add_padding_to_point_cloud function.

    Parameters:
    - point_cloud (np.ndarray): The point cloud to remove padding from.
    - rgb (np.ndarray): The rgb of point cloud to remove padding from.

    Returns:
    - np.ndarray: The point cloud with padding removed.
    """
    special_point = point_cloud[-1]
    if special_point[0] != -1 or special_point[1] != 1:
        # No padding was added
        return point_cloud, rgb
    original_length = int(special_point[2])
    return point_cloud[:original_length], rgb[:original_length]


def calculate_point_offset(
    ee_euler: np.ndarray, home_offset: np.ndarray, euler_adjustment: np.ndarray
) -> np.ndarray:
    """
    Calculate the offset for any point given the end-effector's Euler angles and the point's home offset.

    Parameters:
    - ee_euler: np.ndarray - The Euler angles of the end-effector.
    - home_offset: np.ndarray - The home position offset of the point with respect to the end-effector frame.
    - euler_adjustment: np.ndarray - The adjustment to be made to the Euler angles.

    Returns:
    - np.ndarray - The calculated offset of the point.
    """
    ee_euler_adjustment = ee_euler.copy() - euler_adjustment
    point_offset = R.from_euler("xyz", ee_euler_adjustment).as_matrix() @ home_offset
    return point_offset


def extract_point_cloud(
    obs: Dict[str, np.ndarray],
    intrinsics: Dict[str, np.ndarray],
    transforms: Dict[str, Dict[str, np.ndarray]],
    num_point: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts and processes point clouds from RGB and depth frames for multiple agents.
    Convert all to robot frame.

    Parameters:
    - obs (Dict[str, np.ndarray]): Observations containing RGB and depth frames for each agent.
    - intrinsics (Dict[str, np.ndarray]): Camera intrinsics for each agent.
    - transforms (Dict[str, Dict[str, np.ndarray]]): Camera extrinsics (transforms) for each agent.
    - num_point (int): Number of points to downsample to.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Downsampled points and corresponding colors.
    """
    points_list = []
    colors_list = []

    for view in ["agent1", "agent2"]:
        rgb_frame = obs[f"{view}_image"]
        depth_frame = obs[f"{view}_depth"]
        depth_frame = depth_frame.squeeze()
        agent_intrinsics = intrinsics[view]
        agent_extrinsics = transforms[view]["tcr"]
        points = deproject(depth_frame, agent_intrinsics, agent_extrinsics)
        colors = rgb_frame.reshape(points.shape) / 255.0
        points_list.append(points)
        colors_list.append(colors)

    merged_points = np.vstack(points_list)
    merged_colors = np.vstack(colors_list)

    idxs = crop(
        merged_points, min_bound=[0.35, -0.24, 0.02], max_bound=[0.7, 0.24, 0.3]
    )
    merged_points = merged_points[idxs]
    merged_colors = merged_colors[idxs]

    downsample_idxs = np.random.choice(
        np.arange(len(merged_points)), min(len(merged_points), num_point), replace=False
    )

    points = merged_points[downsample_idxs]
    colors = merged_colors[downsample_idxs]

    return points, colors


def crop_image(image: np.ndarray, dim: int) -> Tuple[np.ndarray, int]:
    """
    Crop the image to a square and resize it to the specified dimension.

    Parameters:
    image (np.ndarray): The input image.
    dim (int): The dimension to resize the image to.

    Returns:
    Tuple[np.ndarray, int]: The cropped and resized image, and the offset used to crop the image.
    """
    H, W, C = image.shape
    # Get the offset to crop the image to a square
    offset = (W - H) // 2
    image = image[:, offset // 2 : -offset // 2, :]
    image = cv2.resize(image, (dim, dim))
    return image, offset


def create_colorbar_image(
    cmap: Any,
    dt: int,
) -> np.ndarray:
    """
    Create a colorbar image using the specified colormap and number of timesteps.

    Parameters:
    cmap (Any): The colormap to use for the colorbar.
    dt (int): The number of timesteps.

    Returns:
    np.ndarray: The colorbar image.
    """
    # Create a figure and axis for the color bar
    fig, ax = plt.subplots(figsize=(1, 5))  # Width is 1 inch, height is 5 inches
    fig.subplots_adjust(left=0.3, right=0.55, top=0.96, bottom=0.05)

    # Create a color bar with the specified colors
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap), cax=ax, ticks=np.linspace(0, 1, dt + 1)
    )
    cbar.ax.invert_yaxis()  # Invert the y-axis to have the color bar in the right orientation
    cbar.ax.set_yticklabels(
        ["t=0" if i == 0 else f"t={dt}" if i == dt else "" for i in range(dt + 1)]
    )

    # Save the color bar to an image file
    fig.canvas.draw()
    colorbar_image = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)  # Close the figure to free memory

    return colorbar_image