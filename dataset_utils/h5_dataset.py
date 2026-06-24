from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union, Optional
from copy import deepcopy
from termcolor import cprint
from collections import namedtuple
import json
import numpy as np
import torch
import h5py
import cv2
from tqdm import tqdm, trange

REPO_ROOT = Path(__file__).resolve().parents[1]

from dataset_utils.common import (
    ACTION_TYPE_EE,
    ACTION_TYPES,
    ACTION_TYPE_TRACKS,
    H5Batch as Batch,
    KPT_TYPE_TO_KEYS,
    ROBOT_POINT_TO_IDX,
    LH_POINT_TO_IDX,
    RH_POINT_TO_IDX,
    get_data_stats,
    normalize_data,
    make_dataset,
    make_dataset_from_episodes,
    make_delta_actions,
    infer_public_embodiment,
    is_human_embodiment,
    canonical_label_for_embodiment,
)


def resolve_calibration_root(raw_root: str | Path | None) -> Path:
    if raw_root is None:
        raise ValueError("calibration_root must be set when overlay_3d_keypoints is enabled")
    root = Path(raw_root).expanduser()
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root.resolve()


@dataclass
class H5DatasetConfig:
    """
    Configuration for H5 dataset loading.
    
    Episode sampling parameters:
    - split_file_name: Name of the split file to use
    """
    # Image settings
    image_dim: int = 128
    normalize_image: bool = True
    transforms_name: str = "medium-geometric"
    apply_segmentation_mask: bool = False
    overlay_3d_keypoints: bool = True
    calibration_root: str = "calib_cornell"

    # Horizon settings
    pred_horizon: int = 8
    action_horizon: int = 8
    obs_horizon: int = field(default_factory=lambda: 1)

    delta_actions: bool = False

    # Action settings
    action_pred_type: str = ACTION_TYPE_TRACKS
    kpt_type: str = "all_low"  # ['all', 'rh', 'lh', 'both3', 'both4']
    jitter_kpts: bool = False
    fixed_kpt_dims: List[str] = field(default_factory=lambda: ["wrist_low"])
    
    # End-effector data settings
    use_ee_data: bool = True  # If True, use ee_pos + ee_euler instead of keypoint tracks for action/state_cond

    # Language settings
    lang_instruction: str = ""

    # Grasp information
    add_grasp_info_to_tracks: bool = True
    grasp_threshold: float = 0.5

    # Data paths
    physical_data_paths: List[Path] = field(default_factory=list)  # Paths to physical agent H5 files
    rgb_seg_data_paths: List[Path] = field(default_factory=list)   # Paths to RGB+segmentation H5 files

    # Additional settings
    add_in_wrist: bool = False
    add_depth_gs: bool = False
    add_depth_colored: bool = False
    human_train_percentage: float = 1.0
    human_filtered_train_percentage: float = 1.0
    robot_train_percentage: float = 1.0
    
    # Episode sampling settings
    split_file_name: Optional[str] = None  # Name of the split file to use

    # Conditioning source: 'tracks_2d', 'tracks_3d', or 'proprio'
    cond_source: str = "tracks_3d"
    
    # Balanced sampling settings
    balanced_sampling_weights: Optional[tuple] = (0.5, 0.5)  # e.g., (0.5, 0.5) for equal sampling from labels 0 and 1
    
    # Image loading settings
    load_images: bool = False  # Whether to load images from the dataset
    camera_views: List[str] = field(default_factory=lambda: ["agent1"])  # Camera view names in the dataset
    image_dim: int = 96  # Target image dimension for resizing

    @property
    def kpt_keys(self) -> List[str]:
        return KPT_TYPE_TO_KEYS[self.kpt_type]

    @property
    def num_points(self) -> int:
        return len(self.kpt_keys)


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg: H5DatasetConfig, partition: str = "train"):
        self.stats = {}
        self.pred_horizon = cfg.pred_horizon
        self.action_horizon = cfg.action_horizon
        self.obs_horizon = cfg.obs_horizon
        self.delta_actions = cfg.delta_actions
        self.action_pred_type = cfg.action_pred_type
        self._add_grasp_info_to_tracks = cfg.add_grasp_info_to_tracks
        self._grasp_threshold = cfg.grasp_threshold
        self._grasp_and_term_dim = 1 + int(cfg.add_grasp_info_to_tracks)
        self._human_train_percentage = cfg.human_train_percentage
        self._human_filtered_train_percentage = cfg.human_filtered_train_percentage
        self._robot_train_percentage = cfg.robot_train_percentage
        self._kpt_keys = cfg.kpt_keys
        self._num_points = cfg.num_points
        self._cond_source = cfg.cond_source
        self._load_images = cfg.load_images
        self._camera_views = cfg.camera_views
        self._image_dim = cfg.image_dim
        self._apply_segmentation_mask = cfg.apply_segmentation_mask
        self._overlay_3d_keypoints = cfg.overlay_3d_keypoints
        self.cfg = cfg
        self.partition = partition
        self._split_file_name = cfg.split_file_name
        if self._split_file_name in (None, "full"):
            raise ValueError("split_file_name must point to a split JSON for X-Diffusion H5 training")
        self.split_index_dict = json.load(open(self._split_file_name))

        # Unified indexing for proper dataset access
        self.unified_indices = []  # List of (dataset_type, local_idx) tuples

        train_data = {
            key: {"human": [], "robot": []}
            for key in ["action", "state_cond", "data_type"]
        }
        
        # Add image data keys if loading images
        if cfg.load_images and cfg.camera_views:
            for camera_view in cfg.camera_views:
                train_data[f"images_{camera_view}"] = {"human": [], "robot": []}

        # Load datasets from physical and RGB+segmentation data
        pbar = tqdm(zip(cfg.physical_data_paths, cfg.rgb_seg_data_paths), desc="Loading H5 datasets", leave=False)
        for physical_path, rgb_seg_path in pbar:
            pbar.set_description(f"Loading {physical_path.parent}")
            dataset = self._load_episodes_from_directory(physical_path, rgb_seg_path, self.partition, self.split_index_dict)
            dataset_type = "human" if is_human_embodiment(infer_public_embodiment(physical_path)) else "robot"
            
            for key in train_data:
                train_data[key][dataset_type].append(dataset[key])

        # Concatenate data from all datasets
        for key in train_data:
            for dtype in ["human", "robot"]:
                if train_data[key][dtype]:
                    if key.startswith("images_"):
                        # For image data, concatenate along the first axis (sequence dimension)
                        train_data[key][dtype] = np.concatenate(
                            train_data[key][dtype], axis=0
                        )
                    else:
                        # For other data, concatenate normally
                        train_data[key][dtype] = np.concatenate(
                            train_data[key][dtype], axis=0
                        )
        
        # Combine human and robot data
        all_data = {}
        for key in train_data:
            if len(train_data[key]["human"]) > 0 and len(train_data[key]["robot"]) > 0:
                if key.startswith("images_"):
                    # For image data, concatenate along the first axis
                    all_data[key] = np.concatenate(
                        [train_data[key]["human"], train_data[key]["robot"]], axis=0
                    )
                else:
                    # For other data, concatenate normally
                    all_data[key] = np.concatenate(
                        [train_data[key]["human"], train_data[key]["robot"]], axis=0
                    )
            elif len(train_data[key]["human"]) > 0:
                all_data[key] = np.array(train_data[key]["human"])
            else:
                all_data[key] = np.array(train_data[key]["robot"])
        
        # Calculate statistics for non-image data
        for key in all_data:
            if not key.startswith("images_"):
                # Only calculate stats for non-image data
                self.stats[key] = get_data_stats(all_data[key])
        
        self.train_data = train_data
        self.all_data = all_data

        # Build unified indices for proper dataset access
        self._build_unified_indices()

        human_count = len(self.train_data["action"]["human"]) if len(self.train_data["action"]["human"]) > 0 else 0
        robot_count = len(self.train_data["action"]["robot"]) if len(self.train_data["action"]["robot"]) > 0 else 0
        cprint(
            f"Loaded H5 dataset with lengths: human={human_count}, robot={robot_count}", color="green", attrs=["bold"]
        )

    def _build_unified_indices(self):
        """Build unified indices that combine human and robot indices for proper dataset access."""
        self.unified_indices = []
        
        # Add human indices first
        human_count = len(self.train_data["action"]["human"]) if len(self.train_data["action"]["human"]) > 0 else 0
        for i in range(human_count):
            self.unified_indices.append(("human", i))
        
        # Add robot indices second
        robot_count = len(self.train_data["action"]["robot"]) if len(self.train_data["action"]["robot"]) > 0 else 0
        for i in range(robot_count):
            self.unified_indices.append(("robot", i))
        
        cprint(f"Built unified indices: {len(self.unified_indices)} total samples "
               f"({human_count} human, {robot_count} robot)", 
               color="cyan", attrs=["bold"])

    @property
    def action_dim(self):
        return self.stats["action"]["max"].shape[-1]

    @property
    def state_cond_dim(self):
        return self.stats["state_cond"]["max"].flatten().shape[0]

    def get_class_distribution(self) -> dict:
        """Get the distribution of samples across classes."""
        human_count = len(self.train_data["action"]["human"]) if len(self.train_data["action"]["human"]) > 0 else 0
        robot_count = len(self.train_data["action"]["robot"]) if len(self.train_data["action"]["robot"]) > 0 else 0
        return {"human": human_count, "robot": robot_count}
    
    def get_label_distribution(self) -> dict:
        """Get the distribution of samples across labels (0=human, 1=robot)."""
        human_count = len(self.train_data["action"]["human"]) if len(self.train_data["action"]["human"]) > 0 else 0
        robot_count = len(self.train_data["action"]["robot"]) if len(self.train_data["action"]["robot"]) > 0 else 0
        return {0: human_count, 1: robot_count}
    
    def create_sample_weights(self, balanced_weights: Optional[tuple] = None) -> torch.Tensor:
        """Create sample weights for balanced sampling based on labels."""
        if balanced_weights is None:
            # Return uniform weights (no balancing)
            return torch.ones(len(self), dtype=torch.float32)
        
        label_dist = self.get_label_distribution()
        total_samples = sum(label_dist.values())
        
        if total_samples == 0:
            return torch.ones(len(self), dtype=torch.float32)
        
        # Create sample weights
        sample_weights = []
        for dataset_type, local_idx in self.unified_indices:
            label = 0 if dataset_type == "human" else 1
            if label < len(balanced_weights):
                # Use the specified weight for this label
                weight = balanced_weights[label]
                # Normalize by the actual count to achieve the desired ratio
                if label_dist[label] > 0:
                    weight = weight * total_samples / (len(balanced_weights) * label_dist[label])
                sample_weights.append(weight)
            else:
                sample_weights.append(1.0)
        
        return torch.tensor(sample_weights, dtype=torch.float32)

    def __len__(self) -> int:
        # Return the total number of samples in the unified dataset
        return len(self.unified_indices)

    def __getitem__(self, idx: int) -> Batch:
        # Use unified indices to determine dataset type and local index
        if idx >= len(self.unified_indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.unified_indices)}")
        
        dataset_type, local_idx = self.unified_indices[idx]
        
        # Get the pre-computed sequences directly from train_data
        action = self.train_data["action"][dataset_type][local_idx]
        state_cond = self.train_data["state_cond"][dataset_type][local_idx]
        data_type = self.train_data["data_type"][dataset_type][local_idx]

        # Normalize the data
        action = normalize_data(action, self.stats["action"])
        state_cond = normalize_data(state_cond, self.stats["state_cond"])
        
        # Create label based on dataset type
        label = 0 if dataset_type == "human" else 1

        # Load images if configured
        images = {}
        if self._load_images and self._camera_views:
            for camera_view in self._camera_views:
                # Get pre-loaded images from train_data
                image_key = f"images_{camera_view}"
                if image_key in self.train_data and len(self.train_data[image_key][dataset_type]) > 0:
                    # Convert numpy array to torch tensor
                    image_sequence = self.train_data[image_key][dataset_type][local_idx]
                    images[camera_view] = torch.from_numpy(image_sequence).float()
                else:
                    # Create dummy images if not available (obs_horizon*3, H, W)
                    images[camera_view] = torch.zeros(self.obs_horizon * 3, self._image_dim, self._image_dim)

        batch_kwargs = {
            "action": action,
            "state_cond": state_cond,
            "label": label,
            "data_type": data_type,
            "images": images,
        }
        
        return Batch(**batch_kwargs)

    def _load_episodes_from_directory(self, physical_path: Path, rgb_seg_path: Path, partition: str = "train", train_idxs_dict: dict = None):
        """Load episodes from directory containing individual demo H5 files."""
        assert physical_path.exists(), f"Physical data path {physical_path} does not exist"
        assert not self._load_images or rgb_seg_path.exists(), f"RGB+segmentation data path {rgb_seg_path} does not exist (required for image loading)"

        embodiment = infer_public_embodiment(physical_path)
        is_human_data = is_human_embodiment(embodiment)
        data_type = canonical_label_for_embodiment(embodiment)
        if isinstance(train_idxs_dict.get("train"), dict) or isinstance(train_idxs_dict.get("val"), dict):
            selected_ids = set(train_idxs_dict.get(partition, {}).get(embodiment, []))
            use_explicit_partition_ids = True
        else:
            selected_ids = set(train_idxs_dict.get(embodiment, []))
            use_explicit_partition_ids = False

        # Find all demo files in the directories
        physical_files = sorted([f for f in physical_path.glob("demo*.h5")])
        rgb_seg_files = sorted([f for f in rgb_seg_path.glob("demo*_imgs.h5")]) if self._load_images else []
        
        # Match physical and RGB files by demo number
        episode_pairs = []
        for physical_file in physical_files:
            demo_name = physical_file.stem  # e.g., "demo00000"
            if ((use_explicit_partition_ids and demo_name in selected_ids) or (not use_explicit_partition_ids and partition == "train" and demo_name in selected_ids)):
                rgb_file = rgb_seg_path / f"{demo_name}_imgs.h5" if self._load_images else None
                
                # Check if required files exist
                rgb_exists = not self._load_images or (rgb_file is not None and rgb_file.exists())
                
                if rgb_exists:
                    episode_pairs.append((physical_file, rgb_file))
            elif ((use_explicit_partition_ids and partition == "val" and demo_name in selected_ids) or (not use_explicit_partition_ids and partition == "val" and demo_name not in selected_ids)):
                rgb_file = rgb_seg_path / f"{demo_name}_imgs.h5" if self._load_images else None
                
                # Check if required files exist
                rgb_exists = not self._load_images or (rgb_file is not None and rgb_file.exists())
                
                if rgb_exists:
                    episode_pairs.append((physical_file, rgb_file))

        cprint(f"Loading {len(episode_pairs)} episodes for {partition} from {physical_path.parent.name}", color="cyan")
        
        # Load all episodes
        episode_data_list = []
        image_data_list = []
        
        for physical_file, rgb_file in episode_pairs:
            # Only pass rgb_file if we're loading images
            rgb_file_to_use = rgb_file if self._load_images else None
            episode_data, episode_images = self._load_single_episode(physical_file, rgb_file_to_use, is_human_data, data_type)
            episode_data_list.append(episode_data)
            if episode_images is not None:
                image_data_list.append(episode_images)
        
        # Use the new make_dataset_from_episodes function
        train_data = make_dataset_from_episodes(
            episode_data_list,
            self.obs_horizon,
            self.pred_horizon,
            self.delta_actions,
            is_ee_data=self.cfg.use_ee_data,
            image_data_list=image_data_list if image_data_list else None,
            camera_views=self._camera_views if self._load_images else [],
        )
        
        return train_data

    def _load_single_episode(self, physical_file: Path, rgb_file: Path, is_human_data: bool, data_type: int):
        """Load a single episode from physical and RGB+segmentation H5 files."""
        # Load physical agent data
        with h5py.File(physical_file, 'r') as f:
            ee_pos = np.array(f['ee_pos'])
            ee_euler = np.array(f['ee_euler'])
            gripper_open = np.array(f['gripper_open'])
            keypoints_3d = np.array(f['3d_tracks'])  # (T, 5, 3)

        # Load RGB+segmentation data only if loading images
        segmentation_data = None
        agent1_images = None
        if self._load_images and rgb_file is not None:
            with h5py.File(rgb_file, 'r') as f:
                segmentation_data = np.array(f['segmentation'])  # (T, H, W)
                agent1_images = np.array(f['agent1_images'])  # (T, H, W, 3)

        # Process grasp data
        if not is_human_data:
            # For robot data, process gripper actions
            grasp_actions = np.zeros_like(gripper_open)
            grasp_actions[0] = 1.0 if gripper_open[0] >= self._grasp_threshold else 0.0
            
            for t in range(1, len(gripper_open)):
                cur_gripper_status = gripper_open[t-1]
                next_gripper_status = gripper_open[t]
                
                if abs(next_gripper_status - cur_gripper_status) < 0.03:
                    grasp_actions[t] = grasp_actions[t-1]
                elif next_gripper_status < cur_gripper_status:
                    grasp_actions[t] = 0.0
                elif next_gripper_status > cur_gripper_status:
                    grasp_actions[t] = 1.0
            
            grasp_data = grasp_actions
        else:
            # For human data, use gripper_open directly
            grasp_data = gripper_open

        # Build action data
        if self.cfg.use_ee_data:
            # Use end-effector data: [ee_pos (3D), ee_euler (3D), grasp_data (1D)]
            action_data = np.concatenate([
                ee_pos, 
                ee_euler, 
                grasp_data.reshape(grasp_data.shape[0], -1)
            ], axis=1)
        else:
            # Use keypoint tracks: [keypoint_tracks (num_points*3D), grasp_data (1D)]
            action_data = self._reshape_action_data_3d(
                keypoints_3d, self._num_points, is_human_data
            )
            action_data = np.concatenate([
                action_data, 
                grasp_data.reshape(grasp_data.shape[0], -1)
            ], axis=1)

        # Build state conditioning data
        state_cond_data = action_data
        # Load image data if configured
        episode_image_data = None
        if self._load_images and self._camera_views and agent1_images is not None:
            episode_image_data = {}
            for camera_view in self._camera_views:
                if camera_view == "agent1":
                    images = agent1_images
                else:
                    # Skip if camera view not available
                    continue
                    
                # Ensure proper shape and normalization
                if images.dtype != np.float32:
                    images = images.astype(np.float32)
                if images.max() > 1.0:
                    images = images / 255.0
                # Ensure correct shape (T, C, H, W) for RGB images
                if images.ndim == 4 and images.shape[-1] == 3:
                    # If shape is (T, H, W, 3), transpose to (T, 3, H, W)
                    images = np.transpose(images, (0, 3, 1, 2))
                elif images.ndim == 3:
                    # If grayscale (T, H, W), convert to RGB by repeating channels
                    images = np.repeat(images[:, np.newaxis, :, :], 3, axis=1)
                
                # Apply segmentation mask if enabled (before resizing for efficiency)
                if self._apply_segmentation_mask and segmentation_data is not None and camera_view == "agent1":
                    # Convert mask to binary (non-zero values become 1)
                    binary_mask = (segmentation_data > 0).astype(np.float32)
                    
                    # Expand mask to match image channels (T, H, W) -> (T, 1, H, W)
                    binary_mask = binary_mask[:, np.newaxis, :, :]
                    
                    # Apply mask to images by multiplication
                    images = images * binary_mask
                
                # Overlay 3D keypoints if enabled
                if self._overlay_3d_keypoints and camera_view == "agent1" and keypoints_3d is not None:
                    images = self._overlay_keypoints_on_images(images, keypoints_3d)
                
                # Save first image for inspection
                # import cv2
                # first_img = images[len(images)*3//4].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                # first_img = (first_img * 255).astype(np.uint8)  # Convert to uint8 for saving
                # cv2.imwrite('debug_image.png', first_img)
                # print(f"Saved debug image to debug_image.png, shape: {first_img.shape}")
                # breakpoint()
                
                # Resize images to target dimension
                if images.shape[-1] != self._image_dim or images.shape[-2] != self._image_dim:
                    import cv2
                    # Batch resize: transpose all images at once (T, C, H, W) -> (T, H, W, C)
                    images_hwc = np.transpose(images, (0, 2, 3, 1))
                    # Resize all images in batch using cv2.resize with interpolation
                    resized_images_hwc = np.array([
                        cv2.resize(img, (self._image_dim, self._image_dim)) 
                        for img in images_hwc
                    ])
                    # Transpose back to (T, C, H, W)
                    images = np.transpose(resized_images_hwc, (0, 3, 1, 2))
                
                episode_image_data[camera_view] = images
                
        episode_data = {
            'action': action_data,
            'state_cond': state_cond_data,
            'data_type': data_type
        }
        
        return episode_data, episode_image_data

    def _reshape_action_data_3d(
        self,
        action: np.ndarray,
        num_points: int,
        is_human_data: bool,
    ) -> np.ndarray:
        """
        Reshapes 3D actions from (N, total_points, 3) to (N, num_points*3).
        """
        assert (
            len(action.shape) == 3 and action.shape[-1] == 3
        ), f"3D action data should be (N, total_points, 3), got {action.shape}"

        output = np.zeros((action.shape[0], num_points, 3), dtype=action.dtype)

        if is_human_data and action.shape[1] > 5:
            # For human data, use appropriate hand mapping
            # Assuming we want to use the first num_points keypoints
            kpt_idxs = list(range(min(num_points, action.shape[1])))
            output = action[:, kpt_idxs, :]
        else:
            # For robot data, use first num_points
            kpt_idxs = list(range(min(num_points, action.shape[1])))
            output = action[:, kpt_idxs, :]

        output = output.reshape(output.shape[0], -1)
        return output.astype(np.float32)

    def _load_calibration(self):
        """Load camera calibration data from a configurable repo-relative or absolute root."""
        calibration_root = resolve_calibration_root(self.cfg.calibration_root)
        agent1_path = calibration_root / "agent1_intrinsics.npy"
        agent2_path = calibration_root / "agent2_intrinsics.npy"
        transforms_path = calibration_root / "transforms_both.npy"

        missing = [str(path) for path in [agent1_path, agent2_path, transforms_path] if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Calibration files not found for 3D keypoint overlay. "
                f"Expected under {calibration_root}: {missing}"
            )

        intrinsics = {
            "agent1": np.load(agent1_path),
            "agent2": np.load(agent2_path),
        }
        transforms = np.load(transforms_path, allow_pickle=True).item()
        return intrinsics, transforms

    def _project_3d_to_2d(self, points_3d, intrinsics, extrinsics):
        """Project 3D points to 2D image coordinates using OpenCV's projectPoints"""
        # Extract rotation and translation from extrinsics matrix
        R_cam = extrinsics[:3, :3]
        t_cam = extrinsics[:3, 3]
        
        # Convert rotation matrix to rotation vector
        rvec, _ = cv2.Rodrigues(R_cam.astype(np.float32))
        tvec = t_cam.astype(np.float32)
        
        # Ensure intrinsics are float32
        K = intrinsics.astype(np.float32)
        
        # Project points using OpenCV
        points_3d_float = points_3d.astype(np.float32)
        pts2d, _ = cv2.projectPoints(points_3d_float, rvec, tvec, K, None)
        pts2d = pts2d.reshape(-1, 2)
        
        return pts2d

    def _overlay_keypoints_on_images(self, images, keypoints_3d):
        """Overlay 3D keypoints on images as yellow dots"""
        try:
            intrinsics, transforms = self._load_calibration()
            agent1_intrinsics = intrinsics["agent1"]
            agent1_extrinsics = transforms["agent1"]["trc"]  # 4x4 transformation matrix
            
            # Get original image dimensions (before any resizing)
            original_height, original_width = images.shape[2], images.shape[3]
            
            # Convert images to HWC format for drawing
            images_hwc = np.transpose(images, (0, 2, 3, 1))  # (T, C, H, W) -> (T, H, W, C)
            
            for t in range(images_hwc.shape[0]):
                # Project 3D keypoints to 2D
                points_2d = self._project_3d_to_2d(
                    keypoints_3d[t], agent1_intrinsics, agent1_extrinsics
                )
                
                # Draw yellow dots for valid keypoints
                for i, (x, y) in enumerate(points_2d):
                    if 0 <= x < original_width and 0 <= y < original_height:
                        # Draw a small yellow circle
                        cv2.circle(images_hwc[t], (int(x), int(y)), 5, (1.0, 1.0, 0.0), -1)
            
            # Convert back to CHW format
            images = np.transpose(images_hwc, (0, 3, 1, 2))  # (T, H, W, C) -> (T, C, H, W)
            
        except Exception as e:
            print(f"Warning: Could not overlay keypoints: {e}")
        
        return images


def create_h5_dataloader(
    partition: str,
    log_dir: str,
    batch_size: int,
    num_workers: int,
    physical_data_paths: List[str],
    rgb_seg_data_paths: List[str],
    dataset_cfg: H5DatasetConfig,
    dataloader_overrides: dict = None,
    no_transforms: bool = False,
    human_only_train: bool = False,
    use_image: bool = True,
    camera_views: List[str] = None,
    return_class_distribution: bool = False,
    create_val_dataloader: bool = False,
):
    """
    Create H5 dataloader(s) for training.
    
    Args:
        partition: "train" or "val"
        log_dir: Directory for logging
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        physical_data_paths: List of physical agent H5 data paths
        rgb_seg_data_paths: List of RGB+segmentation H5 data paths
        dataset_cfg: H5 dataset configuration
        dataloader_overrides: Override dataloader settings
        no_transforms: Whether to disable transforms
        human_only_train: Whether to exclude robot data
        use_image: Whether to use image data
        camera_views: List of camera view names to load from dataset
        return_class_distribution: Whether to return class distribution info
        create_val_dataloader: Whether to create both train and val dataloaders
    
    Returns:
        If create_val_dataloader=False: (dataloader, stats, class_distribution)
        If create_val_dataloader=True: (train_dataloader, val_dataloader, stats, train_class_dist, val_class_dist)
    """
    dataset_cfg.physical_data_paths = []
    dataset_cfg.rgb_seg_data_paths = []
    
    for physical_path, rgb_seg_path in zip(physical_data_paths, rgb_seg_data_paths):
        # Since we don't have partition folders, use the paths directly
        physical_h5_path = Path(physical_path)
        rgb_seg_h5_path = Path(rgb_seg_path)

        # Check if required H5 directories exist
        physical_exists = physical_h5_path.exists()
        rgb_exists = not use_image or rgb_seg_h5_path.exists()
        
        if physical_exists and rgb_exists:
            if partition == "train" and human_only_train and "robot" in str(physical_h5_path):
                continue

            dataset_cfg.physical_data_paths.append(physical_h5_path)
            dataset_cfg.rgb_seg_data_paths.append(rgb_seg_h5_path)
        else:
            missing_parts = []
            if not physical_exists:
                missing_parts.append(f"physical: {physical_h5_path}")
            if not rgb_exists:
                missing_parts.append(f"rgb_seg: {rgb_seg_h5_path}")
            
            cprint(
                f"[WARNING]: Skipping dataset due to missing paths: {', '.join(missing_parts)}", 
                color="yellow"
            )

    if len(dataset_cfg.physical_data_paths) == 0:
        cprint(f"[ERROR]: No valid H5 data paths found for {partition}", color="red")
        if create_val_dataloader:
            return None, None, None, None, None
        elif return_class_distribution:
            return None, None, None
        else:
            return None, None

    if no_transforms:
        original_transforms = dataset_cfg.transforms_name
        dataset_cfg.transforms_name = "none"
    
    # Set camera views and image loading if provided
    if camera_views is not None:
        dataset_cfg.camera_views = camera_views
        dataset_cfg.load_images = use_image
        cprint(f"[H5_DATASET] Loading images for camera views: {camera_views}", color="cyan")
    
    # Create training dataset
    train_dataset = H5Dataset(dataset_cfg, partition=partition)
    train_dataset._use_image = use_image
    
    # Create validation dataset if requested
    val_dataset = None
    if create_val_dataloader:
        val_dataset_cfg = dataset_cfg
        val_dataset = H5Dataset(val_dataset_cfg, partition="val")
        val_dataset._use_image = use_image
    
    if no_transforms:
        dataset_cfg.transforms_name = original_transforms

    dset_str = ",".join([path.parent.name for path in dataset_cfg.physical_data_paths])
    cprint(
        f" Created {partition.capitalize()} H5 dataset with {len(train_dataset)} samples from [{dset_str}]",
        color="cyan",
        attrs=["bold"],
    )
    
    if create_val_dataloader and val_dataset:
        cprint(f" Created validation H5 dataset with {len(val_dataset)} samples from [{dset_str}]",
               color="cyan", attrs=["bold"])
    
    # Get class distribution from the datasets if requested
    train_class_dist = None
    val_class_dist = None
    if return_class_distribution:
        train_class_dist = train_dataset.get_class_distribution()
        cprint(f"Training class distribution: {train_class_dist}", color="cyan", attrs=["bold"])
        
        if create_val_dataloader and val_dataset:
            val_class_dist = val_dataset.get_class_distribution()
            cprint(f"Validation class distribution: {val_class_dist}", color="cyan", attrs=["bold"])
    
    stats = train_dataset.stats
    Path(log_dir, "dataset_stats").mkdir(parents=True, exist_ok=True)
    np.save(f"{log_dir}/dataset_stats/{partition}.npy", stats)

    base_dataloader_kwargs = dict(
        batch_size=batch_size,
        num_workers=0,  # Set to 0 for H5 files to avoid multiprocessing issues
        shuffle=True,
        pin_memory=True,
        persistent_workers=False,
    )
    if dataloader_overrides is not None:
        base_dataloader_kwargs.update(dataloader_overrides)
    
    # Handle balanced sampling if configured
    sampler = None
    if dataset_cfg.balanced_sampling_weights is not None:
        sample_weights = train_dataset.create_sample_weights(dataset_cfg.balanced_sampling_weights)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        # Remove shuffle when using sampler
        base_dataloader_kwargs["shuffle"] = False
        cprint(f"Using balanced sampling with weights: {dataset_cfg.balanced_sampling_weights}", 
               color="cyan", attrs=["bold"])
        
        # Log the label distribution
        label_dist = train_dataset.get_label_distribution()
        cprint(f"Label distribution: {label_dist}", color="yellow")
    
    # Create training dataloader
    train_dataloader_kwargs = base_dataloader_kwargs.copy()
    if sampler is not None:
        train_dataloader_kwargs["sampler"] = sampler
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        **train_dataloader_kwargs,
    )
    
    # Create validation dataloader if requested
    val_dataloader = None
    if create_val_dataloader and val_dataset:
        val_dataloader_kwargs = base_dataloader_kwargs.copy()
        val_dataloader_kwargs["shuffle"] = False  # No shuffling for validation
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            **val_dataloader_kwargs,
        )
    
    # Return appropriate values based on what was requested
    if create_val_dataloader:
        if return_class_distribution:
            return train_dataloader, val_dataloader, stats, train_class_dist, val_class_dist
        else:
            return train_dataloader, val_dataloader, stats
    else:
        if return_class_distribution:
            return train_dataloader, stats, train_class_dist
        else:
            return train_dataloader, stats
