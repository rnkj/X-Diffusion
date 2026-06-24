import torch
import torch.nn as nn
from termcolor import cprint
from pathlib import Path
from common_utils.checkpointer import Checkpointer


class KeypointMapPredictor(nn.Module):
    def __init__(self, num_keypoints: int, keypoint_dim: int = 2):
        """
        Initializes the KeypointPredictor module with an encoder network and a keypoint predictor network.

        Args:
            encoder (nn.Module): The encoder network.
            keypoint_predictor (nn.Module): The keypoint predictor network.
        """
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_keypoints * keypoint_dim, 128),
            nn.Tanh(),
            nn.Linear(128, num_keypoints * keypoint_dim),
        )

    def forward(self, keypoints: torch.Tensor):
        """
        Args:
            keypoints (torch.Tensor): The keypoints tensor of shape (B, num_keypoints * 2).
        """
        return self.fc(keypoints)


def load_keypoint_map_predictor(log_dir: str, num_keypoints: int, keypoint_dim: int = 2):
    # Load and freeze predictor network
    policy = KeypointMapPredictor(num_keypoints, keypoint_dim)
    checkpointer = Checkpointer(Path(log_dir))
    checkpointer.update_checkpoint_from_dir(Path(log_dir, "checkpoints"))
    policy, loaded_checkpoint = Checkpointer.load_checkpoint(
        policy, checkpointer, checkpoint_type="best"
    )

    # Freeze the policy network
    for param in policy.parameters():
        param.requires_grad = False
    policy.eval()

    return policy
