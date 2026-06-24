from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from termcolor import cprint

from dataset_utils.common import H5Batch as Batch
from models.policy_nets.unet import HalfUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from common_utils.data_aug import RandomShiftsAug



@dataclass
class MultiviewHalfUnetConfig:
    pass

class MultiviewHalfUnet:
    def __init__(self, *args, **kwargs):
        raise ImportError("ImageHumanRobotClassifier requires baselines, which is intentionally not included in X-Diffusion classifier-only training.")

@dataclass
class HumanRobotClassifierConfig:
    # HalfUnet1D configuration
    half_unet_down_dims: List[int] = field(default_factory=lambda: [256, 512, 1024])
    half_unet_kernel_size: int = 5
    half_unet_n_groups: int = 8
    half_unet_diffusion_step_embed_dim: int = 256
    
    # Noise scheduler configuration
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    clip_sample: bool = True
    prediction_type: str = "epsilon"
    
    # Noise addition configuration
    add_noise_to_sample: bool = True  # Add noise to actions (sample)
    add_noise_to_state_cond: bool = True  # Add noise to observations (state_cond)


class HumanRobotClassifier(nn.Module):
    def __init__(
        self,
        obs_horizon: int,
        state_cond_dim: int,
        action_dim: int,
        cfg: HumanRobotClassifierConfig,
    ):
        super().__init__()

        self.obs_horizon = obs_horizon
        self.cfg = cfg
        
        # Initialize BCE loss with logits
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

        # Setup HalfUnet1D for feature processing
        self.half_unet = HalfUnet1D(
            input_dim=action_dim,
            cond_dim=state_cond_dim,
            obs_horizon=obs_horizon,
            diffusion_step_embed_dim=cfg.half_unet_diffusion_step_embed_dim,
            down_dims=cfg.half_unet_down_dims,
            kernel_size=cfg.half_unet_kernel_size,
            n_groups=cfg.half_unet_n_groups,
        )
        
        # Final classification layer after HalfUnet1D
        final_channel_dim = cfg.half_unet_down_dims[-1]
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global average pooling: (B, C, T) -> (B, C, 1)
            nn.Flatten(),  # (B, C, 1) -> (B, C)
            nn.Linear(final_channel_dim, 1)  # Final classification layer
        )
        
        # Setup noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.num_train_timesteps,
            beta_schedule=cfg.beta_schedule,
            clip_sample=cfg.clip_sample,
            prediction_type=cfg.prediction_type,
        )

    def add_noise_to_data(
        self,
        data: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to input data using the noise scheduler
        
        Args:
            data: Input data tensor of shape (B, T, D)
            timesteps: Optional pre-computed timesteps to use. If provided, noise_timestep_range is ignored.
        
        Returns:
            noisy_data: Data with noise added
            timesteps: The timesteps used for noise addition
        """
        batch_size = data.shape[0]
        device = data.device
        
        # Generate random noise
        noise = torch.randn(data.shape, device=device)
        
        # Use provided timesteps or sample new ones
        if timesteps is None:
            timesteps = torch.randint(
                low=0,
                high=self.noise_scheduler.config["num_train_timesteps"],
                size=(batch_size,),
                device=device,
            ).long()
        # Add noise using the scheduler
        noisy_data = self.noise_scheduler.add_noise(data, noise, timesteps)
        
        return noisy_data, timesteps

    def forward(
        self,
        batch: Batch,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for classification with optional noise addition
        
        Args:
            sample: Actions tensor of shape (B, obs_horizon, action_dim)
            state_cond: Observations/Proprioceptive state conditioning of shape (B, obs_horizon, state_cond_dim)
            noise_timestep_range: Optional tuple (min_timestep, max_timestep) for noise level
            
        Returns:
            logits: Classification logits of shape (B, 1)
        """
        batch_size = batch.action.shape[0]
        
        # Apply noise based on configuration
        noisy_sample = batch.action
        noisy_state_cond = batch.state_cond

        noisy_sample, timesteps = self.add_noise_to_data(noisy_sample, timesteps=timesteps)
        if self.cfg.add_noise_to_state_cond:
            noisy_state_cond, _ = self.add_noise_to_data(noisy_state_cond, timesteps=timesteps)
        
        # Use HalfUnet1D for feature processing
        # Flatten state_cond for conditioning: (B, obs_horizon, state_cond_dim) -> (B, obs_horizon * state_cond_dim)
        state_cond_flat = noisy_state_cond.reshape(batch_size, -1)
        
        half_unet_features = self.half_unet(
            sample=noisy_sample,  # (B, obs_horizon, action_dim)
            timestep=timesteps if timesteps is not None else 0,  # Use timesteps
            cond=state_cond_flat
        )
        
        logits = self.classifier(half_unet_features)
        
        return logits

    def loss(
        self,
        batch: Batch,
    ) -> torch.Tensor:
        """
        Calculate binary cross entropy loss for classification using BCEWithLogitsLoss
        
        Args:
            batch: Batch containing action, state_cond, and label
            
        Returns:
            loss: Classification loss
        """
        # Ensure we have labels
        if not hasattr(batch, 'label') or batch.label is None:
            raise ValueError("Batch must contain 'label' field for classification training")
        
        # Get predictions (raw logits) with noise
        logits = self.forward(batch)
        # Calculate loss using BCEWithLogitsLoss (handles sigmoid internally)
        # Convert labels to float for BCE loss
        labels_float = batch.label.float()
        loss = self.bce_loss(logits.squeeze(-1), labels_float)
        
        return loss

    def unified_forward(
        self,
        batch: Batch,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for classification with optional noise addition
        """
        batch_size = batch.action.shape[0]
        # Apply noise based on configuration
        sample = batch.action
        state_cond = batch.state_cond
        
        noisy_sample, _ = self.add_noise_to_data(sample, timesteps=timesteps)
        noisy_state_cond, _ = self.add_noise_to_data(state_cond, timesteps=timesteps)

        state_cond_flat = noisy_state_cond.reshape(batch_size, -1)
        
        half_unet_features = self.half_unet(
            sample=noisy_sample,  # (B, obs_horizon, action_dim)
            timestep=timesteps if timesteps is not None else 0,  # Use timesteps if noise was added, otherwise 0
            cond=state_cond_flat  # Conditioning on flattened observations (B, obs_horizon * state_cond_dim)
        )
        # half_unet_features shape: (B, C, T) where C is final channel dimension
        
        # Apply final classification layer
        logits = self.classifier(half_unet_features)
        
        return logits

@dataclass
class ImageHumanRobotClassifierConfig:
    # HalfUnet1D configuration
    half_unet_down_dims: List[int] = field(default_factory=lambda: [256, 512, 1024])
    half_unet_kernel_size: int = 5
    half_unet_n_groups: int = 8
    half_unet_diffusion_step_embed_dim: int = 256
    
    # Noise scheduler configuration
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    clip_sample: bool = True
    prediction_type: str = "epsilon"
    
    # Noise addition configuration
    add_noise_to_sample: bool = True  # Add noise to actions (sample)
    add_noise_to_state_cond: bool = True  # Add noise to observations (state_cond)

    half_unet: MultiviewHalfUnetConfig = field(default_factory=lambda: MultiviewHalfUnetConfig())

    shift_pad: int = 4


class ImageHumanRobotClassifier(nn.Module):
    def __init__(
        self,
        obs_horizon: int,
        obs_shape: tuple,
        prop_dim: int,
        action_dim: int,
        camera_views: List[str],
        cfg: ImageHumanRobotClassifierConfig,
    ):
        super().__init__()

        self.obs_horizon = obs_horizon
        self.obs_shape = obs_shape
        self.prop_dim = prop_dim
        self.action_dim = action_dim
        self.camera_views = camera_views
        self.cfg = cfg
        
        # Initialize BCE loss with logits
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

        # Setup HalfUnet1D for feature processing
        self.half_unet = MultiviewHalfUnet(
            obs_shape=obs_shape,
            obs_horizon=obs_horizon,
            prop_dim=prop_dim,
            cameras=camera_views,
            action_dim=action_dim,
            cfg=cfg.half_unet,
        )        
        self.aug = RandomShiftsAug(pad=cfg.shift_pad)
        # Final classification layer after HalfUnet1D
        final_channel_dim = cfg.half_unet_down_dims[-1]
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global average pooling: (B, C, T) -> (B, C, 1)
            nn.Flatten(),  # (B, C, 1) -> (B, C)
            nn.Linear(final_channel_dim, 1)  # Final classification layer
        )
        
        # Setup noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.num_train_timesteps,
            beta_schedule=cfg.beta_schedule,
            clip_sample=cfg.clip_sample,
            prediction_type=cfg.prediction_type,
        )

    def add_noise_to_data(
        self,
        data: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to input data using the noise scheduler
        
        Args:
            data: Input data tensor of shape (B, T, D)
            timesteps: Optional pre-computed timesteps to use. If provided, noise_timestep_range is ignored.
        
        Returns:
            noisy_data: Data with noise added
            timesteps: The timesteps used for noise addition
        """
        batch_size = data.shape[0]
        device = data.device
        
        # Generate random noise
        noise = torch.randn(data.shape, device=device)

        if timesteps is None:
            timesteps = torch.randint(
                low=0,
                high=self.cfg.num_train_timesteps,
                size=(batch_size,),
                device=device,
            ).long()
        
        # Add noise using the scheduler
        noisy_data = self.noise_scheduler.add_noise(data, noise, timesteps)
        
        return noisy_data, timesteps

    def forward(
        self,
        batch: Batch,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for classification with optional noise addition
        
        Args:
            batch: Actions tensor of shape (B, obs_horizon, action_dim)
            noise_timestep_range: Optional tuple (min_timestep, max_timestep) for noise level
            
        Returns:
            logits: Classification logits of shape (B, 1)
        """
        batch_size = batch.action.shape[0]
        
        # Apply noise based on configuration
        sample = batch.action
        noisy_state_cond = batch.state_cond

        noisy_sample, timesteps = self.add_noise_to_data(sample, timesteps=timesteps)
        noisy_obs = {}
        for k, v in batch.images.items():
            if k in self.camera_views:
                if self.cfg.add_noise_to_state_cond:
                    noisy_obs[k], _ = self.add_noise_to_data(self.aug(v.float()), timesteps=timesteps)
                else:
                    noisy_obs[k] = self.aug(v.float()) if hasattr(self, 'aug') else v
        if self.cfg.add_noise_to_state_cond:
            noisy_obs["prop"], _ = self.add_noise_to_data(batch.state_cond.reshape(batch_size, -1), timesteps=timesteps)
        else:
            noisy_obs["prop"] = batch.state_cond.reshape(batch_size, -1)
        
        half_unet_features = self.half_unet(noisy_obs, noisy_sample, timesteps)
        # half_unet_features shape: (B, C, T) where C is final channel dimension
        
        # Apply final classification layer
        logits = self.classifier(half_unet_features)
        
        return logits

    def loss(
        self,
        batch: Batch,
    ) -> torch.Tensor:
        """
        Calculate binary cross entropy loss for classification using BCEWithLogitsLoss
        
        Args:
            batch: Batch containing action, state_cond, and label
            noise_timestep_range: Optional tuple (min_timestep, max_timestep) for noise level
            
        Returns:
            loss: Classification loss
        """
        # Ensure we have labels
        if not hasattr(batch, 'label') or batch.label is None:
            raise ValueError("Batch must contain 'label' field for classification training")
        
        # Get predictions (raw logits) with noise
        logits = self.forward(batch)
        # Calculate loss using BCEWithLogitsLoss (handles sigmoid internally)
        # Convert labels to float for BCE loss
        labels_float = batch.label.float()
        loss = self.bce_loss(logits.squeeze(-1), labels_float)
        
        return loss

def test():
    """Test the HumanRobotClassifier"""
    print("Testing Human Robot Classifier")
    
    # Define parameters
    obs_horizon = 2
    state_cond_dim = 10
    action_dim = 10
    
    # Create config
    cfg = HumanRobotClassifierConfig(
        half_unet_down_dims=[256, 512, 1024],
        half_unet_kernel_size=5,
        half_unet_n_groups=8,
        half_unet_diffusion_step_embed_dim=256,
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
        add_noise_to_sample=True,
        add_noise_to_state_cond=True,
    )
    
    # Create dummy input tensors
    batch_size = 4
    sample = torch.rand(batch_size, obs_horizon, action_dim)  # Actions as sample
    state_cond = torch.rand(batch_size, obs_horizon, state_cond_dim)  # Observations as state_cond
    labels = torch.randint(0, 2, (batch_size,))  # Binary labels
    
    # Instantiate classifier
    classifier = HumanRobotClassifier(
        obs_horizon=obs_horizon,
        state_cond_dim=state_cond_dim,
        action_dim=action_dim,
        cfg=cfg,
    )
    
    # Test forward pass without noise
    logits = classifier(sample, state_cond)
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (batch_size, 1), f"Expected (4, 1), got {logits.shape}"
    
    # Test forward pass with noise
    logits_with_noise = classifier(sample, state_cond, noise_timestep_range=(0.5, 0.7))
    print(f"Logits with noise shape: {logits_with_noise.shape}")
    assert logits_with_noise.shape == (batch_size, 1), f"Expected (4, 1), got {logits_with_noise.shape}"
    
    # Test loss calculation
    batch = Batch(
        obs=torch.rand(batch_size, obs_horizon, 3, 96, 96),  # Dummy observations
        depth=torch.rand(batch_size, obs_horizon, 1, 96, 96),  # Dummy depth
        action=sample,  # Use the sample (actions) we created
        state_cond=state_cond,
        label=labels,
    )
    
    # Test loss without noise
    loss = classifier.loss(batch)
    print(f"Loss: {loss.item()}")
    assert loss.dim() == 0, f"Expected scalar loss, got {loss.shape}"
    
    # Test loss with noise
    loss_with_noise = classifier.loss(batch, noise_timestep_range=(0.3, 0.8))
    print(f"Loss with noise: {loss_with_noise.item()}")
    assert loss_with_noise.dim() == 0, f"Expected scalar loss, got {loss_with_noise.shape}"
    
    print("Test passed successfully!")


if __name__ == "__main__":
    test()