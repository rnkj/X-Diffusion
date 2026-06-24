from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce
from line_profiler import profile
from termcolor import cprint

from collections import namedtuple

from dataset_utils.common import ACTION_TYPE_TRACKS, ACTION_TYPES, H5Batch as Batch
from models.encoders import SUPPORTED_ENCODERS, get_encoder_by_name
from models.xdiffusion.feature_compressor import FeatureCompressor
from models.xdiffusion.noise_predictor import NoisePredictor
from models.policy_nets import ConditionalUnet1D, TransformerForDiffusion
from models.train_utils import absolute_to_delta_action, delta_to_absolute_action
from models.keypoint_map_predictor import load_keypoint_map_predictor
from common_utils.data_aug import RandomShiftsAug
from baselines.models.action_normalizer import ActionNormalizer
from baselines.models.dp_net import MultiviewCondUnet, MultiviewCondUnetConfig


@dataclass
class DDPMConfig:
    num_train_timesteps: int = 100
    num_inference_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    clip_sample: bool = True
    prediction_type: str = "epsilon"


@dataclass
class DDIMConfig:
    num_train_timesteps: int = 100
    num_inference_timesteps: int = 16
    beta_schedule: str = "squaredcos_cap_v2"
    clip_sample: bool = True
    set_alpha_to_one: bool = True
    steps_offset: int = 0
    prediction_type: str = "epsilon"


@dataclass
class DiffusionPolicyConfig:
    # arch
    encoder: str = "resnet50"
    input_type: str = "image"
    backbone: str = "unet"
    add_depth_gs: bool = False
    add_depth_colored: bool = False
    cam_names: List[str] = field(default_factory=lambda: ["main"])

    # finetuning
    freeze_encoder: bool = False
    pretrained_encoder: bool = True
    num_compress_layers: int = 1
    upsample_prop: bool = False
    layers_to_freeze: List[str] = field(default_factory=lambda: [])

    # auxiliary losses
    learn_prop_fusion: bool = False
    balance_grasp_loss: bool = True
    adv_domain_loss: bool = False
    kpt_loss_weight: float = 0.0
    kl_loss_weight: float = 0.0
    meanvar_loss_weight: float = 0.0
    img_dropout: float = 0.0

    # keypoint mapping
    noise_only_state_cond: bool = True
    keypoint_map_from: Optional[str] = None

    # noise scheduler
    use_ddpm: bool = False
    ddpm: DDPMConfig = field(default_factory=lambda: DDPMConfig())
    ddim: DDIMConfig = field(default_factory=lambda: DDIMConfig())

    # track prediction
    action_pred_type: str = field(default=ACTION_TYPE_TRACKS)
    add_cond_vector: bool = True  # whether to condition on eef or proprio state info
    add_grasp_info_to_tracks: bool = False  # add additional dimension for grasp
    delta_actions: bool = True  # predict delta points instead of absolute

    # legacy
    prop_bias_estimator: bool = False
    train_only_bias: bool = False
    add_in_wrist: bool = False  
    domain_loss_weight: float = 0.0

    # human-specific noise scheduling
    human_min_noise_level: float = 0.0  # minimum noise level for human data (0.0-1.0)

    def __post_init__(self):
        assert self.action_pred_type in ACTION_TYPES
        assert self.input_type in [
            "image",
            "pc",
        ], f"Unknown input type: {self.input_type}"
        assert self.encoder in SUPPORTED_ENCODERS, f"Unknown encoder: {self.encoder}"
        assert self.backbone in ["unet"], f"Unknown backbone: {self.backbone}"

        if (
            self.add_depth_gs or self.add_depth_colored
        ) and "depth" not in self.cam_names:
            self.cam_names = self.cam_names.append("depth")


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        num_points: int,
        action_dim: int,
        obs_horizon: int,
        pred_horizon: int,
        action_horizon: int,
        state_cond_dim: int,
        cfg: DiffusionPolicyConfig,
        use_image: bool = True,
    ):
        super().__init__()

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.num_points = num_points
        self.action_dim = action_dim
        self.cfg = cfg
        self.use_image = use_image

        if cfg.domain_loss_weight > 0.0:
            self.adv_domain_loss = True

        self.noise_predictor: NoisePredictor = self._setup_noise_predictor(
            cfg, action_dim, obs_horizon, pred_horizon, state_cond_dim
        )

        if cfg.use_ddpm:
            scheduler_kwargs = asdict(cfg.ddpm)
            self.num_inference_timesteps = scheduler_kwargs.pop(
                "num_inference_timesteps"
            )
            self.num_train_timesteps = scheduler_kwargs["num_train_timesteps"]
            self.noise_scheduler = DDPMScheduler(**scheduler_kwargs)
            cprint("Using DDPM", "green", attrs=["bold"])
        else:
            scheduler_kwargs = asdict(cfg.ddim)
            self.num_inference_timesteps = scheduler_kwargs.pop(
                "num_inference_timesteps"
            )
            self.num_train_timesteps = scheduler_kwargs["num_train_timesteps"]
            self.noise_scheduler = DDIMScheduler(**scheduler_kwargs)
            cprint("Using DDIM", "green", attrs=["bold"])

    def _setup_noise_predictor(
        self,
        cfg: DiffusionPolicyConfig,
        action_dim: int,
        obs_horizon: int,
        pred_horizon: int,
        state_cond_dim: int,
    ) -> nn.Module:
        cond_dim = int(cfg.add_cond_vector) * state_cond_dim
        if cfg.backbone == "unet":
            noise_pred_net = ConditionalUnet1D(
                input_dim=action_dim,
                obs_horizon=obs_horizon,
                cond_dim=cond_dim,
            )
        elif cfg.backbone == "transformer":
            noise_pred_net = TransformerForDiffusion(
                input_dim=action_dim,
                output_dim=action_dim,
                horizon=pred_horizon,
                n_obs_steps=obs_horizon,
                n_layer=8,
                n_head=4,
                n_emb=256,
                p_drop_emb=0.0,
                p_drop_attn=0.3,
                n_cond_layers=0,
                causal_attn=True,
                time_as_cond=True,
                obs_as_cond=True,
                cond_dim=cond_dim,
            )
        else:
            raise NotImplementedError(f"Missing backbone: {cfg.backbone}")

        noise_predictor = NoisePredictor(
            noise_pred_net, cfg.layers_to_freeze
        )
        return noise_predictor

    @torch.no_grad()
    def act(
        self,
        state_cond: torch.Tensor,
        *,
        first_action_abs: torch.Tensor = None,
        return_full_action: bool = False,
        cpu: bool = True,
    ) -> torch.Tensor:
        """Generate action from noise"""
        assert not self.training
        assert not (
            self.cfg.delta_actions and first_action_abs is None
        ), "first_action_abs is required for delta_actions"
        
        batched = True
        if state_cond is not None and state_cond.dim() == 2:
            # state_cond: (obs_horizon, cond_dim)
            batched = False
            state_cond = state_cond.unsqueeze(0)
            first_action_abs = first_action_abs.unsqueeze(0)

        bsize = state_cond.size(0)
        device = state_cond.device

        # pure noise input to begin with
        noisy_action = torch.randn(
            (bsize, self.pred_horizon, self.action_dim), device=device
        )
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            noise_pred, obs_emb = (
                self.noise_predictor.predict_noise(
                    noisy_action,
                    k,
                    additional_cond=state_cond,
                )
            )
            # inverse diffusion step (remove noise)
            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=noisy_action,  # type: ignore
            ).prev_sample.detach()  # type: ignore

        action = noisy_action


        if not batched:
            action = action.squeeze(0)
        if cpu:
            action = action.cpu()
        return action

    @profile
    def loss(
        self,
        batch: Batch,
        avg: bool = True,
        action_mse: bool = False,
        progress: int = 0,
    ) -> torch.Tensor:
        """
        Calculate loss for the diffusion model.

        If `action_mse` is true, the loss is calculated as the MSE between the predicted
        actions and the ground truth actions. This is used for monitoring only and is not
        used in optimization.
        """
        actions = batch.action
        state_cond = batch.state_cond
        bsize = actions.size(0)
        
        
        noise = torch.randn(actions.shape, device=actions.device)
        # sample a diffusion iteration for each data point
        has_labels = hasattr(batch, 'label') and batch.label is not None
        if has_labels and self.cfg.human_min_noise_level > 0.0:
            # Separate human and robot data for different noise levels
            human_idx = torch.where(batch.label == 0)[0]
            robot_idx = torch.where(batch.label == 1)[0]
            timesteps = torch.zeros(bsize, device=actions.device, dtype=torch.long)
            
            if len(human_idx) > 0:
                # Human data: use minimum noise level
                min_timestep = int(self.cfg.human_min_noise_level * self.noise_scheduler.config["num_train_timesteps"])
                timesteps[human_idx] = torch.randint(
                    low=min_timestep,
                    high=self.noise_scheduler.config["num_train_timesteps"],
                    size=(len(human_idx),),
                    device=actions.device,
                ).long()
            
            if len(robot_idx) > 0:
                # Robot data: use full range
                timesteps[robot_idx] = torch.randint(
                    low=0,
                    high=self.noise_scheduler.config["num_train_timesteps"],
                    size=(len(robot_idx),),
                    device=actions.device,
                ).long()
        else:
            # Default behavior: use full range for all data
            timesteps = torch.randint(
                low=0,
                high=self.noise_scheduler.config["num_train_timesteps"],
                size=(bsize,),
                device=actions.device,
            ).long()
        
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)  # type: ignore
        noise_pred, obs_emb = (
            self.noise_predictor.predict_noise(
                noisy_actions,
                timesteps,
                additional_cond=state_cond if self.cfg.add_cond_vector else None,
            )
        )

        loss_dict = {}

        # Noise prediction loss
        noise_loss = nn.functional.mse_loss(noise_pred, noise, reduction="none")
        action_noise_loss = noise_loss[:, :, :-1]
        grasp_noise_loss = noise_loss[:, :, -1:]
        action_loss = noise_loss.sum(dim=2)

        loss_dict["total/action_noise"] = action_noise_loss.mean()
        loss_dict["total/grasp_noise"] = grasp_noise_loss.mean()

        if avg:
            # Reduction from official DP repo
            action_loss = reduce(action_loss, "b ... -> b (...)", "mean")
            action_loss = action_loss.mean()
        else:
            action_loss = action_loss

        final_loss = (
            action_loss
        )

        return final_loss, loss_dict

    def unified_loss(self, batch: Batch, timesteps: torch.Tensor, loss_masks: torch.Tensor, avg: bool = True) -> torch.Tensor:
        """
        Calculate loss for the diffusion model.
        """
        actions = batch.action
        state_cond = batch.state_cond
        bsize = actions.size(0)
        noise = torch.randn(actions.shape, device=actions.device)
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)  # type: ignore
        noise_pred, obs_emb = (
            self.noise_predictor.predict_noise(
                noisy_actions,
                timesteps,
                additional_cond=state_cond if self.cfg.add_cond_vector else None,
            )
        )
        loss_dict = {}

        # Noise prediction loss
        noise_loss = nn.functional.mse_loss(noise_pred, noise, reduction="none")
        action_noise_loss = noise_loss[:, :, :-1]
        grasp_noise_loss = noise_loss[:, :, -1:]
        action_loss = (noise_loss*loss_masks[:, None, None]).sum(dim=2)

        if avg:
            # Reduction from official DP repo
            action_loss = reduce(action_loss, "b ... -> b (...)", "mean")
            action_loss = action_loss.mean()
        else:
            action_loss = action_loss

        final_loss = (
            action_loss
        )

        loss_dict["total/action_noise"] = action_noise_loss.mean()
        loss_dict["total/grasp_noise"] = grasp_noise_loss.mean()

        return final_loss, loss_dict

    def _symm_kl_embedding_loss(self, emb1, emb2):
        # Ensure emb1 and emb2 have the same batch size
        min_batch = min(emb1.size(0), emb2.size(0))
        emb1 = emb1[:min_batch]
        emb2 = emb2[:min_batch]

        mu1, sigma1 = torch.mean(emb1, dim=1), torch.std(emb1, dim=1)  # (B, emb_dim)
        mu2, sigma2 = torch.mean(emb2, dim=1), torch.std(emb2, dim=1)  # (B, emb_dim)

        # KL Loss from emb1 to emb2
        term1 = torch.log(sigma2 / sigma1)
        term2 = (sigma1**2 + (mu1 - mu2) ** 2) / (2 * sigma2**2)
        kl_loss_1to2 = term1 + term2 - 0.5

        # KL Loss from emb2 to emb1
        term1 = torch.log(sigma1 / sigma2)
        term2 = (sigma2**2 + (mu2 - mu1) ** 2) / (2 * sigma1**2)
        kl_loss_2to1 = term1 + term2 - 0.5

        kl_loss = (kl_loss_1to2 + kl_loss_2to1) / 2.0

        return kl_loss.sum()


# ImageDiffusionPolicy classes based on reference code
@dataclass
class ImageDDPMConfig:
    num_train_timesteps: int = 100
    num_inference_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    clip_sample: int = 1
    prediction_type: str = "epsilon"


@dataclass
class ImageDDIMConfig:
    num_train_timesteps: int = 100
    num_inference_timesteps: int = 10
    beta_schedule: str = "squaredcos_cap_v2"
    clip_sample: int = 1
    set_alpha_to_one: int = 1
    steps_offset: int = 0
    prediction_type: str = "epsilon"


# Use the actual MultiviewCondUnetConfig from baselines


@dataclass
class ImageDiffusionPolicyConfig:
    # algo
    use_ddpm: int = 1
    ddpm: ImageDDPMConfig = field(default_factory=lambda: ImageDDPMConfig())
    ddim: ImageDDIMConfig = field(default_factory=lambda: ImageDDIMConfig())
    action_horizon: int = 8
    prediction_horizon: int = 8
    shift_pad: int = 4
    # arch
    cond_unet: MultiviewCondUnetConfig = field(default_factory=lambda: MultiviewCondUnetConfig())
    
    # human-specific noise scheduling
    human_min_noise_level: float = 0.0  # minimum noise level for human data (0.0-1.0)


class ImageDiffusionPolicy(nn.Module):
    def __init__(
        self,
        obs_horizon,
        obs_shape,
        prop_dim: int,
        action_dim: int,
        camera_views,
        cfg: ImageDiffusionPolicyConfig,
    ):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.obs_shape = obs_shape
        self.prop_dim = prop_dim * obs_horizon
        self.action_dim = action_dim
        self.camera_views = camera_views
        self.cfg = cfg

        # for data augmentation in training
        self.aug = RandomShiftsAug(pad=cfg.shift_pad)

        # we concat image in dataset & env_wrapper
        if self.obs_horizon > 1:
            obs_shape = (obs_shape[0] // self.obs_horizon, obs_shape[1], obs_shape[2])

        self.net = MultiviewCondUnet(
            obs_shape,
            obs_horizon,
            prop_dim,
            camera_views,
            action_dim,
            cfg.cond_unet,
        )

        if cfg.use_ddpm:
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=cfg.ddpm.num_train_timesteps,
                beta_schedule=cfg.ddpm.beta_schedule,
                clip_sample=bool(cfg.ddpm.clip_sample),
                prediction_type=cfg.ddpm.prediction_type,
            )
        else:
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=cfg.ddim.num_train_timesteps,
                beta_schedule=cfg.ddim.beta_schedule,
                clip_sample=bool(cfg.ddim.clip_sample),
                set_alpha_to_one=bool(cfg.ddim.set_alpha_to_one),
                steps_offset=cfg.ddim.steps_offset,
                prediction_type=cfg.ddim.prediction_type,
            )
        self.to("cuda")

    @torch.no_grad()
    def act_demodiffusion(self, obs: dict[str, torch.Tensor], *, cpu=True, start_timestep_pct=None, stop_timestep_pct=None, start_action=None):
        assert not self.training

        unsqueezed = False
        if obs[self.camera_views[0]].dim() == 3:
            unsqueezed = True
            for k, v in obs.items():
                obs[k] = v.unsqueeze(0)

        bsize = obs[self.camera_views[0]].size(0)
        device = obs[self.camera_views[0]].device

        obs["prop"] = obs["prop"].reshape(bsize, -1)

        # pure noise input to begine with
        if start_action is not None:
            noisy_action = start_action
        else:
            noisy_action = torch.randn(
                (bsize, self.cfg.prediction_horizon, self.action_dim), device=device
            )

        if self.cfg.use_ddpm:
            num_inference_timesteps = self.cfg.ddpm.num_inference_timesteps
        else:
            num_inference_timesteps = self.cfg.ddim.num_inference_timesteps

        num_inference_timesteps = 20
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        cached_image_emb = None
        all_actions = []
        num_steps = 0
        for k in self.noise_scheduler.timesteps:
            if start_timestep_pct is not None and k > self.noise_scheduler.timesteps[-1 * int(start_timestep_pct * num_inference_timesteps)]:
                continue
            elif stop_timestep_pct is not None and k <= self.noise_scheduler.timesteps[-1 * int(stop_timestep_pct * num_inference_timesteps)]:
                break

            noise_pred, cached_image_emb = self.net.predict_noise(
                obs, noisy_action, k, cached_image_emb
            )

            # inverse diffusion step (remove noise)
            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=noisy_action  # type: ignore
            ).prev_sample.detach()  # type: ignore

            all_actions.append(noisy_action)
            num_steps += 1

        action = noisy_action
        # when obs_horizon=2, the model was trained as
        # o_0, o_1,
        # a_0, a_1, a_2, a_3, ..., a_{h-1}  -> action_horizon number of predictions
        # so we DO NOT use the first prediction at test time
        action = action[:, self.obs_horizon - 1 : self.cfg.action_horizon]
        print(f"> Number of diffusionsteps: {num_steps}")
        # No action normalization needed - dataset handles it
        if unsqueezed:
            action = action.squeeze(0)
        if cpu:
            action = action.cpu()
        return action


    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor], *, cpu=True):
        assert not self.training

        unsqueezed = False
        if obs[self.camera_views[0]].dim() == 3:
            unsqueezed = True
            for k, v in obs.items():
                obs[k] = v.unsqueeze(0)

        bsize = obs[self.camera_views[0]].size(0)
        device = obs[self.camera_views[0]].device

        obs["prop"] = obs["prop"].reshape(bsize, -1)

        # pure noise input to begine with
        noisy_action = torch.randn(
            (bsize, self.cfg.prediction_horizon, self.action_dim), device=device
        )

        if self.cfg.use_ddpm:
            num_inference_timesteps = self.cfg.ddpm.num_inference_timesteps
        else:
            num_inference_timesteps = self.cfg.ddim.num_inference_timesteps

        num_inference_timesteps = 20
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        cached_image_emb = None
        all_actions = []
        num_steps = 0
        for k in self.noise_scheduler.timesteps:
            noise_pred, cached_image_emb = self.net.predict_noise(
                obs, noisy_action, k, cached_image_emb
            )

            # inverse diffusion step (remove noise)
            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=noisy_action  # type: ignore
            ).prev_sample.detach()  # type: ignore

            all_actions.append(noisy_action)
            num_steps += 1

        action = noisy_action
        # when obs_horizon=2, the model was trained as
        # o_0, o_1,
        # a_0, a_1, a_2, a_3, ..., a_{h-1}  -> action_horizon number of predictions
        # so we DO NOT use the first prediction at test time
        action = action[:, self.obs_horizon - 1 : self.cfg.action_horizon]
        print(f"> Number of diffusionsteps: {num_steps}")
        # No action normalization needed - dataset handles it
        if unsqueezed:
            action = action.squeeze(0)
        if cpu:
            action = action.cpu()
        return action

    def loss(self, batch: Batch, avg: bool = True, action_mse: bool = False, progress: int = 0) -> torch.Tensor:
        """
        Calculate loss for the diffusion model.

        If `action_mse` is true, the loss is calculated as the MSE between the predicted
        actions and the ground truth actions. This is used for monitoring only and is not
        used in optimization.
        """
        actions = batch.action
        bsize = actions.size(0)
        state_cond = batch.state_cond
        # Prepare images for multiview processing
        obs = {}
        for k, v in batch.images.items():
            if k in self.camera_views:
                obs[k] = self.aug(v.float()) if hasattr(self, 'aug') else v

        obs["prop"] = state_cond.reshape(bsize, -1)
        
        noise = torch.randn(actions.shape, device=actions.device)
        # sample a diffusion iteration for each data point
        has_labels = hasattr(batch, 'label') and batch.label is not None
        if has_labels and self.cfg.human_min_noise_level > 0.0:
            # Separate human and robot data for different noise levels
            human_idx = torch.where(batch.label == 0)[0]
            robot_idx = torch.where(batch.label == 1)[0]
            timesteps = torch.zeros(bsize, device=actions.device, dtype=torch.long)
            
            if len(human_idx) > 0:
                # Human data: use minimum noise level
                min_timestep = int(self.cfg.human_min_noise_level * self.noise_scheduler.config["num_train_timesteps"])
                timesteps[human_idx] = torch.randint(
                    low=min_timestep,
                    high=self.noise_scheduler.config["num_train_timesteps"],
                    size=(len(human_idx),),
                    device=actions.device,
                ).long()
            
            if len(robot_idx) > 0:
                # Robot data: use full range
                timesteps[robot_idx] = torch.randint(
                    low=0,
                    high=self.noise_scheduler.config["num_train_timesteps"],
                    size=(len(robot_idx),),
                    device=actions.device,
                ).long()
        else:
            # Default behavior: use full range for all data
            timesteps = torch.randint(
                low=0,
                high=self.noise_scheduler.config["num_train_timesteps"],
                size=(bsize,),
                device=actions.device,
            ).long()
        
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)  # type: ignore
        noise_pred, obs_emb = self.net.predict_noise(obs, noisy_actions, timesteps)

        loss_dict = {}

        # Noise prediction loss
        noise_loss = nn.functional.mse_loss(noise_pred, noise, reduction="none")
        action_noise_loss = noise_loss[:, :, :-1]
        grasp_noise_loss = noise_loss[:, :, -1:]
        action_loss = noise_loss.sum(dim=2)

        loss_dict["total/action_noise"] = action_noise_loss.mean()
        loss_dict["total/grasp_noise"] = grasp_noise_loss.mean()

        if avg:
            # Reduction from official DP repo
            action_loss = reduce(action_loss, "b ... -> b (...)", "mean")
            action_loss = action_loss.mean()
        else:
            action_loss = action_loss

        final_loss = (
            action_loss
        )

        return final_loss, loss_dict

    def unified_loss(self, batch: Batch, timesteps: torch.Tensor, loss_masks: torch.Tensor, avg: bool = True) -> torch.Tensor:
        """
        Calculate loss for the diffusion model with custom timesteps and loss masks.
        """
        actions = batch.action
        state_cond = batch.state_cond
        bsize = loss_masks.size(0)
        
        # Prepare images for multiview processing
        obs = {}
        for k, v in batch.images.items():
            if k in self.camera_views:
                obs[k] = self.aug(v.float()) if hasattr(self, 'aug') else v

        obs["prop"] = state_cond.reshape(bsize, -1)
        noise = torch.randn(actions.shape, device=actions.device)
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)  # type: ignore

        noise_pred, obs_emb = self.net.predict_noise(obs, noisy_actions, timesteps)
        # loss: [batch, num_action, action_dim]
        noise_loss = nn.functional.mse_loss(noise_pred, noise, reduction="none")
        action_noise_loss = noise_loss[:, :, :-1]
        grasp_noise_loss = noise_loss[:, :, -1:]
        action_loss = (noise_loss*loss_masks[:, None, None]).sum(dim=2)

        # Apply loss masks
        # loss = loss * loss_masks[:, None]

        if avg:
            action_loss = reduce(action_loss, "b ... -> b (...)", "mean")
            action_loss = action_loss.mean()
        else:
            action_loss = action_loss

        final_loss = (
            action_loss
        )
        loss_dict = {}
        loss_dict["total/action_noise"] = action_noise_loss.mean()
        loss_dict["total/grasp_noise"] = grasp_noise_loss.mean()

        return final_loss, loss_dict

def test():
    print("Testing Diffusion Policy")
    cprint(
        "[WARNING], addin grasp info to points is not implemented/tested",
        "yellow",
        attrs=["bold", "blink"],
    )
    # Define parameters
    num_points = 5
    action_dim = 10
    obs_horizon = 2
    pred_horizon = 16
    action_horizon = 8
    state_cond_dim = 10
    add_grasp_info_to_tracks = True

    # Create config
    cfg = DiffusionPolicyConfig(
        input_type="image",
        encoder="resnet",
        backbone="unet",
        use_ddpm=True,
        add_cond_vector=True,
        add_grasp_info_to_tracks=add_grasp_info_to_tracks,
        delta_actions=True,
    )

    # Instantiate DiffusionPolicy
    policy = DiffusionPolicy(
        num_points=num_points,
        action_dim=action_dim,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        action_horizon=action_horizon,
        state_cond_dim=state_cond_dim,
        cfg=cfg,
        use_image=True,
    ).cuda()

    # Set to evaluation mode
    policy.eval()

    # Create dummy input tensors
    batch_size = 2
    obs = torch.rand(
        batch_size, obs_horizon, 3, 96, 96
    ).cuda()  # Assuming image input of shape (3, 96, 96)
    state_cond = torch.rand(batch_size, obs_horizon, action_dim).cuda()

    # Test act method
    action = policy.act(
        obs,
        state_cond,
        first_action_abs=torch.rand(batch_size, action_dim).cuda(),
    )
    print(f"Output action shape: {action.shape}")
    expected_shape = (
        batch_size,
        action_horizon,
        action_dim + 1 + int(add_grasp_info_to_tracks),
    )
    assert (
        action.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {action.shape}"

    # Test act method without batch
    action = policy.act(
        obs[0], state_cond[0], first_action_abs=torch.rand(action_dim).cuda()
    )
    print(f"Output action shape: {action.shape}")
    expected_shape = (action_horizon, action_dim)
    assert (
        action.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {action.shape}"

    # Test loss calculation
    policy.train()
    batch = Batch(
        obs=obs,
        action=torch.rand(batch_size, pred_horizon, action_dim).cuda(),
        state_cond=state_cond,
    )
    loss = policy.loss(batch)
    print(f"Loss: {loss.item()}")

    print("Test passed successfully!")


def test_image_diffusion_policy():
    print("Testing Image Diffusion Policy")
    
    # Define parameters
    obs_horizon = 2
    obs_shape = (3, 96, 96)  # RGB image
    prop_dim = 10
    action_dim = 7
    camera_views = ["wrist_view", "front_view"]
    
    # Create config
    cfg = ImageDiffusionPolicyConfig(
        use_ddpm=1,
        action_horizon=8,
        prediction_horizon=16,
        shift_pad=4,
        human_min_noise_level=0.1,
    )
    
    # Instantiate ImageDiffusionPolicy
    policy = ImageDiffusionPolicy(
        obs_horizon=obs_horizon,
        obs_shape=obs_shape,
        prop_dim=prop_dim,
        action_dim=action_dim,
        camera_views=camera_views,
        cfg=cfg,
    ).cuda()
    
    # Set to evaluation mode
    policy.eval()
    
    # Create dummy input tensors
    batch_size = 2
    obs = {}
    for view in camera_views:
        obs[view] = torch.rand(batch_size, obs_horizon, *obs_shape).cuda()
    
    # Test act method
    action = policy.act(obs)
    print(f"Output action shape: {action.shape}")
    expected_shape = (batch_size, cfg.action_horizon, action_dim)
    assert (
        action.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {action.shape}"
    
    # Test act method without batch
    single_obs = {k: v[0] for k, v in obs.items()}
    action = policy.act(single_obs)
    print(f"Output action shape (single): {action.shape}")
    expected_shape = (cfg.action_horizon, action_dim)
    assert (
        action.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {action.shape}"
    
    # Test loss function with batch.images
    from collections import namedtuple
    Batch = namedtuple("Batch", ["action", "state_cond", "label", "data_type", "images"])
    
    batch = Batch(
        action=torch.rand(batch_size, cfg.prediction_horizon, action_dim).cuda(),
        state_cond=torch.rand(batch_size, obs_horizon, prop_dim).cuda(),
        label=torch.tensor([0, 1]).cuda(),  # mixed human/robot
        data_type=torch.tensor([0, 1]).cuda(),
        images=obs
    )
    
    # Test loss calculation
    policy.train()
    loss, loss_dict = policy.loss(batch)
    print(f"Loss: {loss.item()}")
    print(f"Loss dict keys: {loss_dict.keys()}")
    
    print("ImageDiffusionPolicy test passed successfully!")


if __name__ == "__main__":
    test()
    test_image_diffusion_policy()
