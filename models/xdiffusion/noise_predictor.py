"""
This file contains a nn.Module class inteded to wrap around an encoder
network and a noise predictor network. The encoder network is used to
encode the input data, while the noise predictor network is used to
predict noise to be added to the input data.
"""

import torch
import torch.nn as nn
from typing import Optional, List
from termcolor import cprint
from line_profiler import profile


class NoisePredictor(nn.Module):
    def __init__(
        self,
        noise_pred_net: nn.Module,
        layers_to_freeze: List[str] = [],
    ):
        """
        Initializes the NoisePredictor module with an encoder network and a noise predictor network.

        Args:
            encoder (nn.Module): The encoder network.
            feat_mlp (nn.Module): Linear layer after encoder.
            noise_pred_net (nn.Module): The noise predictor network.
            freeze_encoder (bool, optional): Whether to freeze the encoder network. Defaults to False.
        """
        super().__init__()
        self.noise_pred_net = noise_pred_net

        if len(layers_to_freeze) > 0:
            self._freeze_layers(layers_to_freeze)

    def _freeze_layers(self, layers_to_freeze: List[str]):
        """Freeze all layers except the prop_fc in LinearCompress"""
        # cprint(f"Freezing layers: {frozen_layers}", "blue", attrs=["bold"])

        frozen_layer_names = []
        for name, param in self.named_parameters():
            for fl in layers_to_freeze:
                if fl in name:
                    param.requires_grad = False
                    frozen_layer_names.append(name)
                else:
                    param.requires_grad = True
        cprint(
            f"[NOTE] Froze {len(frozen_layer_names)} with names {layers_to_freeze}",
            "blue",
            attrs=["bold"],
        )

    @profile
    def predict_noise(
        self,
        noisy_action: torch.Tensor,
        timestep: int,
        *,
        additional_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predicts noise to be added to the input data.

        Args:
            obs (torch.Tensor): The input data.
            noisy_action (torch.Tensor): The noisy action.
            timestep (int): The timestep.
            lang (str): Optional language conditioning. Defaults to None.
            additional_cond (torch.Tensor, optional): Additional conditioning data. Defaults to None.
            cached_emb (torch.Tensor, optional): The cached embedding. Defaults to None.

        Returns:
            torch.Tensor: The predicted noise.
        """
        obs_emb = additional_cond.flatten(start_dim=1)
        noise_pred = self.noise_pred_net(
            sample=noisy_action, timestep=timestep, cond=obs_emb
        )
        return noise_pred, obs_emb
