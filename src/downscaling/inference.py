"""
Inference-time wrapper: crop coarse forecast + topography to an event's
bbox, upsample, run diffusion sampling, add back the upsampled-coarse field
to get the final fine-resolution physical field.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F

from src.downscaling.diffusion_model import DownscalingUNet, GaussianDiffusion


class Downscaler:
    def __init__(self, model_path: str, base_ch: int = 48, timesteps: int = 1000, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = DownscalingUNet(in_channels=4, base_ch=base_ch).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.diffusion = GaussianDiffusion(timesteps=timesteps, device=self.device)

    @torch.no_grad()
    def downscale_patch(
        self,
        coarse_patch: torch.Tensor,      # (1, 1, h, w) coarse forecast crop, normalised
        topo_patch: torch.Tensor,        # (1, 1, H, W) fine-res topography, already at target resolution
        lsm_patch: torch.Tensor,         # (1, 1, H, W) land-sea mask, target resolution
        target_hw: tuple,                # (H, W) fine grid size, e.g. ~5km cells
    ) -> torch.Tensor:
        upsampled_coarse = F.interpolate(coarse_patch, size=target_hw, mode="bilinear", align_corners=False)
        cond = torch.cat([upsampled_coarse, topo_patch, lsm_patch], dim=1).to(self.device)

        residual_sample = self.diffusion.sample(
            self.model, cond, shape=(1, 1, *target_hw), device=self.device
        )
        fine_field = residual_sample + upsampled_coarse.to(self.device)
        return fine_field.squeeze(0).squeeze(0)  # (H, W)
