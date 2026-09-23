"""
Conditional denoising diffusion model that turns a coarse (~12 km) forecast
patch, conditioned on high-res topography, into a fine (~5 km) field —
CorrDiff-style residual/patch diffusion, kept deliberately lightweight so it
trains on a hackathon timeline/GPU budget.

Design choices for speed:
  - Operates on small cropped patches around each detected event's bbox,
    not the whole domain (see bounding_box.pad_bbox).
  - Predicts the *residual* (fine - bilinear-upsampled-coarse), not the
    absolute field — residual diffusion converges faster and preserves
    extremes better than modelling the raw field.
  - Small UNet, sinusoidal timestep embedding, topography as an extra
    conditioning channel (this is the piece a coarse grid structurally
    cannot give you, so it's the most important conditioning input).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class DownscalingUNet(nn.Module):
    """
    Small UNet predicting noise (epsilon) for the residual field.

    Input channels: [noisy_residual (1), bilinear_upsampled_coarse (1),
                      topography (1), land_sea_mask (1)]  = 4 channels default,
    adjust `in_channels` if you add more conditioning fields.
    """

    def __init__(self, in_channels: int = 4, base_ch: int = 48, t_dim: int = 128):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))

        self.in_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)

        self.down1 = ResBlock(base_ch, base_ch, t_dim)
        self.pool1 = nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1)

        self.down2 = ResBlock(base_ch * 2, base_ch * 2, t_dim)
        self.pool2 = nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)

        self.mid = ResBlock(base_ch * 4, base_ch * 4, t_dim)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.dec2 = ResBlock(base_ch * 4, base_ch * 2, t_dim)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.dec1 = ResBlock(base_ch * 2, base_ch, t_dim)

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, 3, padding=1)

    def forward(self, x, timesteps):
        t_emb = self.t_mlp(sinusoidal_embedding(timesteps, self.t_dim))

        h0 = self.in_conv(x)
        h1 = self.down1(h0, t_emb)
        h1p = self.pool1(h1)

        h2 = self.down2(h1p, t_emb)
        h2p = self.pool2(h2)

        hm = self.mid(h2p, t_emb)

        u2 = self.up2(hm)
        d2 = self.dec2(torch.cat([u2, h2], dim=1), t_emb)

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, h1], dim=1), t_emb)

        out = self.out_conv(F.silu(self.out_norm(d1)))
        return out


class GaussianDiffusion:
    """Standard DDPM noise schedule + training/sampling helpers."""

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02, device="cpu"):
        self.T = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bars[t][:, None, None, None]
        return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise, noise

    def training_loss(self, model: DownscalingUNet, x0_residual: torch.Tensor, cond_channels: torch.Tensor):
        """
        x0_residual   : (B, 1, H, W) target residual = fine_truth - upsampled_coarse
        cond_channels : (B, C-1, H, W) conditioning (upsampled coarse, topography, land-sea mask, ...)
        """
        b = x0_residual.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0_residual.device)
        x_noisy, noise = self.q_sample(x0_residual, t)
        model_in = torch.cat([x_noisy, cond_channels], dim=1)
        noise_pred = model(model_in, t)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, model: DownscalingUNet, cond_channels: torch.Tensor, shape, device):
        """DDPM ancestral sampling. shape = (B, 1, H, W)."""
        x = torch.randn(shape, device=device)
        for t_idx in reversed(range(self.T)):
            t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
            model_in = torch.cat([x, cond_channels], dim=1)
            eps_pred = model(model_in, t)

            alpha = self.alphas[t_idx]
            alpha_bar = self.alpha_bars[t_idx]
            beta = self.betas[t_idx]

            noise = torch.randn_like(x) if t_idx > 0 else torch.zeros_like(x)
            x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * eps_pred) + torch.sqrt(beta) * noise
        return x
