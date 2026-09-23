"""
src/physics/losses.py
=====================
MoES / NCMRWF SIH-26078: Modular Physics-Informed Loss Functions for Extreme
Weather Precipitation Downscaling.

Implements:
1. Moisture Convergence Penalty: Penalizes precipitation generated where moisture
   flux is diverging or insufficient (derived from q, u, v NWP fields).
2. Orographic Gradient Loss: Enforces physical consistency between wind vectors
   and SRTM DEM terrain slope (upslope condensation vs. downslope rain shadow).
3. Extreme Quantile Tail Loss: Prioritizes gradients at the extreme distribution
   tail (> 95th/99th percentile) to prevent regression smoothing.
4. Spatial Smoothness / TV Regularization: Prevents high-frequency checkerboard artifacts.
5. CompositePhysicsDownscalingLoss: Modular, configurable composite loss orchestrator.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. Physical Operators (Finite-Difference Gradients & Divergence)
# =====================================================================

class SpatialDerivatives2D(nn.Module):
    """
    Computes spatial partial derivatives (d/dx, d/dy) using Sobel/central
    difference kernels over latitude/longitude grid cells.
    """

    def __init__(self, dx_km: float = 5.0, dy_km: float = 5.0):
        super().__init__()
        self.dx_m = dx_km * 1000.0
        self.dy_m = dy_km * 1000.0

        # Central difference kernels: [1, 1, 3, 3]
        kx = torch.tensor([[-1.0, 0.0, 1.0],
                           [-2.0, 0.0, 2.0],
                           [-1.0, 0.0, 1.0]], dtype=torch.float32) / (8.0 * self.dx_m)
        ky = torch.tensor([[-1.0, -2.0, -1.0],
                           [ 0.0,  0.0,  0.0],
                           [ 1.0,  2.0,  1.0]], dtype=torch.float32) / (8.0 * self.dy_m)

        self.register_buffer("kernel_x", kx.unsqueeze(0).unsqueeze(0))
        self.register_buffer("kernel_y", ky.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (B, 1, H, W)
        Returns:
            df_dx, df_dy of shape (B, 1, H, W) with replicated padding.
        """
        x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
        df_dx = F.conv2d(x_pad, self.kernel_x)
        df_dy = F.conv2d(x_pad, self.kernel_y)
        return df_dx, df_dy


# =====================================================================
# 2. Physics-Informed Modular Loss Modules
# =====================================================================

class MoistureConvergenceLoss(nn.Module):
    """
    Penalizes precipitation predictions that violate the atmospheric moisture
    budget constraint:
        P <= C_conv * max(0, - div(q * V)) + P_ambient_reservoir

    Where:
        q: Specific humidity (kg/kg)
        u, v: Horizontal wind vector components (m/s)
        div(q * V) = d(q*u)/dx + d(q*v)/dy
        -div(q * V) is the horizontal moisture convergence.
    """

    def __init__(
        self,
        grid_spacing_km: float = 5.0,
        efficiency_factor: float = 0.85,
        ambient_allowance_mm: float = 5.0
    ):
        super().__init__()
        self.derivatives = SpatialDerivatives2D(dx_km=grid_spacing_km, dy_km=grid_spacing_km)
        self.efficiency = efficiency_factor
        self.ambient_allowance = ambient_allowance_mm

    def forward(
        self,
        pred_rain_mm: torch.Tensor,
        q_humidity: torch.Tensor,
        u_wind: torch.Tensor,
        v_wind: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_rain_mm: (B, 1, H, W) in linear mm/period
            q_humidity:   (B, 1, H, W) specific humidity
            u_wind:       (B, 1, H, W) zonal wind
            v_wind:       (B, 1, H, W) meridional wind
        """
        # Horizontal moisture fluxes
        flux_x = q_humidity * u_wind
        flux_y = q_humidity * v_wind

        d_flux_x, _ = self.derivatives(flux_x)
        _, d_flux_y = self.derivatives(flux_y)

        # Horizontal moisture convergence = - div(q * V)
        moisture_convergence = -(d_flux_x + d_flux_y)

        # Scale convergence into equivalent dynamic precipitation potential
        # (scaled for 6-hr / 24-hr accumulation window)
        conv_pot_rain = F.relu(moisture_convergence) * 1e5 * self.efficiency + self.ambient_allowance

        # Physical penalty: penalize predicted rain that exceeds physical dynamic ceiling
        excess = F.relu(pred_rain_mm - conv_pot_rain)
        return torch.mean(excess ** 2)


class OrographicSlopeConsistencyLoss(nn.Module):
    """
    Enforces topographic consistency against SRTM DEM.
    Air forced upslope (V · grad(DEM) > 0) creates condensation / heavy precipitation,
    whereas downslope (V · grad(DEM) < 0) creates rain shadow damping.
    Penalizes extreme anomalies predicted on lee slopes with strong downward motion.
    """

    def __init__(self, grid_spacing_km: float = 5.0):
        super().__init__()
        self.derivatives = SpatialDerivatives2D(dx_km=grid_spacing_km, dy_km=grid_spacing_km)

    def forward(
        self,
        pred_rain_mm: torch.Tensor,
        dem_elevation: torch.Tensor,
        u_wind: torch.Tensor,
        v_wind: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_rain_mm:  (B, 1, H, W)
            dem_elevation: (B, 1, H, W) in meters
            u_wind, v_wind:(B, 1, H, W)
        """
        dz_dx, dz_dy = self.derivatives(dem_elevation)
        # Updraft forcing proxy: V · grad(Z)
        orographic_lift = u_wind * dz_dx + v_wind * dz_dy

        # In strong downslope areas (lift < -0.05), high rain (> 64.5 mm) is physically penalized
        lee_slope_mask = (orographic_lift < -0.05).float()
        unphysical_lee_rain = F.relu(pred_rain_mm - 64.5) * lee_slope_mask

        return torch.mean(unphysical_lee_rain ** 2)


class ExtremeQuantileLoss(nn.Module):
    """
    Asymmetric Pinball / Quantile loss focusing on the upper tail (tau = 0.95 or 0.99).
    Prevents the standard L2/MSE smooth-out failure mode on extreme rainfall peaks.
    """

    def __init__(self, quantile: float = 0.99):
        super().__init__()
        self.quantile = quantile

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = target - pred
        loss = torch.max((self.quantile - 1.0) * diff, self.quantile * diff)
        return torch.mean(loss)


class TotalVariationLoss2D(nn.Module):
    """Preserves spatial coherence and suppresses high-frequency checkerboard noise."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        return torch.mean(diff_h) + torch.mean(diff_w)


# =====================================================================
# 3. Modular Composite Physics Loss Orchestrator
# =====================================================================

@dataclass
class LossWeights:
    """Weights configuration for downscaling loss components."""
    data_loss: float = 1.0            # L1 / Huber reconstruction loss
    extreme_tail: float = 2.5         # Quantile tail boost (Q99)
    moisture_conv: float = 0.35       # Moisture convergence physics constraint
    orographic: float = 0.15          # DEM wind-slope consistency
    total_variation: float = 0.05     # Spatial smoothness regularization


class CompositePhysicsDownscalingLoss(nn.Module):
    """
    Fully modular loss orchestrator that seamlessly switches between pure data-driven
    mode and full physics-informed NWP constraint mode.
    """

    def __init__(
        self,
        weights: Optional[LossWeights] = None,
        grid_spacing_km: float = 5.0,
        tail_quantile: float = 0.99
    ):
        super().__init__()
        self.weights = weights or LossWeights()

        # Core data fidelity
        self.reconstruction_loss = nn.SmoothL1Loss(beta=1.0)

        # Specialized loss modules
        self.tail_loss = ExtremeQuantileLoss(quantile=tail_quantile)
        self.moisture_loss = MoistureConvergenceLoss(grid_spacing_km=grid_spacing_km)
        self.orographic_loss = OrographicSlopeConsistencyLoss(grid_spacing_km=grid_spacing_km)
        self.tv_loss = TotalVariationLoss2D()

    def forward(
        self,
        pred_rain: torch.Tensor,
        target_rain: torch.Tensor,
        physics_conditioning: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the weighted composite loss.

        Args:
            pred_rain:   (B, 1, H, W) downscaled rain field (linear or log1p space)
            target_rain: (B, 1, H, W) ground truth 5 km rain (CHIRPS / IMERG)
            physics_conditioning: Optional dict containing:
                - 'dem': (B, 1, H, W) SRTM elevation
                - 'u':   (B, 1, H, W) 850hPa zonal wind
                - 'v':   (B, 1, H, W) 850hPa meridional wind
                - 'q':   (B, 1, H, W) specific humidity
                - 'is_log1p': bool (if True, exponentiates pred before physics calculation)

        Returns:
            total_loss (torch.Tensor), breakdown_dict (Dict[str, float])
        """
        # 1. Base Data Reconstruction Loss
        l_rec = self.reconstruction_loss(pred_rain, target_rain)
        l_tail = self.tail_loss(pred_rain, target_rain)

        total_loss = self.weights.data_loss * l_rec + self.weights.extreme_tail * l_tail
        breakdown = {
            "loss_rec": float(l_rec.item()),
            "loss_tail": float(l_tail.item()),
            "loss_moisture": 0.0,
            "loss_orographic": 0.0,
            "loss_tv": 0.0,
        }

        # 2. Total Variation (Noise control)
        if self.weights.total_variation > 0.0:
            l_tv = self.tv_loss(pred_rain)
            total_loss += self.weights.total_variation * l_tv
            breakdown["loss_tv"] = float(l_tv.item())

        # 3. Physics-informed terms (when meteorological fields are provided)
        if physics_conditioning is not None:
            # Transform to linear precipitation space (mm) if inputs are in log1p
            if physics_conditioning.get("is_log1p", False):
                linear_rain = torch.expm1(torch.clamp(pred_rain, min=0.0))
            else:
                linear_rain = pred_rain

            q = physics_conditioning.get("q")
            u = physics_conditioning.get("u")
            v = physics_conditioning.get("v")
            dem = physics_conditioning.get("dem")

            # Moisture Convergence constraint
            if self.weights.moisture_conv > 0.0 and q is not None and u is not None and v is not None:
                l_moist = self.moisture_loss(linear_rain, q, u, v)
                total_loss += self.weights.moisture_conv * l_moist
                breakdown["loss_moisture"] = float(l_moist.item())

            # Orographic Wind-Slope constraint
            if self.weights.orographic > 0.0 and dem is not None and u is not None and v is not None:
                l_oro = self.orographic_loss(linear_rain, dem, u, v)
                total_loss += self.weights.orographic * l_oro
                breakdown["loss_orographic"] = float(l_oro.item())

        breakdown["total_loss"] = float(total_loss.item())
        return total_loss, breakdown


# =====================================================================
# 4. Self-Test / Verification Routine
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MoES/NCMRWF SIH-26078: Testing Modular Physics Loss Functions")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W = 4, 1, 16, 16

    # 1. Mock inputs (Batch of 16x16 downscaled patches around Cyclone Amphan core)
    torch.manual_seed(42)
    mock_pred = torch.rand((B, C, H, W), device=device) * 150.0  # mm
    mock_target = torch.rand((B, C, H, W), device=device) * 150.0
    mock_target[:, :, 6:10, 6:10] += 120.0  # Extreme peak > 200 mm

    mock_physics = {
        "dem": torch.rand((B, C, H, W), device=device) * 1200.0,       # 0 to 1200m elevation
        "u": (torch.rand((B, C, H, W), device=device) - 0.5) * 35.0,   # -17 to +17 m/s
        "v": (torch.rand((B, C, H, W), device=device) - 0.5) * 35.0,
        "q": torch.rand((B, C, H, W), device=device) * 0.022,          # up to 22 g/kg
        "is_log1p": False
    }

    # 2. Instantiate with default SIH weights
    custom_weights = LossWeights(
        data_loss=1.0,
        extreme_tail=2.0,
        moisture_conv=0.4,
        orographic=0.2,
        total_variation=0.05
    )
    loss_fn = CompositePhysicsDownscalingLoss(weights=custom_weights, grid_spacing_km=5.0).to(device)

    # 3. Compute loss and verify backward pass
    mock_pred.requires_grad_(True)
    loss, breakdown = loss_fn(mock_pred, mock_target, mock_physics)
    loss.backward()

    print("[+] Loss computation successful.")
    for k, v in breakdown.items():
        print(f"    - {k:<18}: {v:.4f}")

    assert mock_pred.grad is not None, "Gradient did not backpropagate!"
    print(f"\n[+] Gradients validated. Mean grad norm: {mock_pred.grad.norm().item():.4f}")
    print("[+] Verification complete: Modular physics-informed loss is fully operational.")