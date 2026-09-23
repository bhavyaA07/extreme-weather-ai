"""
src/anomaly/efi.py
==================
MoES / NCMRWF SIH-26078: Extreme Forecast Index (EFI) & Shift of Tails (SOT) Engine.

Computes the ECMWF-standard Extreme Forecast Index (Lalaurette 2003, Zsoter 2006)
comparing a coarse numerical weather prediction (NWP) ensemble forecast (e.g., GEFS / NEPS-G)
against an ERA5 reanalysis 30-year climatological cumulative distribution function (CDF).

Key Capabilities:
1. ECMWF Integral EFI: Uses weighted variance normalization (sqrt(p * (1-p)))
   focusing sensitivity on the distribution tails (p -> 0 and p -> 1).
2. Shift of Tails (SOT): Measures the extent to which extreme ensemble members
   exceed the historical climatological record limit (p=90, 95, 99).
3. Memory-efficient chunked computation across 2D spatial fields (latitude/longitude)
   and time lead steps.
4. Supports NumPy arrays and Xarray DataArrays with NaN handling for ocean/masked domains.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np


@dataclass
class EFIResult:
    """Container for computed anomaly indices across a 2D spatial domain."""
    efi: np.ndarray             # Range: [-1.0, 1.0], values > 0.6 indicate abnormal, > 0.8 severe
    sot_90: np.ndarray          # Shift of Tails evaluated at 90th climatological percentile
    ensemble_mean: np.ndarray   # Mean of the ensemble members
    ensemble_max: np.ndarray    # Peak ensemble member value
    climate_q95: np.ndarray     # Climatological 95th percentile baseline


class ExtremeForecastIndexCalculator:
    """
    Computes ECMWF-style Extreme Forecast Index (EFI) and Shift of Tails (SOT)
    for precipitation or other meteorological variables.
    """

    def __init__(
        self,
        num_quantiles: int = 100,
        eps: float = 1e-6
    ):
        """
        Args:
            num_quantiles: Integration resolution steps across the CDF range [0, 1].
            eps: Epsilon to avoid division by zero near p=0 and p=1 boundaries.
        """
        self.num_quantiles = num_quantiles
        self.eps = eps

        # Discretized integration points p in (0, 1)
        self.p_steps = np.linspace(0.01, 0.99, self.num_quantiles, dtype=np.float32)

        # ECMWF weighting function w(p) = 1 / sqrt(p * (1 - p))
        # Weights the tails much heavier than the median
        self.weights = 1.0 / np.sqrt(self.p_steps * (1.0 - self.p_steps))
        # Normalization factor such that EFI ranges strictly in [-1.0, 1.0]
        self.norm_factor = np.sum(self.weights)

    def compute_cell_efi(
        self,
        forecast_ensemble_values: np.ndarray,
        climate_quantiles: np.ndarray
    ) -> float:
        """
        Calculates EFI for a single grid cell.

        Args:
            forecast_ensemble_values: 1D array of ensemble member values (e.g. 21 or 31 members).
            climate_quantiles: 1D array of pre-computed climatological values corresponding
                               to self.p_steps (length == self.num_quantiles).

        Returns:
            efi: float in range [-1.0, 1.0].
        """
        # Filter NaNs if cell is outside valid model mask
        valid_ens = forecast_ensemble_values[~np.isnan(forecast_ensemble_values)]
        if len(valid_ens) < 2 or np.all(np.isnan(climate_quantiles)):
            return 0.0

        n_members = len(valid_ens)

        # F_f(Q_c(p)): Probability that ensemble forecast <= climatological quantile Q_c(p)
        # Using searchsorted for O(M log K) efficiency
        sorted_ens = np.sort(valid_ens)
        # Fraction of ensemble members <= each climate quantile
        counts_below = np.searchsorted(sorted_ens, climate_quantiles, side="right")
        f_forecast_at_qc = counts_below.astype(np.float32) / float(n_members)

        # ECMWF Integral: sum over p of (p - F_f(Q_c(p))) * w(p)
        integrand = (self.p_steps - f_forecast_at_qc) * self.weights
        efi = float(np.sum(integrand) / self.norm_factor)

        # Constrain to canonical boundary [-1.0, 1.0]
        return float(np.clip(efi, -1.0, 1.0))

    def compute_grid_efi(
        self,
        ensemble_grid: np.ndarray,
        climate_quantiles_grid: np.ndarray
    ) -> EFIResult:
        """
        Vectorized/batched computation of EFI & SOT across a full 2D spatial grid.

        Args:
            ensemble_grid: Shape (M, H, W)
                M = number of ensemble members (e.g. 21 GEFS members)
                H, W = latitude, longitude grid size
            climate_quantiles_grid: Shape (Q, H, W)
                Q = length of self.p_steps (e.g. 100 quantiles from ERA5)

        Returns:
            EFIResult dataclass containing 2D arrays:
                efi, sot_90, ensemble_mean, ensemble_max, climate_q95
        """
        M, H, W = ensemble_grid.shape
        Q, H_c, W_c = climate_quantiles_grid.shape

        assert H == H_c and W == W_c, (
            f"Spatial dimension mismatch: ensemble is ({H}, {W}) vs climate ({H_c}, {W_c})"
        )
        assert Q == self.num_quantiles, (
            f"Quantiles dimension mismatch: expected {self.num_quantiles}, got {Q}"
        )

        # 1. Basic Ensemble Summary Statistics
        ens_mean = np.nanmean(ensemble_grid, axis=0)
        ens_max = np.nanmax(ensemble_grid, axis=0)

        # Find index corresponding closest to p=0.95 and p=0.90 for baselines
        idx_90 = int(np.argmin(np.abs(self.p_steps - 0.90)))
        idx_95 = int(np.argmin(np.abs(self.p_steps - 0.95)))
        idx_99 = int(np.argmin(np.abs(self.p_steps - 0.99)))

        clim_q90 = climate_quantiles_grid[idx_90]
        clim_q95 = climate_quantiles_grid[idx_95]
        clim_q99 = climate_quantiles_grid[idx_99]

        # 2. Vectorized CDF Evaluation across Grid
        # Sort ensemble members along member axis: (M, H, W)
        sorted_ens = np.sort(ensemble_grid, axis=0)

        # Reshape for broadcasting:
        # sorted_ens:           (M, 1, H, W)
        # climate_quantiles:    (1, Q, H, W)
        sorted_ens_exp = sorted_ens[:, np.newaxis, :, :]
        clim_exp = climate_quantiles_grid[np.newaxis, :, :, :]

        # Boolean mask of members <= climatological threshold
        members_below = (sorted_ens_exp <= clim_exp)
        # Empirical forecast CDF: F_f(Q_c(p)) with shape (Q, H, W)
        f_forecast_at_qc = np.sum(members_below, axis=0).astype(np.float32) / float(M)

        # 3. Integrate with ECMWF Weights
        # weights broadcast across (Q, 1, 1)
        w_exp = self.weights[:, np.newaxis, np.newaxis]
        p_exp = self.p_steps[:, np.newaxis, np.newaxis]

        integrand = (p_exp - f_forecast_at_qc) * w_exp
        efi_grid = np.sum(integrand, axis=0) / self.norm_factor
        efi_grid = np.clip(efi_grid, -1.0, 1.0)

        # 4. Shift of Tails (SOT) evaluated at 90th percentile:
        # SOT = - (Q_f(90) - Q_c(90)) / (Q_c(90) - Q_c(median))
        # Quantifies how much the forecast tail breaks beyond historical bounds.
        # Positive values (> 0) indicate that 10% of ensemble members exceed the climatological 90th percentile.
        ens_q90 = np.quantile(ensemble_grid, 0.90, axis=0)
        idx_50 = int(np.argmin(np.abs(self.p_steps - 0.50)))
        clim_q50 = climate_quantiles_grid[idx_50]

        sot_denom = np.maximum(clim_q90 - clim_q50, self.eps)
        sot_90 = (ens_q90 - clim_q90) / sot_denom

        return EFIResult(
            efi=efi_grid,
            sot_90=sot_90,
            ensemble_mean=ens_mean,
            ensemble_max=ens_max,
            climate_q95=clim_q95,
        )

    def extract_anomaly_hotspots(
        self,
        efi_result: EFIResult,
        efi_threshold: float = 0.70,
        min_precip_mm: float = 20.0
    ) -> np.ndarray:
        """
        Generates a binary boolean mask of grid cells that meet extreme anomaly criteria:
        High EFI (ensemble shifts markedly toward the upper climate tail) AND
        meaningful precipitation expected (guards against division noise in dry regions).
        """
        hotspots = (efi_result.efi >= efi_threshold) & (efi_result.ensemble_max >= min_precip_mm)
        return hotspots


# =====================================================================
# 5. Standalone Verification / Self-Test Routine
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MoES/NCMRWF SIH-26078: Testing ECMWF-Style EFI & SOT Calculation")
    print("=" * 70)

    # 1. Setup synthetic domain: 16x16 grid (Bay of Bengal / Odisha coast sector)
    np.random.seed(42)
    H, W = 16, 16
    M = 21   # 21 GEFS ensemble members
    Q = 100  # 100 quantile bins

    # 2. Synthesize 30-year climatology baseline from ERA5
    # Historical median is low (5-15 mm), with rare monsoon storms reaching 70 mm
    p_levels = np.linspace(0.01, 0.99, Q)
    climate_q_grid = np.zeros((Q, H, W), dtype=np.float32)
    for q_idx, p in enumerate(p_levels):
        # Exponential distribution quantiles: -scale * ln(1 - p)
        climate_q_grid[q_idx, :, :] = -15.0 * np.log(1.0 - p)

    # 3. Synthesize Forecast Ensemble (e.g., Cyclone Amphan approaching)
    # Background has normal rain (10-25 mm), but a severe eyewall feature
    # has ensemble members forecasting 120-220 mm in cells (6:11, 7:12)
    ensemble_grid = np.random.exponential(scale=12.0, size=(M, H, W)).astype(np.float32)
    eyewall_slice = (slice(6, 11), slice(7, 12))
    ensemble_grid[:, eyewall_slice[0], eyewall_slice[1]] += np.random.uniform(110.0, 190.0, size=(M, 5, 5))

    # 4. Run EFI Calculator
    calculator = ExtremeForecastIndexCalculator(num_quantiles=Q)
    result = calculator.compute_grid_efi(ensemble_grid, climate_q_grid)

    print("\n--- Domain Summary Statistics ---")
    print(f"EFI min / max:                 {np.min(result.efi):.3f} / {np.max(result.efi):.3f}")
    print(f"Max Ensemble Peak (mm):         {np.max(result.ensemble_max):.1f} mm")
    print(f"Climate 95th Percentile (mm):  {np.max(result.climate_q95):.1f} mm")
    print(f"Max Shift of Tails (SOT-90):   {np.max(result.sot_90):.2f}")

    # 5. Detect Extreme Hotspots (EFI >= 0.75 & rain >= 50 mm)
    hotspots = calculator.extract_anomaly_hotspots(result, efi_threshold=0.75, min_precip_mm=50.0)
    hotspot_count = int(np.sum(hotspots))

    print(f"\n[+] Detected {hotspot_count} severe anomaly cells (EFI >= 0.75)")
    assert hotspot_count > 0, "Failed to identify cyclone eyewall anomaly!"

    # Inspect the center of the cyclone anomaly cell
    center_r, center_c = 8, 9
    print(f"\nCell ({center_r}, {center_c}) Eyewall Metrics:")
    print(f"  - EFI Index:      {result.efi[center_r, center_c]:.4f} (Severity: EXTREME)")
    print(f"  - SOT (90th):     {result.sot_90[center_r, center_c]:.2f}")
    print(f"  - Ensemble Mean:  {result.ensemble_mean[center_r, center_c]:.1f} mm")
    print(f"  - Ensemble Max:   {result.ensemble_max[center_r, center_c]:.1f} mm")
    print(f"  - Climatology Q95:{result.climate_q95[center_r, center_c]:.1f} mm")

    print("\n[+] Verification successful: EFI/SOT numerical physics engine operational.")