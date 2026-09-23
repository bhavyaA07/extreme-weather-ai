"""
Training loop for the downscaling diffusion model.
    python -m src.downscaling.train --epochs 100

Expects a Dataset yielding dicts:
    {
      "upsampled_coarse": (1, H, W)  bilinear-upsampled coarse forecast, physical units, normalised
      "topography":        (1, H, W)  static high-res elevation, normalised
      "land_sea_mask":      (1, H, W)  0/1
      "fine_truth":         (1, H, W)  ERA5-Land / IMD gridded truth at 5 km, normalised
    }
Residual target = fine_truth - upsampled_coarse (computed here).
"""

from __future__ import annotations
import argparse
import torch
from torch.utils.data import DataLoader

from src.downscaling.diffusion_model import DownscalingUNet, GaussianDiffusion


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: point this at your processed data/processed/downscaling/ pairs
    from src.downscaling.dataset import DownscalingPatchDataset  # user-provided
    train_ds = DownscalingPatchDataset(split="train")
    val_ds = DownscalingPatchDataset(split="val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)

    model = DownscalingUNet(in_channels=4, base_ch=args.base_ch).to(device)
    diffusion = GaussianDiffusion(timesteps=args.timesteps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            coarse = batch["upsampled_coarse"].to(device)
            topo = batch["topography"].to(device)
            lsm = batch["land_sea_mask"].to(device)
            fine = batch["fine_truth"].to(device)

            residual = fine - coarse
            cond = torch.cat([coarse, topo, lsm], dim=1)  # (B, 3, H, W)

            optimizer.zero_grad()
            loss = diffusion.training_loss(model, residual, cond)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                coarse = batch["upsampled_coarse"].to(device)
                topo = batch["topography"].to(device)
                lsm = batch["land_sea_mask"].to(device)
                fine = batch["fine_truth"].to(device)
                residual = fine - coarse
                cond = torch.cat([coarse, topo, lsm], dim=1)
                val_loss += diffusion.training_loss(model, residual, cond).item()

        print(f"epoch {epoch:03d}  train_loss {total_loss/max(1,len(train_loader)):.4f}  "
              f"val_loss {val_loss/max(1,len(val_loader)):.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"  -> saved best model to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base_ch", type=int, default=48)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--save_path", type=str, default="models/diffusion/downscaler.pt")
    args = parser.parse_args()
    train(args)
