"""
Training loop for StormTrackGNN. Run with:
    python -m src.gnn.train --epochs 50 --lr 1e-3
"""

from __future__ import annotations
import argparse
import torch
from torch_geometric.loader import DataLoader

from src.gnn.model import StormTrackGNN, track_loss
from src.gnn.dataset import StormTrackDataset


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: replace with real loaders that build forecast_runs / reference_tracks
    # from your processed ERA5/NWP events + IMD/IBTrACS reference tracks.
    from src.data.build_training_graphs import load_training_runs  # user-provided loader
    forecast_runs, reference_tracks = load_training_runs()

    dataset = StormTrackDataset(forecast_runs, reference_tracks)
    n_val = max(1, int(0.15 * len(dataset)))
    train_set = [dataset[i] for i in range(len(dataset) - n_val)]
    val_set = [dataset[i] for i in range(len(dataset) - n_val, len(dataset))]

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    model = StormTrackGNN(in_dim=6, hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            if batch.edge_index.numel() == 0:
                continue
            optimizer.zero_grad()
            edge_logits, motion_pred, _ = model(batch.x, batch.edge_index, batch.edge_attr)
            loss, edge_l, motion_l = track_loss(
                edge_logits, batch.edge_labels, motion_pred, batch.motion_target, batch.motion_mask
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                if batch.edge_index.numel() == 0:
                    continue
                edge_logits, motion_pred, _ = model(batch.x, batch.edge_index, batch.edge_attr)
                loss, _, _ = track_loss(
                    edge_logits, batch.edge_labels, motion_pred, batch.motion_target, batch.motion_mask
                )
                val_loss += loss.item()

        print(f"epoch {epoch:03d}  train_loss {total_loss/max(1,len(train_loader)):.4f}  "
              f"val_loss {val_loss/max(1,len(val_loader)):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"  -> saved best model to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--save_path", type=str, default="models/gnn/storm_track_gnn.pt")
    args = parser.parse_args()
    train(args)
