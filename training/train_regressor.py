import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from training.datasets import RegressorDataset
from training.models import RegressorNet


def train_regressor(data_dir: str,
                    save_path: str,
                    in_features: int,
                    hidden_sizes: list,
                    activation: str,
                    batch_size: int,
                    lr: float,
                    epochs: int,
                    device: str):
    # Prepare dataset and loader
    dataset = RegressorDataset(data_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)  # batch_size=1 since each sample is variable-sized
    # Instantiate model
    model = RegressorNet(in_features=in_features,
                         hidden_sizes=hidden_sizes,
                         activation=activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for features, target in loader:
            # features: list of (N_i, F) tensors, but DataLoader with batch_size=1 returns a tensor of shape (1, N, F)
            # we squeeze the batch dimension
            x = features.squeeze(0).to(device)          # (N, F)
            y_true = target.squeeze(0).to(device)       # (N,)

            optimizer.zero_grad()
            y_pred = model(x)                           # (N,)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"[Epoch {epoch}/{epochs}] Avg MSE Loss: {avg_loss:.6f}")

    # Save checkpoint
    ckpt = {
        'state_dict': model.state_dict(),
        'in_features': in_features,
        'hidden_sizes': hidden_sizes,
        'activation': activation
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(ckpt, save_path)
    print(f"Model checkpoint saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train step-size regressor MLP")
    parser.add_argument('--data-dir', '-d', required=True,
                        help="Directory with .npz regressor training samples")
    parser.add_argument('--save-path', '-s', default='training/checkpoints/regressor.pth',
                        help="Output path for model checkpoint")
    parser.add_argument('--in-features', type=int, default=2,
                        help="Number of input features per vertex")
    parser.add_argument('--hidden-sizes', nargs='+', type=int, default=[64, 32],
                        help="Hidden layer widths")
    parser.add_argument('--activation', choices=['relu', 'tanh'], default='relu',
                        help="Activation function")
    parser.add_argument('--batch-size', type=int, default=1,
                        help="Batch size (samples per gradient step)")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument('--epochs', type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument('--device', default='cpu',
                        help="Torch device (e.g., 'cpu' or 'cuda')")
    args = parser.parse_args()

    train_regressor(
        data_dir=args.data_dir,
        save_path=args.save_path,
        in_features=args.in_features,
        hidden_sizes=args.hidden_sizes,
        activation=args.activation,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device
    )
