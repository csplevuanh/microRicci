#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressorMLP(nn.Module):
    """
    A small feed‐forward MLP that takes a feature vector per vertex and
    outputs a scalar step size.
    """
    def __init__(self,
                 in_features: int = 2,
                 hidden_sizes: list = [32, 16],
                 activation: str = 'relu'):
        """
        Args:
            in_features: number of input features (e.g., [curvature, degree]).
            hidden_sizes: list of hidden layer widths.
            activation: name of activation ('relu' or 'tanh').
        """
        super().__init__()
        layers = []
        prev = in_features
        act = F.relu if activation == 'relu' else F.tanh

        # Hidden layers
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            prev = h

        # Final output layer -> scalar
        self.layers = nn.ModuleList(layers)
        self.out_layer = nn.Linear(prev, 1)
        self.activation = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (N, in_features)
        Returns:
            Tensor of shape (N,) containing predicted δu for each vertex.
        """
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.out_layer(x)
        return x.view(-1)

    @staticmethod
    def load(path: str, device: str = 'cpu') -> 'RegressorMLP':
        """
        Load a saved regressor from disk.

        Args:
            path: path to the .pt or .pth file with state_dict.
            device: torch device.
        Returns:
            model: RegressorMLP with loaded weights.
        """
        checkpoint = torch.load(path, map_location=device)
        # If you saved hyperparams in checkpoint, adapt accordingly:
        in_f = checkpoint.get('in_features', None)
        hiddens = checkpoint.get('hidden_sizes', None)
        act = checkpoint.get('activation', 'relu')

        # Instantiate model
        if in_f and hiddens:
            model = RegressorMLP(in_features=in_f,
                                 hidden_sizes=hiddens,
                                 activation=act)
        else:
            # Fallback to default if hyperparams not saved
            model = RegressorMLP()

        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        return model


if __name__ == "__main__":
    # Quick sanity check
    model = RegressorMLP(in_features=2, hidden_sizes=[32, 16])
    dummy = torch.rand(5, 2)
    out = model(dummy)
    print(f"Dummy output shape: {out.shape}")  # should be (5,)
