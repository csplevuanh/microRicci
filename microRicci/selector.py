import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SelectorMLP(nn.Module):
    """
    A small feed‐forward MLP that takes a feature vector per vertex and
    outputs a scalar score for selection.
    """
    def __init__(self,
                 in_features: int = 2,
                 hidden_sizes: list = [32, 16],
                 activation: str = 'relu'):
        """
        Args:
            in_features: number of input features (e.g., [abs_residual, degree]).
            hidden_sizes: list of hidden layer widths.
            activation: name of activation ('relu' or 'tanh').
        """
        super().__init__()
        layers = []
        prev = in_features
        self.activation = F.relu if activation == 'relu' else F.tanh

        # Hidden layers
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            prev = h

        self.layers = nn.ModuleList(layers)
        # Final output layer -> scalar score per vertex
        self.out_layer = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (N, in_features)
        Returns:
            Tensor of shape (N,) containing raw scores.
        """
        for layer in self.layers:
            x = self.activation(layer(x))
        scores = self.out_layer(x).view(-1)
        return scores

    def select_vertex(self, features: np.ndarray, device: str = 'cpu') -> int:
        """
        Given a NumPy array of per-vertex features, returns the index
        of the vertex with the highest score.

        Args:
            features: (N, in_features) array.
            device: torch device.
        Returns:
            idx: int, selected vertex index.
        """
        self.to(device)
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(features.astype(np.float32)).to(device)
            scores = self.forward(x)
            idx = int(torch.argmax(scores).cpu().item())
        return idx

    @staticmethod
    def load(path: str, device: str = 'cpu') -> 'SelectorMLP':
        """
        Load a saved selector from disk.

        Args:
            path: path to the .pt or .pth file with state_dict.
            device: torch device.
        Returns:
            model: SelectorMLP with loaded weights.
        """
        checkpoint = torch.load(path, map_location=device)
        in_f = checkpoint.get('in_features', None)
        hiddens = checkpoint.get('hidden_sizes', None)
        act = checkpoint.get('activation', 'relu')

        if in_f and hiddens:
            model = SelectorMLP(in_features=in_f,
                                hidden_sizes=hiddens,
                                activation=act)
        else:
            model = SelectorMLP()

        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        return model


if __name__ == "__main__":
    # Sanity check
    model = SelectorMLP(in_features=2, hidden_sizes=[32, 16])
    dummy = np.random.rand(5, 2)
    idx = model.select_vertex(dummy)
    print(f"Selected vertex index: {idx}")  # should be in [0,4]
