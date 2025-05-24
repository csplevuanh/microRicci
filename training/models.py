import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectorNet(nn.Module):
    """
    MLP for selecting the next vertex to update.
    Input: per-vertex feature vector of size F_sel.
    Output: scalar score per vertex.
    """
    def __init__(self, in_features: int = 2, hidden_sizes: list = [64, 32], activation: str = 'relu'):
        """
        Args:
            in_features: number of input features per vertex.
            hidden_sizes: list of hidden layer sizes.
            activation: 'relu' or 'tanh'.
        """
        super().__init__()
        self.activation = F.relu if activation == 'relu' else F.tanh

        layers = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            prev = h
        self.hidden_layers = nn.ModuleList(layers)
        self.out_layer = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (N, in_features)
        Returns:
            scores: tensor of shape (N,) raw selection scores
        """
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        scores = self.out_layer(x).squeeze(-1)
        return scores


class RegressorNet(nn.Module):
    """
    MLP for predicting step-sizes per vertex.
    Input: per-vertex feature vector of size F_reg.
    Output: scalar delta_u per vertex.
    """
    def __init__(self, in_features: int = 2, hidden_sizes: list = [64, 32], activation: str = 'relu'):
        """
        Args:
            in_features: number of input features per vertex.
            hidden_sizes: list of hidden layer sizes.
            activation: 'relu' or 'tanh'.
        """
        super().__init__()
        self.activation = F.relu if activation == 'relu' else F.tanh

        layers = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            prev = h
        self.hidden_layers = nn.ModuleList(layers)
        self.out_layer = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (N, in_features)
        Returns:
            delta_u: tensor of shape (N,) predicted step-sizes
        """
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        delta_u = self.out_layer(x).squeeze(-1)
        return delta_u


if __name__ == '__main__':
    # Quick sanity tests
    sel = SelectorNet(in_features=2, hidden_sizes=[64, 32])
    x = torch.randn(10, 2)
    scores = sel(x)
    print(f"SelectorNet output shape: {scores.shape}")  # (10,)

    reg = RegressorNet(in_features=2, hidden_sizes=[64, 32])
    delta = reg(x)
    print(f"RegressorNet output shape: {delta.shape}")  # (10,)
