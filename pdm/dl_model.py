"""
dl_model.py
-----------
The actual deep-learning autoencoder (PyTorch), as specified in the synopsis
("Apply Autoencoder (Deep Learning) for anomaly detection... using PyTorch or
TensorFlow").
"""

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Simple symmetric feed-forward autoencoder.

    input -> 12 -> bottleneck -> 12 -> input, ReLU activations on hidden
    layers, linear output (reconstruction).
    """

    def __init__(self, n_features: int, bottleneck: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 12),
            nn.ReLU(),
            nn.Linear(12, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 12),
            nn.ReLU(),
            nn.Linear(12, n_features),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def export_weights(self):
        """Return the trained weights/biases as plain numpy arrays, in the
        exact order the NumPy replica (numpy_autoencoder.py) expects."""
        layers = [
            self.encoder[0], self.encoder[2],  # Linear layers in encoder
            self.decoder[0], self.decoder[2],  # Linear layers in decoder
        ]
        weights = {}
        for i, layer in enumerate(layers):
            weights[f"W{i}"] = layer.weight.detach().numpy()  # (out, in)
            weights[f"b{i}"] = layer.bias.detach().numpy()
        return weights
