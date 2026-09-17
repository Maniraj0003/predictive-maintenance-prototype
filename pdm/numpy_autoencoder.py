import numpy as np


def relu(x):
    return np.maximum(0, x)


class NumpyAutoencoder:
    def __init__(self, weights_path: str):
        w = np.load(weights_path)
        self.W = [w["W0"], w["W1"], w["W2"], w["W3"]]
        self.b = [w["b0"], w["b1"], w["b2"], w["b3"]]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """x: (n_samples, n_features) -> reconstructed (n_samples, n_features)"""
        h = x
        # Linear layers 0,1 (encoder) use ReLU; layer 2 (decoder hidden) uses ReLU;
        # layer 3 (decoder output) is linear.
        for i in range(3):
            h = relu(h @ self.W[i].T + self.b[i])
        out = h @ self.W[3].T + self.b[3]
        return out
