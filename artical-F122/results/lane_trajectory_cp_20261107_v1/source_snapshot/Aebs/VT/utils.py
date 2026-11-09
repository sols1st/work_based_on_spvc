import torch
import torch.nn as nn
import torch.nn.functional as F

def triangular(shape, device="cuda", dtype=torch.float32):
    """
    Generate triangular-distributed noise over the interval [-1, 1].
    """
    # Uniform distribution over [0, 1)
    U = torch.rand(shape, device=device, dtype=dtype)
    p1 = -1 + torch.sqrt(2 * U)
    p2 = 1 - torch.sqrt(2 * (1 - U))
    return torch.where(U <= 0.5, p1, p2)

def martingale_loss(l, l_next, eps=0.0):
    """
    Compute the martingale loss:
        loss = mean(max(l_next - l + eps, 0))
    """
    diff = l_next - l
    return torch.mean(torch.clamp(diff + eps, min=0.0))


# -------------------------
# B function
# -------------------------
class MLP(nn.Module):
    def __init__(self, features, activation="relu", square_output=False):
        super().__init__()
        self.features = features
        self.activation = activation
        self.square_output = square_output
        
        # Construct list of linear layers
        layers = []
        for in_feat, out_feat in zip(features[:-1], features[1:]):
            layers.append(nn.Linear(in_feat, out_feat))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # Iterate through all layers except the last one
        for layer in self.layers[:-1]:
            x = layer(x)
            if self.activation == "relu":
                x = F.relu(x)
            elif self.activation == "tanh":
                x = torch.tanh(x)
            else:
                raise ValueError(f"Unsupported activation: {self.activation}")
        
        # Final layer
        x = self.layers[-1](x)
        if self.square_output:
            x = F.softplus(x)
            # x = x ** 2
            # x = F.relu(x)
        return x
