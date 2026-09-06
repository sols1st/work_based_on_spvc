import torch
import torch.nn as nn
import torch.nn.functional as F

# Power iteration function
def power_iteration(W, u, n_iterations=1, eps=1e-12):
    for _ in range(n_iterations):
        v = torch.matmul(W.T, u)
        v = v / (v.norm() + eps)
        u = torch.matmul(W, v)
        u = u / (u.norm() + eps)
    return u, v

def max_singular_value(W, u, v):
    return (u.T @ W @ v).item()


# Fully Connected Layer with Spectral Normalization
class DenseSN(nn.Module):
    def __init__(self, in_features, out_features, activation=None, n_iterations=1):
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_features, in_features))
        self.b = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.W)
        self.activation = activation if activation is not None else lambda x: x
        self.n_iterations = n_iterations
        self.register_buffer("u", torch.randn(out_features, 1))

    def forward(self, x):
        u, v = power_iteration(self.W, self.u, self.n_iterations)
        sigma = max_singular_value(self.W, u, v)
        # Spectral Normalization (by Computing the Maximum Singular Value)
        W_sn = self.W / sigma

        x = F.linear(x, W_sn, self.b)
        return self.activation(x)


# Convolutional Layer with Spectral Normalization
class ConvSN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, activation=None, n_iterations=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=True)
        self.activation = activation if activation is not None else lambda x: x
        self.n_iterations = n_iterations

        # Initialize power iteration vector u
        out_dim = out_channels
        self.register_buffer("u", torch.randn(out_dim, 1))

    def forward(self, x):
        # Flatten the convolutional weights into a 2D matrix
        W = self.conv.weight
        W_mat = W.view(W.size(0), -1)  # [out_channels, in_channels * kH * kW]

        # Compute the largest singular value via power iteration
        u, v = power_iteration(W_mat, self.u, self.n_iterations)
        sigma = max_singular_value(W_mat, u, v)

        W_sn = W / sigma

        x = F.conv2d(x, W_sn, self.conv.bias, stride=self.conv.stride,
                     padding=self.conv.padding, dilation=self.conv.dilation)
        return self.activation(x)
