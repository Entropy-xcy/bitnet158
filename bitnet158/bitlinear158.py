import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch


def absmax_quantize(x: Tensor, bits: int = 8):
    """
    Absmax Quantization

    Args:
        x (torch.Tensor): Input tensor
        bits (int, optional): Number of bits. Defaults to 8.

    """
    Qb = 2 ** (bits - 1) - 1
    scale = Qb / torch.max(torch.abs(x))
    quant = (scale * x).round()
    dequant = quant / scale
    return quant.to(torch.int8), dequant


class BitLinear158(nn.Module):
    """
    BitLinear implements a fully connected layer with ternary weight quantization.
    Weights are quantized to -1, 0, or +1 using an absmean quantization approach.
    """

    def __init__(self, in_features, out_features, bias=True):
        """
        Initializes the BitLinear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
        """
        super(BitLinear158, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.bias = None
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.eps = 1e-6  # Small epsilon for numerical stability

    def forward(self, x):
        """
        Forward pass through the BitLinear layer.

        Args:
            x (Tensor): Input tensor of shape (..., in_features).

        Returns:
            Tensor: Output tensor of shape (..., out_features).
        """
        # x = torch.sign(x)
        quantized_weight = self.quantize_weights(self.weight)
        return F.linear(x, quantized_weight)

    def quantize_weights(self, W):
        """
        Quantizes the weights using the absmean quantization function.

        Args:
            W (Tensor): The weight tensor to be quantized.

        Returns:
            Tensor: Quantized weight tensor.
        """
        gamma = torch.mean(torch.abs(W)) + self.eps
        W_scaled = W / gamma
        W_quantized = torch.sign(W_scaled) * torch.clamp(
            torch.abs(W_scaled).round(), max=1.0
        )
        return W_quantized

    def extra_repr(self):
        """
        Provides additional information for debugging and logging.
        """
        return "in_features={}, out_features={}, quantization=ternary".format(
            self.in_features, self.out_features
        )
