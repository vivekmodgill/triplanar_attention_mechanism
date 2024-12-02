import torch
import torch.nn as nn
import torch.nn.functional as F


class TriplanarAttentionLayer(nn.Module):
    """
    Simplified 3D convolutional layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size for convolution.
        stride (int, optional): Stride for convolution. Defaults to 1.
        padding (int, optional): Padding for convolution. Defaults to 0.
        bias (bool, optional): Whether to include bias. Defaults to True.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True):
        super(TriplanarAttentionLayer, self).__init__()
        self.conv3d = nn.Conv3d(
            in_channels =in_channels,
            out_channels=out_channels,
            kernel_size =kernel_size,
            stride      =stride,
            padding     =padding,
            bias        =bias
        )
        self.bn         = nn.BatchNorm3d(out_channels)  # Optional: Batch normalization
        self.activation = nn.ReLU()             # Optional: Activation function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor (N, in_channels, D_in, H_in, W_in).

        Returns:
            torch.Tensor: Output tensor (N, out_channels, D_out, H_out, W_out).
        """
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

