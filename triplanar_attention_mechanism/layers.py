import torch
import torch.nn as nn
import torch.nn.functional as F


class TriplanarAttentionLayer(nn.Module):
    """
    Simplified 3D convolutional layer as a replacement for the triplanar attention mechanism.

    Attributes:
        conv3d (nn.Conv3d): A single 3D convolution layer to process the input tensor.
    """

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int = 1, 
        padding: int = 0, 
        output_padding: int = 0, 
        bias: bool = True, 
        dilation: int = 1, 
        padding_mode: str = 'zeros', 
        tr_flag: bool = False
    ) -> None:
        """
        Initialize the layer.

        Args:
            in_channels    (int): Number of input channels.
            out_channels   (int): Number of output channels.
            kernel_size    (int): Kernel size for convolution.
            stride         (int): Stride for convolution. Defaults to 1.
            padding        (int): Padding for convolution. Defaults to 0.
            output_padding (int): Output padding for transpose convolution. Defaults to 0.
            bias           (bool): Whether to use bias in convolution. Defaults to True.
            dilation       (int): Dilation rate for convolution. Defaults to 1.
            padding_mode   (str): Padding mode for convolution. Defaults to 'zeros'.
            tr_flag        (bool): Transpose convolution flag. Defaults to False.
        """
        super().__init__()
        _kwargs = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'dilation': dilation,
            **({'output_padding': output_padding} if tr_flag else {}),
        }
        self.conv3d = (nn.Conv3d if not tr_flag else nn.ConvTranspose3d)(
            in_channels=in_channels, 
            out_channels=out_channels, 
            bias=bias, 
            padding_mode=padding_mode, 
            **_kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor (N, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor (N, out_channels, D_out, H_out, W_out).
        """
        return self.conv3d(x)
