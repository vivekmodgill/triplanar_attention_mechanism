# triplanar_attention_mechanism/networks.py

from typing import Optional, Tuple

import torch
import torch.nn as nn

from triplanar_attention_mechanism import TriplanarAttentionLayer


class TriplanarAttentionNetwork(nn.Sequential):
    """
    Triplanar attention network module.

    Args:
        in_channels    (Tuple[int, ...]): Input channels.
        out_channels   (Optional[int]):   Output channels.
        kernel_size    (int):             Kernel size of triplanar attention layer.
        stride         (int,   optional): Stride of triplanar attention layer.           Defaults to 1.
        padding        (int,   optional): Padding of triplanar attention layer.          Defaults to 0.
        #output_padding (int,   optional): Output padding of triplanar attention layer.   Defaults to 0.
        #dilation       (int,   optional): Dilation of triplanar attention layer.         Defaults to 1.
        #padding_mode   (str,   optional): Padding mode of triplanar attention layer.     Defaults to 'zeros'.
        #negative_slope (float, optional): Negative slope of leaky rectified linear unit. Defaults to .01.
    """
    def __init__(self, in_channels: Tuple[int, ...], out_channels: Optional[int], kernel_size: int, stride: int = 1, padding: int = 0) -> None:
        #output_padding: int = 0, dilation: int = 1, padding_mode: str = 'zeros', negative_slope: float = .01) -> None:
        _factor = min(len(in_channels[1:]), 2)
        _kwargs = {'kernel_size': kernel_size, 'stride': stride, 'padding': padding}
        #'output_padding': output_padding, 'dilation': dilation, 'padding_mode': padding_mode}
        _module = lambda: nn.Identity() if _factor == 1 else TriplanarAttentionNetwork(in_channels[1:], None, **_kwargs)
        #negative_slope=negative_slope, **_kwargs)
        super().__init__(*(
            TriplanarAttentionLayer(in_channels[0], in_channels[1], **_kwargs),                                     # Input layer.
            nn.LeakyReLU(negative_slope),                                                                           # Input activation.
            _module(),
            TriplanarAttentionLayer(_factor * in_channels[1], out_channels, tr_flag=True, **_kwargs),               # Output layer.
            nn.Identity()                                                                                           # Output activation.
        ) if out_channels else (
            TriplanarAttentionLayer(in_channels[0], in_channels[1], bias=False, **_kwargs),                         # Contraction layer.
            nn.BatchNorm3d(in_channels[1]),                                                                         # Contraction normalisation.
            nn.LeakyReLU(negative_slope),                                                                           # Contraction activation.
            _module(),
            TriplanarAttentionLayer(_factor * in_channels[1], in_channels[0], bias=False, tr_flag=True, **_kwargs), # Expansion layer.
            nn.BatchNorm3d(in_channels[0]),                                                                         # Expansion normalisation.
            nn.ReLU()                                                                                               # Expansion activation.
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor (N, in_channels[0], D, H, W).

        Returns:
            torch.Tensor: Output tensor (N, out_channels or 2 * in_channels[0], D, H, W).
        """
        return super().forward(x) if isinstance(list(self.children())[-1], nn.Identity) else torch.cat((x, super().forward(x)), 1)
