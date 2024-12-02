# triplanar_attention_mechanism/layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class TriplanarAttentionLayer(nn.Module):
    """
    Triplanar attention layer module.

    Attributes:
        embeddings (nn.ModuleList): Embedding transforms (three channel-wise and spatially-factorised convolution (transpose) 3d modules that transform an input tensor to its three planar embedding tensors).
        key        (nn.Conv3d):     Key transform        (a point-wise convolution 3d module that transforms a planar embedding tensor to its key tensor).
        query      (nn.Conv3d):     Query transform      (a point-wise convolution 3d module that transforms a planar embedding tensor to its query tensor).
        value      (nn.Conv3d):     Value transform      (a point-wise convolution 3d module that transforms a planar embedding tensor to its value tensor).

    Args:
        in_channels    (int):            Input channels.
        out_channels   (int):            Output channels.
        kernel_size    (int):            Kernel size.
        stride         (int,  optional): Stride.          Defaults to 1.
        padding        (int,  optional): Padding.         Defaults to 0.
        output_padding (int,  optional): Output padding.  Defaults to 0.
        bias           (bool, optional): Bias.            Defaults to True.
        dilation       (int,  optional): Dilation.        Defaults to 1.
        padding_mode   (str,  optional): Padding mode.    Defaults to 'zeros'.
        tr_flag        (bool, optional): Transpose flag.  Defaults to False.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, bias: bool = True, dilation: int = 1, padding_mode: str = 'zeros', tr_flag: bool = False) -> None:
        super().__init__()
        _kwargs = {'kernel_size': kernel_size, 'stride': stride, 'padding': padding, **({'output_padding': output_padding} if tr_flag else {}), 'dilation': dilation}
        _module = lambda kwargs: (nn.Conv3d if not tr_flag else nn.ConvTranspose3d)(in_channels, in_channels, groups=in_channels, bias=bias, padding_mode=padding_mode, **kwargs)
        self.embeddings = nn.ModuleList([
            nn.Sequential(_module({k: (0, v, v) if 'padding' in k else (1, v, v) for k, v in _kwargs.items()}), _module({k: (v, 0, 0) if 'padding' in k else (v, 1, 1) for k, v in _kwargs.items()})), # XY-embedding transform.
            nn.Sequential(_module({k: (v, 0, v) if 'padding' in k else (v, 1, v) for k, v in _kwargs.items()}), _module({k: (0, v, 0) if 'padding' in k else (1, v, 1) for k, v in _kwargs.items()})), # XZ-embedding transform.
            nn.Sequential(_module({k: (v, v, 0) if 'padding' in k else (v, v, 1) for k, v in _kwargs.items()}), _module({k: (0, 0, v) if 'padding' in k else (1, 1, v) for k, v in _kwargs.items()}))  # YZ-embedding transform.
        ])
        self.key = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.query = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.value = nn.Conv3d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor (N, in_channels, D_in, H_in, W_in).

        Returns:
            torch.Tensor: Output tensor (N, out_channels, D_out, H_out, W_out), where
                          *_out = floor((*_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)) if not tr_flag,
                                  (*_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1 otherwise.
        """
        keys, queries, values = zip(*[(self.key(e), self.query(e), self.value(e)) for e in (e(x) for e in self.embeddings)])
        attentions = [k.size(1)**-0.5 * (k * q).sum(1, True) for (k, q) in zip(keys, queries)]
        return (F.softmax(torch.stack(attentions, 5), 5) * torch.stack(values, 5)).sum(5)
