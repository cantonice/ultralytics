# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Spatial-frequency parallel modules."""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv


class FrequencyBranch(nn.Module):
    """A lightweight frequency-domain branch using FFT -> 1x1 mixing -> iFFT."""

    def __init__(self, c1, c2, s=1, act=True):
        """Initialize frequency branch.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            s (int): Optional spatial downsampling stride.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.down = nn.AvgPool2d(kernel_size=s, stride=s) if s > 1 else nn.Identity()
        self.mix = nn.Conv2d(2 * c1, 2 * c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(2 * c2)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply frequency branch to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            (torch.Tensor): Output tensor of shape [B, C2, H/s, W/s].
        """
        x = self.down(x)
        h, w = x.shape[-2:]

        xf = torch.fft.rfft2(x, norm="ortho")
        ri = torch.cat((xf.real, xf.imag), dim=1)
        ri = self.act(self.bn(self.mix(ri)))

        real, imag = ri.chunk(2, dim=1)
        return torch.fft.irfft2(torch.complex(real, imag), s=(h, w), norm="ortho")


class SFParallelConv(nn.Module):
    """Spatial + frequency parallel convolution block.

    The block keeps `Conv`-like YAML arguments, with an extra optional `branch` argument:
    - "both": run spatial and frequency branches in parallel, then fuse
    - "spatial": run only the spatial branch
    - "freq": run only the frequency branch
    """

    def __init__(self, c1, c2, k=3, s=1, branch="both", p=None, g=1, act=True):
        """Initialize spatial-frequency parallel conv block.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Spatial branch kernel size.
            s (int): Stride for both branches.
            branch (str): One of {"both", "spatial", "freq"}.
            p (int, optional): Spatial branch padding.
            g (int): Spatial branch groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        if branch not in {"both", "spatial", "freq"}:
            raise ValueError(f"Invalid branch='{branch}'. Expected one of ['both', 'spatial', 'freq'].")

        self.branch = branch
        self.spatial = Conv(c1, c2, k, s, p, g, act=act) if branch in {"both", "spatial"} else None
        self.freq = FrequencyBranch(c1, c2, s=s, act=act) if branch in {"both", "freq"} else None
        self.fuse = Conv(2 * c2, c2, k=1, s=1, act=act) if branch == "both" else nn.Identity()

    def forward(self, x):
        """Forward pass."""
        if self.branch == "both":
            return self.fuse(torch.cat((self.spatial(x), self.freq(x)), dim=1))
        if self.branch == "spatial":
            return self.spatial(x)
        return self.freq(x)
