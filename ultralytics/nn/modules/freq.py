# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Spatial-frequency parallel modules."""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv


class FrequencyBranch(nn.Module):
    """A lightweight frequency-domain branch using FFT -> 1x1 mixing -> iFFT."""

    def __init__(
        self,
        c1,
        c2,
        s=1,
        act=True,
        gate=False,
        gate_reduction=4,
        mode="vanilla",
        residual=False,
        alpha=0.5,
    ):
        """Initialize frequency branch.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            s (int): Optional spatial downsampling stride.
            act (bool | nn.Module): Activation function.
            gate (bool): Whether to apply channel gating on frequency features.
            gate_reduction (int): Channel reduction ratio in gate MLP.
            mode (str): One of {"vanilla", "highpass"} for pre-FFT processing.
            residual (bool): Whether to add residual blending from downsampled input.
            alpha (float): Initial scale for frequency residual output.
        """
        super().__init__()
        if mode not in {"vanilla", "highpass"}:
            raise ValueError(f"Invalid mode='{mode}'. Expected one of ['vanilla', 'highpass'].")

        self.mode = mode
        self.residual = bool(residual)
        self.down = nn.AvgPool2d(kernel_size=s, stride=s) if s > 1 else nn.Identity()
        self.hp = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.mix = nn.Conv2d(2 * c1, 2 * c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(2 * c2)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.use_gate = bool(gate)
        hidden = max(c2 // max(int(gate_reduction), 1), 8)
        self.gate = (
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c2, hidden, kernel_size=1, bias=True),
                nn.SiLU(),
                nn.Conv2d(hidden, c2, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
            if self.use_gate
            else nn.Identity()
        )
        self.res_proj = Conv(c1, c2, k=1, s=1, act=False) if self.residual and c1 != c2 else nn.Identity()
        self.alpha = nn.Parameter(torch.tensor(float(alpha))) if self.residual else None

    def forward(self, x):
        """Apply frequency branch to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            (torch.Tensor): Output tensor of shape [B, C2, H/s, W/s].
        """
        x = self.down(x)
        h, w = x.shape[-2:]

        x_fft = x - self.hp(x) if self.mode == "highpass" else x
        xf = torch.fft.rfft2(x_fft, norm="ortho")
        ri = torch.cat((xf.real, xf.imag), dim=1)
        ri = self.act(self.bn(self.mix(ri)))

        real, imag = ri.chunk(2, dim=1)
        out = torch.fft.irfft2(torch.complex(real, imag), s=(h, w), norm="ortho")
        out = out * self.gate(out)
        if self.residual:
            return self.res_proj(x) + torch.tanh(self.alpha) * out
        return out


class SFParallelConv(nn.Module):
    """Spatial + frequency parallel convolution block.

    The block keeps `Conv`-like YAML arguments, with an extra optional `branch` argument:
    - "both": run spatial and frequency branches in parallel, then fuse
    - "spatial": run only the spatial branch
    - "freq": run only the frequency branch
    """

    def __init__(
        self,
        c1,
        c2,
        k=3,
        s=1,
        branch="both",
        p=None,
        g=1,
        act=True,
        fuse_mode="concat",
        freq_gate=False,
        gate_reduction=4,
        freq_mode="vanilla",
        freq_residual=False,
        freq_alpha=0.5,
    ):
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
            fuse_mode (str): One of {"concat", "sum", "gated"} when branch="both".
            freq_gate (bool): Enable channel gate in frequency branch.
            gate_reduction (int): Channel reduction ratio in gate MLP.
            freq_mode (str): One of {"vanilla", "highpass"} for frequency branch.
            freq_residual (bool): Whether to add residual blending in frequency branch.
            freq_alpha (float): Initial frequency residual scale.
        """
        super().__init__()
        if branch not in {"both", "spatial", "freq"}:
            raise ValueError(f"Invalid branch='{branch}'. Expected one of ['both', 'spatial', 'freq'].")
        if fuse_mode not in {"concat", "sum", "gated"}:
            raise ValueError(f"Invalid fuse_mode='{fuse_mode}'. Expected one of ['concat', 'sum', 'gated'].")

        self.branch = branch
        self.fuse_mode = fuse_mode
        self.spatial = Conv(c1, c2, k, s, p, g, act=act) if branch in {"both", "spatial"} else None
        self.freq = (
            FrequencyBranch(
                c1,
                c2,
                s=s,
                act=act,
                gate=freq_gate,
                gate_reduction=gate_reduction,
                mode=freq_mode,
                residual=freq_residual,
                alpha=freq_alpha,
            )
            if branch in {"both", "freq"}
            else None
        )
        self.fuse = Conv(2 * c2, c2, k=1, s=1, act=act) if branch == "both" and fuse_mode == "concat" else None
        self.mix_gate = (
            nn.Conv2d(2 * c2, 2 * c2, kernel_size=1, bias=True)
            if branch == "both" and fuse_mode == "gated"
            else None
        )

    def forward(self, x):
        """Forward pass."""
        if self.branch == "both":
            xs, xf = self.spatial(x), self.freq(x)
            if self.fuse_mode == "concat":
                return self.fuse(torch.cat((xs, xf), dim=1))
            if self.fuse_mode == "sum":
                return 0.5 * (xs + xf)
            b, c, h, w = xs.shape
            gate = self.mix_gate(torch.cat((xs, xf), dim=1)).view(b, 2, c, h, w)
            gate = torch.softmax(gate, dim=1)
            return gate[:, 0] * xs + gate[:, 1] * xf
        if self.branch == "spatial":
            return self.spatial(x)
        return self.freq(x)
