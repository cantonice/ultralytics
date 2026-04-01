# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Depthwise Separable Convolution modules for lightweight YOLO models."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .conv import Conv, DWConv, autopad

__all__ = ("DSConv", "DSConv2d", "BottleneckDS", "C2fDS", "C3k2DS")


class DSConv2d(nn.Module):
    """Depthwise Separable Convolution 2D.
    
    Combines depthwise convolution (grouped convolution with groups=in_channels)
    and pointwise convolution (1x1 convolution) to reduce parameters.
    
    Attributes:
        depthwise (nn.Conv2d): Depthwise convolution layer.
        pointwise (nn.Conv2d): Pointwise convolution layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=None, d=1, act=True):
        """Initialize DSConv2d layer.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            s (int): Stride.
            p (int, optional): Padding.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        # Depthwise convolution: groups = c1 (each input channel is convolved separately)
        self.depthwise = nn.Conv2d(
            c1, c1, k, s, autopad(k, p, d), groups=c1, dilation=d, bias=False
        )
        # Pointwise convolution: 1x1 conv to combine channels
        self.pointwise = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply depthwise separable convolution.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            (torch.Tensor): Output tensor.
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.bn(x))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            (torch.Tensor): Output tensor.
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(x)


class DSConv(Conv):
    """Standard convolution replaced with Depthwise Separable Convolution.
    
    This is a drop-in replacement for the standard Conv class that uses
    depthwise separable convolution instead of regular convolution.
    
    Attributes:
        conv (DSConv2d): Depthwise separable convolution layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize DSConv layer.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups (ignored for DSConv, kept for interface compatibility).
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g, d, act)
        # Replace standard conv with depthwise separable conv
        self.conv = DSConv2d(c1, c2, k, s, p, d, act=False)
        # Re-create BN and act
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()


class BottleneckDS(nn.Module):
    """Standard bottleneck with Depthwise Separable Convolution.
    
    Replaces the second 3x3 convolution in the bottleneck with DSConv.
    """

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize BottleneckDS module.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups (ignored).
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # First conv remains standard Conv
        self.cv1 = Conv(c1, c_, k[0], 1)
        # Second conv uses DSConv instead of standard Conv
        self.cv2 = DSConv(c_, c2, k[1], 1, act=True)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DS bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2fDS(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions and DSConv.
    
    Replaces some Conv layers with DSConv in the C2f structure.
    
    Modes:
        - 'none': No DSConv, standard C2f
        - 'partial': Replace bottleneck convs with DSConv
        - 'full': Replace all applicable convs with DSConv
    """

    def __init__(
        self, 
        c1: int, 
        c2: int, 
        n: int = 1, 
        shortcut: bool = False, 
        g: int = 1, 
        e: float = 0.5,
        ds_mode: str = "partial"  # 'none', 'partial', 'full'
    ):
        """Initialize C2fDS layer.
        
        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            ds_mode (str): DSConv replacement mode.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.ds_mode = ds_mode
        
        # First conv: usually keep as standard Conv for channel adjustment
        if ds_mode == "full":
            self.cv1 = DSConv(c1, 2 * self.c, 1, 1)
            self.cv2 = DSConv((2 + n) * self.c, c2, 1)
        else:
            self.cv1 = Conv(c1, 2 * self.c, 1, 1)
            self.cv2 = Conv((2 + n) * self.c, c2, 1)
        
        # Bottlenecks: use DSConv if mode is 'partial' or 'full'
        BottleneckClass = BottleneckDS if ds_mode in ("partial", "full") else Bottleneck
        self.m = nn.ModuleList(
            BottleneckClass(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) 
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2fDS layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2DS(nn.Module):
    """C3k2 with Depthwise Separable Convolution option.
    
    This is the YOLO11/YOLO26 version of C2f with C3k blocks.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        ds_mode: str = "none",  # 'none', 'partial', 'full'
    ):
        """Initialize C3k2DS module.
        
        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            attn (bool): Whether to use attention blocks.
            g (int): Groups.
            shortcut (bool): Whether to use shortcut.
            ds_mode (str): DSConv replacement mode.
        """
        from .block import C3k, PSABlock
        
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.ds_mode = ds_mode
        
        # Input/output convs
        if ds_mode == "full":
            self.cv1 = DSConv(c1, 2 * self.c, 1, 1)
            self.cv2 = DSConv((2 + n) * self.c, c2, 1)
        else:
            self.cv1 = Conv(c1, 2 * self.c, 1, 1)
            self.cv2 = Conv((2 + n) * self.c, c2, 1)
        
        # Build blocks
        if ds_mode in ("partial", "full") and not attn and not c3k:
            # Use DSConv bottlenecks
            self.m = nn.ModuleList(
                BottleneckDS(self.c, self.c, shortcut, g)
                for _ in range(n)
            )
        else:
            # Use original blocks
            self.m = nn.ModuleList(
                nn.Sequential(
                    Bottleneck(self.c, self.c, shortcut, g),
                    PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
                )
                if attn
                else C3k(self.c, self.c, 2, shortcut, g)
                if c3k
                else Bottleneck(self.c, self.c, shortcut, g)
                for _ in range(n)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C3k2DS layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
