"""
CSPDarknet Backbone Implementation
Based on the paper: Target Detection Algorithm for Drone Aerial Images based on Deep Learning
"""

import torch
import torch.nn as nn


class SiLU(nn.Module):
    """SiLU activation function: x * sigmoid(x)"""
    
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class BaseConv(nn.Module):
    """Standard convolution with BatchNorm and activation"""
    
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = SiLU() if act == "silu" else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutions and skip connection
    As described in the paper: backbone with 1×1 conv and 3×3 conv, residual edge uses skip connections
    """
    
    def __init__(self, in_channels):
        super().__init__()
        hidden_channels = in_channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1)
        self.conv2 = BaseConv(hidden_channels, in_channels, 3, stride=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return out


class CSPLayer(nn.Module):
    """
    CSP Layer as described in the paper:
    - Main branch: stacks N residual blocks
    - Residual branch: directly concatenated with main branch after minimal processing
    """
    
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        
        # Split input into two branches
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1)
        
        # Main branch with N residual blocks
        self.m = nn.Sequential(*[ResidualBlock(hidden_channels) for _ in range(n)])
        
        # Final convolution after concatenation
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1)

    def forward(self, x):
        # Main branch with residual blocks
        x_1 = self.conv1(x)
        x_1 = self.m(x_1)
        
        # Residual branch (minimal processing)
        x_2 = self.conv2(x)
        
        # Concatenate and process
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus module to downsample input efficiently"""
    
    def __init__(self, in_channels, out_channels, ksize=1, stride=1):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride)

    def forward(self, x):
        # Shape: [B, C, H, W] -> [B, 4C, H/2, W/2]
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_top_right, patch_bot_left, patch_bot_right), dim=1)
        return self.conv(x)


class SPPBottleneck(nn.Module):
    """Spatial Pyramid Pooling layer"""
    
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13)):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1)
        self.m = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPDarknet(nn.Module):
    """
    CSPDarknet backbone as described in the paper.
    Consists of multiple CSPLayer structures with residual convolution.
    Outputs feature maps at multiple scales: C2, C3, C4, C5
    """
    
    def __init__(self, dep_mul=0.33, wid_mul=0.5, in_channels=3):
        super().__init__()
        
        # Depth and width multipliers for different model sizes
        # For YOLOX-S: dep_mul=0.33, wid_mul=0.5
        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)

        # Stem: Focus layer
        self.stem = Focus(in_channels, base_channels, ksize=3, stride=1)

        # Stage 1: Output C2 (80x80 for 640x640 input)
        self.dark2 = nn.Sequential(
            BaseConv(base_channels, base_channels * 2, 3, 2),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                shortcut=True,
            ),
        )

        # Stage 2: Output C3 (40x40)
        self.dark3 = nn.Sequential(
            BaseConv(base_channels * 2, base_channels * 4, 3, 2),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                shortcut=True,
            ),
        )

        # Stage 3: Output C4 (20x20)
        self.dark4 = nn.Sequential(
            BaseConv(base_channels * 4, base_channels * 8, 3, 2),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                shortcut=True,
            ),
        )

        # Stage 4: Output C5 (10x10)
        self.dark5 = nn.Sequential(
            BaseConv(base_channels * 8, base_channels * 16, 3, 2),
            SPPBottleneck(base_channels * 16, base_channels * 16),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
            ),
        )

    def forward(self, x):
        outputs = {}
        
        x = self.stem(x)
        outputs["stem"] = x
        
        # C2: 80x80 feature map (for small object detection)
        x = self.dark2(x)
        outputs["dark2"] = x
        
        # C3: 40x40 feature map
        x = self.dark3(x)
        outputs["dark3"] = x
        
        # C4: 20x20 feature map
        x = self.dark4(x)
        outputs["dark4"] = x
        
        # C5: 10x10 feature map
        x = self.dark5(x)
        outputs["dark5"] = x
        
        return outputs


def build_cspdarknet(model_size="s"):
    """
    Build CSPDarknet backbone with different sizes
    
    Args:
        model_size: Model size - 's' (small), 'm' (medium), 'l' (large), 'x' (extra large)
    """
    size_config = {
        's': {'dep_mul': 0.33, 'wid_mul': 0.5},
        'm': {'dep_mul': 0.67, 'wid_mul': 0.75},
        'l': {'dep_mul': 1.0, 'wid_mul': 1.0},
        'x': {'dep_mul': 1.33, 'wid_mul': 1.25},
    }
    
    config = size_config.get(model_size.lower(), size_config['s'])
    return CSPDarknet(**config)


if __name__ == "__main__":
    # Test the backbone
    model = build_cspdarknet("s")
    x = torch.randn(1, 3, 640, 640)
    outputs = model(x)
    
    print("CSPDarknet Backbone Output Shapes:")
    for name, feat in outputs.items():
        print(f"{name}: {feat.shape}")
