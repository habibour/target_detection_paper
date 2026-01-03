"""
ASFF (Adaptive Spatial Feature Fusion) Neck Module
Based on the paper: Target Detection Algorithm for Drone Aerial Images based on Deep Learning

The ASFF module replaces traditional PANet and performs adaptive feature fusion 
by learning weights for different feature layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import BaseConv


class ASFF(nn.Module):
    """
    Adaptive Spatial Feature Fusion Module
    
    As described in the paper:
    1. Feature adjustment: maps features from other scales to corresponding scale
    2. Adaptive fusion: learns important weight parameters α, β, γ for three different feature layers
    
    Formula: y_l = α_l→l * f_l→l + β_l * f_2→l + γ_l * f_3→l
    where α_l + β_l + γ_l = 1
    """
    
    def __init__(self, level, channels, multiplier=1):
        """
        Args:
            level: Output level (1, 2, or 3 corresponding to P2, P3, P4)
            channels: Number of channels for each level [c1, c2, c3]
            multiplier: Channel multiplier for intermediate convolutions
        """
        super(ASFF, self).__init__()
        self.level = level
        self.dim = channels[level]
        self.inter_dim = int(self.dim * multiplier)
        
        # Compression convolutions for each input level
        compress_c = 8
        
        if level == 0:  # P2 level (highest resolution)
            # P2 -> P2 (no change needed)
            self.stride_level_1 = BaseConv(channels[1], self.inter_dim, 3, 2)  # P3 -> P2
            self.stride_level_2 = nn.Sequential(
                BaseConv(channels[2], self.inter_dim, 3, 2),  # P4 -> intermediate
                BaseConv(self.inter_dim, self.inter_dim, 3, 2)  # intermediate -> P2
            )
            
            # Weight learning branches
            self.weight_level_0 = BaseConv(self.inter_dim, compress_c, 1, 1)
            self.weight_level_1 = BaseConv(self.inter_dim, compress_c, 1, 1)
            self.weight_level_2 = BaseConv(self.inter_dim, compress_c, 1, 1)
            
            self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
            
        elif level == 1:  # P3 level (medium resolution)
            # P3 -> P3 (no change needed)
            self.stride_level_0 = nn.ConvTranspose2d(channels[0], self.inter_dim, 
                                                      kernel_size=2, stride=2)  # P2 -> P3 (upsample)
            self.stride_level_2 = BaseConv(channels[2], self.inter_dim, 3, 2)  # P4 -> P3
            
            # Weight learning branches
            self.weight_level_0 = BaseConv(self.inter_dim, compress_c, 1, 1)
            self.weight_level_1 = BaseConv(self.inter_dim, compress_c, 1, 1)
            self.weight_level_2 = BaseConv(self.inter_dim, compress_c, 1, 1)
            
            self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
            
        elif level == 2:  # P4 level (lowest resolution)
            # P4 -> P4 (no change needed)
            self.stride_level_0 = nn.Sequential(
                nn.ConvTranspose2d(channels[0], self.inter_dim, kernel_size=2, stride=2),
                nn.ConvTranspose2d(self.inter_dim, self.inter_dim, kernel_size=2, stride=2)
            )  # P2 -> P4 (upsample 4x)
            self.stride_level_1 = nn.ConvTranspose2d(channels[1], self.inter_dim,
                                                      kernel_size=2, stride=2)  # P3 -> P4 (upsample)
            
            # Weight learning branches
            self.weight_level_0 = BaseConv(self.inter_dim, compress_c, 1, 1)
            self.weight_level_1 = BaseConv(self.inter_dim, compress_c, 1, 1)
            self.weight_level_2 = BaseConv(self.inter_dim, compress_c, 1, 1)
            
            self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        
        # Final expansion convolution
        self.expand = BaseConv(self.inter_dim, self.dim, 3, 1)

    def forward(self, x_level_0, x_level_1, x_level_2):
        """
        Args:
            x_level_0, x_level_1, x_level_2: Feature maps from three scales
            
        Returns:
            Fused feature map at target level
        """
        # Feature adjustment - resize all inputs to target level
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 1:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = x_level_2
        
        # Learn adaptive weights for each level
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        
        # Concatenate weight features
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), dim=1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)  # Normalize weights (sum to 1)
        
        # Adaptive fusion using learned weights
        fused_out_reduced = (
            level_0_resized * levels_weight[:, 0:1, :, :] +
            level_1_resized * levels_weight[:, 1:2, :, :] +
            level_2_resized * levels_weight[:, 2:, :, :]
        )
        
        # Expand to target dimension
        out = self.expand(fused_out_reduced)
        
        return out


class ASFFNeck(nn.Module):
    """
    Complete ASFF-based neck network for HE-YOLOX
    
    Takes feature maps from backbone (C2, C3, C4, C5) and produces
    fused feature maps (P2, P3, P4, P5) for detection heads.
    """
    
    def __init__(self, in_channels_list, out_channels=256):
        """
        Args:
            in_channels_list: List of input channels [C2, C3, C4, C5] from backbone
            out_channels: Output channels for all feature levels
        """
        super(ASFFNeck, self).__init__()
        
        # Path Aggregation - Bottom-up path
        self.lateral_conv0 = BaseConv(in_channels_list[3], out_channels, 1, 1)  # C5
        self.C3_p5 = BaseConv(in_channels_list[2], out_channels, 1, 1)  # C4
        self.C3_n5 = BaseConv(out_channels, out_channels, 3, 1)
        
        self.C3_p4 = BaseConv(in_channels_list[1], out_channels, 1, 1)  # C3
        self.C3_n4 = BaseConv(out_channels, out_channels, 3, 1)
        
        self.C3_p3 = BaseConv(in_channels_list[0], out_channels, 1, 1)  # C2
        self.C3_n3 = BaseConv(out_channels, out_channels, 3, 1)
        
        # Upsample layers
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        
        # Bottom-up path (P2 -> P5)
        self.bu_conv4 = BaseConv(out_channels, out_channels, 3, 2)  # P2 -> P3
        self.C3_n6 = BaseConv(out_channels, out_channels, 3, 1)
        
        self.bu_conv5 = BaseConv(out_channels, out_channels, 3, 2)  # P3 -> P4
        self.C3_n7 = BaseConv(out_channels, out_channels, 3, 1)
        
        # ASFF modules for adaptive fusion
        channels = [out_channels, out_channels, out_channels]
        self.asff_p2 = ASFF(level=0, channels=channels)
        self.asff_p3 = ASFF(level=1, channels=channels)
        self.asff_p4 = ASFF(level=2, channels=channels)

    def forward(self, input):
        """
        Args:
            input: Dictionary with keys 'dark2', 'dark3', 'dark4', 'dark5' from backbone
            
        Returns:
            List of feature maps [P2, P3, P4, P5] for detection
        """
        [C2, C3, C4, C5] = [input[f] for f in ['dark2', 'dark3', 'dark4', 'dark5']]
        
        # Top-down pathway
        # P5 (from C5)
        P5 = self.lateral_conv0(C5)
        
        # P4 (from C4)
        P5_upsample = self.upsample(P5)
        P4 = self.C3_p5(C4)
        P4 = torch.cat([P5_upsample, P4], dim=1)
        P4 = self.C3_n5(P4)
        
        # P3 (from C3)
        P4_upsample = self.upsample(P4)
        P3 = self.C3_p4(C3)
        P3 = torch.cat([P4_upsample, P3], dim=1)
        P3 = self.C3_n4(P3)
        
        # P2 (from C2) - Small target detection layer
        P3_upsample = self.upsample(P3)
        P2 = self.C3_p3(C2)
        P2 = torch.cat([P3_upsample, P2], dim=1)
        P2 = self.C3_n3(P2)
        
        # Bottom-up pathway with ASFF
        # Enhanced P3
        P2_downsample = self.bu_conv4(P2)
        P3_enhanced = torch.cat([P2_downsample, P3], dim=1)
        P3_enhanced = self.C3_n6(P3_enhanced)
        
        # Enhanced P4
        P3_downsample = self.bu_conv5(P3_enhanced)
        P4_enhanced = torch.cat([P3_downsample, P4], dim=1)
        P4_enhanced = self.C3_n7(P4_enhanced)
        
        # Apply ASFF for adaptive fusion
        P2_final = self.asff_p2(P2, P3, P4)
        P3_final = self.asff_p3(P2, P3_enhanced, P4)
        P4_final = self.asff_p4(P2, P3_enhanced, P4_enhanced)
        
        return [P2_final, P3_final, P4_final, P5]


if __name__ == "__main__":
    # Test ASFF module
    print("Testing ASFF Module...")
    asff = ASFF(level=1, channels=[256, 256, 256])
    x0 = torch.randn(2, 256, 80, 80)  # P2
    x1 = torch.randn(2, 256, 40, 40)  # P3
    x2 = torch.randn(2, 256, 20, 20)  # P4
    out = asff(x0, x1, x2)
    print(f"ASFF output shape: {out.shape}")
    
    # Test complete neck
    print("\nTesting ASFF Neck...")
    neck = ASFFNeck(in_channels_list=[128, 256, 512, 1024], out_channels=256)
    inputs = {
        'dark2': torch.randn(2, 128, 80, 80),
        'dark3': torch.randn(2, 256, 40, 40),
        'dark4': torch.randn(2, 512, 20, 20),
        'dark5': torch.randn(2, 1024, 10, 10),
    }
    outputs = neck(inputs)
    print("ASFF Neck output shapes:")
    for i, out in enumerate(outputs):
        print(f"P{i+2}: {out.shape}")
