"""
HE-YOLOX: Highly Efficient YOLOX with ASFF
Complete model architecture combining CSPDarknet backbone, ASFF neck, and detection head
"""

import torch
import torch.nn as nn
from models.backbone import build_cspdarknet
from models.neck import ASFFNeck
from models.head import DecoupledHead


class HEYOLOX(nn.Module):
    """
    HE-YOLOX-ASFF: Target Detection Algorithm for Drone Aerial Images
    
    Architecture:
    1. Backbone: CSPDarknet with residual connections
    2. Neck: ASFF (Adaptive Spatial Feature Fusion)
    3. Head: Decoupled detection head
    
    Features:
    - Multi-scale feature extraction with small target layer (P2)
    - Adaptive spatial feature fusion
    - Optimized for drone aerial image detection
    """
    
    def __init__(
        self,
        num_classes=13,  # VisDrone2019 has 13 classes
        depth=0.33,
        width=0.5,
        act="silu",
        model_size="s"
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Depth and width multipliers
        depth_mul = depth
        width_mul = width
        
        # Calculate channel sizes based on width multiplier
        base_channels = int(width_mul * 64)
        in_channels = [
            base_channels * 2,   # C2: dark2 output
            base_channels * 4,   # C3: dark3 output
            base_channels * 8,   # C4: dark4 output
            base_channels * 16   # C5: dark5 output
        ]
        
        # Backbone: CSPDarknet
        self.backbone = build_cspdarknet(model_size)
        
        # Neck: ASFF-based feature fusion
        out_channels = int(256 * width_mul)
        self.neck = ASFFNeck(
            in_channels_list=in_channels,
            out_channels=out_channels
        )
        
        # Head: Decoupled detection head for 4 scales (P2, P3, P4, P5)
        self.head = DecoupledHead(
            num_classes=num_classes,
            width=width_mul,
            in_channels=[out_channels] * 4,
            act=act
        )

    def forward(self, x):
        # Backbone forward pass
        backbone_out = self.backbone(x)
        
        # Neck forward pass (ASFF fusion)
        neck_out = self.neck(backbone_out)
        
        # Head forward pass (detection)
        outputs = self.head(neck_out)
        
        return outputs


def build_he_yolox(model_size="s", num_classes=13):
    """
    Build HE-YOLOX model with different sizes
    
    Args:
        model_size: Model size - 's' (small), 'm' (medium), 'l' (large), 'x' (extra large)
        num_classes: Number of classes (default 13 for VisDrone2019)
        
    Returns:
        HE-YOLOX model
    """
    size_config = {
        's': {'depth': 0.33, 'width': 0.5},
        'm': {'depth': 0.67, 'width': 0.75},
        'l': {'depth': 1.0, 'width': 1.0},
        'x': {'depth': 1.33, 'width': 1.25},
    }
    
    config = size_config.get(model_size.lower(), size_config['s'])
    
    return HEYOLOX(
        num_classes=num_classes,
        depth=config['depth'],
        width=config['width'],
        model_size=model_size
    )


def get_model_info(model, img_size=640):
    """Get model information including parameters and FLOPs"""
    from thop import profile
    
    stride = 64
    img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
    img = torch.zeros((1, 3, img_size[0], img_size[1]), device=next(model.parameters()).device)
    
    flops, params = profile(model, inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    
    print(f"Model Summary:")
    print(f"  Parameters: {params:.2f}M")
    print(f"  FLOPs: {flops:.2f}G")
    
    return params, flops


if __name__ == "__main__":
    # Test the complete model
    print("Building HE-YOLOX-S model...")
    model = build_he_yolox("s", num_classes=13)
    
    # Test forward pass
    x = torch.randn(2, 3, 640, 640)
    
    # Training mode
    model.train()
    outputs_train = model(x)
    print(f"\nTraining mode outputs (multi-scale):")
    for i, out in enumerate(outputs_train):
        print(f"  Scale {i}: {out.shape}")
    
    # Inference mode
    model.eval()
    with torch.no_grad():
        outputs_infer = model(x)
    print(f"\nInference mode output: {outputs_infer.shape}")
    
    # Model info
    print("\n" + "="*50)
    try:
        get_model_info(model, img_size=640)
    except:
        print("thop not installed. Skipping FLOPs calculation.")
        print("Install with: pip install thop")
    
    # Count parameters manually
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable Parameters: {trainable_params / 1e6:.2f}M")
