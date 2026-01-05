"""
HE-YOLOX: Highly Efficient YOLOX with ASFF
Complete model architecture combining CSPDarknet backbone, ASFF neck, and detection head
"""

import torch
import torch.nn as nn
import os
from models.backbone import build_cspdarknet
from models.neck import ASFFNeck
from models.head import DecoupledHead


# YOLOX pretrained weights URLs (from MEGVII)
PRETRAINED_URLS = {
    's': 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth',
    'm': 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth',
    'l': 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth',
    'x': 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth',
}


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


def build_he_yolox(model_size="s", num_classes=13, pretrained=True):
    """
    Build HE-YOLOX model with different sizes
    
    Args:
        model_size: Model size - 's' (small), 'm' (medium), 'l' (large), 'x' (extra large)
        num_classes: Number of classes (default 13 for VisDrone2019)
        pretrained: If True, load pretrained YOLOX backbone weights (COCO pretrained)
        
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
    
    model = HEYOLOX(
        num_classes=num_classes,
        depth=config['depth'],
        width=config['width'],
        model_size=model_size
    )
    
    # Load pretrained weights
    if pretrained:
        model = load_pretrained_weights(model, model_size)
    
    return model


def load_pretrained_weights(model, model_size="s"):
    """
    Load pretrained YOLOX backbone weights
    
    This loads the official MEGVII YOLOX weights trained on COCO dataset
    and transfers the backbone weights to our model.
    """
    import torch.hub
    
    url = PRETRAINED_URLS.get(model_size.lower())
    if url is None:
        print(f"No pretrained weights for size '{model_size}', training from scratch")
        return model
    
    print(f"Loading pretrained YOLOX-{model_size.upper()} weights...")
    
    # Download weights
    cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
    os.makedirs(cache_dir, exist_ok=True)
    weight_file = os.path.join(cache_dir, f"yolox_{model_size}.pth")
    
    if not os.path.exists(weight_file):
        print(f"Downloading pretrained weights from {url}")
        torch.hub.download_url_to_file(url, weight_file)
    
    # Load checkpoint
    ckpt = torch.load(weight_file, map_location="cpu")
    if "model" in ckpt:
        pretrained_dict = ckpt["model"]
    else:
        pretrained_dict = ckpt
    
    # Debug: print some keys from pretrained model
    pretrained_keys = list(pretrained_dict.keys())
    print(f"  Pretrained model has {len(pretrained_keys)} keys")
    print(f"  Sample keys: {pretrained_keys[:5]}")
    
    # Map YOLOX backbone keys to our backbone
    model_dict = model.state_dict()
    our_keys = list(model_dict.keys())
    print(f"  Our model has {len(our_keys)} keys")
    print(f"  Sample keys: {our_keys[:5]}")
    
    # Key mapping: YOLOX uses 'backbone.backbone.X' but ours uses 'backbone.X'
    loaded_keys = []
    for key, value in pretrained_dict.items():
        new_key = key
        
        # Fix backbone key mapping: 'backbone.backbone.X' -> 'backbone.X'
        if key.startswith('backbone.backbone.'):
            new_key = key.replace('backbone.backbone.', 'backbone.')
        
        # Skip head.cls_preds (different num_classes: 80 vs 13)
        if 'cls_preds' in new_key:
            continue
            
        if new_key in model_dict:
            if model_dict[new_key].shape == value.shape:
                model_dict[new_key] = value
                loaded_keys.append(new_key)
            else:
                print(f"  Shape mismatch: {new_key} - pretrained {value.shape} vs ours {model_dict[new_key].shape}")
    
    model.load_state_dict(model_dict)
    print(f"✅ Loaded {len(loaded_keys)} pretrained layers")
    
    # Show what was loaded
    backbone_loaded = sum(1 for k in loaded_keys if 'backbone' in k)
    neck_loaded = sum(1 for k in loaded_keys if 'neck' in k)
    head_loaded = sum(1 for k in loaded_keys if 'head' in k)
    print(f"   Backbone: {backbone_loaded} layers")
    print(f"   Neck: {neck_loaded} layers") 
    print(f"   Head: {head_loaded} layers (excluding cls_preds)")
    
    return model


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
