"""
Inference script for HE-YOLOX
Run detection on images or videos
"""

import os
import argparse
import yaml
import cv2
import torch
import numpy as np
from pathlib import Path

from models import build_he_yolox
from utils import ValTransform


# VisDrone class names
CLASS_NAMES = [
    'ignored', 'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor',
    'others', 'tree', 'building'
]

# Colors for visualization (BGR format)
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (64, 0, 0), (0, 64, 0)
]


def parse_args():
    parser = argparse.ArgumentParser(description='HE-YOLOX Inference')
    parser.add_argument('--config', type=str, default='configs/he_yolox_asff.yaml',
                        help='Path to config file')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--source', type=str, required=True,
                        help='Image file, directory, or video')
    parser.add_argument('--output', type=str, default='./output',
                        help='Output directory')
    parser.add_argument('--conf_thresh', type=float, default=0.3,
                        help='Confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.65,
                        help='NMS threshold')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--save_img', action='store_true',
                        help='Save detection results')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def preprocess_image(image, input_size):
    """Preprocess image for inference"""
    h, w = image.shape[:2]
    
    # Resize with letterbox
    r = min(input_size[0] / h, input_size[1] / w)
    new_h, new_w = int(h * r), int(w * r)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image
    padded = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    pad_h = (input_size[0] - new_h) // 2
    pad_w = (input_size[1] - new_w) // 2
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    
    # Convert to tensor
    image_tensor = torch.from_numpy(padded.transpose(2, 0, 1)).float()
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, r, (pad_h, pad_w)


def postprocess(outputs, conf_thresh=0.3, nms_thresh=0.65):
    """Post-process model outputs"""
    # outputs shape: [batch, num_anchors, 5 + num_classes]
    
    batch_detections = []
    
    for output in outputs:
        # Extract boxes, scores, and classes
        boxes = output[:, :4]
        obj_conf = output[:, 4]
        cls_conf = output[:, 5:].max(dim=-1)[0]
        cls_pred = output[:, 5:].argmax(dim=-1)
        
        # Combined confidence
        conf = obj_conf * cls_conf
        
        # Filter by confidence threshold
        mask = conf > conf_thresh
        boxes = boxes[mask]
        conf = conf[mask]
        cls_pred = cls_pred[mask]
        
        if len(boxes) == 0:
            batch_detections.append(None)
            continue
        
        # NMS
        keep_idx = nms(boxes, conf, nms_thresh)
        boxes = boxes[keep_idx]
        conf = conf[keep_idx]
        cls_pred = cls_pred[keep_idx]
        
        detections = torch.cat([boxes, conf.unsqueeze(1), cls_pred.unsqueeze(1).float()], dim=1)
        batch_detections.append(detections)
    
    return batch_detections


def nms(boxes, scores, threshold):
    """Non-Maximum Suppression"""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)
    
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        
        i = order[0].item()
        keep.append(i)
        
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        idx = (iou <= threshold).nonzero().squeeze()
        if idx.numel() == 0:
            break
        order = order[idx + 1]
    
    return torch.LongTensor(keep)


def visualize(image, detections, ratio, padding):
    """Visualize detections on image"""
    if detections is None:
        return image
    
    pad_h, pad_w = padding
    
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        
        # Convert back to original coordinates
        x1 = (x1 - pad_w) / ratio
        y1 = (y1 - pad_h) / ratio
        x2 = (x2 - pad_w) / ratio
        y2 = (y2 - pad_h) / ratio
        
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls = int(cls)
        
        # Draw box
        color = COLORS[cls % len(COLORS)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{CLASS_NAMES[cls]}: {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image


@torch.no_grad()
def detect_image(model, image_path, args, device):
    """Run detection on a single image"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return
    
    # Preprocess
    img_tensor, ratio, padding = preprocess_image(image, (args.img_size, args.img_size))
    img_tensor = img_tensor.to(device)
    
    # Inference
    outputs = model(img_tensor)
    
    # Post-process
    detections = postprocess(outputs, args.conf_thresh, args.nms_thresh)[0]
    
    # Visualize
    if detections is not None:
        image = visualize(image, detections.cpu(), ratio, padding)
        print(f"Detected {len(detections)} objects in {image_path}")
    else:
        print(f"No objects detected in {image_path}")
    
    # Save result
    if args.save_img:
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(args.output, Path(image_path).name)
        cv2.imwrite(output_path, image)
        print(f"Saved result to {output_path}")
    
    return image


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    config = load_config(args.config)
    
    # Build model
    print("Building model...")
    model = build_he_yolox(
        model_size=config['model']['size'],
        num_classes=config['model']['num_classes']
    )
    
    # Load weights
    print(f"Loading weights from: {args.weights}")
    checkpoint = torch.load(args.weights, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Run inference
    source = Path(args.source)
    
    if source.is_file():
        # Single image
        if source.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            print(f"Processing image: {source}")
            detect_image(model, str(source), args, device)
        else:
            print(f"Unsupported file format: {source.suffix}")
    
    elif source.is_dir():
        # Directory of images
        image_files = list(source.glob('*.jpg')) + list(source.glob('*.png'))
        print(f"Found {len(image_files)} images")
        
        for img_path in image_files:
            detect_image(model, str(img_path), args, device)
    
    else:
        print(f"Source not found: {source}")
    
    print("Inference completed!")


if __name__ == '__main__':
    main()
