"""
Evaluation script for HE-YOLOX
"""

import os
import argparse
import yaml
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from models import build_he_yolox
from utils import VisDroneDataset, collate_fn, ValTransform, MeanAveragePrecision


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate HE-YOLOX')
    parser.add_argument('--config', type=str, default='configs/he_yolox_asff.yaml',
                        help='Path to config file')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Dataset directory (overrides config)')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


@torch.no_grad()
def evaluate(model, dataloader, device, num_classes):
    """Evaluate model on dataset"""
    model.eval()
    
    # Initialize metrics
    metric = MeanAveragePrecision(num_classes=num_classes)
    
    pbar = tqdm(dataloader, desc="Evaluating")
    
    for images, targets in pbar:
        images = images.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Process predictions
        # outputs shape: [batch_size, num_anchors, 5 + num_classes]
        # 5 = [x, y, w, h, objectness]
        
        for i, output in enumerate(outputs):
            # Extract predictions
            pred_boxes = output[:, :4]
            pred_scores = output[:, 4]
            pred_labels = output[:, 5:].argmax(dim=-1)
            
            # Filter by confidence threshold
            conf_mask = pred_scores > 0.01
            pred_boxes = pred_boxes[conf_mask]
            pred_scores = pred_scores[conf_mask]
            pred_labels = pred_labels[conf_mask]
            
            # Get ground truth
            target = targets[i]
            target_boxes = target['boxes']
            target_labels = target['labels']
            
            # Update metrics
            metric.update(
                pred_boxes, pred_scores, pred_labels,
                target_boxes, target_labels
            )
    
    # Compute metrics
    results = metric.compute()
    
    return results


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config
    if args.data_dir is not None:
        config['data']['data_dir'] = args.data_dir
    if args.device is not None:
        config['device'] = args.device
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    # Build dataloader
    print(f"Loading {args.split} dataset...")
    transform = ValTransform()
    dataset = VisDroneDataset(
        data_dir=config['data']['data_dir'],
        img_size=config['model']['input_size'][0],
        split=args.split,
        preproc=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['val']['batch_size'],
        shuffle=False,
        num_workers=config['val']['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Evaluate
    print("Starting evaluation...")
    results = evaluate(model, dataloader, device, config['model']['num_classes'])
    
    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print("=" * 50)
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
    
    # Save results
    results_path = os.path.join(config['output']['results_dir'], 'eval_results.txt')
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    
    with open(results_path, 'w') as f:
        f.write("Evaluation Results\n")
        f.write("=" * 50 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value:.4f}\n")
    
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
