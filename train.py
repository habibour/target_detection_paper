"""
Training script for HE-YOLOX
"""

import os
import sys
import yaml
import argparse
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import build_he_yolox
from utils import VisDroneDataset, collate_fn, TrainTransform, ValTransform, YOLOXLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Train HE-YOLOX')
    parser.add_argument('--config', type=str, default='configs/he_yolox_asff.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Dataset directory (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_dataloader(config, split='train'):
    """Build dataloader for training or validation"""
    data_dir = config['data']['data_dir']
    img_size = config['model']['input_size'][0]
    
    if split == 'train':
        transform = TrainTransform(
            flip_prob=config['train']['augmentation']['flip_prob'],
            hsv_prob=config['train']['augmentation']['hsv_prob']
        )
        batch_size = config['train']['batch_size']
        num_workers = config['train']['num_workers']
        shuffle = True
    else:
        transform = ValTransform()
        batch_size = config['val']['batch_size']
        num_workers = config['val']['num_workers']
        shuffle = False
    
    dataset = VisDroneDataset(
        data_dir=data_dir,
        img_size=img_size,
        split=split,
        preproc=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return dataloader


def build_optimizer(config, model):
    """Build optimizer"""
    optimizer_name = config['train']['optimizer']
    lr = config['train']['lr']
    weight_decay = config['train']['weight_decay']
    
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config['train']['momentum'],
            weight_decay=weight_decay,
            nesterov=True
        )
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def build_scheduler(config, optimizer, total_steps):
    """Build learning rate scheduler"""
    scheduler_name = config['train']['scheduler']
    
    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=config['train']['lr'] * config['train']['min_lr_ratio']
        )
    elif scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=total_steps // 3,
            gamma=0.1
        )
    else:
        scheduler = None
    
    return scheduler


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, writer, config):
    """Train for one epoch"""
    model.train()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    total_loss = 0.0
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss, loss_dict = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        total_loss += loss.item()
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
        })
        
        # Log to tensorboard
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('train/loss', loss.item(), global_step)
            for key, value in loss_dict.items():
                writer.add_scalar(f'train/{key}', value.item(), global_step)
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, epoch, writer):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Validation")
        
        for images, targets in pbar:
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss, loss_dict = criterion(outputs, targets)
            
            total_loss += loss.item()
            pbar.set_postfix({'val_loss': f"{total_loss / (len(pbar)):.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    
    if writer is not None:
        writer.add_scalar('val/loss', avg_loss, epoch)
    
    return avg_loss


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_dir is not None:
        config['data']['data_dir'] = args.data_dir
    if args.batch_size is not None:
        config['train']['batch_size'] = args.batch_size
        config['val']['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['train']['epochs'] = args.epochs
    if args.device is not None:
        config['device'] = args.device
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(config['output']['model_dir'], exist_ok=True)
    os.makedirs(config['output']['log_dir'], exist_ok=True)
    
    # Build model
    print("Building model...")
    model = build_he_yolox(
        model_size=config['model']['size'],
        num_classes=config['model']['num_classes']
    )
    model = model.to(device)
    
    # Build dataloaders
    print("Building dataloaders...")
    train_loader = build_dataloader(config, split='train')
    val_loader = build_dataloader(config, split='val')
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(config, model)
    total_steps = config['train']['epochs'] * len(train_loader)
    scheduler = build_scheduler(config, optimizer, total_steps)
    
    # Build loss criterion
    criterion = YOLOXLoss(num_classes=config['model']['num_classes'])
    
    # Tensorboard writer
    writer = SummaryWriter(log_dir=config['output']['log_dir'])
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
        # Restore scheduler state if available
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        elif scheduler is not None:
            # Manually step scheduler to correct position
            for _ in range(start_epoch * len(train_loader)):
                scheduler.step()
        
        # Restore best validation loss if available
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
        
        print(f"Resumed from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")
    
    # Training loop
    print("Starting training...")
    
    for epoch in range(start_epoch, config['train']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['train']['epochs']}")
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, writer, config
        )
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Validate
        if (epoch + 1) % config['train']['eval_interval'] == 0:
            val_loss = validate(model, val_loader, criterion, device, epoch, writer)
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(config['output']['model_dir'], 'best.pth')
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                }
                if scheduler is not None:
                    save_dict['scheduler_state_dict'] = scheduler.state_dict()
                torch.save(save_dict, save_path)
                print(f"Saved best model to {save_path}")
        
        # Save checkpoint
        if (epoch + 1) % config['train']['save_interval'] == 0:
            save_path = os.path.join(config['output']['model_dir'], f'epoch_{epoch + 1}.pth')
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'best_val_loss': best_val_loss,
            }
            if scheduler is not None:
                save_dict['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(save_dict, save_path)
            print(f"Saved checkpoint to {save_path}")
    
    print("Training completed!")
    writer.close()


if __name__ == '__main__':
    main()
