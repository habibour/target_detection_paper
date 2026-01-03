"""
VisDrone2019 Dataset Loader
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class VisDroneDataset(Dataset):
    """
    VisDrone2019 Dataset for object detection
    
    Dataset contains 13 categories:
    0: ignored regions
    1: pedestrian
    2: people
    3: bicycle
    4: car
    5: van
    6: truck
    7: tricycle
    8: awning-tricycle
    9: bus
    10: motor
    11: others (ignored)
    12: tree (ignored)
    13: building (ignored)
    """
    
    CLASSES = [
        'ignored',      # 0
        'pedestrian',   # 1
        'people',       # 2
        'bicycle',      # 3
        'car',          # 4
        'van',          # 5
        'truck',        # 6
        'tricycle',     # 7
        'awning-tricycle',  # 8
        'bus',          # 9
        'motor',        # 10
        'others',       # 11 (ignored)
        'tree',         # 12 (ignored)
        'building'      # 13 (ignored)
    ]
    
    # Classes used for training (excluding ignored classes)
    TRAIN_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    def __init__(
        self,
        data_dir,
        img_size=640,
        split='train',
        preproc=None,
        cache_images=False
    ):
        """
        Args:
            data_dir: Root directory of VisDrone dataset
            img_size: Input image size
            split: 'train', 'val', or 'test'
            preproc: Preprocessing function
            cache_images: Cache images in memory
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split
        self.preproc = preproc
        
        # Image and annotation directories
        if split == 'train':
            self.img_dir = os.path.join(data_dir, 'VisDrone2019-DET-train', 'images')
            self.ann_dir = os.path.join(data_dir, 'VisDrone2019-DET-train', 'annotations')
        elif split == 'val':
            self.img_dir = os.path.join(data_dir, 'VisDrone2019-DET-val', 'images')
            self.ann_dir = os.path.join(data_dir, 'VisDrone2019-DET-val', 'annotations')
        else:  # test
            self.img_dir = os.path.join(data_dir, 'VisDrone2019-DET-test-dev', 'images')
            self.ann_dir = None  # Test set has no annotations
        
        # Get all image files
        self.img_files = sorted([
            os.path.join(self.img_dir, f)
            for f in os.listdir(self.img_dir)
            if f.endswith(('.jpg', '.png'))
        ])
        
        # Cache images if requested
        self.imgs = [None] * len(self.img_files)
        if cache_images:
            print(f"Caching images for {split} set...")
            for i, img_path in enumerate(self.img_files):
                self.imgs[i] = cv2.imread(img_path)
                if (i + 1) % 1000 == 0:
                    print(f"  Cached {i + 1}/{len(self.img_files)} images")
    
    def __len__(self):
        return len(self.img_files)
    
    def load_annotation(self, idx):
        """
        Load annotation for image at index
        
        VisDrone annotation format (per line):
        <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
        
        Returns:
            boxes: numpy array of shape (n, 4) in format [x1, y1, x2, y2]
            labels: numpy array of shape (n,) with class labels
        """
        img_path = self.img_files[idx]
        img_name = os.path.basename(img_path).replace('.jpg', '.txt').replace('.png', '.txt')
        ann_path = os.path.join(self.ann_dir, img_name)
        
        boxes = []
        labels = []
        
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 6:
                        continue
                    
                    x, y, w, h = map(float, parts[:4])
                    score = int(parts[4])  # confidence score (0 or 1)
                    category = int(parts[5])
                    
                    # Skip ignored regions and low-confidence detections
                    if category in [0, 11] or score == 0:
                        continue
                    
                    # Skip tree and building (background objects)
                    if category in [12, 13]:
                        continue
                    
                    # Convert to [x1, y1, x2, y2] format
                    x1, y1 = x, y
                    x2, y2 = x + w, y + h
                    
                    # Skip invalid boxes
                    if w <= 0 or h <= 0:
                        continue
                    
                    boxes.append([x1, y1, x2, y2])
                    # Remap category to 0-based index
                    labels.append(category - 1)  # Subtract 1 to make 0-based
        
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
        
        return boxes, labels
    
    def load_image(self, idx):
        """Load image at index"""
        if self.imgs[idx] is not None:
            img = self.imgs[idx]
        else:
            img = cv2.imread(self.img_files[idx])
        
        return img
    
    def __getitem__(self, idx):
        """Get item at index"""
        img = self.load_image(idx)
        
        if self.ann_dir is not None:
            boxes, labels = self.load_annotation(idx)
            
            # Create target dict
            target = {
                'boxes': boxes,
                'labels': labels,
                'image_id': idx,
                'orig_size': img.shape[:2]
            }
        else:
            target = None
        
        # Apply preprocessing
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.img_size)
        
        return img, target


def collate_fn(batch):
    """Custom collate function for dataloader"""
    imgs, targets = zip(*batch)
    
    # Stack images
    imgs = torch.stack(imgs, 0)
    
    # Keep targets as list
    return imgs, list(targets)


if __name__ == "__main__":
    # Test dataset loading
    data_dir = "/path/to/VisDrone2019"
    
    if os.path.exists(data_dir):
        dataset = VisDroneDataset(data_dir, split='train')
        print(f"Dataset size: {len(dataset)}")
        
        # Test loading first image
        img, target = dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Number of objects: {len(target['boxes'])}")
        print(f"Classes: {target['labels']}")
    else:
        print(f"Dataset directory not found: {data_dir}")
        print("Please download VisDrone2019 dataset from:")
        print("http://aiskyeye.com/download/object-detection-2/")
