"""
Data augmentation and preprocessing utilities
"""

import cv2
import numpy as np
import torch
import random
from PIL import Image


class TrainTransform:
    """
    Training data augmentation and preprocessing
    """
    
    def __init__(self, max_labels=100, flip_prob=0.5, hsv_prob=1.0):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob

    def __call__(self, image, target, input_dim):
        """
        Args:
            image: numpy array (H, W, 3) in BGR format
            target: dict with 'boxes' and 'labels'
            input_dim: int or tuple (height, width)
        """
        # Handle both int and tuple input_dim
        if isinstance(input_dim, int):
            input_dim = (input_dim, input_dim)
        
        boxes = target['boxes'].copy()
        labels = target['labels'].copy()
        
        # Get original size
        height, width = image.shape[:2]
        
        # Resize and letterbox
        image, r_scale = self.preprocess_image(image, input_dim)
        
        # Scale boxes
        boxes[:, 0::2] *= r_scale  # x coordinates
        boxes[:, 1::2] *= r_scale  # y coordinates
        
        # Random horizontal flip
        if random.random() < self.flip_prob:
            image = image[:, ::-1, :]
            boxes[:, [0, 2]] = input_dim[1] - boxes[:, [2, 0]]
        
        # HSV augmentation
        if random.random() < self.hsv_prob:
            image = self.augment_hsv(image)
        
        # Convert to tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        # Create padded target tensor
        labels_out = np.zeros((self.max_labels, 5))
        if len(boxes) > 0:
            # Convert to [class, cx, cy, w, h] format normalized by image size
            boxes_xyxy = boxes.copy()
            boxes_xywh = np.zeros_like(boxes)
            boxes_xywh[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2 / input_dim[1]  # cx
            boxes_xywh[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2 / input_dim[0]  # cy
            boxes_xywh[:, 2] = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / input_dim[1]      # w
            boxes_xywh[:, 3] = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / input_dim[0]      # h
            
            n_labels = min(len(boxes), self.max_labels)
            labels_out[:n_labels, 0] = labels[:n_labels]
            labels_out[:n_labels, 1:5] = boxes_xywh[:n_labels]
        
        target_out = {
            'labels': torch.from_numpy(labels_out).float(),
            'image_id': target['image_id'],
            'orig_size': target['orig_size']
        }
        
        return image, target_out
    
    def preprocess_image(self, image, input_size):
        """
        Resize image with letterbox padding
        
        Returns:
            padded_image: resized and padded image
            r_scale: resize ratio
        """
        h, w = image.shape[:2]
        
        # Handle both int and tuple input_size
        if isinstance(input_size, int):
            input_h, input_w = input_size, input_size
        else:
            input_h, input_w = input_size
        
        # Calculate resize ratio
        r = min(input_h / h, input_w / w)
        
        # Resize image
        resized_h, resized_w = int(h * r), int(w * r)
        resized_image = cv2.resize(
            image,
            (resized_w, resized_h),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.uint8)
        
        # Create padded image
        padded_image = np.ones((input_h, input_w, 3), dtype=np.uint8) * 114
        
        # Calculate padding
        pad_h = (input_h - resized_h) // 2
        pad_w = (input_w - resized_w) // 2
        
        # Place resized image in center
        padded_image[pad_h:pad_h + resized_h, pad_w:pad_w + resized_w] = resized_image
        
        return padded_image, r
    
    def augment_hsv(self, image, hgain=0.015, sgain=0.7, vgain=0.4):
        """
        Apply HSV color space augmentation
        """
        # Ensure image is contiguous
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
        
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        dtype = image.dtype
        
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        
        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        # Don't use dst parameter, just return the result
        image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        
        return image


class ValTransform:
    """
    Validation data preprocessing (no augmentation)
    """
    
    def __call__(self, image, target, input_dim):
        """
        Args:
            image: numpy array (H, W, 3) in BGR format
            target: dict with 'boxes' and 'labels'
            input_dim: int or tuple (height, width)
        """
        # Handle both int and tuple input_dim
        if isinstance(input_dim, int):
            input_dim = (input_dim, input_dim)
        
        boxes = target['boxes'].copy()
        labels = target['labels'].copy()
        
        # Resize and letterbox
        image, r_scale = self.preprocess_image(image, input_dim)
        
        # Scale boxes
        boxes[:, 0::2] *= r_scale  # x coordinates
        boxes[:, 1::2] *= r_scale  # y coordinates
        
        # Convert to tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        target_out = {
            'boxes': torch.from_numpy(boxes).float(),
            'labels': torch.from_numpy(labels).long(),
            'image_id': target['image_id'],
            'orig_size': target['orig_size']
        }
        
        return image, target_out
    
    def preprocess_image(self, image, input_size):
        """Resize image with letterbox padding"""
        h, w = image.shape[:2]
        
        # Handle both int and tuple input_size
        if isinstance(input_size, int):
            input_h, input_w = input_size, input_size
        else:
            input_h, input_w = input_size
        
        # Calculate resize ratio
        r = min(input_h / h, input_w / w)
        
        # Resize image
        resized_h, resized_w = int(h * r), int(w * r)
        resized_image = cv2.resize(
            image,
            (resized_w, resized_h),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.uint8)
        
        # Create padded image
        padded_image = np.ones((input_h, input_w, 3), dtype=np.uint8) * 114
        
        # Calculate padding
        pad_h = (input_h - resized_h) // 2
        pad_w = (input_w - resized_w) // 2
        
        # Place resized image in center
        padded_image[pad_h:pad_h + resized_h, pad_w:pad_w + resized_w] = resized_image
        
        return padded_image, r
