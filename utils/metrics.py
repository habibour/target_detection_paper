"""
Evaluation metrics for object detection
"""

import numpy as np
import torch
from collections import defaultdict


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Calculate IoU between two bounding boxes
    
    Args:
        box1: [x1, y1, x2, y2] or [cx, cy, w, h]
        box2: [x1, y1, x2, y2] or [cx, cy, w, h]
        x1y1x2y2: if True, boxes are in [x1, y1, x2, y2] format
    
    Returns:
        iou: intersection over union
    """
    if not x1y1x2y2:
        # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Intersection area
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # Union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area + 1e-16

    iou = inter_area / union_area
    return iou


def ap_per_class(tp, conf, pred_cls, target_cls):
    """
    Calculate average precision per class
    
    Args:
        tp: true positives
        conf: confidence scores
        pred_cls: predicted classes
        target_cls: target classes
    
    Returns:
        ap: average precision for each class
        p: precision
        r: recall
    """
    # Sort by confidence
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]

    # Create precision-recall curve
    px, py = np.linspace(0, 1, 1000), []
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)

            # Precision
            precision = tpc / (tpc + fpc)
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 score
    f1 = 2 * p * r / (p + r + 1e-16)

    return ap, p, r, f1, unique_classes.astype(int)


def compute_ap(recall, precision):
    """
    Compute average precision given precision and recall curves
    
    Args:
        recall: recall curve
        precision: precision curve
    
    Returns:
        ap: average precision
        mpre: modified precision
        mrec: modified recall
    """
    # Append sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap, mpre, mrec


class MeanAveragePrecision:
    """
    Calculate mAP (mean Average Precision) for object detection
    """
    
    def __init__(self, num_classes, iou_threshold=0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """Reset metrics"""
        self.predictions = []
        self.targets = []
    
    def update(self, pred_boxes, pred_scores, pred_labels, target_boxes, target_labels):
        """
        Update with batch predictions
        
        Args:
            pred_boxes: predicted boxes [N, 4]
            pred_scores: confidence scores [N]
            pred_labels: predicted labels [N]
            target_boxes: ground truth boxes [M, 4]
            target_labels: ground truth labels [M]
        """
        self.predictions.append({
            'boxes': pred_boxes,
            'scores': pred_scores,
            'labels': pred_labels
        })
        
        self.targets.append({
            'boxes': target_boxes,
            'labels': target_labels
        })
    
    def compute(self):
        """Compute mAP"""
        # This is a simplified version
        # For full implementation, use pycocotools or torchmetrics
        
        ap_per_class = {}
        for cls in range(self.num_classes):
            cls_aps = []
            
            for pred, target in zip(self.predictions, self.targets):
                # Filter by class
                pred_mask = pred['labels'] == cls
                target_mask = target['labels'] == cls
                
                if pred_mask.sum() == 0 or target_mask.sum() == 0:
                    continue
                
                pred_boxes = pred['boxes'][pred_mask]
                pred_scores = pred['scores'][pred_mask]
                target_boxes = target['boxes'][target_mask]
                
                # Calculate IoU and count TPs
                if len(pred_boxes) > 0 and len(target_boxes) > 0:
                    # Simplified: just count how many predictions match targets
                    pass
            
        return {'mAP': 0.0}  # Placeholder


if __name__ == "__main__":
    # Test metrics
    print("Metrics module loaded successfully")
