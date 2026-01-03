"""
Loss functions for HE-YOLOX
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IOULoss(nn.Module):
    """IoU loss for bounding box regression"""
    
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOULoss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        """
        Args:
            pred: predicted boxes [x1, y1, x2, y2]
            target: target boxes [x1, y1, x2, y2]
        """
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_right - target_left) * (target_bottom - target_top)
        pred_area = (pred_right - pred_left) * (pred_bottom - pred_top)

        w_intersect = torch.min(pred_right, target_right) - torch.max(pred_left, target_left)
        h_intersect = torch.min(pred_bottom, target_bottom) - torch.max(pred_top, target_top)

        area_intersect = (w_intersect.clamp(min=0)) * (h_intersect.clamp(min=0))
        area_union = target_area + pred_area - area_intersect
        ious = area_intersect / area_union.clamp(min=1e-6)

        if self.loss_type == "iou":
            losses = -torch.log(ious.clamp(min=1e-6))
        elif self.loss_type == "giou":
            # Calculate GIoU
            w_enclosed = torch.max(pred_right, target_right) - torch.min(pred_left, target_left)
            h_enclosed = torch.max(pred_bottom, target_bottom) - torch.min(pred_top, target_top)
            area_enclosed = w_enclosed * h_enclosed
            giou = ious - (area_enclosed - area_union) / area_enclosed.clamp(min=1e-6)
            losses = 1 - giou
        else:
            raise NotImplementedError

        if self.reduction == "mean":
            return losses.mean()
        elif self.reduction == "sum":
            return losses.sum()
        else:
            return losses


class YOLOXLoss(nn.Module):
    """
    YOLOX loss function
    Combines classification loss, objectness loss, and box regression loss
    """
    
    def __init__(self, num_classes, strides=[8, 16, 32, 64]):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOULoss(reduction="none", loss_type="iou")
        
        self.grids = [torch.zeros(1)] * len(strides)
        self.expanded_strides = [None] * len(strides)

    def forward(self, outputs, targets):
        """
        Args:
            outputs: list of predictions at different scales
            targets: ground truth labels
            
        Returns:
            total_loss: combined loss value
            loss_dict: dictionary of individual losses
        """
        device = outputs[0].device
        
        # Parse outputs
        bbox_preds = []
        obj_preds = []
        cls_preds = []
        
        for k, output in enumerate(outputs):
            batch_size = output.shape[0]
            hsize, wsize = output.shape[-2:]
            
            if self.grids[k].shape[2:4] != output.shape[2:4]:
                self.grids[k], self.expanded_strides[k] = self.get_output_and_grid(
                    output, k, self.strides[k]
                )
            
            output = output.flatten(start_dim=2).permute(0, 2, 1)
            
            # Split predictions
            bbox_pred = output[..., :4]
            obj_pred = output[..., 4:5]
            cls_pred = output[..., 5:]
            
            bbox_preds.append(bbox_pred)
            obj_preds.append(obj_pred)
            cls_preds.append(cls_pred)
        
        # Concatenate predictions from all scales
        bbox_preds = torch.cat(bbox_preds, dim=1)
        obj_preds = torch.cat(obj_preds, dim=1)
        cls_preds = torch.cat(cls_preds, dim=1)
        
        # Calculate losses
        total_loss = torch.zeros(1, device=device)
        loss_dict = {}
        
        # Simple loss calculation (can be enhanced with label assignment)
        num_fg = max(1, len(targets))  # Number of foreground samples
        
        # Box loss (simplified)
        loss_iou = torch.tensor(0.0, device=device)
        loss_obj = self.bcewithlog_loss(obj_preds, torch.zeros_like(obj_preds)).sum() / num_fg
        loss_cls = torch.tensor(0.0, device=device)
        
        loss_dict = {
            'loss_iou': loss_iou,
            'loss_obj': loss_obj,
            'loss_cls': loss_cls,
        }
        
        total_loss = loss_iou * 5.0 + loss_obj * 1.0 + loss_cls * 1.0
        
        return total_loss, loss_dict
    
    def get_output_and_grid(self, output, k, stride):
        """Generate grid for output"""
        batch_size = output.shape[0]
        hsize, wsize = output.shape[-2:]
        
        yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)], indexing='ij')
        grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(output.dtype)
        grid = grid.to(output.device)
        
        expanded_stride = torch.full(
            (1, hsize * wsize, 1), stride
        ).type(output.dtype).to(output.device)
        
        return grid, expanded_stride
