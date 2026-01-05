"""
Detection Head for HE-YOLOX
"""

import torch
import torch.nn as nn
from models.backbone import BaseConv


class DecoupledHead(nn.Module):
    """
    Decoupled detection head for YOLOX
    Separates classification and regression branches
    """
    
    def __init__(self, num_classes, width=1.0, in_channels=[256, 256, 256, 256], act="silu"):
        super().__init__()
        self.num_classes = num_classes
        self.decode_in_inference = True
        
        Conv = BaseConv
        
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        
        for i in range(len(in_channels)):
            # in_channels already width-adjusted from neck, don't multiply again
            self.stems.append(
                Conv(
                    in_channels=in_channels[i],
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

    def forward(self, xin):
        outputs = []
        
        for k, x in enumerate(xin):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x
            
            cls_feat = self.cls_convs[k](cls_x)
            cls_output = self.cls_preds[k](cls_feat)
            
            reg_feat = self.reg_convs[k](reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        
        if self.training:
            return outputs
        else:
            # Inference mode
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
            
            if self.decode_in_inference:
                return self.decode_outputs(outputs)
            else:
                return outputs

    def decode_outputs(self, outputs):
        """Decode model outputs to boxes"""
        grids = []
        strides = []
        
        for (hsize, wsize), stride in zip(self.hw, [8, 16, 32, 64]):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)], indexing='ij')
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(outputs.dtype).to(outputs.device)
        strides = torch.cat(strides, dim=1).type(outputs.dtype).to(outputs.device)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        
        return outputs
