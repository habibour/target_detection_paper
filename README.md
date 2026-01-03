# HE-YOLOX-ASFF: Highly Efficient YOLOX with Adaptive Spatial Feature Fusion

Implementation of "Target Detection Algorithm for Drone Aerial Images based on Deep Learning"

## Overview

This is a PyTorch implementation of the HE-YOLOX-ASFF algorithm for target detection in drone aerial images. The algorithm improves upon YOLOX-S with:
- CSPDarknet backbone with residual connections
- Multi-scale feature extraction with additional small target layer (P2)
- ASFF (Adaptive Spatial Feature Fusion) module replacing traditional PANet
- Optimized for VisDrone2019 dataset

## Key Features

- **Backbone**: CSPDarknet with residual blocks and SiLU activation
- **Neck**: ASFF module for adaptive multi-scale feature fusion
- **Detection Head**: Enhanced for small object detection
- **Performance**: Optimized for cars (81.2% AP), buses (66.4% AP), trucks (55.6% AP)

## Dataset

The implementation uses the VisDrone2019 dataset with 10,209 high-altitude drone images:
- 6,471 training images
- 548 validation images
- 3,190 test images
- 13 object categories

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python train.py --config configs/he_yolox_asff.yaml
```

### Evaluation
```bash
python eval.py --config configs/he_yolox_asff.yaml --weights checkpoints/best.pth
```

### Inference
```bash
python inference.py --image path/to/image.jpg --weights checkpoints/best.pth
```

## Project Structure

```
├── models/
│   ├── backbone.py        # CSPDarknet backbone
│   ├── neck.py           # ASFF module
│   ├── head.py           # Detection head
│   └── he_yolox.py       # Complete model
├── utils/
│   ├── dataset.py        # VisDrone dataset loader
│   ├── augmentation.py   # Data augmentation
│   ├── loss.py           # Loss functions
│   └── metrics.py        # Evaluation metrics
├── configs/
│   └── he_yolox_asff.yaml # Model configuration
├── train.py              # Training script
├── eval.py               # Evaluation script
└── inference.py          # Inference script
```

## Citation

```
@inproceedings{liu2024target,
  title={Target Detection Algorithm for Drone Aerial Images based on Deep Learning},
  author={Liu, Tao and Zhang, Bohan},
  booktitle={2024 International Conference on Distributed Systems, Computer Networks and Cybersecurity (ICDSCNC)},
  pages={1--5},
  year={2024},
  organization={IEEE}
}
```
