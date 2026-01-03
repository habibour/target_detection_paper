# Quick Start Guide for HE-YOLOX

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

1. Download the VisDrone2019 dataset from: http://aiskyeye.com/download/object-detection-2/

2. Organize the dataset as follows:
```
data/VisDrone2019/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── annotations/
├── VisDrone2019-DET-val/
│   ├── images/
│   └── annotations/
└── VisDrone2019-DET-test-dev/
    └── images/
```

3. Update the `data_dir` in `configs/he_yolox_asff.yaml` to point to your dataset location.

## Training

### Train from scratch
```bash
python train.py --config configs/he_yolox_asff.yaml --data_dir ./data/VisDrone2019
```

### Resume training
```bash
python train.py --config configs/he_yolox_asff.yaml --resume checkpoints/epoch_50.pth
```

### Custom batch size and epochs
```bash
python train.py --config configs/he_yolox_asff.yaml --batch_size 16 --epochs 300
```

## Evaluation

```bash
python eval.py --config configs/he_yolox_asff.yaml --weights checkpoints/best.pth --split val
```

## Inference

### Single image
```bash
python inference.py --config configs/he_yolox_asff.yaml \
                   --weights checkpoints/best.pth \
                   --source image.jpg \
                   --save_img
```

### Directory of images
```bash
python inference.py --config configs/he_yolox_asff.yaml \
                   --weights checkpoints/best.pth \
                   --source ./test_images/ \
                   --save_img
```

### Custom parameters
```bash
python inference.py --config configs/he_yolox_asff.yaml \
                   --weights checkpoints/best.pth \
                   --source image.jpg \
                   --conf_thresh 0.5 \
                   --nms_thresh 0.65 \
                   --save_img
```

## Model Testing

Test the model architecture:
```bash
python models/he_yolox.py
```

Expected output:
```
Building HE-YOLOX-S model...

Training mode outputs (multi-scale):
  Scale 0: torch.Size([2, 18, 80, 80])
  Scale 1: torch.Size([2, 18, 40, 40])
  Scale 2: torch.Size([2, 18, 20, 20])
  Scale 3: torch.Size([2, 18, 10, 10])

Inference mode output: torch.Size([2, 8400, 18])

==================================================
Total Parameters: X.XXM
Trainable Parameters: X.XXM
```

## Expected Results (from paper)

### Average Precision (AP) by Class:
- Car: 81.2%
- Bus: 66.4%
- Truck: 55.6%
- Trucks: 47.0%
- Motor: 45.5%
- Pedestrian: 42.6%
- Tricycle: 29.8%
- Tricycle with canopy: 27.1%
- People: 30.6%
- Bicycle: 19.4%

### Performance Metrics:
- Best on large vehicles (cars, buses, trucks)
- FPS varies by target type (38-371 depending on complexity)
- Parameter count varies by class (4.9-65.1 based on paper Table 1)

## Monitoring Training

View training progress with TensorBoard:
```bash
tensorboard --logdir logs/
```

Then open http://localhost:6006 in your browser.

## Directory Structure After Training

```
implement/
├── checkpoints/          # Saved model weights
│   ├── best.pth
│   ├── epoch_10.pth
│   └── ...
├── logs/                # TensorBoard logs
├── results/             # Evaluation results
└── output/              # Inference output images
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in config or command line: `--batch_size 4`
- Reduce input image size: modify `input_size` in config

### Dataset Not Found
- Verify `data_dir` path in config file
- Check dataset structure matches expected format

### Slow Training
- Increase `num_workers` in config (CPU cores available)
- Enable mixed precision training (set `mixed_precision: true` in config)
- Use smaller model size: change `size` to 's' instead of 'm' or 'l'

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@inproceedings{liu2024target,
  title={Target Detection Algorithm for Drone Aerial Images based on Deep Learning},
  author={Liu, Tao and Zhang, Bohan},
  booktitle={2024 International Conference on Distributed Systems, Computer Networks and Cybersecurity (ICDSCNC)},
  pages={1--5},
  year={2024},
  organization={IEEE}
}
```
