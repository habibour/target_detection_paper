# Setup and Installation Guide

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- At least 16GB RAM
- 50GB+ free disk space for dataset

## Step-by-Step Installation

### 1. Install Python Dependencies

First, install PyTorch with CUDA support (for GPU training):

```bash
# For CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only (not recommended for training)
pip3 install torch torchvision torchaudio
```

Then install other dependencies:

```bash
pip3 install -r requirements.txt
```

### 2. Verify Installation

Test if PyTorch is installed correctly:

```bash
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch version: 2.x.x
CUDA available: True  # (or False if using CPU)
```

### 3. Download Dataset

**Option A: Automatic Download (requires wget)**
```bash
chmod +x download_dataset.sh
./download_dataset.sh
```

**Option B: Manual Download**
1. Visit: http://aiskyeye.com/download/object-detection-2/
2. Download these files:
   - VisDrone2019-DET-train.zip
   - VisDrone2019-DET-val.zip
   - VisDrone2019-DET-test-dev.zip
3. Extract them into `data/VisDrone2019/` directory

### 4. Verify Dataset Structure

```bash
ls -R data/VisDrone2019/
```

Expected structure:
```
data/VisDrone2019/
├── VisDrone2019-DET-train/
│   ├── images/          # 6,471 images
│   └── annotations/     # 6,471 annotation files
├── VisDrone2019-DET-val/
│   ├── images/          # 548 images
│   └── annotations/     # 548 annotation files
└── VisDrone2019-DET-test-dev/
    └── images/          # 3,190 images
```

### 5. Test Model Architecture

```bash
python3 models/he_yolox.py
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

## Quick Start Training

### Minimal Training Command

```bash
python3 train.py --config configs/he_yolox_asff.yaml \
                 --data_dir ./data/VisDrone2019 \
                 --batch_size 8 \
                 --epochs 300
```

### Training with Custom Settings

```bash
python3 train.py --config configs/he_yolox_asff.yaml \
                 --data_dir ./data/VisDrone2019 \
                 --batch_size 16 \
                 --epochs 300 \
                 --device cuda
```

### Resume Training

```bash
python3 train.py --config configs/he_yolox_asff.yaml \
                 --resume checkpoints/epoch_50.pth
```

## Monitor Training Progress

Start TensorBoard:

```bash
tensorboard --logdir logs/
```

Then open http://localhost:6006 in your browser.

## Common Issues and Solutions

### Issue 1: CUDA Out of Memory

**Solution:**
- Reduce batch size: `--batch_size 4` or `--batch_size 2`
- Modify in config file: `train.batch_size: 4`

### Issue 2: Dataset Not Found

**Error:** `FileNotFoundError: [Errno 2] No such file or directory: './data/VisDrone2019'`

**Solution:**
- Check dataset path in config: `configs/he_yolox_asff.yaml`
- Update `data.data_dir` to correct path
- Or use `--data_dir` flag: `--data_dir /path/to/dataset`

### Issue 3: Python Module Not Found

**Error:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
# Install PyTorch
pip3 install torch torchvision

# Verify installation
python3 -c "import torch; print(torch.__version__)"
```

### Issue 4: Slow Training

**Solutions:**
1. Increase number of workers:
   - Edit `configs/he_yolox_asff.yaml`
   - Set `train.num_workers: 8` (based on CPU cores)

2. Enable mixed precision:
   - Set `mixed_precision: true` in config

3. Use smaller model:
   - Change `model.size: "s"` (small) instead of "m" or "l"

### Issue 5: Permission Denied (download_dataset.sh)

**Solution:**
```bash
chmod +x download_dataset.sh
./download_dataset.sh
```

## Hardware Requirements

### Minimum Requirements (for testing):
- CPU: 4 cores
- RAM: 8GB
- GPU: Not required (CPU mode)
- Storage: 20GB

### Recommended for Training:
- CPU: 8+ cores
- RAM: 16GB+
- GPU: NVIDIA GPU with 8GB+ VRAM (GTX 1080, RTX 2080, etc.)
- Storage: 50GB+

### Optimal Setup:
- CPU: 16+ cores
- RAM: 32GB+
- GPU: NVIDIA GPU with 11GB+ VRAM (RTX 3080, RTX 4090, etc.)
- Storage: 100GB+ SSD

## Expected Training Time

On different hardware:

- **RTX 3090 (24GB):** ~12 hours for 300 epochs
- **RTX 3080 (10GB):** ~18 hours for 300 epochs
- **GTX 1080 Ti (11GB):** ~24 hours for 300 epochs
- **CPU only:** Not recommended (would take days)

## Next Steps

After successful installation:

1. **Read documentation:**
   - `README.md` - Project overview
   - `IMPLEMENTATION_SUMMARY.md` - Technical details
   - `QUICKSTART.md` - Usage examples

2. **Start training:**
   ```bash
   python3 train.py --config configs/he_yolox_asff.yaml
   ```

3. **Monitor progress:**
   - Check terminal output for loss values
   - Use TensorBoard for detailed metrics

4. **Evaluate model:**
   ```bash
   python3 eval.py --weights checkpoints/best.pth
   ```

5. **Run inference:**
   ```bash
   python3 inference.py --weights checkpoints/best.pth --source test.jpg
   ```

## Support and Troubleshooting

If you encounter any issues:

1. Check Python version: `python3 --version` (should be 3.8+)
2. Check PyTorch version: `python3 -c "import torch; print(torch.__version__)"`
3. Verify CUDA: `python3 -c "import torch; print(torch.cuda.is_available())"`
4. Check disk space: `df -h`
5. Monitor GPU usage: `nvidia-smi` (if using GPU)

## Package Versions (requirements.txt)

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
opencv-python>=4.8.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
tqdm>=4.65.0
tensorboard>=2.13.0
pyyaml>=6.0
pillow>=10.0.0
albumentations>=1.3.0
pycocotools>=2.0.6
thop>=0.1.1
```

---

**Happy Training! 🚀**

For questions or issues, refer to:
- Paper: DOI 10.1109/ICDSCNC62492.2024.10939462
- Implementation: Check IMPLEMENTATION_SUMMARY.md
