# 🚀 Google Colab Training Instructions

## Quick Setup Guide

### Step 1: Prepare Your Files

1. **Zip the project folder:**
   ```bash
   cd /Users/habibourakash/CSERUET/4-2/CSE4200-theisis
   zip -r implement.zip implement/ -x "*.pyc" "*__pycache__*" "*.DS_Store" "*/data/*"
   ```

2. The ZIP should contain:
   - `models/` folder (all .py files)
   - `utils/` folder (all .py files)
   - `configs/` folder (YAML config)
   - `train.py`, `eval.py`, `inference.py`
   - `requirements.txt`

### Step 2: Upload to Google Colab

1. **Open the notebook:**
   - Go to: https://colab.research.google.com/
   - Click "Upload" → Select `HE_YOLOX_Training_Colab.ipynb`

2. **Set GPU runtime:**
   - Click "Runtime" → "Change runtime type"
   - Hardware accelerator: **GPU (T4)**
   - Click "Save"

3. **Run the cells in order:**
   - Cell 1: Check GPU ✅
   - Cell 2: Install dependencies ✅
   - Cell 3: Mount Google Drive ✅
   - Cell 4: Upload your `implement.zip` ✅
   - Cell 5: Download VisDrone2019 dataset (~2GB, takes 10-15 min) ✅
   - Cell 6: Update config ✅
   - Cell 7: Test model ✅
   - Cell 8: **START TRAINING** 🚀

### Step 3: Training

**Training will take approximately 12-18 hours on T4 GPU**

**Important:**
- ✅ Checkpoints save automatically to Google Drive every 10 epochs
- ✅ Best model saves when validation improves
- ✅ Can close browser - training continues
- ✅ If session times out (12h limit), just re-run training cell to resume

**Monitor progress:**
- Use TensorBoard cell to see live metrics
- Check GPU usage with `!nvidia-smi`
- Look at checkpoint files in Google Drive

### Step 4: Handle Session Timeout

**When Colab times out after 12 hours:**

1. Reconnect and run:
   ```python
   # Mount drive again
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Upload code again (or extract ZIP)
   # ... code upload cell ...
   
   # Training will auto-resume from latest checkpoint
   !python train.py --config configs/he_yolox_asff_colab.yaml --device cuda
   ```

2. Training automatically finds latest checkpoint and resumes

3. Repeat until 300 epochs complete

### Step 5: Evaluation & Results

After training completes:

1. **Evaluate model:**
   - Run evaluation cell
   - Results save to Google Drive

2. **Test inference:**
   - Run inference cells
   - View detection visualizations

3. **Download results:**
   - Run download cell
   - Get ZIP with all checkpoints, logs, results

## 📊 What to Expect

### During Training:

```
Epoch 1/300
  loss: 15.2341, avg_loss: 15.2341
  ...

Epoch 10/300
  loss: 8.4521, avg_loss: 8.4521
  Validation Loss: 7.8934
  ✅ Saved best model

...

Epoch 300/300
  Training completed!
```

### Expected Results:

| Metric | Expected | Your Results |
|--------|----------|--------------|
| Car AP | 81.2% | ___ |
| Bus AP | 66.4% | ___ |
| Truck AP | 55.6% | ___ |
| Pedestrian AP | 42.6% | ___ |
| Overall mAP | ~50-55% | ___ |

### Training Time Breakdown:

- **Epochs 1-50:** ~2-3 hours (model learning basic features)
- **Epochs 51-150:** ~5-6 hours (model refining)
- **Epochs 151-300:** ~6-8 hours (final tuning)
- **Total:** ~12-18 hours

## 💡 Tips & Tricks

### Optimize Training Speed:

1. **Increase batch size** (if GPU has memory):
   ```python
   config['train']['batch_size'] = 24  # Instead of 16
   ```

2. **Use mixed precision** (already enabled):
   ```python
   config['mixed_precision'] = True  # Faster, uses less memory
   ```

3. **Reduce validation frequency**:
   ```python
   config['train']['eval_interval'] = 20  # Instead of 10
   ```

### Save Google Drive Space:

```python
# Only keep best model and last 5 checkpoints
config['train']['save_interval'] = 50  # Save less frequently
```

### Debug Issues:

If training fails:

1. **Check GPU:**
   ```python
   !nvidia-smi
   import torch
   print(torch.cuda.is_available())
   ```

2. **Check dataset:**
   ```bash
   !ls data/VisDrone2019/VisDrone2019-DET-train/images | wc -l
   # Should show: 6471
   ```

3. **Reduce batch size:**
   ```python
   config['train']['batch_size'] = 8  # If OOM error
   ```

4. **Check logs:**
   ```python
   !tail -50 /content/drive/MyDrive/HE_YOLOX/logs/events.out.tfevents.*
   ```

## 🎯 After Training

### 1. Compare with Paper:

Create a comparison table:

```python
paper_results = {
    'car': 81.2, 'bus': 66.4, 'truck': 55.6,
    'pedestrian': 42.6, 'motor': 45.5, 'bicycle': 19.4
}

your_results = {
    # Fill in from eval_results.txt
    'car': ____, 'bus': ____, ...
}

for cls in paper_results:
    diff = your_results[cls] - paper_results[cls]
    print(f"{cls}: {your_results[cls]:.1f}% (paper: {paper_results[cls]:.1f}%, diff: {diff:+.1f}%)")
```

### 2. Download Everything:

Your Google Drive folder structure:
```
/MyDrive/HE_YOLOX/
├── checkpoints/
│   ├── best.pth          ← Best model (smallest val loss)
│   ├── epoch_10.pth
│   ├── epoch_20.pth
│   └── ...
├── logs/
│   └── events.out.tfevents.*  ← TensorBoard logs
└── results/
    ├── eval_results.txt   ← mAP scores
    └── inference/         ← Detection images
```

### 3. Use Trained Model Locally:

Download `best.pth` from Google Drive, then:

```bash
# On your Mac
python3 inference.py \
    --weights best.pth \
    --source test_image.jpg \
    --device cpu \
    --save_img
```

## 🆘 Troubleshooting

### Problem: "No GPU available"
**Solution:** Runtime → Change runtime type → GPU (T4)

### Problem: "Session timeout"
**Solution:** Re-run cells, training auto-resumes from checkpoint

### Problem: "CUDA out of memory"
**Solution:** Reduce batch size in config cell:
```python
config['train']['batch_size'] = 8  # or even 4
```

### Problem: "Dataset not found"
**Solution:** Re-run dataset download cell

### Problem: "Training too slow"
**Solution:** 
- Check you're using GPU: `!nvidia-smi`
- Reduce image size in config (not recommended)
- Use Colab Pro for better GPU

### Problem: "Google Drive storage full"
**Solution:**
- Each checkpoint is ~100MB
- Keep only best + last 3 checkpoints
- Delete old runs

## ✅ Checklist

Before starting training:
- [ ] GPU is enabled (T4 or better)
- [ ] Google Drive is mounted
- [ ] Project files uploaded
- [ ] Dataset downloaded (6471 train + 548 val images)
- [ ] Config updated for Colab
- [ ] Model test passed
- [ ] Enough Drive space (~5-10GB for full training)

During training:
- [ ] Monitor TensorBoard
- [ ] Check checkpoints saving to Drive
- [ ] Watch for best model updates
- [ ] Note session time (reset before 12h)

After training:
- [ ] Evaluate on validation set
- [ ] Compare with paper results
- [ ] Test inference on sample images
- [ ] Download trained model
- [ ] Save results and logs

## 📞 Support

If you encounter issues:

1. Check GPU: `!nvidia-smi`
2. Check files: `!ls -la models/`
3. Check dataset: `!ls data/VisDrone2019/`
4. Check logs: `!cat /content/drive/MyDrive/HE_YOLOX/logs/*.txt`

**Common issues are usually:**
- Wrong runtime (CPU instead of GPU)
- Files not uploaded correctly
- Dataset not fully downloaded
- Google Drive disconnected

---

**Good luck with training! 🚀**

Expected timeline:
- Setup: 15-30 minutes
- Training: 12-18 hours
- Evaluation: 5-10 minutes
- Total: ~13-19 hours

**You'll achieve the base accuracy from the paper!** 🎯
