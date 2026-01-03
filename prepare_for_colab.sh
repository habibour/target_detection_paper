#!/bin/bash

# Script to prepare files for Google Colab upload
# Run this to create a clean ZIP file

echo "🚀 Preparing HE-YOLOX for Google Colab..."
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create temporary directory for clean files
echo "📦 Creating clean package..."
TMP_DIR="implement_colab_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TMP_DIR"

# Copy necessary files
echo "📋 Copying project files..."
cp -r models/ "$TMP_DIR/"
cp -r utils/ "$TMP_DIR/"
cp -r configs/ "$TMP_DIR/"
cp train.py "$TMP_DIR/"
cp eval.py "$TMP_DIR/"
cp inference.py "$TMP_DIR/"
cp requirements.txt "$TMP_DIR/"

# Create empty directories for outputs
mkdir -p "$TMP_DIR/checkpoints"
mkdir -p "$TMP_DIR/logs"
mkdir -p "$TMP_DIR/results"

# Remove any Python cache files
echo "🧹 Cleaning cache files..."
find "$TMP_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find "$TMP_DIR" -type f -name "*.pyc" -delete 2>/dev/null
find "$TMP_DIR" -type f -name ".DS_Store" -delete 2>/dev/null

# Create ZIP file
echo "📦 Creating ZIP archive..."
ZIP_NAME="implement_colab.zip"
cd "$TMP_DIR/.."
zip -r "$ZIP_NAME" "$(basename $TMP_DIR)" -q

# Move to original directory
mv "$ZIP_NAME" "$SCRIPT_DIR/"

# Cleanup
cd "$SCRIPT_DIR"
rm -rf "$TMP_DIR"

# Get file size
SIZE=$(du -h "$ZIP_NAME" | cut -f1)

echo ""
echo "✅ Package ready for Google Colab!"
echo ""
echo "📦 File: $ZIP_NAME"
echo "📏 Size: $SIZE"
echo ""
echo "Next steps:"
echo "1. Open: https://colab.research.google.com/"
echo "2. Upload: HE_YOLOX_Training_Colab.ipynb"
echo "3. In the notebook, upload: $ZIP_NAME"
echo "4. Follow the instructions in COLAB_INSTRUCTIONS.md"
echo ""
echo "Expected training time: 12-18 hours on T4 GPU"
echo ""
echo "🚀 Happy training!"
