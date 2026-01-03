#!/bin/bash

# Download VisDrone2019 Dataset Script
# This script downloads the VisDrone2019 dataset for object detection

echo "Downloading VisDrone2019 Dataset..."
echo "===================================="

# Create data directory
mkdir -p data/VisDrone2019
cd data/VisDrone2019

# Download training set
echo "Downloading training set..."
wget https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-DET-train.zip

# Download validation set
echo "Downloading validation set..."
wget https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-DET-val.zip

# Download test set
echo "Downloading test set..."
wget https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-DET-test-dev.zip

# Unzip all files
echo "Extracting files..."
unzip -q VisDrone2019-DET-train.zip
unzip -q VisDrone2019-DET-val.zip
unzip -q VisDrone2019-DET-test-dev.zip

# Clean up zip files
echo "Cleaning up..."
rm VisDrone2019-DET-train.zip
rm VisDrone2019-DET-val.zip
rm VisDrone2019-DET-test-dev.zip

echo ""
echo "Download complete!"
echo "Dataset structure:"
tree -L 2 .

echo ""
echo "Dataset statistics:"
echo "Training images: $(ls VisDrone2019-DET-train/images | wc -l)"
echo "Validation images: $(ls VisDrone2019-DET-val/images | wc -l)"
echo "Test images: $(ls VisDrone2019-DET-test-dev/images | wc -l)"

cd ../..
