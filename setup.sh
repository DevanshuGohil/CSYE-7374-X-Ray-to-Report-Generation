#!/bin/bash

echo "Setting up X-Ray Report Generation Environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment
echo "Creating conda environment: xray_report_env"
conda create -n xray_report_env python=3.8 -y

# Activate conda environment
echo "Activating conda environment"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xray_report_env

# Install pip dependencies from requirements.txt
echo "Installing dependencies from requirements.txt"
pip install -r requirements.txt

# Check if Kaggle API credentials exist
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Kaggle API credentials not found."
    echo "Please visit https://www.kaggle.com/account and download your API token."
    echo "Then place the kaggle.json file in ~/.kaggle/ and run this script again."
    
    # Create the directory if it doesn't exist
    mkdir -p ~/.kaggle
    
    exit 1
fi

# Make sure the credentials file has the right permissions
chmod 600 ~/.kaggle/kaggle.json

# Create data directory
mkdir -p data

# Download dataset using curl
echo "Downloading Indiana University Chest X-rays dataset"
curl -L -o ./data/chest-xrays-indiana-university.zip \
  "https://www.kaggle.com/api/v1/datasets/download/raddar/chest-xrays-indiana-university" \
  --header "Authorization: Basic $(base64 -w0 ~/.kaggle/kaggle.json)"

# Unzip the dataset
echo "Extracting dataset..."
unzip -q ./data/chest-xrays-indiana-university.zip -d ./data

echo "Setup complete! Activate the environment with: conda activate xray_report_env"
echo "Dataset is extracted in the ./data directory" 