#!/bin/bash

# Update package list
sudo apt-get update

# Install dependencies
sudo apt-get install -y wget gnupg2 software-properties-common python3-pip python3-dev build-essential

# Add CUDA repository
wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin | sudo tee /etc/apt/preferences.d/cuda-repository-pin-600
wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub | sudo apt-key add -
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

# Install CUDA toolkit
sudo apt-get update
sudo apt-get install -y cuda-toolkit-11-8

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt
