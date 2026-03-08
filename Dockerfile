# Base image: includes Python, CUDA, cuDNN, and PyTorch preinstalled
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Core ML stack (PINNED — DO NOT REINSTALL LATER)
RUN pip install \
    transformers==4.44.2 \
    peft==0.18.1 \
    accelerate==0.33.0 \
    bitsandbytes==0.43.1 \
    safetensors==0.4.3 \
    sentencepiece==0.2.0 \
    datasets==2.19.0

# General scientific stack
RUN pip install \
    numpy==1.26.4 \
    pandas \
    matplotlib \
    scikit-learn \
    scipy \
    h5py \
    tqdm \
    wandb \
    protobuf \
    einops \
    rich \
    opencv-python-headless \
    grad-cam

# Set working directory
WORKDIR /workspace

# Default command
CMD ["bash"]