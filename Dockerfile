# Base image: includes Python, CUDA, cuDNN, and PyTorch preinstalled
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Install basic system tools and update pip
RUN apt-get update && apt-get install -y \
    git wget curl vim build-essential && \
    pip install --upgrade pip

# Core ML stack (PINNED — DO NOT REINSTALL LATER)
RUN pip install \
    torch==2.3.1 \
    torchvision==0.18.1 \
    torchaudio==2.3.1 \
    transformers==4.44.2 \
    peft==0.18.1 \
    accelerate==0.30.1 \
    bitsandbytes==0.43.1 \
    safetensors==0.4.3 \
    sentencepiece==0.2.0 \
    datasets==2.19.0

# General scientific stack (safe to float)
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
    rich


# Set a working directory inside the container
WORKDIR /workspace

# Default command — drop you into a bash shell
CMD ["bash"]