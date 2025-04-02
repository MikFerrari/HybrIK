# Start with NVIDIA CUDA 12.8 on Ubuntu 22.04
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set up NVIDIA runtime environment variables
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compute,utility

# Install necessary packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        git \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        pkg-config \
        python3-dev \
        python3-pip \
        python3-setuptools \
        wget \
        ffmpeg \
        libavcodec-extra && \
    rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /workspace

# Clone HybrIK repository
RUN git clone https://github.com/Jeff-sjtu/HybrIK.git /workspace/HybrIK

# Set up Python environment
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    scikit-image \
    opencv-python \
    matplotlib \
    pycocotools \
    tqdm \
    tensorboardX \
    easydict \
    pyyaml

# Install PyTorch with CUDA support (compatible with CUDA 12.x)
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Set up PyTorch3D for visualization (optional)
RUN pip3 install --no-cache-dir fvcore iopath
RUN pip3 install --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install utilities to download and unzip files
RUN apt-get update && apt-get install -y unzip python3-pip && \
    pip install gdown

# Download models
RUN gdown https://drive.google.com/uc?id=1un9yAGlGjDooPwlnwFpJrbGHRiLaBNzV -O /tmp/model_files.zip

# Unzip the model files
RUN unzip /tmp/model_files.zip -d /workspace/HybrIK/ && \
    rm /tmp/model_files.zip

# Set the working directory
WORKDIR /workspace/HybrIK

# Download the model weights and place them in the pretrained_models directory
RUN mkdir -p pretrained_models && \
    gdown https://drive.google.com/file/d/1bKIPD60z_Im4S3W2-rew6YtOtUGff6-v -O pretrained_models/hybrikx_hrnet.pth && \
    gdown https://drive.google.com/file/d/1R0WbySXs_vceygKg_oWeLMNAZCEoCadG -O pretrained_models/hybrikx_rle_hrnet.pth

# Modify setup.py to remove the fixed OpenCV version
RUN sed -i "s/'opencv-python==[^']*'/'opencv-python'/" setup.py

# Install required dependencies for HybrIK
RUN pip3 install --no-cache-dir -e .

# Manually fix chumpy's import issue
RUN sed -i 's/from numpy import bool, int, float, complex, object, unicode, str, nan, inf/import numpy as np\nbool = np.bool_\nint = np.int_\nfloat = np.float64\ncomplex = np.complex128\nobject = np.object_\nstr = np.str_\nnan = np.nan\ninf = np.inf/' \
    /usr/local/lib/python3.10/dist-packages/chumpy/__init__.py

# Set the default command
CMD ["/bin/bash"]
