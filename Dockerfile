# CUDA 11.1 기반 이미지 사용 (torch==1.8.1+cu111 호환성을 위해)
FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04

# 비대화형 설치 설정
ENV DEBIAN_FRONTEND=noninteractive

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 파이썬 패키지 설치
RUN pip3 install -U pip setuptools wheel

# PyTorch 설치
RUN pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# 추가 패키지 설치
RUN pip3 install torch-scatter==2.0.8 -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
RUN pip3 install nuscenes-devkit argparse pyyaml

# MIM 패키지 설치
RUN pip3 install -U openmim

# MMDetection 관련 라이브러리 설치
RUN mim install mmengine==0.9.0
RUN mim install mmcv==2.1.0
RUN mim install mmdet==3.2.0
RUN mim install mmdet3d==1.3.0

# CUDA 환경 변수 설정
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 기본 명령 설정
CMD ["/bin/bash"]