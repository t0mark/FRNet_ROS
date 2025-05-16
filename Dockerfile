# NVIDIA CUDA 기반 이미지 (CUDA 11.1)
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# 비대화 모드 설정
ENV DEBIAN_FRONTEND=noninteractive

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3-pip \
    python3-opencv \
    libopencv-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    wget \
    git \
    ffmpeg \
    build-essential \
    ninja-build \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 기본 python3를 python3.8로 설정
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# 작업 디렉토리 설정
WORKDIR /app/FRNet

# pip 최신화
RUN pip3 install --no-cache-dir -U pip setuptools wheel

# PyTorch + torchvision 설치 (CUDA 11.1 대응 버전)
RUN pip3 install --no-cache-dir torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# 디버깅: PyTorch + CUDA 상태 확인
RUN python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"

# MIM 기반 MMEngine, MMCV, MMDetection, MMDetection3D 설치
RUN pip3 install --no-cache-dir -U openmim
RUN mim install mmengine==0.9.0
RUN mim install mmcv==2.1.0
RUN mim install mmdet==3.2.0
RUN mim install mmdet3d==1.3.0

# ✅ torch-scatter (CUDA 지원) 공식 호환 버전 설치
RUN pip3 install --no-cache-dir torch-scatter==2.0.8 -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html

# 디버깅: scatter_max CUDA 연산 가능 여부 확인
RUN python3 -c "import torch_scatter; print('scatter_max:', hasattr(torch_scatter, 'scatter_max')); print('Location:', torch_scatter.__file__)"

# NuScenes devkit 및 기타 의존성 설치
RUN pip3 install --no-cache-dir nuscenes-devkit
RUN pip3 install --no-cache-dir argparse pyyaml

# 환경변수 설정 (명시적)
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# ROS Noetic 설치를 위한 준비
RUN apt-get update && apt-get install -y \
    lsb-release \
    curl \
    gnupg2 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ROS 저장소 설정 (Ubuntu 20.04 Focal용)
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list'

# ROS 키 설정
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# ROS 패키지 설치 - RViz 및 GUI 지원 포함
RUN apt-get update && apt-get install -y \
    ros-noetic-desktop \
    ros-noetic-pcl-ros \
    ros-noetic-pcl-conversions \
    ros-noetic-rviz \
    ros-noetic-rqt \
    ros-noetic-rqt-common-plugins \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    build-essential \
    x11-apps \
    mesa-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ROS 초기화
RUN rosdep init && rosdep update

# ROS 환경 설정
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# Python3와 ROS 호환성 패키지 설치
RUN pip3 install rospkg catkin_pkg

# 작업 디렉토리 생성
WORKDIR /app/catkin_ws

# 기본 실행 명령
CMD ["/bin/bash"]