# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
# ---------------------------------------------------------------

FROM nvcr.io/nvidia/pytorch:23.04-py3
ARG PYTHON_VERSION=3.8
RUN apt-get update
RUN apt-get install libgl1-mesa-glx
RUN apt-get install -y build-essential zsh tmux wget
RUN python -m pip install scipy open3d
RUN python -m pip install h5py
RUN python -m pip install cupy-cuda12x
RUN python -m pip install nestargs
RUN python -m pip install flatten_dict
RUN python -m pip install scikit-image
RUN python -m pip install torchmetrics monai pip install imageio[ffmpeg] imageio[pyav]
RUN python -m pip install opencv-python==4.8.0.74
RUN python -m pip install pytorch_lightning resnest
RUN python -m pip install -r https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/requirements-dev.txt

# RUN export PATH=/usr/local/cuda-11.1/bin:$PATH
# RUN export CPATH=/usr/local/cuda-11.1/include:$CPATH
# RUN export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH
# RUN export DYLD_LIBRARY_PATH=/usr/local/cuda-11.1/lib:$DYLD_LIBRARY_PATH

RUN python -m pip install git+https://github.com/openai/CLIP.git
RUN python -m pip install loguru einops comet_ml calmsize diffusers
RUN python -m pip install tensorboard 

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs
RUN git lfs install

ARG USER=lumargot
ARG UID=338730
ARG GID=1001

RUN useradd --system --create-home --home-dir /home/${USER} \
    --shell /bin/bash --uid ${UID} ${USER}

USER root
