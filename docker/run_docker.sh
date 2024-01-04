#!/bin/bash

# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
# ---------------------------------------------------------------

NAME="TMRnet-gbp"

docker="lumargot/tmrnet-pytorch2:latest"
code_path=/home/lumargot/

docker run --gpus all \
            --ipc=host \
            -d \
            --rm \
            --init \
       	    --name $NAME-docker \
            -v $code_path:$code_path \
            -v /CMF/data/:/CMF/data/ \
            -it $docker bash
