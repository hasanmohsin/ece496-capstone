#!/bin/bash

export LD_LIBRARY_PATH=/pkgs/cuda-10.1/lib64:/pkgs/cudnn-10.1-v7.6.3.30/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} 
export PATH=/pkgs/anaconda3/bin:$PATH 
export CUDA_HOME=/pkgs/cuda-10.2/
export PYTHONPATH=$PYTHONPATH:~/ece496-capstone/train/utils
source activate /scratch/ssd001/home/$USER/capstone/
source ./env/bin/activate
source ./scripts/env.sh
source ./scripts/config.sh
