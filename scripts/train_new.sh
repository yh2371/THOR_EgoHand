#!/bin/bash

# export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_LAUNCH_BLOCKING=1
python main_THOR.py \
 --root ../\
 --batch_size 1 \
 --hid_size 96 \
 --dataset_name ego4d\
 --gpu 0
  # --pretrained_model ./checkpoints/h2o-2d/model-3.pkl
