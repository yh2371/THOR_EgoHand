#!/bin/bash

# export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_LAUNCH_BLOCKING=1
python main_THOR.py \
 --root ../\
 --batch_size 16 \
 --hid_size 96 \
 --dataset_name ego4d\
 --gpu 1\
 --output_file ./checkpoints/finetune/model- \
 --pretrained_model ./checkpoints/automatic/model-1.pkl \
 --freeze
