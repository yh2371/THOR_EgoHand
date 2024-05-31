#!/bin/bash
python train.py \
 --cfg_file ./configs/ego4d/finetune_ego4d.yaml \
 --output_file ./checkpoints/finetune/model- \
 --pretrained_model ./checkpoints/auto/model.pkl \
 --freeze
