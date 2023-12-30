# -*- coding: utf-8 -*-
import argparse
import yaml
from easydict import EasyDict as edict

def parse_args_function():
    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    
    parser.add_argument(
        "--cfg_file",
        default='configs/ego4d/manual_ego4d.yaml',
        help="Config file path"
    )
    
    parser.add_argument(
        "--output_file",
        default='./checkpoints/model-',
        help="Prefix of output pkl filename"
    )

    # Optional arguments.
    parser.add_argument(
        "--pretrained_model",
        default='',
        help="Load trained model weights file."
    )
  
    parser.add_argument(
        "--freeze",
        action='store_true',
        help="freeze components for finetuning"
    )

    # For Inference
    parser.add_argument(
        "--output_folder",
        default='./outputs',
        help="Folder of output json filename"
    )

    parser.add_argument(
        "--output_prefix",
        default='',
        help="Prefix of output json filename"
    )

    # For Testing
    parser.add_argument(
        "--seq", 
        default='MPM13', 
        help="Sequence Name"
    )

    parser.add_argument(
        "--split",
        default='test', 
        help="Which subset to evaluate on"
    )

    args = parser.parse_args()
    return args

def update_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config