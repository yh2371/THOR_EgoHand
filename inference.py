from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from GraFormer.progress.bar import Bar
from GraFormer.common.utils import AverageMeter

""" import libraries"""
import numpy as np
import torch
import torch.nn as nn
import time

from utils.options import parse_args_function, update_config
from utils.utils import create_loader
from models.thor_net import create_thor
import json
from tqdm import tqdm

CUDA_LAUNCH_BLOCKING=1

def evaluate(data_loader, dataset, model_pos, device, output_pth):

    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    model_pos.eval()
    end = time.time()

    res = {} #result storage
    for i, data_dict in tqdm(enumerate(data_loader)):
        '''
        format:
        data_dict = {
            'original_image': img,
            'inputs': inputs,
            'point2d': curr_2d_kpts,
            'point3d': curr_3d_kpts_cam_offset,
            'bb': bb,
            'boxes': boxes,
            'labels': labels,
            'keypoints': keypoints,
            'keypoints3d': keypoints3d,
            'valid': vis_flag,
            'meta': meta
        }
        '''   
        meta = data_dict[0]['meta']
        take_res = res.get(meta['take_uid'],{})
        frame_res = take_res.get(meta['frame_number'], {'left_hand':[], 'right_hand': []})
        hand_order = meta['hand_order']

        # Forward
        inputs_2d = [t['inputs'].to(device) for t in data_dict]
        outputs_3d = model_pos(inputs_2d)[0]['keypoints3d'][0].detach().cpu() # 21 x 3
        outputs_3d = outputs_3d * dataset.joint_std + dataset.joint_mean

        # Store results
        frame_res[f'{hand_order}_hand'] = (outputs_3d/1000).tolist()
        take_res[meta['frame_number']] = frame_res
        res[meta['take_uid']] = take_res

    with open(output_pth, "w") as outfile: 
        json.dump(res, outfile)

torch.multiprocessing.set_sharing_strategy('file_system')

args = parse_args_function()
cfg = update_config(args.cfg_file)
    
# Define device
device = torch.device(f'cuda:{cfg.GPUS}' if torch.cuda.is_available() else 'cpu')

use_cuda = torch.cuda.is_available()

num_kps2d, num_kps3d = cfg.MODEL.NUM_JOINTS, cfg.MODEL.NUM_JOINTS
print("2D", num_kps2d, "3D", num_kps3d)

""" load datasets """

valloader, dataset = create_loader(args.split, batch_size=cfg.TEST.BATCH_SIZE, anno_type="manual", cfg=cfg)
num_classes = cfg.MODEL.NUM_CLASS #hand -> 1, backgroud->0
graph_input = cfg.MODEL.GRAPH_INPUT

""" load model """
model = create_thor(num_kps2d=num_kps2d, num_kps3d=num_kps3d, num_classes=num_classes, rpn_post_nms_top_n_train=num_classes-1, device=device, num_features=cfg.MODEL.NUM_FEATURES, hid_size=cfg.MODEL.HID_SIZE, graph_input=graph_input, dataset_name=cfg.DATASET.DATASET)
print('THOR is loaded')

if torch.cuda.is_available():
    model = model.to(device)

""" load saved model"""
pretrained_model = args.pretrained_model
model.load_state_dict(torch.load(pretrained_model, map_location=f'cuda:{cfg.GPUS}'), strict=False)
model = model.eval()
keys = ['boxes', 'labels', 'keypoints', 'keypoints3d']
print("Checkpoint loaded")

""" evaluation """
print("Evaluating...")

output_pth = f'./{args.output_folder}/{args.output_prefix}.json'
evaluate(valloader, dataset, model, device, output_pth)

print("Inference Complete.")