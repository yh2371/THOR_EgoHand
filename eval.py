from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import os.path as path
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from GraFormer.progress.bar import Bar
from GraFormer.common.log import Logger, savefig
from GraFormer.common.utils import AverageMeter, lr_decay, save_ckpt
from GraFormer.common.data_utils import fetch, read_3d_data, create_2d_data, create_edges
from GraFormer.common.generators import PoseGenerator
from GraFormer.common.loss import mpjpe, p_mpjpe
from GraFormer.network.GraFormer import GraFormer, adj_mx_from_edges
from tqdm import tqdm

""" import libraries"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import sys
import os
import time

from utils.options import parse_args_function
from utils.utils import freeze_component, calculate_keypoints, create_loader
from utils.h2o_utils.h2o_dataset_utils import load_tar_split
from utils.h2o_utils.h2o_preprocessing_utils import MyPreprocessor

from models.thor_net import create_thor

CUDA_LAUNCH_BLOCKING=1

def evaluate(data_loader, model_pos, device, seq_length=1):
    joint_mean = torch.from_numpy(np.array([[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
                                    [-3.9501650e+00, -8.6685377e-01,  2.4517984e+01],
                                    [-1.3187613e+01,  1.2967486e+00,  4.7673504e+01],
                                    [-2.2936522e+01,  1.5275195e+00,  7.2566208e+01],
                                    [-3.1109295e+01,  1.9404153e+00,  9.5952751e+01],
                                    [-4.8375599e+01,  4.6012049e+00,  6.7085617e+01],
                                    [-5.9843365e+01,  5.9568534e+00,  9.3948418e+01],
                                    [-5.7148232e+01,  5.7935758e+00,  1.1097713e+02],
                                    [-5.1052166e+01,  4.9937048e+00,  1.2502338e+02],
                                    [-5.1586624e+01,  2.5471370e+00,  7.2120811e+01],
                                    [-6.5926834e+01,  3.0671554e+00,  9.8404510e+01],
                                    [-6.1979191e+01,  2.8341565e+00,  1.1610429e+02],
                                    [-5.4618130e+01,  2.5274558e+00,  1.2917862e+02],
                                    [-4.6503471e+01,  3.3559692e-01,  7.3062035e+01],
                                    [-5.9186893e+01,  2.6649246e-02,  9.6192421e+01],
                                    [-5.6693432e+01, -8.4625520e-02,  1.1205978e+02],
                                    [-5.1260197e+01,  3.4378145e-02,  1.2381713e+02],
                                    [-3.5775276e+01, -1.0368422e+00,  7.0583588e+01],
                                    [-4.3695080e+01, -1.9620019e+00,  8.8694397e+01],
                                    [-4.4897186e+01, -2.6101866e+00,  1.0119468e+02],
                                    [-4.4571526e+01, -3.3564034e+00,  1.1180748e+02]])).cuda().unsqueeze(0)
    joint_std = torch.from_numpy(np.array([[ 0.      ,  0.      ,  0.      ],
                                [17.266953, 44.075836, 14.078445],
                                [24.261362, 65.793236, 18.580193],
                                [25.479671, 74.18796 , 19.767653],
                                [30.458921, 80.729996, 23.553158],
                                [21.826715, 45.61571 , 18.80888 ],
                                [26.570208, 54.434124, 19.955523],
                                [30.757236, 60.084938, 23.375763],
                                [35.174015, 64.042404, 31.206692],
                                [21.586899, 28.31489 , 16.090088],
                                [29.26384 , 35.83172 , 18.48644 ],
                                [35.396465, 40.93173 , 26.987226],
                                [40.40074 , 45.358475, 37.419308],
                                [20.73408 , 21.591717, 14.190551],
                                [28.290194, 27.946808, 18.350618],
                                [34.42277 , 31.388414, 28.024563],
                                [39.819054, 35.205494, 38.80897 ],
                                [19.79841 , 29.38799 , 14.820373],
                                [26.476702, 34.7448  , 20.027615],
                                [31.811651, 37.06962 , 27.742807],
                                [36.893555, 38.98199 , 36.001797]])+ 1e-8).cuda().unsqueeze(0)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()
    epoch_loss_3d_pos_procrustes = AverageMeter()

    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    model_pos.eval()
    end = time.time()

    bar = Bar('Eval ', max=len(data_loader))

    for i, tr_data in enumerate(data_loader):
        
        # get the inputs
        data_dict = tr_data
        '''
        format:
        data = {
            'path': img_path,
            'original_image': img,
            'inputs': inputs,
            'point2d': curr_2d_kpts,
            'point3d': curr_3d_kpts_cam_offset,
            'bb': bb,
            'boxes': boxes,
            'labels': labels,
            'keypoints': keypoints,
            'keypoints3d': keypoints3d,
        }
        '''      
        # Measure data loading time
        data_time.update(time.time() - end)  
        # Forward
        targets_3d = [t['point3d'] for t in data_dict][0].unsqueeze(0).cuda()
        #print(targets_3d)

        inputs_2d = [t['inputs'].to(device) for t in data_dict]
        targets = [{k: v.to(device) for k, v in t.items() if k in keys} for t in data_dict]

        #targets_3d = targets['point3d']

        num_poses = len(targets_3d)

        # [:, -29:] i.e. Use last pose if more than 1 pose, otherwise use the only pose
        outputs_3d = model_pos(inputs_2d)[0]['keypoints3d'][0:1] #2d/3d
        bs, num_joints, dim = outputs_3d.shape 
        # print(outputs_3d)
        # print(outputs_3d.shape)
        # print(targets_3d.shape)
        mask = ~torch.isnan(targets_3d)
        if len(targets_3d[mask]) == 0:
            continue
        indices = []
        for i in range(targets_3d.shape[1]):
            if torch.count_nonzero(torch.isnan(targets_3d[0,i])) == 0:
                indices.append(i)
        
        outputs_3d = outputs_3d #* (joint_std) + joint_mean
        targets_3d = targets_3d #* (joint_std) + joint_mean
        #curr_3d_kpts_cam_offset = (curr_3d_kpts_cam_offset - self.joint_mean) / (self.joint_std + 1e-8)

        epoch_loss_3d_pos.update(mpjpe(outputs_3d[:,indices,:], targets_3d.float().to(device)[:,indices,:]).item(), num_poses)
        # print("TGT",targets_3d.view(bs,-1,dim))
        # print("MASKED",targets_3d[mask].view(bs,-1,dim))
        
        epoch_loss_3d_pos_procrustes.update(p_mpjpe(outputs_3d[:,indices,:], targets_3d.float().to(device)[:,indices,:]).item(), num_poses)

        #epoch_loss_3d_pos.update(mpjpe(outputs_3d[mask].reshape(bs,-1,dim), targets_3d.float().to(device)[mask].view(bs,-1,dim)).item(), num_poses)
        # print("TGT",targets_3d.view(bs,-1,dim))
        # print("MASKED",targets_3d[mask].view(bs,-1,dim))
        #epoch_loss_3d_pos_procrustes.update(p_mpjpe(outputs_3d[mask].reshape(bs,-1,dim), targets_3d[mask].view(bs,-1,dim)).item(), num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, e1=epoch_loss_3d_pos.avg, e2=epoch_loss_3d_pos_procrustes.avg)
        bar.next()

    bar.finish()
    return epoch_loss_3d_pos.avg, epoch_loss_3d_pos_procrustes.avg


torch.multiprocessing.set_sharing_strategy('file_system')

args = parse_args_function()

# project_name = 'THOR'
# run = wandb.init(
#     # Set the project where this run will be logged
#     project=project_name,
#     # Track hyperparameters and run metadata
#     config=args)
    
# Define device
device = torch.device(f'cuda:{args.gpu_number[0]}' if torch.cuda.is_available() else 'cpu')

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True

num_kps2d, num_kps3d, num_verts = calculate_keypoints(args.dataset_name, args.object) #21, 21, _
print("2D", num_kps2d, "3D", num_kps3d)


""" load datasets """

valloader = create_loader(args.dataset_name, args.root, args.split, batch_size=args.batch_size,anno_type="annotation")
num_classes = 2 #hand -> 1, backgroud->0
graph_input = 'heatmaps'

""" load model """
model = create_thor(num_kps2d=num_kps2d, num_kps3d=num_kps3d, num_classes=num_classes, rpn_post_nms_top_n_train=num_classes-1, device=device, num_features=args.num_features, hid_size=args.hid_size, graph_input=graph_input, dataset_name=args.dataset_name)
print('THOR is loaded')

if torch.cuda.is_available():
    model = model.cuda(args.gpu_number[0])
    model = nn.DataParallel(model, device_ids=args.gpu_number)

""" load saved model"""
pretrained_model = f'./checkpoints/{args.checkpoint_folder}/model-{args.checkpoint_id}.pkl'
model.load_state_dict(torch.load(pretrained_model, map_location=f'cuda:{args.gpu_number[0]}'), strict=False)
model = model.eval()
keys = ['boxes', 'labels', 'keypoints', 'keypoints3d']
print("Checkpoint loaded")

""" evaluation """
print("Evaluating...")

errors_p1, errors_p2 = evaluate(valloader, model, device)

print('Protocol #1   (MPJPE) action-wise average: {:.2f} (mm)'.format(np.mean(errors_p1).item()))
print('Protocol #2 (P-MPJPE) action-wise average: {:.2f} (mm)'.format(np.mean(errors_p2).item()))