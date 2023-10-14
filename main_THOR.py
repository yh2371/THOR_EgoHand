# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

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
### WANDB Setup ###
import wandb
wandb.login()

torch.multiprocessing.set_sharing_strategy('file_system')

args = parse_args_function()

project_name = 'THOR'
run = wandb.init(
    # Set the project where this run will be logged
    project=project_name,
    # Track hyperparameters and run metadata
    config=args)
    
# Define device
device = torch.device(f'cuda:{args.gpu_number[0]}' if torch.cuda.is_available() else 'cpu')

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True

num_kps2d, num_kps3d, num_verts = calculate_keypoints(args.dataset_name, args.object) #21, 21, _
print("2D", num_kps2d, "3D", num_kps3d)

""" Configure a log """

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.output_file[:-6], 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


""" load datasets """

trainloader = create_loader(args.dataset_name, args.root, 'train', batch_size=args.batch_size, num_kps3d=num_kps3d)
valloader = create_loader(args.dataset_name, args.root, 'val', batch_size=args.batch_size)
num_classes = 2 #hand -> 1, backgroud->0
graph_input = 'heatmaps'

""" load model """
model = create_thor(num_kps2d=num_kps2d, num_kps3d=num_kps3d, num_classes=num_classes, rpn_post_nms_top_n_train=num_classes-1, device=device, num_features=args.num_features, hid_size=args.hid_size, graph_input=graph_input, dataset_name=args.dataset_name)
print('THOR is loaded')

if torch.cuda.is_available():
    model = model.cuda(args.gpu_number[0])
    model = nn.DataParallel(model, device_ids=args.gpu_number)

""" load saved model"""

if args.pretrained_model != '':
    model.load_state_dict(torch.load(args.pretrained_model, map_location=f'cuda:{args.gpu_number[0]}'))
    losses = np.load(args.pretrained_model[:-4] + '-losses.npy').tolist()
    start = len(losses)
else:
    losses = []
    start = 0

"""define optimizer"""

criterion = nn.MSELoss()
# criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_step_gamma)
scheduler.last_epoch = start

keys = ['boxes', 'labels', 'keypoints', 'keypoints3d']

""" training """

logging.info('Begin training the network...')

for epoch in range(start, args.num_iterations):  # loop over the dataset multiple times
    start_time = time.time()
    train_loss2d = 0.0
    running_loss2d = 0.0
    running_loss3d = 0.0

    for i, tr_data in enumerate(trainloader):
        
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

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward
        targets = [{k: v.to(device) for k, v in t.items() if k in keys} for t in data_dict]
        #print(targets[0])
        inputs = [t['inputs'].to(device) for t in data_dict]
        loss_dict = model(inputs, targets)
        
        # Calculate Loss
        loss = sum(loss for _, loss in loss_dict.items())
        
        # Backpropagate
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss2d += loss_dict['loss_keypoint'].data
        running_loss2d += loss_dict['loss_keypoint'].data
        running_loss3d += loss_dict['loss_keypoint3d'].data

        if (i+1) % args.log_batch == 0:    # print every log_iter mini-batches
            logging.info('[%d, %5d] loss 2d: %.4f, loss 3d: %.4f' % 
            (epoch + 1, i + 1, running_loss2d / args.log_batch, running_loss3d / args.log_batch))
            wandb.log({'Loss 2D': running_loss2d/ args.log_batch, 'Loss 3D': running_loss3d/ args.log_batch})
            
            running_loss2d = 0.0
            running_loss3d = 0.0

    losses.append((train_loss2d / (i+1)).cpu().numpy())
    
    if (epoch+1) % args.snapshot_epoch == 0:
        torch.save(model.state_dict(), args.output_file+str(epoch+1)+'.pkl')
        np.save(args.output_file+str(epoch+1)+'-losses.npy', np.array(losses))

    if (epoch+1) % args.val_epoch == 0:
        val_loss2d = 0.0
        val_loss3d = 0.0

        for v, val_data in enumerate(valloader):
            
            # get the inputs
            data_dict = val_data
        
            # wrap them in Variable
            targets = [{k: v.to(device) for k, v in t.items() if k in keys} for t in data_dict]
            inputs = [t['inputs'].to(device) for t in data_dict]    
            loss_dict = model(inputs, targets)
            
            val_loss2d += loss_dict['loss_keypoint'].data
            val_loss3d += loss_dict['loss_keypoint3d'].data
            
        logging.info('val loss 2d: %.4f, val loss 3d: %.4f' % (val_loss2d / (v+1), val_loss3d / (v+1)))   
        wandb.log({'Val Loss 2D': val_loss2d/(v+1), 'Val Loss 3D': val_loss3d/(v+1)})     
    
    if args.freeze and epoch == 0: #Apply pretrained Keypoint RCNN
        logging.info('Freezing Keypoint RCNN ..')            
        freeze_component(model.module.backbone)
        freeze_component(model.module.rpn)
        freeze_component(model.module.roi_heads)

    # Decay Learning Rate
    scheduler.step()

logging.info('Finished Training')
