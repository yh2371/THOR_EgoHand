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

from utils.options import parse_args_function, update_config
from utils.utils import freeze_component, create_loader

from models.thor_net import create_thor

CUDA_LAUNCH_BLOCKING=1

torch.multiprocessing.set_sharing_strategy('file_system')

args = parse_args_function()
cfg = update_config(args.cfg_file)
    
# Define device
device = torch.device(f'cuda:{cfg.GPUS}' if torch.cuda.is_available() else 'cpu')
use_cuda = torch.cuda.is_available()

num_kps2d, num_kps3d = cfg.MODEL.NUM_JOINTS, cfg.MODEL.NUM_JOINTS

""" Configure a log """

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.output_file[:-6], 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

""" load datasets """

trainloader, _ = create_loader('train', batch_size=cfg.TRAIN.BATCH_SIZE, anno_type=cfg.DATASET.ANNO_TYPE, cfg=cfg)
valloader, _ = create_loader('val', batch_size=cfg.TEST.BATCH_SIZE, anno_type="manual", cfg=cfg) #validation on manual only
num_classes = cfg.MODEL.NUM_CLASS #hand -> 1, background->0
graph_input = cfg.MODEL.GRAPH_INPUT

""" load model """
model = create_thor(num_kps2d=num_kps2d, num_kps3d=num_kps3d, num_classes=num_classes, rpn_post_nms_top_n_train=num_classes-1, device=device, num_features=cfg.MODEL.NUM_FEATURES, hid_size=cfg.MODEL.HID_SIZE, graph_input=graph_input, dataset_name=cfg.DATASET.DATASET)
print('THOR is loaded')
print("DATA", len(trainloader))

if torch.cuda.is_available():
    model = model.to(device)

""" load saved model"""

if args.pretrained_model != '':
    model.load_state_dict(torch.load(args.pretrained_model, map_location=f'cuda:{cfg.GPUS}'), strict=False)
losses = []
start = 0

"""define optimizer"""

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_GAMMA)
scheduler.last_epoch = start

keys = ['boxes', 'labels', 'keypoints', 'keypoints3d','valid']

""" training """

logging.info('Begin training the network...')

if args.freeze:
    logging.info('Freezing Keypoint RCNN ..')            
    freeze_component(model.backbone)

best_loss3d = float("inf")
for epoch in range(start, cfg.TRAIN.NUM_EPOCH):  # loop over the dataset multiple times
    start_time = time.time()
    train_loss3d = 0.0
    running_loss2d = 0.0
    running_loss3d = 0.0
    running_proj = 0.0
    for i, data_dict in enumerate(trainloader):
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
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward
        targets = []
        for t in data_dict:
            d = {}
            for k,v in t.items():
                if k in keys:
                    try:
                        d[k] = torch.from_numpy(v).to(device).float()
                    except:
                        d[k] = v.to(device)
            targets.append(d)

        inputs = [t['inputs'].to(device) for t in data_dict]

        loss_dict = model(inputs, targets)
        
        # Calculate Loss
        loss = sum(loss for _, loss in loss_dict.items())
        
        # Backpropagate
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss3d += loss_dict['loss_keypoint3d'].data
        running_loss2d += loss_dict['loss_keypoint'].data
        running_loss3d += loss_dict['loss_keypoint3d'].data

        if (i+1) % cfg.TRAIN.LOG_BATCH == 0:    # print every log_iter mini-batches
            logging.info('[%d, %5d] loss 2d: %.4f, loss 3d: %.4f' %
            (epoch + 1, i + 1, running_loss2d / cfg.TRAIN.LOG_BATCH, running_loss3d / cfg.TRAIN.LOG_BATCH))
            running_loss2d = 0.0
            running_loss3d = 0.0

        if (i+1) % cfg.TRAIN.SAVE_MID == 0:
            torch.save(model.state_dict(), args.output_file+str(i+1)+'.pkl')

    losses.append((train_loss3d / (i+1)).cpu().numpy())

    if (epoch+1) % cfg.TRAIN.VAL_EPOCH == 0:
        val_loss2d = 0.0
        val_loss3d = 0.0
        val_proj = 0.0

        for v, val_data in enumerate(valloader):
            
            # get the inputs
            data_dict = val_data
        
            # wrap them in Variable
            targets = [{k: v.to(device) for k, v in t.items() if k in keys} for t in data_dict]
            inputs = [t['inputs'].to(device) for t in data_dict]    
            loss_dict = model(inputs, targets)
            
            val_loss2d += loss_dict['loss_keypoint'].data
            val_loss3d += loss_dict['loss_keypoint3d'].data

        if cfg.TRAIN.SAVE_INTER:
            torch.save(model.state_dict(), args.output_file+str(epoch+1)+'.pkl')
            np.save(args.output_file+str(epoch+1)+'-losses.npy', np.array(losses))
        if val_loss3d / (v+1) < best_loss3d:
            best_loss3d = val_loss3d / (v+1)
            torch.save(model.state_dict(), args.output_file+'best.pkl')
            np.save(args.output_file+'best-losses.npy', np.array(losses))

        logging.info('val loss 2d: %.4f, val loss 3d: %.4f' % (val_loss2d / (v+1), val_loss3d / (v+1)))   
    
    # Decay Learning Rate
    scheduler.step()

logging.info('Finished Training')
