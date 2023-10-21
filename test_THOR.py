from posixpath import split
import torch 
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import argparse


from torch.utils.data import DataLoader
from GraFormer.progress.bar import Bar
from GraFormer.common.log import Logger, savefig
from GraFormer.common.utils import AverageMeter, lr_decay, save_ckpt
from GraFormer.common.data_utils import fetch, read_3d_data, create_2d_data, create_edges
from GraFormer.common.generators import PoseGenerator
from GraFormer.common.loss import mpjpe, p_mpjpe
from GraFormer.network.GraFormer import GraFormer, adj_mx_from_edges
from tqdm import tqdm

# matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt

from utils.ego4d_dataset import ego4dDataset
from utils.vis_utils import *
from tqdm import tqdm
from models.thor_net import create_thor
from utils.utils import *
from utils.h2o_utils.h2o_dataset_utils import load_tar_split
from utils.h2o_utils.h2o_preprocessing_utils import MyPreprocessor
from utils.options import parse_args_function

left_hand_faces, right_hand_faces, obj_faces = load_faces()

def visualize2d(img, predictions, labels=None, filename=None, palm=None, evaluate=False):
    
    fig = plt.figure(figsize=(20, 10))
    
    H = 1
    if evaluate:
        H = 2
    W = 3

    plot_id = 1
    fig_config = (fig, H, W)
    # idx = list(predictions['labels']).index(1) #[0]
    # Plot GT bounding boxes
    if evaluate:
        plot_bb_ax(img, labels, fig_config, plot_id, 'GT BB')
        plot_id += 1
        
        # Plot GT 2D keypoints
        plot_pose2d(img, labels, 0, fig_config, plot_id, plot_txt = 'Projected GT 3D pose', center=palm, pose2d = False)
        plot_id += 1

        # # Plot GT 3D keypoints
        # plot_pose2d(img, labels, 0, fig_config, plot_id, plot_txt = 'Projected GT 2D pose', center=palm, pose2d = True)
        plot_id += 1
        
        
        # Plot GT 3D Keypoints
        # plot_pose3d(labels, fig_config, plot_id, 'GT 3D pose', center=palm)
        # plot_id += 1

        # # Plot GT 3D mesh
        # plot_mesh3d(labels, right_hand_faces, obj_faces, fig_config, plot_id, 'GT 3D mesh', center=palm, left_hand_faces=left_hand_faces)
        # plot_id += 1

        # # Save textured mesh
        # texture = generate_gt_texture(img, labels['mesh3d'][0][:, :3])
        # save_mesh(labels, filename, right_hand_faces, obj_faces, texture=texture, shape_dir='mesh_gt', left_hand_faces=left_hand_faces)

    # Plot predicted bounding boxes
    plot_bb_ax(img, predictions, fig_config, plot_id, 'Predicted Bounding box')
    plot_id += 1

    # Plot predicted 2D keypoints
    plot_pose2d(img, predictions, 0, fig_config, plot_id, 'Predicted 2D pose', palm, pose2d = True)
    plot_id += 1

    # Plot predicted 2D keypoints
    plot_pose2d(img, predictions, 0, fig_config, plot_id, 'Projected Predicted 3D pose', palm, pose2d = False)
    plot_id += 1

    # plot_pose_heatmap(img, predictions, idx, palm, fig_config, plot_id)
    # plot_id += 1

    # # Plot predicted 3D keypoints
    # plot_pose3d(predictions, fig_config, plot_id, '3D pose', center=palm)
    # plot_id += 1

    # # Plot predicted 3D Mesh
    # plot_mesh3d(predictions, right_hand_faces, obj_faces, fig_config, plot_id, '3D mesh', center=palm, left_hand_faces=left_hand_faces)
    # plot_id += 1

    # # Save textured mesh
    # predicted_texture = predictions['mesh3d'][0][:, 3:]
    # save_mesh(predictions, filename, right_hand_faces, obj_faces, texture=predicted_texture, left_hand_faces=left_hand_faces)
    
    fig.tight_layout()
    plt.show()
    plt.savefig(filename)
    plt.clf()
    plt.close(fig)

# Input parameters
args = parse_args_function()

# Transformer function
transform_function = transforms.Compose([transforms.ToTensor()])

num_kps2d, num_kps3d, num_verts = calculate_keypoints(args.dataset_name, args.object)

# Create Output directory

# Dataloader

testset = ego4dDataset(root=args.root, split=args.split, anno_type='annotation',transform=transform_function)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=16, collate_fn=ho3d_collate_fn)
num_classes = 2
graph_input='heatmaps'

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True

# Define device
device = torch.device(f'cuda:{args.gpu_number[0]}' if torch.cuda.is_available() else 'cpu')

# Define model
model = create_thor(pretrained=False, num_classes=num_classes, device=device,
                                num_kps2d=num_kps2d, num_kps3d=num_kps3d, num_verts=num_verts,
                                rpn_post_nms_top_n_test=num_classes-1, 
                                box_score_thresh=0.0,
                                photometric=args.photometric, graph_input=graph_input, dataset_name=args.dataset_name,
                                num_features=args.num_features, hid_size=args.hid_size)

if torch.cuda.is_available():
    model = model.cuda(device=args.gpu_number[0])
    model = nn.DataParallel(model, device_ids=args.gpu_number)

### Load model
pretrained_model = f'./checkpoints/{args.checkpoint_folder}/model-{args.checkpoint_id}.pkl'
model.load_state_dict(torch.load(pretrained_model, map_location='cuda:0'),strict=False)
model = model.eval()
print(model)
print('model loaded!')

keys = ['boxes', 'labels', 'keypoints', 'keypoints3d']
# if args.dataset_name == 'ho3d':
#     keys.append('palm')
c = 0
# supporting_dicts = (pickle.load(open('./rcnn_outputs/rcnn_outputs_778_test_3d.pkl', 'rb')),
#                     pickle.load(open('./rcnn_outputs_mesh/rcnn_outputs_778_test_3d.pkl', 'rb')))
supporting_dicts = None
output_dicts = ({}, {})

evaluate = True
errors = [[], [], [], []]
# if args.split == 'val' or (args.dataset_name == 'h2o' and args.split == 'test'):  
#     evaluate = True  

# rgb_errors = []

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

count = 0
for i, ts_data in tqdm(enumerate(testloader)):
        
    data_dict = ts_data

    #print(data_dict)
    path = data_dict[0]['path'].split('/')[-1]
    # if args.seq not in data_dict[0]['path']:
    #     continue
    if '_' in path:
        path = path.split('_')[-1]
    frame_num = int(path.split('.')[0])
    
    ### Run inference
    inputs = [t['inputs'].to(device) for t in data_dict]
    outputs = model(inputs)
    img = inputs[0].cpu().detach().numpy()
    
    predictions, img, palm, labels = prepare_data_for_evaluation(data_dict, outputs, img, keys, device, args.split, mean = joint_mean, std = joint_std)

    ### Visualization
    if args.visualize: 

        name = path.split('/')[-1]

        if (num_classes == 2 and 1 in predictions['labels']) or (num_classes == 4 and set([1, 2, 3]).issubset(predictions['labels'])):
            #print("daf")
            #print(f'./visual_results/{args.seq}/{name}')
            visualize2d(img, predictions, labels, filename=f'./visual_results/{args.seq}/{name}', palm=palm, evaluate=evaluate)
        else:
            cv2.imwrite(f'./visual_results/{args.seq}/{name}', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    ### Evaluation
    # if evaluate:
    #     c = save_calculate_error(predictions, labels, path, errors, output_dicts, c, num_classes, args.dataset_name, obj=args.object, generate_mesh=False)
        targets_3d = [t['point3d'] for t in data_dict][0].unsqueeze(0).cuda()
        targets_2d = [t['point2d'] for t in data_dict][0].unsqueeze(0).cuda()
        #print(targets_3d)

        inputs_2d = [t['inputs'].to(device) for t in data_dict]
        targets = [{k: v.to(device) for k, v in t.items() if k in keys} for t in data_dict]

        #targets_3d = targets['point3d']

        num_poses = len(targets_3d)

        # [:, -29:] i.e. Use last pose if more than 1 pose, otherwise use the only pose
        outputs_3d = outputs[0]['keypoints3d'][0:1] #2d/3d
        outputs_2d = outputs[0]['keypoints'][0:1] #2d/3d
        bs, num_joints, dim = outputs_3d.shape 

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
        print("Path:", path)
        print("MPJPE",mpjpe(outputs_3d[:,indices,:], targets_3d.float().to(device)[:,indices,:]).item())
        # print("TGT",targets_3d.view(bs,-1,dim))
        # print("MASKED",targets_3d[mask].view(bs,-1,dim))

        print("PMPJPE",p_mpjpe(outputs_3d[:,indices,:], targets_3d.float().to(device)[:,indices,:]).item())

        indices = []
        for i in range(targets_2d.shape[1]):
            if torch.count_nonzero(torch.isnan(targets_2d[0,i])) == 0:
                indices.append(i)
        print("MPJPE-2D",mpjpe(outputs_2d[:,indices,:], targets_2d.float().to(device)[:,indices,:]).item())
        print()
        count += 1
        if count>=10:
            break

# if evaluate:
#     names = ['lh pose', 'rh pose']

#     for i in range(len(errors)):
#         avg_error = np.average(np.array(errors[i]))
#         print(f'{names[i]} average error on test set:', avg_error)

    # avg_error = np.average(np.array(errors))
    # print('Hand shape average error on validation set:', avg_error)

    # avg_rgb_error = np.average(np.array(rgb_errors))
    # print('Texture average error on validation set:', avg_rgb_error)

# save_dicts(output_dicts, args.split)