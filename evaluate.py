import argparse
import os
import json
import cv2
import numpy as np
import torch
import time
from easydict import EasyDict as edict
import torchvision.transforms as transforms
from tqdm import tqdm

"""
Performs ego-pose model performance evaluation and report MPJPE & PA-MPJPE as metric
Input:
    - private validation or test annotated JSON file
    - model inference output JSON file
"""

def parse_args_function():
    parser = argparse.ArgumentParser(
        """
        Performs ego-pose model performance evaluation and report MPJPE & PA-MPJPE as metric. The GT annotation
        is original 3D hand kpts (in Aria cam coordinate system) without offset. Please add --offset if the user 
        inference result is offset by hand wrist.
        """
    )

    parser.add_argument(
        "--pred_path",
        help="Path of user inference prediction JSON file"
    )
    parser.add_argument(
        "--gt_path",
        help="Path of ground truth annotation JSON file"
    )
    parser.add_argument(
        "--offset",
        action="store_true",
        help="Whether the user inference result is offset by hand wrist"
    )

    args = parser.parse_args()
    return args


def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance) from 
    https://github.com/zhaoweixi/GraFormer/blob/main/common/loss.py.
    Modified s.t. it could compute MPJPE for only those valid keypoints (where 
    # of visible keypoints = num)
    """
    assert predicted.shape == target.shape
    pjpe = torch.norm(predicted - target, dim=len(target.shape) - 1)
    mpjpe = torch.mean(pjpe)
    return mpjpe


def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    # Convert to Numpy because this metric needs numpy array
    predicted = predicted.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    pjpe = np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)
    p_mpjpe = np.mean(pjpe)
    return p_mpjpe


def main(args):
    # Load pred and gt to compute metric
    pred_anno = json.load(open(args.pred_path))
    gt_anno = json.load(open(args.gt_path))

    ######### Inferece #########
    epoch_loss_3d_pos = []
    epoch_loss_3d_pos_procrustes = []

    for take_uid, take_anno in gt_anno.items():
        for frame_number, curr_frame_gt_anno in take_anno.items():
            for hand_order in ["right", "left"]:
                # Only evaluate valid GT hand
                if len(curr_frame_gt_anno[f"{hand_order}_hand"]) != 0:
                    # Get GT 3D hand joints
                    gt_3d_kpts = np.array(curr_frame_gt_anno[f"{hand_order}_hand"]) * 1000
                    # Get 3D hand joints prediction
                    curr_frame_pred = np.array(pred_anno[take_uid][frame_number][f"{hand_order}_hand"]) * 1000
                    # Add back hand wrist if prediction is offset by hand wrist
                    if args.offset:
                        curr_frame_pred += gt_3d_kpts[0]
                    # Get valid flag
                    vis_flag = torch.from_numpy(np.array(curr_frame_gt_anno[f"{hand_order}_hand_valid"]))
    
                    # Compute MPJPE and PA-MPJPE
                    valid_pred_3d_kpts = torch.from_numpy(curr_frame_pred)
                    valid_pred_3d_kpts = valid_pred_3d_kpts[vis_flag].view(1,-1,3)
                    valid_pose_3d_gt = torch.from_numpy(gt_3d_kpts)
                    valid_pose_3d_gt = valid_pose_3d_gt[vis_flag].view(1,-1,3)
                    epoch_loss_3d_pos.append(mpjpe(valid_pred_3d_kpts, valid_pose_3d_gt).item())
                    epoch_loss_3d_pos_procrustes.append(p_mpjpe(valid_pred_3d_kpts, valid_pose_3d_gt))

    epoch_loss_3d_pos_avg = np.mean(epoch_loss_3d_pos)
    epoch_loss_3d_pos_procrustes_avg = np.mean(epoch_loss_3d_pos_procrustes)
    print(f'MPJPE: {epoch_loss_3d_pos_avg:.2f} (mm)')
    print(f'P-MPJPE: {epoch_loss_3d_pos_procrustes_avg:.2f} (mm)')


if __name__ == '__main__':
    args = parse_args_function()
    main(args)
