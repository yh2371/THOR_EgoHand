import glob
import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import imageio
import torch
import matplotlib.pyplot as plt
from projectaria_tools.core import calibration
from torch.utils.data import Dataset
import cv2
import copy
from .rcnn_utils import calculate_bounding_box, create_rcnn_data


class ego4dDataset(Dataset):
    """
    Load Ego4D dataset with only Ego(Aria) images for 3D hand pose estimation
    Reference: https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
    """
    def __init__(self, cfg, anno_type, split, transform=None, use_preset=False, load_2d=True):
        self.dataset_root = cfg.DATASET.ROOT
        self.load_2d = load_2d
        self.anno_type = anno_type
        self.split = split
        self.num_joints = cfg.MODEL.NUM_JOINTS                          # Number of joints for one single hand
        self.pixel_std = 200                                            # Pixel std to define scale factor for image resizing
        self.undist_img_dim = np.array(cfg.DATASET.ORIGINAL_IMAGE_SIZE) # Size of undistorted aria image
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)                # Size of input image to the model
        gt_anno_path = os.path.join(self.dataset_root, 
                                    "ego4d_baseline_data", 
                                    "annotation", 
                                    self.anno_type, 
                                    f"ego_pose_gt_anno_{self.split}_public.json")
        # self.img_dir = os.path.join(self.dataset_root, 'image', split)
        # TODO: uncomment to use relative path e.g. [img_root]/image/test
        self.img_dir = "/mnt/volume2/Data/Ego4D/aria_undistorted_images" 
        self.cam_pose_dir = os.path.join(self.dataset_root, 'annotations/ego_pose/hand/camera_pose')
        self.db = self.load_all_data(gt_anno_path)
        self.pred_temp = self.generate_pred_temp(gt_anno_path)
        self.transform = transform
        # For 2D reprojected keypoint gt
    def __getitem__(self, idx):
        """
        Return transformed images, normalized & offset 3D hand GT pose, valid hand joint flag and metadata.
        """
        curr_db = copy.deepcopy(self.db[idx])

        # Define parameters for affine transformation of hand image
        c, s = xyxy2cs(*curr_db['bbox'], self.undist_img_dim, self.pixel_std)
        r = 0
        trans = get_affine_transform(c, s, r, self.image_size)
        # Load image
        metadata = curr_db['metadata']
        img_path = os.path.join(self.img_dir, f"{metadata['take_name']}", f"{metadata['frame_number']:06d}.jpg")
        img = imageio.imread(img_path, pilmode='RGB')
        # Get affine transformed hand image
        input = cv2.warpAffine(
            img,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        # Apply Pytorch transform if needed
        if self.transform:
            input = self.transform(input)

        if self.split.lower() != 'test':
            # Load ground truth 3D hand joints and valid flag info for train and val, omit for test
            curr_3d_kpts_cam = curr_db['joints_3d']
            curr_3d_kpts_cam = curr_3d_kpts_cam * 1000 # m to mm
            curr_3d_kpts_cam_offset = curr_3d_kpts_cam - curr_3d_kpts_cam[0]
            curr_3d_kpts_cam_offset[~curr_db['valid_flag']] = None
            curr_3d_kpts_cam_offset = torch.from_numpy(curr_3d_kpts_cam_offset.astype(np.float32))

            # Generate valid joints flag
            vis_flag = torch.from_numpy(curr_db['valid_flag'])

            # Optional 2D annotation loading
            if self.load_2d:
                curr_2d_kpts = curr_db['joints_2d']
                curr_2d_kpts = affine_transform(curr_2d_kpts, trans)
                curr_2d_kpts = torch.from_numpy(curr_2d_kpts.astype(np.float32))
            else:
                curr_2d_kpts = torch.Tensor([])

            #print(curr_db['bbox'])
            #print(np.array(curr_db['bbox']).reshape((2,2)))
            bb = affine_transform(np.array(curr_db['bbox']).reshape((2,2)), trans)[:,:2].reshape((4,))
            #print(bb)
            #.reshape((4,))
            boxes, labels, keypoints, keypoints3d = create_rcnn_data(bb, curr_2d_kpts, curr_3d_kpts_cam_offset, num_keypoints=self.num_joints, vis_flag = vis_flag)
        else:
            # Return only input if split is test
            bb = np.array([])
            boxes, labels, keypoints, keypoints3d = torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([])
            curr_2d_kpts = torch.Tensor([])
            curr_3d_kpts_cam_offset = torch.Tensor([])
            vis_flag = torch.Tensor([])
        
        data = {
            'original_image': img,
            'inputs': input,
            'point2d': curr_2d_kpts,
            'point3d': curr_3d_kpts_cam_offset,
            'bb': bb,
            'boxes': boxes,
            'labels': labels,
            'keypoints': keypoints,
            'keypoints3d': keypoints3d,
            'valid': vis_flag,
            'meta': metadata
        }
        
        return data
        #return input, curr_3d_kpts_cam_offset, vis_flag, meta


    def __len__(self):
        return len(self.db)


    def load_all_data(self, gt_anno_path):
        """
        Store each valid hand's annotation per frame separately, with 
        dict key based on split:
        Train & val:
            - joints_3d
            - valid_flag
            - bbox
            - metadata
        Test:
            - bbox
            - metadata
        """
        # Load ground truth annotation
        gt_anno = json.load(open(gt_anno_path))

        # Load gt annotation for train & val
        if self.split in ["train", "val"]:
            all_frame_anno = []
            for _, curr_take_anno in tqdm(gt_anno.items()):
                curr_take_uid = None
                for _, curr_f_anno in curr_take_anno.items():
                    # Load cam pose for target take
                    if curr_take_uid is None:
                        curr_take_uid = curr_f_anno["metadata"]['take_uid']
                        curr_take_cam_pose_path = os.path.join(self.cam_pose_dir, f"{curr_take_uid}.json")
                        cam_pose = json.load(open(curr_take_cam_pose_path))
                    for hand_order in ["right", "left"]:
                        single_hand_anno = {}
                        if len(curr_f_anno[f"{hand_order}_hand"]) != 0:
                            single_hand_anno["joints_3d"] = np.array(curr_f_anno[f"{hand_order}_hand"])
                            single_hand_anno["valid_flag"] = np.array(curr_f_anno[f"{hand_order}_hand_valid"])
                            single_hand_anno["bbox"] = np.array(curr_f_anno[f"{hand_order}_hand_bbox"])
                            single_hand_anno["metadata"] = curr_f_anno["metadata"]

                            # Optional 2D annotation loading
                            if self.load_2d:
                                frame_idx = str(curr_f_anno["metadata"]['frame_number'])
                                # Check if current frame has corresponding camera pose
                                if frame_idx not in cam_pose.keys() or 'aria01' not in cam_pose[frame_idx].keys():
                                    curr_intri = None
                                else:
                                    # Build camera projection matrix
                                    curr_intri = np.array(cam_pose[frame_idx]['aria01']['camera_intrinsics'])
                                one_hand_2d_kpts_original = cam_to_img(single_hand_anno["joints_3d"], curr_intri)
                                one_hand_2d_kpts_extracted = aria_original_to_extracted(one_hand_2d_kpts_original, self.undist_img_dim)
                                single_hand_anno["joints_2d"] = np.array(one_hand_2d_kpts_extracted)

                            all_frame_anno.append(single_hand_anno)
        # Load un-annotated test JSON file for evaluation
        else:
            all_frame_anno = []
            for _, curr_take_anno in gt_anno.items():
                for _, curr_f_anno in curr_take_anno.items():
                    for hand_order in ["right", "left"]:
                        single_hand_anno = {}
                        if len(curr_f_anno[f"{hand_order}_hand_bbox"]) != 0:
                            # Load bbox regardless of whether it's empty or not
                            single_hand_anno["bbox"] = np.array(curr_f_anno[f"{hand_order}_hand_bbox"])
                            single_hand_anno["metadata"] = copy.deepcopy(curr_f_anno["metadata"])
                            single_hand_anno["metadata"]["hand_order"] = hand_order
                            all_frame_anno.append(single_hand_anno)
        return all_frame_anno
    
    def load_cam_intrinsics(self, metadata):
        curr_take_uid = metadata['take_uid']
        frame_idx = str(metadata['frame_number'])
        curr_take_cam_pose_path = os.path.join(self.cam_pose_dir, f"{curr_take_uid}.json")
        cam_pose = json.load(open(curr_take_cam_pose_path))
        frame_idx = str(metadata['frame_number'])
        # Check if current frame has corresponding camera pose
        if frame_idx not in cam_pose.keys() or 'aria01' not in cam_pose[frame_idx].keys():
            return None
        # Build camera projection matrix
        curr_cam_intrinsic = np.array(cam_pose[frame_idx]['aria01']['camera_intrinsics'])
        return curr_cam_intrinsic


    def generate_pred_temp(self, gt_anno_path):
        """
        Generate empty prediction template with specicifed JSON format:
        {
            "<take_uid>": {
                "frame_number": {
                        "left_hand":  [],
                        "right_hand": []       
                }
            }
        }
        """
        # Load ground truth annotation
        gt_anno = json.load(open(gt_anno_path))
        pred_template = copy.deepcopy(gt_anno)
        # Create empty template for each frame in each take
        pred_temp = {}
        for take_uid, take_anno in gt_anno.items():
            curr_take_pred_temp = {}
            for frame_number in take_anno.keys():
                curr_frame_pred_temp = {
                    "left_hand": [],
                    "right_hand": []
                }
                curr_take_pred_temp[str(frame_number)] = curr_frame_pred_temp
            pred_temp[take_uid] = curr_take_pred_temp
        return pred_temp


    def init_split(self):
        # Get tain/val/test df
        train_df = self.takes_df[self.takes_df['split']=='TRAIN']
        val_df = self.takes_df[self.takes_df['split']=='VAL']
        test_df = self.takes_df[self.takes_df['split']=='TEST']
        # Get train/val/test uid
        all_train_uid = list(train_df['take_uid'])
        all_val_uid = list(val_df['take_uid'])
        all_test_uid = list(test_df['take_uid'])
        return {'train':all_train_uid, 'val':all_val_uid, 'test':all_test_uid}


def xyxy2cs(x1, y1, x2, y2, img_shape, pixel_std):
    aspect_ratio = img_shape[1] * 1.0 / img_shape[0]

    center = np.zeros((2), dtype=np.float32)
    center[0] = (x1 + x2) / 2
    center[1] = (y1 + y2) / 2

    w = x2 - x1
    h = y2 - y1

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std],dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_dir(src_point, rot_rad):
    """Rotate the point by `rot_rad` degree."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    """Return vector c that perpendicular to (a - b)."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def cam_to_img(kpts, intri):
    """
    Project points in camera coordinate system to image plane
    Input:
        kpts: (N,3)
    Output:
        new_kpts: (N,2)
    """
    none_idx = np.any(np.isnan(kpts), axis=1)
    new_kpts = kpts.copy()
    new_kpts[none_idx] = -1
    new_kpts = intri @ new_kpts.T # (3,N)
    new_kpts = new_kpts / new_kpts[2,:]
    new_kpts = new_kpts[:2,:].T
    new_kpts[none_idx] = None
    return new_kpts

def aria_original_to_extracted(kpts, img_shape=(1408, 1408)):
    """
    Rotate kpts coordinates from original view (hand horizontal) to extracted view (hand vertical)
    img_shape is the shape of original view image
    """
    # assert len(kpts.shape) == 2, "Only can rotate 2D arrays"
    H, _ = img_shape
    none_idx = np.any(np.isnan(kpts), axis=1)
    new_kpts = kpts.copy()
    new_kpts[~none_idx, 0] = H - kpts[~none_idx, 1] - 1
    new_kpts[~none_idx, 1] = kpts[~none_idx, 0]
    return new_kpts

def affine_transform(kpts, trans):
    """
    Affine transformation of 2d kpts
    Input:
        kpts: (N,2)
        trans: (3,3)
    Output:
        new_kpts: (N,2)
    """
    if trans.shape[0] == 2:
        trans = np.concatenate((trans, [[0,0,1]]), axis=0)
    new_kpts = kpts.copy()
    none_idx = np.any(np.isnan(new_kpts), axis=1)
    new_kpts[none_idx] = 0
    new_kpts = np.append(new_kpts, np.ones((new_kpts.shape[0], 1)), axis=1)
    new_kpts = (trans @ new_kpts.T).T
    new_kpts[none_idx] = None
    return new_kpts