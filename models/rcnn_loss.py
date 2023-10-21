import torch
import torch.nn.functional as F
import os
from torchvision.ops import roi_align
from typing import Optional, List, Dict, Tuple
from torch import nn, Tensor
import cv2
import numpy as np
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
        trans = torch.cat((trans, torch.tensor([[0, 0, 1]], dtype=torch.float32).cuda(1)), dim=0)
    none_idx = torch.any(torch.isnan(kpts), dim=-1).cuda(1)
    kpts[none_idx] = 0
    kpts = torch.cat((kpts, torch.ones((kpts.shape[0],1)).cuda(1)), dim=1) 
    kpts = (trans @ kpts.T).T
    kpts[none_idx] = torch.nan
    return kpts

def aria_original_to_extracted(kpts, img_shape=(1408, 1408)):
    """
    Rotate kpts coordinates from original view (hand horizontal) to extracted view (hand vertical)
    img_shape is the shape of original view image
    """
    # assert len(kpts.shape) == 2, "Only can rotate 2D arrays"
    H, _ = img_shape
    #print((kpts == None).shape)
    #none_idx = torch.from_numpy(np.any(kpts == None, axis=1)).cuda()
    none_idx = torch.any(torch.isnan(kpts), dim=-1).cuda(1)
    kpts[~none_idx, 0], kpts[~none_idx, 1] = H - kpts[~none_idx, 1] - 1, kpts[~none_idx, 0]
    return kpts

def project_3D_points(cam_mat, pts3D, is_OpenGL_coords=True):
    '''
    Function for projecting 3d points to 2d
    :param camMat: camera matrix
    :param pts3D: 3D points
    :param isOpenGLCoords: If True, hand/object along negative z-axis. If False hand/object along positive z-axis
    :return:
    '''
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2
    # print(pts3D.shape, cam_mat.shape)

    proj_pts = pts3D @ cam_mat.T
    proj_pts = proj_pts[:,0:2]/proj_pts[:,2:]#torch.cat((proj_pts[:,0:1]/proj_pts[:,2:], proj_pts[:,1:2]/proj_pts[:,2:]),dim=1)

    #print(proj_pts.shape)
    proj_pts = aria_original_to_extracted(proj_pts,np.array([512,512]))

    assert len(proj_pts.shape) == 2

    return proj_pts

def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    #classification_loss = F.cross_entropy(class_logits, labels) #remove classification loss

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    
    sampled_pos_inds_subset = torch.where(labels > 0)[0].clone().cuda(1)
    
    labels_pos = labels[sampled_pos_inds_subset]
    
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
    
    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction='sum',
    )
    
    box_loss = box_loss / labels.numel()

    #return classification_loss, box_loss
    return box_loss

def keypoints_to_heatmap(keypoints, rois, heatmap_size):
    # type: (Tensor, Tensor, int) -> Tuple[Tensor, Tensor]
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()

    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()

    lin_ind = y * heatmap_size + x
    heatmaps = lin_ind * valid

    return heatmaps, valid

def heatmaps_to_keypoints(maps, rois):
    """Extract predicted keypoint locations from heatmaps. Output has shape
    (#rois, 4, #keypoints) with the 4 rows corresponding to (x, y, logit, prob)
    for each keypoint.
    """
    # This function converts a discrete image coordinate in a HEATMAP_SIZE x
    # HEATMAP_SIZE image to a continuous keypoint coordinate. We maintain
    # consistency with keypoints_to_heatmap_labels by using the conversion from
    # Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a
    # continuous coordinate.
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]

    widths = rois[:, 2] - rois[:, 0]
    heights = rois[:, 3] - rois[:, 1]
    widths = widths.clamp(min=1)
    heights = heights.clamp(min=1)
    widths_ceil = widths.ceil()
    heights_ceil = heights.ceil()

    num_keypoints = maps.shape[1]

    xy_preds = torch.zeros((len(rois), 3, num_keypoints), dtype=torch.float32, device=maps.device)
    end_scores = torch.zeros((len(rois), num_keypoints), dtype=torch.float32, device=maps.device)
    for i in range(len(rois)):
        roi_map_width = int(widths_ceil[i].item())
        roi_map_height = int(heights_ceil[i].item())
        width_correction = widths[i] / roi_map_width
        height_correction = heights[i] / roi_map_height
        roi_map = F.interpolate(
            maps[i][:, None], size=(roi_map_height, roi_map_width), mode='bicubic', align_corners=False)[:, 0]
        # roi_map_probs = scores_to_probs(roi_map.copy())
        w = roi_map.shape[2]
        pos = roi_map.reshape(num_keypoints, -1).argmax(dim=1)

        x_int = pos % w
        y_int = torch.div(pos - x_int, w, rounding_mode='floor')
        # assert (roi_map_probs[k, y_int, x_int] ==
        #         roi_map_probs[k, :, :].max())
        x = (x_int.float() + 0.5) * width_correction
        y = (y_int.float() + 0.5) * height_correction
        xy_preds[i, 0, :] = x + offset_x[i]
        xy_preds[i, 1, :] = y + offset_y[i]
        xy_preds[i, 2, :] = 1
        end_scores[i, :] = roi_map[torch.arange(num_keypoints, device=roi_map.device), y_int, x_int]

    return xy_preds.permute(0, 2, 1), end_scores

def keypointrcnn_loss(keypoint_logits, proposals, gt_keypoints, keypoint_matched_idxs, 
                    keypoint3d_pred=None, keypoint3d_gt=None, original_images=None, palms_gt=None, num_classes=4, dataset_name='h2o', wrist=None, cam_mat =None, trans=None):

    N, K, H, W = keypoint_logits.shape
    assert H == W
    discretization_size = H
    heatmaps = []
    valid = []
    kps3d = []
    images = []
    palms = []
    heatmaps_3d = []
    valid_3d = []
    if palms_gt is None:
        palms_gt = [None] * len(proposals)

    zipped_data = zip(proposals, gt_keypoints, keypoint3d_gt, original_images, keypoint_matched_idxs, palms_gt)
    # zipped_data = zip(proposals, gt_keypoints, keypoint_matched_idxs, original_images)
    for proposals_per_image, gt_kp_in_image, gt_kp3d_in_image, image, midx, palm_in_image in zipped_data:
        kp = gt_kp_in_image[midx]

        if palm_in_image is not None:
            palm = palm_in_image[midx]
            palms.append(palm.view(-1))

        num_regions = midx.shape[0]
        
        heatmaps_per_image, valid_per_image = keypoints_to_heatmap(kp, proposals_per_image, discretization_size)

        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))
        
        kp3d = gt_kp3d_in_image[midx]
        kps3d.append(kp3d.view(-1))

        for m in midx:
            keypoint_proj = project_3D_points(cam_mat[m].float(), (kp3d[0].clone()+wrist[m].float())/1000, is_OpenGL_coords=False)
            kp_pseudo = affine_transform(keypoint_proj, trans[m].float())

            heatmaps_per_image_3d, valid_per_image_3d = keypoints_to_heatmap(kp_pseudo.unsqueeze(0), proposals_per_image, discretization_size)

            heatmaps_3d.append(heatmaps_per_image_3d.view(-1))
            valid_3d.append(valid_per_image_3d.view(-1))

        images.extend([image] * num_regions)
  
    keypoint_targets = torch.cat(heatmaps, dim=0)
    valid = torch.cat(valid, dim=0).to(dtype=torch.uint8)    
    valid = torch.where(valid)[0]
    
    # torch.mean (in binary_cross_entropy_with_logits) does'nt
    # accept empty tensors, so handle it sepaartely
    
    if keypoint_targets.numel() == 0 or len(valid) == 0:
        print("ISSUE", len(valid))
        return keypoint_logits.sum() * 0, keypoint_logits.sum() * 0

    keypoint_logits = keypoint_logits.view(N * K, H * W)

    # Heatmap Loss
    mask = ~torch.isnan(keypoint_targets[valid])
    if len(keypoint_targets[valid][mask]) == 0:
        keypoint_loss = torch.tensor(0, dtype = torch.float32).cuda(1)
    else:
        keypoint_loss = F.cross_entropy(keypoint_logits[valid][mask], keypoint_targets[valid][mask])

    keypoint_targets_pseudo = torch.cat(heatmaps_3d, dim=0).float()

    if keypoint_targets_pseudo.numel() == 0 or len(valid_3d) == 0:
        print("ISSUE", len(valid_3d))
        return 0,0

    # Heatmap Loss
    mask = ~torch.isnan(keypoint_targets[valid])
    if len(keypoint_targets[valid][mask]) == 0:
        rcnn_loss_proj = torch.tensor(0, dtype = torch.float32).cuda(1)
    else:
        rcnn_loss_proj = F.cross_entropy(keypoint_targets_pseudo[valid][mask], keypoint_targets[valid][mask].float())
    rcnn_loss_proj = torch.tensor(0, dtype = torch.float32).cuda(1)

    keypoint3d_pred = keypoint3d_pred.view(N * K, 3)

    # 3D pose Loss
    keypoint3d_targets = torch.cat(kps3d, dim=0).view(N, K, 3)

    N, K, D = keypoint3d_targets.shape
    keypoint3d_pred = keypoint3d_pred.view(N * K, 3)
    keypoint3d_targets = keypoint3d_targets.view(N * K, 3)
    mask = ~torch.isnan(keypoint3d_targets)
    if len(keypoint3d_targets[mask]) == 0:
        keypoint3d_loss = torch.tensor(0, dtype = torch.float32).cuda(1)
    else:
        keypoint3d_loss = F.mse_loss(keypoint3d_pred[mask], keypoint3d_targets[mask]) / 1000
    # Print the losses
   

    return keypoint_loss, keypoint3d_loss, rcnn_loss_proj

def keypointrcnn_inference(x, boxes):
    # type: (Tensor, List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
    kp_probs = []
    kp_scores = []

    boxes_per_image = [box.size(0) for box in boxes]
    x2 = x.split(boxes_per_image, dim=0)

    for xx, bb in zip(x2, boxes):
        kp_prob, scores = heatmaps_to_keypoints(xx, bb)
        kp_probs.append(kp_prob)
        kp_scores.append(scores)

    return kp_probs, kp_scores

def append_rois_shapes(keypoint_proposals, image_shapes, kps, scale):
    rois_with_shapes = []

    for i, p in enumerate(keypoint_proposals):
        n_rois = p.shape[0]
        img_shape = torch.Tensor(image_shapes[i]).unsqueeze(axis=0).repeat(n_rois, 1).to(p.device)
        roi_with_shape = torch.cat((p, img_shape), axis=1)
        rois_with_shapes.append(roi_with_shape)

    rois_tensor = torch.cat(rois_with_shapes, dim=0).unsqueeze(axis=1)
    rois_tensor = rois_tensor.repeat(1, kps, 1) / scale
    
    return rois_tensor

def filter_rois(keypoint_proposals, training, labels=None):
    new_keypoint_proposals = []

    if training:
        for i in range(len(keypoint_proposals)):
            new_keypoint_proposals.append(keypoint_proposals[i][-3:])

    else:
        for i in range(len(keypoint_proposals)):
            
            labels_list = labels[i].tolist()
            if not set([1, 2, 3]).issubset(labels_list):
                return None
            lh_roi = keypoint_proposals[i][labels_list.index(1)]
            rh_roi = keypoint_proposals[i][labels_list.index(2)]
            obj_roi = keypoint_proposals[i][labels_list.index(3)]
            rois = torch.stack([lh_roi, rh_roi, obj_roi], dim=0)
            new_keypoint_proposals.append(rois)

    return new_keypoint_proposals

