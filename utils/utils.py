import torch
import numpy as np
import pickle
import torchvision.transforms as transforms

from .h2o_utils.h2o_datapipe_pt_1_12 import create_datapipe
#from .dataset import Dataset
from .ego4d_dataset import ego4dDataset
    

def collate_fn(batch):
    return batch

def create_loader(split, batch_size, anno_type ='manual', cfg = None):

    transform = transforms.Compose([transforms.ToTensor()]) #to tensor transformation
    dataset = ego4dDataset(cfg, anno_type=anno_type, split = split, transform = transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)      
        
    return loader

def freeze_component(model):
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))

def save_dicts(output_dicts, split):
    
    output_dict = dict(sorted(output_dicts[0].items()))
    output_dict_mesh = dict(sorted(output_dicts[1].items()))
    print('Total number of predictions:', len(output_dict.keys()))

    with open(f'./outputs/rcnn_outputs/rcnn_outputs_21_{split}_3d_v3.pkl', 'wb') as f:
        pickle.dump(output_dict, f)

    with open(f'./outputs/rcnn_outputs/rcnn_outputs_778_{split}_3d_v3.pkl', 'wb') as f:
        pickle.dump(output_dict_mesh, f)

def get_p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    # Convert to Numpy because this metric needs numpy array
    # predicted = predicted.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    mask = ~np.isnan(target)
    if len(target[mask]) == 0:
        return predicted 
    orig = predicted
    predicted = predicted[mask].reshape((1,-1,3))
    target = target[mask].reshape((1,-1,3))
    
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
    predicted_aligned = a * np.matmul(orig, R) + t
    
    return predicted_aligned

def prepare_data_for_evaluation(data_dict, outputs, img, keys, device, split, mean = None, std = None):
    """Postprocessing function"""

    targets = [{k: v.to(device) for k, v in t.items() if k in keys} for t in data_dict]
    labels = {k: v.cpu().detach().numpy() for k, v in targets[0].items()}
    predictions = {k: v.cpu().detach().numpy() for k, v in outputs[0].items()}

    predictions['keypoints3d'] =  get_p_mpjpe(predictions['keypoints3d'],targets[0]['keypoints3d']) #* (std).detach().cpu().numpy() + mean.detach().cpu().numpy()
    targets[0]['keypoints3d'] =  targets[0]['keypoints3d'] #* (std)+ mean

    if split == 'test':
        labels = None

    img = img.transpose(1, 2, 0) * 255
    img = np.ascontiguousarray(img, np.uint8) 

    return predictions, img, palm, labels

def project_3D_points(pts3D):

    cam_mat = np.array(
        [[617.343,0,      312.42],
        [0,       617.343,241.42],
        [0,       0,       1]])

    proj_pts = pts3D.dot(cam_mat.T)
    proj_pts = np.stack([proj_pts[:,0] / proj_pts[:,2], proj_pts[:,1] / proj_pts[:,2]], axis=1)
    return proj_pts
