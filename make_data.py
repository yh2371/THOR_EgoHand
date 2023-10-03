import os
import numpy as np
import pickle
import torch
import cv2
import os
import argparse
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from manopth.manolayer import ManoLayer
#import joblib
import glob
from tqdm import tqdm
import json

# Change this path

# Input parameters
parser = argparse.ArgumentParser()

# Loading dataset    
parser.add_argument("--root", required=True, help="HO3D dataset folder")
#parser.add_argument("--mano_root", required=True, help="Path to MANO models")
#parser.add_argument("--YCBModelsDir", default='./datasets/ycb_models', help="Path to YCB object meshes folder")
parser.add_argument("--dataset_path", default='./datasets/ho3d', help="Where to store dataset files")

args = parser.parse_args()

#evaluation = os.path.join(root, 'evaluation')
#train = os.path.join(root, 'training')

def load_json_files(path = "./", split = 'training'):
  with open(f'{path}/{split}_K.json') as K_fp, open(f'{path}/{split}_xyz.json') as xyz_fp:
    K_array = json.load(K_fp)
    xyz_array = json.load(xyz_fp)
  return K_array, xyz_array


def projectPoints(xyz,K):
  xyz = np.array(xyz)
  K = np.array(K)
  uv = np.matmul(K, xyz.T).T
  return uv[:,:2]/uv[:,-1:]
  
def project_batch(xyz_array, K_array):
  uv_array = []
  for i in tqdm(range(len(K_array))):
    uv = projectPoints(xyz_array[i], K_array[i]).astype(np.int32)
    uv_array.append(uv)
  return uv_array
  
def output(pth, split, img, xyz, uv):
  with open(os.path.join(pth,f'{split}_images.npy'), 'wb') as f1, open(os.path.join(pth,f'{split}_points3d.npy'), 'wb') as f2, open(os.path.join(pth,f'{split}_points2d.npy'), 'wb') as f3:
    np.save(f1, img)
    np.save(f2, np.array(xyz))
    np.save(f3, np.array(uv))
  total = len(img)
  print(f"{split} set has {total} samples.") 

  
if __name__ == '__main__':
  
  train_K, train_xyz = load_json_files(args.root, "training")
  val_K, val_xyz = load_json_files(args.root, "evaluation")
  
  train_uv = project_batch(train_xyz, train_K)
  val_uv = project_batch(val_xyz, val_K)
  
  train_img = list(sorted(glob.glob(os.path.join(args.root, "training/rgb/*.jpg"))))[:len(train_K)] #assume green screen version only
  #print(len(train_img))
  #print(train_img[-4:])
  val_img = list(sorted(glob.glob(os.path.join(args.root, "evaluation/rgb/*.jpg"))))
  
  output(args.dataset_path, "train", train_img, train_xyz, train_uv)
  output(args.dataset_path, "val", val_img, val_xyz, val_uv)

  
 
