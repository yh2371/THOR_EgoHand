# THOR-Net - Ego-Exo4D Hand Pose Estimation Baseline Model

A modified version of the PyTorch implementation for **THOR-Net: End-to-end Graformer-based Realistic Two Hands and Object Reconstruction with Self-supervision** [Original Repo](https://github.com/ATAboukhadra/THOR-Net) used in the [Ego-Exo4D](https://github.com/facebookresearch/Ego4d) hand ego pose benchmark.

## Data preparation
Follow instructions [here](https://github.com/EGO4D/ego-exo4d-egopose/tree/main/handpose/data_preparation) to get:
- ground truth annotation files in `$gt_output_dir/annotation/manual` or `$gt_output_dir/annotation/auto` if using automatic annotations,
referred as `gt_anno_dir` below
- corresponding undistorted Aria images in `$gt_output_dir/image/undistorted`, 
referred as `aria_img_dir` below

## Setup

Install the dependencies listed in `conda env create -f environment.yml`

## Training
To train THOR-Net on automatic annotations:
``` bash
./scripts/train_auto.sh
```
To train THOR-Net on manual annotations:
``` bash
./scripts/train_manual.sh
```
To finetune THOR-Net on manual annotations:
``` bash
./scripts/finetune_auto.sh
```

## Evaluation
Download pretrained ([EvalAI baseline](https://eval.ai/web/challenges/challenge-page/2249/overview)) model weights from [here](https://drive.google.com/drive/folders/17FllgdZuFrtR1KlFQXyyQzVyivqofDIW?usp=sharing).

To obtain inference results:
``` bash
./scripts/inference.sh
```
To evaluate inference results:
``` bash
./scripts/eval.sh
```

## Citation 

```
@InProceedings{Aboukhadra_2023_WACV,
    author    = {Aboukhadra, Ahmed Tawfik and Malik, Jameel and Elhayek, Ahmed and Robertini, Nadia and Stricker, Didier},
    title     = {THOR-Net: End-to-End Graformer-Based Realistic Two Hands and Object Reconstruction With Self-Supervision},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {1001-1010}
}
```
