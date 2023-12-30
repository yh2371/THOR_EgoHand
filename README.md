# THOR-Net - Ego Hand Pose Estimation

This repo contains a modified version of the PyTorch implementation for **THOR-Net: End-to-end Graformer-based Realistic Two Hands and Object Reconstruction with Self-supervision** published in WACV 2023 [[Paper](https://openaccess.thecvf.com/content/WACV2023/html/Aboukhadra_THOR-Net_End-to-End_Graformer-Based_Realistic_Two_Hands_and_Object_Reconstruction_With_WACV_2023_paper.html), [Video](https://www.youtube.com/watch?v=TLPvs1shMAM&t=240s), [Poster](https://video.vast.uccs.edu/WACV23/1967-wacv-post.pdf)]

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
