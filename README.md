# TwoStageAlign
The official codes of our CVPR2022 paper: A Differentiable Two-stage Alignment Scheme for Burst Image Reconstruction with Large Shift

[Paper](https://arxiv.org/pdf/2203.09294.pdf) | [Supp](https://github.com/GuoShi28/2StageAlign/blob/main/03788-supp.pdf)

## Abstract
Denoising and demosaicking are two essential steps to reconstruct a clean full-color image from the raw data. Recently, joint denoising and demosaicking (JDD) for burst images, namely JDD-B, has attracted much attention by using multiple raw images captured in a short time to reconstruct a single high-quality image. One key challenge of JDD-B lies in the robust alignment of image frames. State-of-the-art alignment methods in feature domain cannot effectively utilize the temporal information of burst images, where large shifts commonly exist due to camera and object motion. In addition, the higher resolution (e.g., 4K) of modern imaging devices results in larger displacement between frames. To address these challenges, we design a differentiable two-stage alignment scheme sequentially in patch and pixel level for effective JDD-B. The input burst images are firstly aligned in the patch level by using a differentiable progressive block matching method, which can estimate the offset between distant frames with small computational cost. Then we perform implicit pixel-wise alignment in full-resolution feature domain to refine the alignment results. The two stages are jointly trained in an end-to-end manner. Extensive experiments demonstrate the significant improvement of our method over existing JDD-B methods.

## Framework
![Framework of 2 Stage Align](https://github.com/GuoShi28/2StageAlign/blob/main/network.png)

## Test
[Pretrain models](https://github.com/GuoShi28/2StageAlign/tree/main/experiments/J0007_JDDB_PBMNet_wgr_ncc_gt_s2h/models)

#### REDS4

* we only put an example of REDS4 in dataset folder, please download the full testset in official website, [RED](https://seungjunnah.github.io/Datasets/reds.html). 
* More detail can refer to [data preparation](https://github.com/xinntao/EDVR/blob/master/docs/DatasetPreparation.md)

```
python /codes/test_Vid4_REDS4_joint_2stage_REDS4.py
```

#### Videezy

* To evaluate the performance on 4K burst images/video, we collect several clips from website. 
* Dataset: [Google Drive](https://drive.google.com/file/d/1YDljUONvyKUO24smTx__CUH_4Zxhle09/view?usp=sharing)

```
python /codes/test_Vid4_REDS4_joint_2stage_Videezy4K.py
```

#### SC_burst (Smartphone burst) Dataset

* Please refer to [GCP-Net](https://github.com/GuoShi28/GCP-Net). 
* Whole dataset: [BaiduYun](https://pan.baidu.com/s/1gRQ1im6Qa7vZiuOv9eO2Qw) with password d8u8.

```
python /codes/test_Vid4_REDS4_joint_2stage_RealCaptured.py
```

## Train
* training data preparation: Please refer to the "Video Super-Resolution" part of [data preparation](https://github.com/xinntao/EDVR/blob/master/docs/DatasetPreparation.md). To create LMDB dataset, please run [create_lmdb.py](https://github.com/GuoShi28/2StageAlign/blob/main/codes/data_scripts/create_lmdb.py).

* change training options in [train_burst_JDD_2stage.yml](https://github.com/GuoShi28/2StageAlign/blob/main/codes/options/train/train_burst_JDD_2stage.yml)

```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4540 train.py -opt options/train/train_GCP_Net.yml --launcher pytorch
```

## Environment
* Refer to the [requirement.txt](https://github.com/GuoShi28/2StageAlign/blob/main/requirements.txt)
* We utilize pytorch 1.2 and the deformable version does not support pytorch > 1.3. Thus when you use newest pytorch, please replace deformable version to newest (refer to [BasicSR](https://github.com/xinntao/BasicSR)). 

## Citation
```
@article{guo2022differentiable,
  title={A Differentiable Two-stage Alignment Scheme for Burst Image Reconstruction with Large Shift},
  author={Guo, Shi and Yang, Xi and Ma, Jianqi and Ren, Gaofeng and Zhang, Lei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

## Acknowledgement
This repo is built upon the framework of [EDVR](https://github.com/xinntao/EDVR), and we borrow some code from [Unprocessing denoising](https://github.com/timothybrooks/unprocessing), thanks for their excellent work!
