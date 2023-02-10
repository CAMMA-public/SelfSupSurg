<div align="center">
<a href="http://camma.u-strasbg.fr/">
<img src="static/camma_logo_tr.png" width="30%">
</a>
</div>


## **Dissecting Self-Supervised Learning Methods for Surgical Computer Vision**

_Sanat Ramesh, Vinkle Srivastav, Deepak Alapatt, Tong Yu, Aditya Murali, Luca Sestini, Chinedu Innocent Nwoye, Idris Hamoud, Saurav Sharma, Antoine Fleurentin, Georgios Exarchakis, Alexandros Karargyris, Nicolas Padoy_, 2022

[![arXiv](https://img.shields.io/badge/arxiv-2207.00449-red)](https://arxiv.org/abs/2207.00449)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dissecting-self-supervised-learning-methods/surgical-phase-recognition-on-cholec80-1)](https://paperswithcode.com/sota/surgical-phase-recognition-on-cholec80-1?p=dissecting-self-supervised-learning-methods)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dissecting-self-supervised-learning-methods/surgical-tool-detection-on-cholec80)](https://paperswithcode.com/sota/surgical-tool-detection-on-cholec80?p=dissecting-self-supervised-learning-methods)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dissecting-self-supervised-learning-methods/semantic-segmentation-on-endoscapes)](https://paperswithcode.com/sota/semantic-segmentation-on-endoscapes?p=dissecting-self-supervised-learning-methods)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dissecting-self-supervised-learning-methods/surgical-tool-detection-on-heichole-benchmark)](https://paperswithcode.com/sota/surgical-tool-detection-on-heichole-benchmark?p=dissecting-self-supervised-learning-methods)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dissecting-self-supervised-learning-methods/action-triplet-recognition-on-cholect50-1)](https://paperswithcode.com/sota/action-triplet-recognition-on-cholect50-1?p=dissecting-self-supervised-learning-methods)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dissecting-self-supervised-learning-methods/surgical-phase-recognition-on-heichole)](https://paperswithcode.com/sota/surgical-phase-recognition-on-heichole?p=dissecting-self-supervised-learning-methods)

### Introduction
<div style="text-align: left">
The field of surgical computer vision has undergone considerable breakthroughs in recent years with the rising popularity of deep neural network-based methods. However, standard fully-supervised approaches for training such models require vast amounts of annotated data, imposing a prohibitively high cost; especially in the clinical domain. Self-Supervised Learning (SSL) methods, which have begun to gain traction in the general computer vision community, represent a potential solution to these annotation costs, allowing to learn useful representations from only unlabeled data. Still, the effectiveness of SSL methods in more complex and impactful domains, such as medicine and surgery, remains limited and unexplored. In this work, we address this critical need by investigating four state-of-the-art SSL methods (MoCo v2, SimCLR, DINO, SwAV) in the context of surgical computer vision. We present an extensive analysis of the performance of these methods on the Cholec80 dataset for two fundamental and popular tasks in surgical context understanding, phase recognition and tool presence detection. We examine their parameterization, then their behavior with respect to training data quantities in semi-supervised settings. Correct transfer of these methods to surgery, as described and conducted in this work, leads to substantial performance gains over generic uses of SSL - up to 7.4% on phase recognition and 20% on tool presence detection - as well as state-of-the-art semi-supervised phase recognition approaches by up to 14%. Further results obtained on a highly diverse selection of surgical datasets exhibit strong generalization properties.
</div>
<p float="center"> <img src="static/arch_720p.gif" width="90%" /> </p>

## Main takeaways from the paper
**[1]** Benchmarking of four state-of-the-art SSL methods ( [**MoCo v2**](https://arxiv.org/abs/2003.04297), [**SimCLR**](https://arxiv.org/abs/2002.05709), [**SwAV**](https://arxiv.org/abs/2006.09882), and [**DINO**](https://arxiv.org/abs/2104.14294)) in the surgical domain.

**[2]** Thorough experimentation (**∼200** experiments, **7000** GPU hours) and analysis of different design settings - data augmentations, batch size, training duration, frame rate, and initialization - highlighting a need for and intuitions towards designing principled approaches for domain transfer of SSL methods.

**[3]** In-depth analysis on the adaptation of these methods, originally developed using other datasets and tasks, to the surgical domain with a comprehensive set of evaluation protocols, spanning 10 surgical vision tasks in total performed on 6 datasets: [**Cholec80**](https://arxiv.org/abs/1602.03012), [**CholecT50**](https://cholectriplet2021.grand-challenge.org/), [**HeiChole**](https://www.synapse.org/#!Synapse:syn25101790/wiki/610013), [**Endoscapes**](https://arxiv.org/abs/2112.13815), [**CATARACTS**](https://discovery.ucl.ac.uk/id/eprint/10068008/1/CATARACTS.pdf), and [**CaDIS**](https://www.sciencedirect.com/science/article/pii/S1361841521000992).

**[4]** Extensive evaluation (**∼280** experiments, **2000** GPU hours) of the scalability of these methods to various amounts of labeled and unlabeled data through an exploration of both fully and semi-supervised settings.


#### In this repo we provide:
- Self-supervised weights trained on cholec80 dataset using four state-of-the-art SSL methods (MOCO V2, SimCLR, SwAV, and DINO).
- Self-supervised pre-training scripts.
- Downstream fine-tuning scripts for surgical phase recognition (linear fine-tuning and TCN fine-tuning).
- Downstream fine-tuning scripts for surgical tool recognition (linear fine-tuning).

# Get Started

## Datasets and imagenet checkpoints
Follow the steps for cholec80 dataset preparation and setting up imagenet checkpoints:

```bash
# 1. Cholec80 phase and tool labels for different splits
> git clone https://github.com/CAMMA-public/SelfSupSurg
> SelfSupSurg=$(pwd)/SelfSupSurg
> cd $SelfSupSurg/datasets/cholec80
> wget https://s3.unistra.fr/camma_public/github/selfsupsurg/ch80_labels.zip
> unzip -q ch80_labels.zip && rm ch80_labels.zip
# 2. Cholec80 frames:  
# Download cholec80 videos from CAMMA website: (https://camma.u-strasbg.fr/datasets/cholec80)
# Copy the videos in datasets/cholec80/videos 
# Extract frames using the following script (you need OpenCV and numpy)
> cd $SelfSupSurg
> python utils/extract_frames_ch80.py
# 3. Download Imagenet fully supervised and self-supervised weights
> cd $SelfSupSurg/checkpoints/defaults/resnet_50
> wget https://s3.unistra.fr/camma_public/github/selfsupsurg/imagenet_ckpts.zip
> unzip -q imagenet_ckpts.zip && rm imagenet_ckpts.zip

```
- Directory structure should look as follows.
```shell
$SelSupSurg/
└── datasets/cholec80/
    ├── frames/
        ├── train/
            └── video01/
            └── video02/
            ...
        ├── val/
            └── video41/
            └── video42/
            ...
        ├── test/
            └── video49/
            └── video50/
            ...
    ├── labels/
        ├── train/
            └── 1fps_12p5_0.pickle
            └── 1fps_12p5_1.pickle
            ...
        ├── val/
            └── 1fps.pickle
            └── 3fps.pickle
            ...
        ├── test/
            └── 1fps.pickle
            └── 3fps.pickle
            ...        
    └── classweights/
        ├── train/
            └── 1fps_12p5_0.pickle
            └── 1fps_12p5_1.pickle
                ...
    ...
    └── checkpoints/defaults/resnet_50/
        └── resnet50-19c8e357.pth
        └── moco_v2_800ep_pretrain.pth.tar
        └── simclr_rn50_800ep_simclr_8node_resnet_16_07_20.7e8feed1.torch
        └── swav_in1k_rn50_800ep_swav_8node_resnet_27_07_20.a0a6b676.torch
        └── dino_resnet50_pretrain.pth
```


## Installation
You need to have a [Anaconda3](https://www.anaconda.com/products/individual#linux) installed for the setup. We developed the code on the Ubuntu 20.04, Python 3.8, PyTorch 1.7.1, and CUDA 10.2 using V100 GPU.
```sh
> cd $SelfSupSurg
> conda create -n selfsupsurg python=3.8 && conda activate selfsupsurg
# install dependencies 
(selfsupsurg)>conda install -y pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.2 -c pytorch 
(selfsupsurg)>pip install opencv-python
(selfsupsurg)>pip install openpyxl==3.0.7
(selfsupsurg)>pip install pandas==1.3.2
(selfsupsurg)>pip install scikit-learn
(selfsupsurg)>pip install easydict
(selfsupsurg)>pip install apex -f https://dl.fbaipublicfiles.com/vissl/packaging/apexwheels/py38_cu102_pyt171/download.html
(selfsupsurg)>cd $SelfSupSurg/ext_libs
(selfsupsurg)>git clone https://github.com/facebookresearch/ClassyVision.git && cd ClassyVision
(selfsupsurg)>git checkout 659d7f788c941a8c0d08dd74e198b66bd8afa7f5 && pip install -e .
(selfsupsurg)>cd ../ && git clone --recursive https://github.com/facebookresearch/vissl.git && cd ./vissl/
(selfsupsurg)>git checkout 65f2c8d0efdd675c68a0dfb110aef87b7bb27a2b
(selfsupsurg)>pip install --progress-bar off -r requirements.txt
(selfsupsurg)>pip install -e .[dev] && cd $SelfSupSurg
(selfsupsurg)>cp -r ./vissl/vissl/* $SelfSupSurg/ext_libs/vissl/vissl/
```
#### Modify `$SelfSupSurg/ext_libs/vissl/configs/config/dataset_catalog.json` by appending the following key/value pair to the end of the dictionary
```json
"surgery_datasets": {
    "train": ["<img_path>", "<lbl_path>"],
    "val": ["<img_path>", "<lbl_path>"],
    "test": ["<img_path>", "<lbl_path>"]
}
```


## Pre-training
Run the folllowing code for the pre-training of MoCo v2, SimCLR, SwAV, and DINO methods on the Cholec80 dataset with 4 GPUS.
```sh
# MoCo v2
(selfsupsurg)>cfg=hparams/cholec80/pre_training/cholec_to_cholec/series_01/h001.yaml
(selfsupsurg)>python main.py -hp $cfg -m self_supervised
# SimCLR
(selfsupsurg)>cfg=hparams/cholec80/pre_training/cholec_to_cholec/series_01/h002.yaml
(selfsupsurg)>python main.py -hp $cfg -m self_supervised
# SwAV
(selfsupsurg)>cfg=hparams/cholec80/pre_training/cholec_to_cholec/series_01/h003.yaml
(selfsupsurg)>python main.py -hp $cfg -m self_supervised
# DINO 
(selfsupsurg)>cfg=hparams/cholec80/pre_training/cholec_to_cholec/series_01/h004.yaml
(selfsupsurg)>python main.py -hp $cfg -m self_supervised
```

## Model Weights for the **pre-training** experiments

|   Model      |  Model Weights |
| :----------: | :-----:   |
| [MoCo V2](configs/config/hparams/cholec80/pre_training/cholec_to_cholec/series_01/h001.yaml)| [download](https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_moco_v2_surg.torch) |
| [SimCLR](configs/config/hparams/cholec80/pre_training/cholec_to_cholec/series_01/h002.yaml)| [download](https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_simclr_surg.torch) |
| [SwAV](configs/config/hparams/cholec80/pre_training/cholec_to_cholec/series_01/h003.yaml)| [download](https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_swav_surg.torch) |
| [DINO](configs/config/hparams/cholec80/pre_training/cholec_to_cholec/series_01/h004.yaml)| [download](https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_dino_surg.torch) |


## Downstream finetuning
First perform pre-training using the above scripts or download the [pre-trained weights](#model-weights-for-the-pre-training-experiments) and copy them into the appropriate directories, shown below
```sh
# MoCo v2
(selfsupsurg)>mkdir -p runs/cholec80/pre_training/cholec_to_cholec/series_01/run_001/ \
               && cp model_final_checkpoint_moco_v2_surg.torch runs/cholec80/pre_training/cholec_to_cholec/series_01/run_001/
# SimCLR
(selfsupsurg)>mkdir -p runs/cholec80/pre_training/cholec_to_cholec/series_01/run_002/ \
               && cp model_final_checkpoint_simclr_surg.torch runs/cholec80/pre_training/cholec_to_cholec/series_01/run_002/
# SwAV
(selfsupsurg)>mkdir -p runs/cholec80/pre_training/cholec_to_cholec/series_01/run_003/ \
               && cp model_final_checkpoint_swav_surg.torch runs/cholec80/pre_training/cholec_to_cholec/series_01/run_003/
# DINO 
(selfsupsurg)>mkdir -p runs/cholec80/pre_training/cholec_to_cholec/series_01/run_004/ \
               && cp model_final_checkpoint_dino_surg.torch runs/cholec80/pre_training/cholec_to_cholec/series_01/run_004/
```
### 1. Surgical phase recognition (Linear Finetuning)
The config files for the surgical phase recognition **linear finetuning** experiments are in [cholec80 pre-training init](configs/config/hparams/cholec80/finetuning/cholec_to_cholec/series_01/test/phase) and [imagenet init](configs/config/hparams/cholec80/finetuning/imagenet_to_cholec/series_01/test/phase). The config files are organized as follows:
```sh
# config files for the proposed pre-training init from cholec80 are oraganized as follows:
├── cholec_to_cholec/series_01/test/phase
    ├── 100 #(100 % of cholec 80)
    │   └── 0 #(split 0)
    │       ├── h001.yaml # MoCo V2 Surg
    │       ├── h002.yaml # SimCLR Surg
    │       ├── h003.yaml # SwAV Surg
    │       └── h004.yaml # DINO Surg
    ├── 12.5 #(12.5 % of cholec 80 dataset)
    │   ├── 0 #(split 0)
    │   │   ├── h001.yaml # MoCo V2 Surg
    │   │   ├── h002.yaml # SimCLR Surg
    │   │   ├── h003.yaml # SwAV Surg
    │   │   └── h004.yaml # DINO Surg
    │   ├── 1 #(split 1)
    │   │   ├── h001.yaml # MoCo V2 Surg
    │   │   ├── h002.yaml # SimCLR Surg
    │   │   ├── h003.yaml # SwAV Surg
    │   │   └── h004.yaml # DINO Surg
    │   ├── 2 #(split 2)
    │   │   ├── h001.yaml # MoCo V2 Surg
    │   │   ├── h002.yaml # SimCLR Surg
    │   │   ├── h003.yaml # SwAV Surg
    │   │   └── h004.yaml # DINO Surg
    └── 25 #(25 % of cholec 80 dataset)
        ├── 0 #(split 0)
        │   ├── h001.yaml # MoCo V2 Surg
        │   ├── h002.yaml # SimCLR Surg
        │   ├── h003.yaml # SwAV Surg
        │   └── h004.yaml # DINO Surg
        ├── 1 #(split 1)
        │   ├── h001.yaml # MoCo V2 Surg
        │   ├── h002.yaml # SimCLR Surg
        │   ├── h003.yaml # SwAV Surg
        │   └── h004.yaml # DINO Surg
        ├── 2 #(split 2)
        │   ├── h001.yaml # MoCo V2 Surg
        │   ├── h002.yaml # SimCLR Surg
        │   ├── h003.yaml # SwAV Surg
        │   └── h004.yaml # DINO Surg
# config files for the baselines imagenet to cholec80 are oraganized as follows:
├── imagenet_to_cholec/series_01/test/phase
    ├── 100 #(100 % of cholec 80)
    │   └── 0 #(split 0)
    │       ├── h001.yaml # Fully-supervised imagenet
    │       ├── h002.yaml # MoCo V2 imagenet
    │       ├── h003.yaml # SimCLR imagenet
    │       ├── h004.yaml # SwAV imagenet
    │       └── h005.yaml # DINO imagenet
    ├── 12.5 #(12.5 % of cholec 80 dataset)
    │   ├── 0 #(split 0)
    │   │   ├── h001.yaml # Fully-supervised imagenet
    │   │   ├── h002.yaml # MoCo V2 imagenet
    │   │   ├── h003.yaml # SimCLR  imagenet
    │   │   ├── h004.yaml # SwAV  imagenet
    │   │   └── h005.yaml # DINO imagenet
    │   ├── 1 #(split 1)
    │   │   ├── h001.yaml # Fully-supervised imagenet 
    │   │   ├── h002.yaml # MoCo V2 imagenet
    │   │   ├── h003.yaml # SimCLR imagenet
    │   │   ├── h004.yaml # SwAV imagenet
    │   │   └── h005.yaml # DINO imagenet
    │   ├── 2 #(split 2)
    │   │   ├── h001.yaml # Fully-supervised imagenet 
    │   │   ├── h002.yaml # MoCo V2 imagenet
    │   │   ├── h003.yaml # SimCLR imagenet
    │   │   ├── h004.yaml # SwAV imagenet
    │   │   └── h005.yaml # DINO imagenet
    └── 25 #(25 % of cholec 80 dataset)
        ├── 0 #(split 0)
        │   ├── h001.yaml # Fully-supervised imagenet
        │   ├── h002.yaml # MoCo V2 imagenet
        │   ├── h003.yaml # SimCLR imagenet
        │   ├── h004.yaml # SwAV imagenet
        │   └── h005.yaml # DINO imagenet
        ├── 1 #(split 1)
        │   ├── h001.yaml # Fully-supervised imagenet
        │   ├── h002.yaml # MoCo V2 imagenet
        │   ├── h003.yaml # SimCLR imagenet
        │   ├── h004.yaml # SwAV imagenet
        │   ├── h005.yaml # DINO imagenet
        ├── 2 #(split 2)
        │   ├── h001.yaml # Fully-supervised imagenet
        │   ├── h002.yaml # MoCo V2 imagenet
        │   ├── h003.yaml # SimCLR imagenet
        │   ├── h004.yaml # SwAV imagenet
        │   └── h005.yaml # DINO imagenet
```
Examples commands for surgical phase linear fine-tuning. It uses 4 GPUS for the training
```sh
# Example 1, run the following command for linear fine-tuning, initialized with MoCO V2 weights 
# on 25% of cholec80 data (split 0).
(selfsupsurg)>cfg=hparams/cholec80/finetuning/cholec_to_cholec/series_01/test/phase/25/0/h001.yaml
(selfsupsurg)>python main.py -hp $cfg -m supervised

# Example 2, run the following command for linear fine-tuning, initialized with SimCLR weights 
# on 12.5% of cholec80 data (split 1).
(selfsupsurg)>cfg=hparams/cholec80/finetuning/cholec_to_cholec/series_01/test/phase/12.5/1/h002.yaml
(selfsupsurg)>python main.py -hp $cfg -m supervised

# Example 3, run the following command for linear fine-tuning, initialized with 
# imagenet MoCO v2 weights on 12.5% of cholec80 data (split 2).
(selfsupsurg)>cfg=hparams/cholec80/finetuning/imagenet_to_cholec/series_01/test/phase/12.5/2/h002.yaml
(selfsupsurg)>python main.py -hp $cfg -m supervised
```

### 2. Surgical phase recognition (TCN Finetuning)

The config files for the surgical phase recognition **TCN finetuning** experiments are in [cholec80 pre-training init](configs/config/hparams/cholec80/finetuning/cholec_to_cholec/series_01/test/phase_tcn) and [imagenet init](configs/config/hparams/cholec80/finetuning/imagenet_to_cholec/series_01/test/phase_tcn). The config files are organized as follows:
```sh
# config files for the proposed pre-training init from cholec80 are oraganized as follows:
├── cholec_to_cholec/series_01/test/phase_tcn
    ├── 100 #(100 % of cholec 80)
    │   └── 0 #(split 0)
    │       ├── h001.yaml # MoCo V2 Surg
    │       ├── h002.yaml # SimCLR Surg
    │       ├── h003.yaml # SwAV Surg
    │       └── h004.yaml # DINO Surg
    ├── 12.5 #(12.5 % of cholec 80 dataset)
    │   ├── 0 #(split 0)
    │   │   ├── h001.yaml # MoCo V2 Surg
    │   │   ├── h002.yaml # SimCLR Surg
    │   │   ├── h003.yaml # SwAV Surg
    │   │   └── h004.yaml # DINO Surg
    │   ├── 1 #(split 1)
    │   │   ├── h001.yaml # MoCo V2 Surg
    │   │   ├── h002.yaml # SimCLR Surg
    │   │   ├── h003.yaml # SwAV Surg
    │   │   └── h004.yaml # DINO Surg
    │   ├── 2 #(split 2)
    │   │   ├── h001.yaml # MoCo V2 Surg
    │   │   ├── h002.yaml # SimCLR Surg
    │   │   ├── h003.yaml # SwAV Surg
    │   │   └── h004.yaml # DINO Surg
    └── 25 #(25 % of cholec 80 dataset)
        ├── 0 #(split 0)
        │   ├── h001.yaml # MoCo V2 Surg
        │   ├── h002.yaml # SimCLR Surg
        │   ├── h003.yaml # SwAV Surg
        │   └── h004.yaml # DINO Surg
        ├── 1 #(split 1)
        │   ├── h001.yaml # MoCo V2 Surg
        │   ├── h002.yaml # SimCLR Surg
        │   ├── h003.yaml # SwAV Surg
        │   └── h004.yaml # DINO Surg
        ├── 2 #(split 2)
        │   ├── h001.yaml # MoCo V2 Surg
        │   ├── h002.yaml # SimCLR Surg
        │   ├── h003.yaml # SwAV Surg
        │   └── h004.yaml # DINO Surg
# config files for the baselines imagenet to cholec80 are oraganized as follows:
├── imagenet_to_cholec/series_01/test/phase_tcn
    ├── 100 #(100 % of cholec 80)
    │   └── 0 #(split 0)
    │       ├── h001.yaml # Fully-supervised imagenet
    │       ├── h002.yaml # MoCo V2 imagenet
    │       ├── h003.yaml # SimCLR imagenet
    │       ├── h004.yaml # SwAV imagenet
    │       └── h005.yaml # DINO imagenet
    ├── 12.5 #(12.5 % of cholec 80 dataset)
    │   ├── 0 #(split 0)
    │   │   ├── h001.yaml # Fully-supervised imagenet
    │   │   ├── h002.yaml # MoCo V2 imagenet
    │   │   ├── h003.yaml # SimCLR  imagenet
    │   │   ├── h004.yaml # SwAV  imagenet
    │   │   └── h005.yaml # DINO imagenet
    │   ├── 1 #(split 1)
    │   │   ├── h001.yaml # Fully-supervised imagenet 
    │   │   ├── h002.yaml # MoCo V2 imagenet
    │   │   ├── h003.yaml # SimCLR imagenet
    │   │   ├── h004.yaml # SwAV imagenet
    │   │   └── h005.yaml # DINO imagenet
    │   ├── 2 #(split 2)
    │   │   ├── h001.yaml # Fully-supervised imagenet 
    │   │   ├── h002.yaml # MoCo V2 imagenet
    │   │   ├── h003.yaml # SimCLR imagenet
    │   │   ├── h004.yaml # SwAV imagenet
    │   │   └── h005.yaml # DINO imagenet
    └── 25 #(25 % of cholec 80 dataset)
        ├── 0 #(split 0)
        │   ├── h001.yaml # Fully-supervised imagenet
        │   ├── h002.yaml # MoCo V2 imagenet
        │   ├── h003.yaml # SimCLR imagenet
        │   ├── h004.yaml # SwAV imagenet
        │   └── h005.yaml # DINO imagenet
        ├── 1 #(split 1)
        │   ├── h001.yaml # Fully-supervised imagenet
        │   ├── h002.yaml # MoCo V2 imagenet
        │   ├── h003.yaml # SimCLR imagenet
        │   ├── h004.yaml # SwAV imagenet
        │   ├── h005.yaml # DINO imagenet
        ├── 2 #(split 2)
        │   ├── h001.yaml # Fully-supervised imagenet
        │   ├── h002.yaml # MoCo V2 imagenet
        │   ├── h003.yaml # SimCLR imagenet
        │   ├── h004.yaml # SwAV imagenet
        │   └── h005.yaml # DINO imagenet
```
Examples commands for TCN fine-tuning. We first extract the features for the `train`, `val` and `test` set and then perform the TCN fine-tuning
```sh
# Example 1, run the following command for TCN fine-tuning, initialized with MoCO V2 weights 
# on 25% of cholec80 data (split 0).
# 1) feature extraction for the train, val and test set
(selfsupsurg)>cfg=hparams/cholec80/finetuning/cholec_to_cholec/series_01/test/phase/25/0/h001.yaml
(selfsupsurg)>python main.py -hp $cfg -m  feature_extraction -s train -f Trunk
(selfsupsurg)>python main.py -hp $cfg -m  feature_extraction -s val -f Trunk
(selfsupsurg)>python main.py -hp $cfg -m  feature_extraction -s test -f Trunk                            
# 2) TCN fine-tuning        
(selfsupsurg)>cfg=hparams/cholec80/finetuning/cholec_to_cholec/series_01/test/phase_tcn/25/0/h001.yaml
(selfsupsurg)>python main_ft_phase_tcn.py -hp $cfg -t test

# Example 2, run the following command for TCN fine-tuning, initialized with SimCLR weights 
# on 12.5% of cholec80 data (split 1).
# 1) feature extraction for the train, val and test set
(selfsupsurg)>cfg=hparams/cholec80/finetuning/cholec_to_cholec/series_01/test/phase/12.5/1/h002.yaml
(selfsupsurg)>python main.py -hp $cfg -m  feature_extraction -s train -f Trunk
(selfsupsurg)>python main.py -hp $cfg -m  feature_extraction -s val -f Trunk
(selfsupsurg)>python main.py -hp $cfg -m  feature_extraction -s test -f Trunk                            
# 2) TCN fine-tuning        
(selfsupsurg)>cfg=hparams/cholec80/finetuning/cholec_to_cholec/series_01/test/phase_tcn/12.5/1/h002.yaml
(selfsupsurg)>python main_ft_phase_tcn.py -hp $cfg -t test

# Example 3, run the following command for TCN fine-tuning, initialized with imagenet MoCO v2 weights 
# on 12.5% of cholec80 data (split 2).
# 1) feature extraction for the train, val and test set
(selfsupsurg)>cfg=hparams/cholec80/finetuning/imagenet_to_cholec/series_01/test/phase/12.5/2/h002.yaml
(selfsupsurg)>python main.py -hp $cfg -m  feature_extraction -s train -f Trunk
(selfsupsurg)>python main.py -hp $cfg -m  feature_extraction -s val -f Trunk
(selfsupsurg)>python main.py -hp $cfg -m  feature_extraction -s test -f Trunk                            
# 2) TCN fine-tuning        
(selfsupsurg)>cfg=hparams/cholec80/finetuning/imagenet_to_cholec/series_01/test/phase_tcn/12.5/2/h002.yaml
(selfsupsurg)>python main_ft_phase_tcn.py -hp $cfg -t test
```

### 3. Surgical tool recognition

The config files for the surgical tool recognition experiments are in [cholec80 pre-training init](configs/config/hparams/cholec80/finetuning/cholec_to_cholec/series_01/test/tools) and [imagenet init](configs/config/hparams/cholec80/finetuning/imagenet_to_cholec/series_01/test/tools). The config files are organized as follows:
```sh
# config files for the proposed pre-training init from cholec80 are oraganized as follows:
├── cholec_to_cholec/series_01/test/tools
    ├── 100 #(100 % of cholec 80)
    │   └── 0 #(split 0)
    │       ├── h001.yaml # MoCo V2 Surg
    │       ├── h002.yaml # SimCLR Surg
    │       ├── h003.yaml # SwAV Surg
    │       └── h004.yaml # DINO Surg
    ├── 12.5 #(12.5 % of cholec 80 dataset)
    │   ├── 0 #(split 0)
    │   │   ├── h001.yaml # MoCo V2 Surg
    │   │   ├── h002.yaml # SimCLR Surg
    │   │   ├── h003.yaml # SwAV Surg
    │   │   └── h004.yaml # DINO Surg
    │   ├── 1 #(split 1)
    │   │   ├── h001.yaml # MoCo V2 Surg
    │   │   ├── h002.yaml # SimCLR Surg
    │   │   ├── h003.yaml # SwAV Surg
    │   │   └── h004.yaml # DINO Surg
    │   ├── 2 #(split 2)
    │   │   ├── h001.yaml # MoCo V2 Surg
    │   │   ├── h002.yaml # SimCLR Surg
    │   │   ├── h003.yaml # SwAV Surg
    │   │   └── h004.yaml # DINO Surg
    └── 25 #(25 % of cholec 80 dataset)
        ├── 0 #(split 0)
        │   ├── h001.yaml # MoCo V2 Surg
        │   ├── h002.yaml # SimCLR Surg
        │   ├── h003.yaml # SwAV Surg
        │   └── h004.yaml # DINO Surg
        ├── 1 #(split 1)
        │   ├── h001.yaml # MoCo V2 Surg
        │   ├── h002.yaml # SimCLR Surg
        │   ├── h003.yaml # SwAV Surg
        │   └── h004.yaml # DINO Surg
        ├── 2 #(split 2)
        │   ├── h001.yaml # MoCo V2 Surg
        │   ├── h002.yaml # SimCLR Surg
        │   ├── h003.yaml # SwAV Surg
        │   └── h004.yaml # DINO Surg
# config files for the baselines imagenet to cholec80 are oraganized as follows:
├── imagenet_to_cholec/series_01/test/tools
    ├── 100 #(100 % of cholec 80)
    │   └── 0 #(split 0)
    │       ├── h001.yaml # Fully-supervised imagenet
    │       ├── h002.yaml # MoCo V2 imagenet
    │       ├── h003.yaml # SimCLR imagenet
    │       ├── h004.yaml # SwAV imagenet
    │       └── h005.yaml # DINO imagenet
    ├── 12.5 #(12.5 % of cholec 80 dataset)
    │   ├── 0 #(split 0)
    │   │   ├── h001.yaml # Fully-supervised imagenet
    │   │   ├── h002.yaml # MoCo V2 imagenet
    │   │   ├── h003.yaml # SimCLR  imagenet
    │   │   ├── h004.yaml # SwAV  imagenet
    │   │   └── h005.yaml # DINO imagenet
    │   ├── 1 #(split 1)
    │   │   ├── h001.yaml # Fully-supervised imagenet 
    │   │   ├── h002.yaml # MoCo V2 imagenet
    │   │   ├── h003.yaml # SimCLR imagenet
    │   │   ├── h004.yaml # SwAV imagenet
    │   │   └── h005.yaml # DINO imagenet
    │   ├── 2 #(split 2)
    │   │   ├── h001.yaml # Fully-supervised imagenet 
    │   │   ├── h002.yaml # MoCo V2 imagenet
    │   │   ├── h003.yaml # SimCLR imagenet
    │   │   ├── h004.yaml # SwAV imagenet
    │   │   └── h005.yaml # DINO imagenet
    └── 25 #(25 % of cholec 80 dataset)
        ├── 0 #(split 0)
        │   ├── h001.yaml # Fully-supervised imagenet
        │   ├── h002.yaml # MoCo V2 imagenet
        │   ├── h003.yaml # SimCLR imagenet
        │   ├── h004.yaml # SwAV imagenet
        │   └── h005.yaml # DINO imagenet
        ├── 1 #(split 1)
        │   ├── h001.yaml # Fully-supervised imagenet
        │   ├── h002.yaml # MoCo V2 imagenet
        │   ├── h003.yaml # SimCLR imagenet
        │   ├── h004.yaml # SwAV imagenet
        │   ├── h005.yaml # DINO imagenet
        ├── 2 #(split 2)
        │   ├── h001.yaml # Fully-supervised imagenet
        │   ├── h002.yaml # MoCo V2 imagenet
        │   ├── h003.yaml # SimCLR imagenet
        │   ├── h004.yaml # SwAV imagenet
        │   └── h005.yaml # DINO imagenet
```
Examples commands for surgical tool recognition **linear fine-tuning**. It uses 4 GPUS for the training
```sh
# Example 1, run the following command for linear fine-tuning, initialized with MoCO V2 weights 
# on 25% of cholec80 data (split 0).
(selfsupsurg)>cfg=hparams/cholec80/finetuning/cholec_to_cholec/series_01/test/tools/25/0/h001.yaml
(selfsupsurg)>python main.py -hp $cfg -m supervised

# Example 2, run the following command for linear fine-tuning, initialized with SimCLR weights 
# on 12.5% of cholec80 data (split 1).
(selfsupsurg)>cfg=hparams/cholec80/finetuning/cholec_to_cholec/series_01/test/tools/12.5/1/h002.yaml
(selfsupsurg)>python main.py -hp $cfg -m supervised

# Example 3, run the following command for linear fine-tuning, initialized with 
# imagenet MoCO v2 weights on 12.5% of cholec80 data (split 2).
(selfsupsurg)>cfg=hparams/cholec80/finetuning/imagenet_to_cholec/series_01/test/tools/12.5/2/h002.yaml
(selfsupsurg)>python main.py -hp $cfg -m supervised
```


## Citation
```bibtex
@article{ramesh2022dissecting,
  title={Dissecting Self-Supervised Learning Methods for Surgical Computer Vision},
  author={Ramesh, Sanat and Srivastav, Vinkle and Alapatt, Deepak and Yu, Tong and Murali, Aditya and Sestini, Luca and Innocent Nwoye, Chinedu and Hamoud, Idris and Fleurentin, Antoine and Exarchakis, Georgios and Karargyris, Alexandros and Padoy, Nicolas},
  journal={arXiv e-prints},
  pages={arXiv--2207},
  year={2022}
}
```




### References
The project uses [VISSL](https://github.com/facebookresearch/vissl). We thank the authors of VISSL for releasing the library. If you use VISSL, consider citing it using the following BibTeX entry.
```bibtex
@misc{goyal2021vissl,
  author =       {Priya Goyal and Quentin Duval and Jeremy Reizenstein and Matthew Leavitt and Min Xu and
                  Benjamin Lefaudeux and Mannat Singh and Vinicius Reis and Mathilde Caron and Piotr Bojanowski and
                  Armand Joulin and Ishan Misra},
  title =        {VISSL},
  howpublished = {\url{https://github.com/facebookresearch/vissl}},
  year =         {2021}
}
```
The project also leverages following research works. We thank the authors for releasing their codes.
- [TeCNO](https://github.com/tobiascz/TeCNO)

## License
This code, models, and datasets are available for non-commercial scientific research purposes as defined in the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). By downloading and using this code you agree to the terms in the [LICENSE](LICENSE). Third-party codes are subject to their respective licenses.
