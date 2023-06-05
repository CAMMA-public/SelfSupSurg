
## Surgical triplet recognition (finetuning experiments)

### Pre-requisites:
`pip install ivtmetrics` for evaluation on CholecT45 test set.


### Downloads 
Download the pre-trained weights and copy to the `$SelfSupSurg/downstream_triplet/checkpoints/ssl_no_phase`

|   Model      |  Model Weights |
| :----------: | :-----:   |
| [MoCo V2](configs/config/hparams/cholec80/pre_training/cholec_to_cholec/series_01/h001.yaml)| [download](https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_moco_v2_surg.torch) |
| [SimCLR](configs/config/hparams/cholec80/pre_training/cholec_to_cholec/series_01/h002.yaml)| [download](https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_simclr_surg.torch) |
| [SwAV](configs/config/hparams/cholec80/pre_training/cholec_to_cholec/series_01/h003.yaml)| [download](https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_swav_surg.torch) |
| [DINO](configs/config/hparams/cholec80/pre_training/cholec_to_cholec/series_01/h004.yaml)| [download](https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_dino_surg.torch) |

or use the following commands to download the weights
```sh
(selfsupsurg)>mkdir -p $SelfSupSurg/downstream_triplet/checkpoints/ssl_no_phase
(selfsupsurg)>cd $SelfSupSurg/downstream_triplet/checkpoints/ssl_no_phase
(selfsupsurg)>wget https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_moco_v2_surg.torch
(selfsupsurg)>wget https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_simclr_surg.torch
(selfsupsurg)>wget https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_swav_surg.torch
(selfsupsurg)>wget https://s3.unistra.fr/camma_public/github/selfsupsurg/models/model_final_checkpoint_dino_surg.torch
```

### CholecT50 dataset

Follow [CholecT50 Dataset](https://github.com/CAMMA-public/cholect50) to download the CholecT50 dataset. The dataset should be in `$SelfSupSurg/downstream_triplet/datasets/CholecT50`

### Config files
The config files for the surgical triplet recognition experiments are structured as follows:
<details>
<summary>config_files</summary>
```sh
├── cholec_to_triplet/series_01/
    ├── 100 #(100 % of CholecT45)
    │   └── 0 #(split 0)
    │       ├── moco.yaml 
    │       ├── simclr.yaml 
    │       ├── swav.yaml 
    │       ├── dino.yaml
    │       └── imagenet.yaml
    ├── 12.5 #(12.5 % of CholecT45 dataset)
    │   ├── 0 #(split 0)
    │   │   ├── moco.yaml 
    │   │   ├── simclr.yaml 
    │   │   ├── swav.yaml 
    │   │   ├── dino.yaml     
    │   │   └── imagenet.yaml 
    │   ├── 1 #(split 1)
    │   │   ├── moco.yaml 
    │   │   ├── simclr.yaml 
    │   │   ├── swav.yaml 
    │   │   ├── dino.yaml     
    │   │   └── imagenet.yaml 
    │   ├── 2 #(split 2)
    │   │   ├── moco.yaml 
    │   │   ├── simclr.yaml 
    │   │   ├── swav.yaml 
    │   │   ├── dino.yaml     
    │   │   └── imagenet.yaml 
    ├── 25 #(25 % of CholecT45 dataset)
    │   ├── 0 #(split 0)
    │   │   ├── moco.yaml 
    │   │   ├── simclr.yaml 
    │   │   ├── swav.yaml 
    │   │   ├── dino.yaml     
    │   │   └── imagenet.yaml 
    │   ├── 1 #(split 1)
    │   │   ├── moco.yaml 
    │   │   ├── simclr.yaml 
    │   │   ├── swav.yaml 
    │   │   ├── dino.yaml     
    │   │   └── imagenet.yaml 
    │   ├── 2 #(split 2)
    │   │   ├── moco.yaml 
    │   │   ├── simclr.yaml 
    │   │   ├── swav.yaml 
    │   │   ├── dino.yaml     
    │   │   └── imagenet.yaml 
```
</details>


#### Training:

```sh
# Example 1, run the following command for fine-tuning on the 100% of triplet dataset, initialized with MoCO V2 weights
(selfsupsurg)>cd $SelfSupSurg/downstream_triplet
(selfsupsurg)>python main_triplet.py --exp_mode train --en ft_moco_v2_100p --cf cholec_to_triplet/series_01/100/0/moco.yaml

# Example 2, run the following command for fine-tuning on the 25% of triplet dataset (split 0), initialized with MoCO V2 weights
(selfsupsurg)>cd $SelfSupSurg/downstream_triplet
(selfsupsurg)>python main_triplet.py --exp_mode train --en ft_moco_v2_100p --cf cholec_to_triplet/series_01/25/0/moco.yaml
```

#### Evaluation:

```sh
(selfsupsurg)>cd $SelfSupSurg/downstream_triplet
(selfsupsurg)>python main_triplet.py --exp_mode eval --en evaluate_triplet --cf cholec_to_triplet/series_01/100/0/moco.yaml --ckp_n moco_v2-ft-100-0
```