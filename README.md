
# WAS: Weakly Supervised Assistance for Unsupervised Domain Adaptive Object Detection


This is the official PyTorch implementation of our WAS paper:

**[Weakly Supervised Assistance for Unsupervised Domain Adaptive Object Detection](https://arxiv.org/abs/2305.03034)**

[Xinhe Ouyang](https://shengcao-cao.github.io/), [Yanting Pei](https://research.ibm.com/people/dhiraj-joshi), [Minhao Hao](https://cs.illinois.edu/about/people/faculty/lgui), [Fan Yang](https://yxw.web.illinois.edu/)

![WAS-pipeline](CMT-pipeline.png)

## Overview

In this repository, we include the implementation of Contrastive Mean Teacher, integrated with both base methods Contrastive Mean Teacher (CMT, [[code](https://github.com/Shengcao-Cao/CMT)] [[paper](https://arxiv.org/abs/2305.03034)]) and H2FA (H2FA, [[code](https://github.com/XuYunqiu/H2FA_R-CNN)] [[paper](https://ieeexplore.ieee.org/document/9878659)]). Our code is based on the publicly available implementation of these two methods.

## Environment and Dataset Setup

We follow CMT and H2FA original instructions to set up the environment and datasets. The details are included in the README files.

## Usage

Here is an example script for reproducing our results of WAS on Cityscapes -> Foggy Cityscapes (all splits):

```bash

#prepare pseudo_labels(these pseudo-labels are generate by CMT,
#if you want to try other UDAOD pseudo-labels you can construct yours)
#I have provide foggy2city_train_pseudo_label.json and voc2clipart_train_pseudo_label.json in WAS/pseudo_labels

# enter the code directory for WAS
cd UDAOD/h2fa

# activate AT environment
conda activate h2fa

python train_net.py \
--config-file ../configs/CrossDomain-Detection/h2fa_rcnn_R_101_DC5_foggycityscapes.yaml \
--num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
SOLVER.STEPS 48000,64000 SOLVER.MAX_ITER 72000

```


- Other configuration options may be found in `configs`.
- To resume the training, simply add `--resume` to the command.
- To evaluate an existing model checkpoint, add `--eval-only` and specify `MODEL.WEIGHTS path/to/your/weights.pth` in the command.

## Model Weights

Here we list the model weights for the results included in our paper:

| Dataset                                    | Method    | mAP (AP50) | Weights |
| ------------------------------------------ |-----------|------------|---------|
| Cityscapes -> Foggy Cityscapes | CMT + WAS | 55.1       | further |
| Pascal VOC -> Clipart1k        | CMT + WAS | 48.1       | further |

## Additional Changes


## Citation



## License

This project is released under the [Apache 2.0 license](./LICENSE). Other codes from open source repository follows the original distributive licenses.






