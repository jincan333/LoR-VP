# [LoR-VP: Low-Rank Visual Prompting for Efficient Vision Model Adaptation](https://arxiv.org/pdf/2502.00896)
[![LICENSE](https://img.shields.io/badge/LICENSE-MIT-4caf50.svg)](https://github.com/jincan333/LoR-VP)


## Table of Contents

[**Overview**](#overview) | [**Requirements**](#install-requirements) | [**WANDB**](#configure-wandb) | [**Implementation**](#run-lot) | [**Contributor**](#contributors) | [**Citation**](#citation)


## Overview

Visual prompting has gained popularity as a method for adapting pre-trained models to specific tasks, particularly in the realm of parameter-efficient tuning. However, existing visual prompting techniques often pad the prompt parameters around the image, limiting the interaction between the visual prompts and the original image to a small set of patches while neglecting the inductive bias present in shared information across different patches. In this study, we conduct a thorough preliminary investigation to identify and address these limitations. We propose a novel visual prompt design, introducing Low-Rank matrix multiplication for Visual Prompting (**LoR-VP**), which enables shared and patch-specific information across rows and columns of image pixels. Extensive experiments across seven network architectures and four datasets demonstrate significant improvements in both performance and efficiency compared to state-of-the-art visual prompting methods, achieving up to 6 times faster training times, utilizing 18 times fewer visual prompt parameters, and delivering a 3.1% improvement in performance.

Authors: [Can Jin](https://jincan333.github.io/), Ying Li, Mingyu Zhao, Shiyu Zhao, Zhenting Wang, Xiaoxiao He, Ligong Han, Tong Che, Dimitris N. Metaxas

![LoR-VP](LoR-VP.png)

We resize the image to a resolution of $L \times L$ and initialize two low-rank matrices $\textbf{B}$ and $\textbf{A}$ as tunable parameters. The product $\textbf{B} \cdot \textbf{A}$ serves as the visual prompt and is directly added to the resized images. This design allows for shared information in rows and columns while also permitting patch-specific information across different patches.

## Install Requirements: 
```
conda create -n LoR-VP python=3.10
conda activate LoR-VP
pip install -r requirements.txt
```

## Configure WANDB

Configure WANDB USER_NAME and API_KEY in the environment variables.

## Run LoR-VP

### Image Classification
Run the following command for ViT-B/16-21K on Tiny-ImageNet.
```
bash run/run_lorvp.sh
```

The meaning of the parameters in the `run_lorvp.sh` file is as follows:
- network: the name of the network architecture.
- dataset: the name of the dataset.
- downstream_mapping: the name of the downstream mapping, including `lp`, `ilm`, `flm`, and `fm`.
- mapping_freq: the frequency of the mapping.
- prompt_method: the name of the prompt method.
- bar_width: the rank of the low-rank visual prompts.
- init_method: the name of the initialization method.
- train_batch_size: the batch size of the training.
- randomcrop: whether to use random crop.
- optimizer: the name of the optimizer.
- scheduler: the name of the scheduler.
- lr: the learning rate.
- epochs: the number of epochs.
- weight_decay: the weight decay.
- gpu: the GPU number.
- seed: the seed.
- eval_frequency: the epoch frequency of the evaluation.

change the hyperparameters following our [paper](https://arxiv.org/pdf/2502.00896) to run other datasets and networks.

## Contributors
Some of the code in this repository is based on the following amazing works.

* [Visual Prompting Upgrades Neural Network Sparsification: A Data-Model Perspective (VPNs)](https://github.com/UNITES-Lab/VPNs) (Jin et al., 2025)
* [Iterative-Label-Mapping Visual Prompting (ILM-VP)](https://github.com/OPTML-Group/ILM-VP) (Chen et al., 2023)

## Citation
We encourage citing our paper if our findings are used in your research.
```bibtex
@inproceedings{
jin2025lorvp,
title={LoR-{VP}: Low-Rank Visual Prompting for Efficient Vision Model Adaptation},
author={Can Jin and Ying Li and Mingyu Zhao and Shiyu Zhao and Zhenting Wang and Xiaoxiao He and Ligong Han and Tong Che and Dimitris N. Metaxas},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=5btFIv2PNb}
}
```