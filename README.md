# TG-Net
Reconstruct visual wood texture with semantic attention. Image Inpainting. 

[paper](https://www.sciencedirect.com/science/article/abs/pii/S0097849321001928?via%3Dihub)

![image](https://github.com/NEFUJoeyChen/TG-Net/blob/main/img/train/Graphical%20Abstract.jpg)

## Requirements
- Python3.7
- Pytorch1.3.1
- tqdm
- PIL

## Usage
- You can find more information on ./configs and change the relevant training parameters in the config.yaml
- batch_size
- image_shape
- mask_shape
- `lr`, `beta1`, `beta2` are the parameters of the Adam optimiser, for more information you can consult the manual on the torch website(https://pytorch.org/docs/1.1.0/_modules/torch/optim/adam.html).

## Train
- We provide `WoodDataset` for you to train your network. Download on [Mendeley Data](https://data.mendeley.com/datasets/2w3wy6ctvr/1) or Baidu Netdisk：(https://pan.baidu.com/s/1MO_iG0YpG9ZjjovTA5lwgA 
提取码：1219). You can also use a face dataset or street view or your own dataset to train the model. Unfortunately, our trained models larger than 100M cannot be uploaded to git.
- set images under `./img` and mask image is placed under the main directory of the folder. THen,

```python train.py```

## Test
![image](https://github.com/NEFUJoeyChen/TG-Net/blob/main/img/train/ex1.jpg)
- You can retrain your own model using your dataset.
- Modify the corresponding parameter of ArgumentParser(), then

```python test_single.py```

## Quantitative experiments
- PSNR.py provides three quantitative metrics - MSE, SSIM and PSNR - to allow you to observe the performance of inpainting.

```python PSNR.py```

## Citation
```
@article{CHEN2021,
title = {TG-Net: Reconstruct visual wood texture with semantic attention},
journal = {Computers & Graphics},
year = {2021},
issn = {0097-8493},
doi = {https://doi.org/10.1016/j.cag.2021.09.006},
url = {https://www.sciencedirect.com/science/article/pii/S0097849321001928},
author = {Jiahao Chen and Yilin Ge and Quan Wang and Yizhuo Zhang},
keywords = {Wood defect treatment, Image inpainting, Generative adversarial networks, Attention mechanism}
}
```
