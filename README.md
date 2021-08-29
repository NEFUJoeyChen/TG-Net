# TG-Net
Reconstruct visual wood texture with semantic attention. Image Inpainting. 


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
- We provide `WoodDataset` for you to train your network. Download on Baidu Netdisk：(https://pan.baidu.com/s/1MO_iG0YpG9ZjjovTA5lwgA 
提取码：1219). You can also use a face dataset or street view or your own dataset to train the model. Unfortunately, our trained models larger than 100M cannot be uploaded to git.
- set images under `./img` and mask image is placed under the main directory of the folder. THen,
    python train.py

## Test
- You can retrain your own model using your dataset.
- Modify the corresponding parameter of ArgumentParser(), then
    python test_single.py

## Quantitative experiments
- PSNR.py provides three quantitative metrics - MSE, SSIM and PSNR - to allow you to observe the performance of inpainting.
    python PSNR.py
