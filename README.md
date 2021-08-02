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
- lr, beta1, beta2 are the parameters of the Adam optimiser, for more information you can consult the manual on the torch website(https://pytorch.org/docs/1.1.0/_modules/torch/optim/adam.html).

## Train
- set images under ./img and mask image is placed under the main directory of the folder. THen,
'python train.py'

## Test
- You can load our trained model or retrain your own model using your dataset.
- Modify the corresponding parameter of ArgumentParser(), then
'python test_single.py'

## Quantitative experiments
- PSNR.py provides three quantitative metrics - MSE, SSIM and PSNR - to allow you to observe the performance of inpainting.
'python PSNR.py'
