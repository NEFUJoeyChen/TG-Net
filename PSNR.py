# _*_ coding:utf-8 _*_
# 开发人员：Joey
# 开发时间：2020/12/113:51
# 文件名称：PSRN.py
# 开发工具：PyCharm
import math
import numpy as np
from skimage import io
from scipy.signal import convolve2d

def compute_psnr(img1, img2):
    if isinstance(img1,str):
        img1=io.imread(img1)
    if isinstance(img2,str):
        img2=io.imread(img2)
    mse = np.mean( (img1/255. - img2/255.) ** 2 )
    if mse < 1.0e-10:
       return 1000000000000
    PIXEL_MAX = 1
    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return mse, psnr


def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))

    return np.mean(np.mean(ssim_map))

if __name__ == "__main__":
    from PIL import Image
    # gray images
    # pred = np.asarray(Image.open('./img1.png'))
    # gt = np.asarray(Image.open('./img2.png'))
    # # if not:
    img1 = Image.open('./gt1.png').convert('L')
#     img1 = img1.resize((256, 256), Image.BILINEAR)
    img1 = np.asarray(img1)
    img2 = Image.open('./v2_1.jpg').convert('L')
#     img2 = img2.resize((256, 256), Image.BILINEAR)
    img2 = np.asarray(img2)
    mse, psnr = compute_psnr(img2, img1)
    ssim = compute_ssim(img2, img1)

    print('mse = %.4f, psnr = %.4f, ssim = %.4f' % (mse, psnr, ssim))
