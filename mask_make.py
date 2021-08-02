# _*_ coding:utf-8 _*_
# 开发人员：Joey
# 开发时间：2021/4/2422:48
# 文件名称：mask_make.py
# 开发工具：PyCharm
import numpy as np
import cv2

# image = cv2.imread("img/street.jpg")  # 读图
#cv2.imshow("Oringinal", image) #显示原图
# print(image.shape[:2])

# 输入图像是RGB图像，故构造一个三维数组，四个二维数组是mask四个点的坐标，
site = np.array([[[203, 53],[53, 53],[53,203],[203,203]]], dtype=np.int32)

# im = np.zeros([256,256], dtype="uint8")  # 生成image大小的全白图
im = np.full([256,256], 0,dtype="uint8")
cv2.imshow("Oringinal", im)
cv2.polylines(im, site, 1, 0)  # 在im上画site大小的线，1表示线段闭合，255表示线段颜色
cv2.fillPoly(im, site, 255)  # 在im的site区域，填充颜色为255

mask = im
#cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)  # 可调整窗口大小，不加这句不可调整
cv2.imshow("Mask", mask)
# masked = cv2.bitwise_and(image, image, mask=mask)  # 在模板mask上，将image和image做“与”操作
#cv2.namedWindow('Mask to Image', cv2.WINDOW_NORMAL)  # 同上
# cv2.imshow("Mask to Image", masked)
# cv2.imwrite('1.jpg',masked)
cv2.waitKey(0)  # 图像一直显示，键盘按任意键即可关闭窗口
cv2.imwrite('mask_24.png', mask)