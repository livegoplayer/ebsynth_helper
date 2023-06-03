import cv2
import numpy as np


def infer2(img_path: str, mask_path: str = '', out_img_path: str = ''):

    # 使用opencv叠加图片
    img1 = cv2.imread(img_path)
    img2 = cv2.imread(mask_path)
    # 反转
    matrix = 255 - np.asarray(img2)
    image = cv2.add(img1, matrix)
    print("save image to " + out_img_path)
    cv2.imwrite(out_img_path, image)