import os

import numpy as np
from PIL import Image
import cv2

from MODNet_entry.MODNet.src.models.modnet import MODNet


_here = os.path.dirname(os.path.abspath(__file__))


def infer2(img_path: str, mask_path: str = '', out_img_path: str = ''):

    # 使用opencv叠加图片
    img1 = cv2.imread(img_path)
    img2 = cv2.imread(mask_path)

    image = cv2.add(img1, img2)
    print("save image to " + out_img_path)
    cv2.imwrite(out_img_path, image)