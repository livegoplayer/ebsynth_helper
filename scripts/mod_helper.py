import shutil

import cv2
import numpy as np
import os


def infer2(img_path: str, mask_path: str = '', out_img_path: str = ''):
    # 使用opencv叠加图片
    img1 = cv2.imread(img_path, -1)
    img2 = cv2.imread(mask_path, -1)
    # 反转
    image = cutout_by_mask(img1, img2)

    save(out_img_path, to_transparent(image))


def split_by_mask(img_path: str, mask_path: str = '', output_dir: str = ''):
    # 使用opencv叠加图片
    img1 = cv2.imread(img_path, -1)
    img2 = cv2.imread(mask_path, -1)
    # 反转
    matrix = 255 - np.asarray(img2)
    mainImage = cutout_by_mask(img1, img2)
    subImage = cutout_by_mask(img1, matrix)

    mainImagePath = os.path.join(output_dir, "main_image")
    subImagePath = os.path.join(output_dir, "sub_image")

    if os.path.exists(mainImagePath):
        shutil.rmtree(mainImagePath, ignore_errors=True)
    os.makedirs(mainImagePath)

    if os.path.exists(subImagePath):
        shutil.rmtree(subImagePath, ignore_errors=True)
    os.makedirs(subImagePath)
    save(mainImagePath, to_transparent(mainImage))
    save(subImagePath, to_transparent(subImage))


def save(img_path, img):
    print("save image to " + img_path)
    cv2.imwrite(img_path, img)


# 白色转透明
def to_transparent(image):
    bigimg4 = np.ones((image.shape[0], image.shape[1], 4)) * 255  # 4通道底图，第4个通道设为透明度
    xs, ys = np.where(np.sum(image, axis=2) == 255 * 3)  # 前三个通道均为 255 的像素点，设为透明

    for x, y in zip(xs, ys):
        bigimg4[x, y, 3] = 0
    return bigimg4


def adjust_mask(mainPreImagePath, preSubImagePath, adjustmentImagePath, output_dir):
    mainPreImage = cv2.imread(mainPreImagePath, -1)
    preSubImage = cv2.imread(preSubImagePath, -1)
    adjustmentImage = cv2.imread(adjustmentImagePath, -1)

    # 校验
    height = adjustmentImage.shape[0]  # 将tuple中的元素取出，赋值给height，width，channels
    width = adjustmentImage.shape[1]

    if preSubImage.shape[0] != height or preSubImage.shape[1] != width:
        print(
            "the preSubImage " + preSubImage + "does not match the size of " + "[" + width + "," + height + "]" + " path= " + adjustmentImage)
        print("skip")

    if mainPreImage.shape[0] != height or mainPreImage.shape[1] != width:
        print(
            "the mainPreImage " + mainPreImage + "does not match the size of " + "[" + width + "," + height + "]" + " path= " + adjustmentImage)
        print("skip")

    adjustmentImageMask = foreground_to_mask(adjustmentImage)
    preSubImageMask = foreground_to_mask(preSubImage)
    mainPreImageMask = foreground_to_mask(mainPreImage)

    height = adjustmentImageMask.shape[0]
    width = adjustmentImageMask.shape[1]

    # 这里是针对main图像的命名方法，到sub中会取反
    white_change_group = []
    black_change_group = []
    for row in range(height):  # 遍历每一行
        for col in range(width):
            if adjustmentImageMask[row][col] != mainPreImage[row][col]:
                if np.sum(adjustmentImageMask[row][col]) < 255 * 3:
                    if np.sum(mainPreImageMask[row][col]) == 255 * 3:
                        black_change_group.append([row, col])
                else:
                    if np.sum(mainPreImageMask[row][col]) < 255 * 3:
                        white_change_group.append([row, col])

    for x, y in white_change_group:
        preSubImageMask[x, y] = [0, 0, 0]

    for x, y in black_change_group:
        preSubImageMask[x, y] = [255, 255, 255]

    outputMainMask = adjustmentImageMask
    outputSubMask = preSubImageMask

    mainImagePath = os.path.join(output_dir, "main_mask")
    subImagePath = os.path.join(output_dir, "sub_mask")

    if os.path.exists(mainImagePath):
        shutil.rmtree(mainImagePath, ignore_errors=True)
    os.makedirs(mainImagePath)

    if os.path.exists(subImagePath):
        shutil.rmtree(subImagePath, ignore_errors=True)
    os.makedirs(subImagePath)
    save(mainImagePath, outputMainMask)
    save(subImagePath, outputSubMask)


def generate_sub_by_foreground_img(imgPath, mainImgPath, subOutputPath):
    img = cv2.imread(imgPath, -1)
    imgMask = foreground_to_mask(img)

    if not os.path.exists(mainImgPath):
        mainImg = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        mainMask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    else:
        mainImg = cv2.imread(mainImgPath, -1)
        mainMask = foreground_to_mask(mainImg)

    imgPathDir, filename = os.path.split(imgPath)
    imgMaskDir = os.path.join(imgPathDir, "mask")
    if not os.path.exists(imgMaskDir):
        os.makedirs(imgMaskDir)
    imgMaskDirPath = os.path.join(imgMaskDir, filename)
    if os.path.exists(imgMaskDirPath):
        os.remove(imgMaskDirPath)

    mainImgPathDir, filename = os.path.split(mainImgPath)
    mainMaskDir = os.path.join(mainImgPathDir, "mask")
    if not os.path.exists(mainMaskDir):
        os.makedirs(mainMaskDir)
    mainMaskDirPath = os.path.join(mainMaskDir, filename)
    if os.path.exists(mainMaskDirPath):
        os.remove(mainMaskDirPath)

    subOutputPathDir, filename = os.path.split(subOutputPath)
    subMaskDir = os.path.join(subOutputPathDir, "mask")
    if not os.path.exists(subMaskDir):
        os.makedirs(subMaskDir)
    subMaskDirPath = os.path.join(subMaskDir, filename)

    height = img.shape[0]
    width = img.shape[1]
    if mainImg.shape[0] != height or mainImg.shape[1] != width:
        print(
            "the mainImg " + mainImgPath + "does not match the size of " + "[" + width + "," + height + "]" + " path= " + imgPath)
        print("skip")

    if not os.path.exists(mainImgPath):
        print("the main img for " + imgPath + " is not exists path = " + mainImgPath)
        print("use the raw img as sub img")
        subOutputImg = img
    else:
        # main 反过来，然后使用main的反遮罩取raw的mask
        subOutputImg = cutout_by_mask(img, mainMask)

    subOutputImgMask = foreground_to_mask(subOutputImg)

    save(imgMaskDirPath, imgMask)
    save(mainMaskDirPath, mainMask)
    save(subOutputPath, subOutputImg)
    save(subMaskDirPath, subOutputImgMask)

def cutout_by_mask(image, mask):
    # 如果三通道，就取白色
    # bigimg4 = np.ones((image.shape[0], image.shape[1], 3))
    bigimg4 = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    height, width, channels = mask.shape
    # 黑白遮罩
    if channels == 3:
        bigimg4[:, :, :3] = image
        for i in range(height):
            for j in range(width):
                if mask[i, j].tolist() != [255.0, 255.0, 255.0]:
                    bigimg4[i, j, 3] = 255

    # 透明遮罩
    if channels == 4:
        bigimg4[:, :, :3] = image
        for i in range(height):
            for j in range(width):
                if mask[i, j, 3] == 0:
                    bigimg4[i, j, 3] = 255

def foreground_to_mask(image):
    # 如果三通道，就取白色
    # bigimg4 = np.ones((image.shape[0], image.shape[1], 3))
    bigimg4 = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    height, width, channels = image.shape
    print(channels)
    if channels == 3:
        bigimg4[:, :, :] = image
        for i in range(height):
            for j in range(width):
                if image[i, j, :].tolist() != [255.0, 255.0, 255.0]:
                    bigimg4[i, j, :] = np.array([0, 0, 0])

    # 如果四通道，先判断透明度是否有透明，再决定取值
    if channels == 4:
        not_show_pixels = np.where(image[:, :, 3] != 255)
        print(not_show_pixels)
        if len(list(not_show_pixels)) >= 0 and len(list(not_show_pixels[0])) > 0:
            bigimg4[not_show_pixels] = [0, 0, 0]
        else:
            height, width, channels = image.shape
            bigimg4[:, :, :] = image[:, :, :3]
            for i in range(height):
                for j in range(width):
                    if image[i, j, :].tolist() != [255.0, 255.0, 255.0]:
                        bigimg4[i, j, :] = np.array([0, 0, 0])

        # for x, y in zip(xs, ys):
        #   bigimg4[x, y] = [255, 255, 255]
    return bigimg4
