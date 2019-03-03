import skimage.io as io
import skimage.color as color
import glob
import random
import os
from skimage import filters, util
import numpy as np

def hsvtransform(img):
    hsv_img = color.rgb2hsv(img)
    hsv_img[:, :, 0] += random.randint(-25, 25) / 255
    hsv_img[:, :, 1] += random.randint(-10, 10) / 255
    hsv_img[:, :, 2] += random.randint(-10, 10) / 255
    new_img = color.hsv2rgb(hsv_img) * 255
    return new_img.astype(np.uint8)


img_frames = sorted(glob.glob("./data/train/image/*"))
mask_frames = sorted(glob.glob("./data/train/mask/*"))

img_saved_path = "./data/train_aug/image/"
mask_saved_path = "./data/train_aug/mask/"

img_counter = 0

img_num = 0

for (img_path, mask_path) in zip(img_frames, mask_frames):
    img = io.imread(img_path)
    mask = io.imread(mask_path)
    # 保存原图像
    io.imsave(os.path.join(img_saved_path, "%d.png" % img_counter), img)
    io.imsave(os.path.join(mask_saved_path, "%d.png" % img_counter), mask)
    img_counter += 1
    # 高斯滤波
    gaussian_img = filters.gaussian(img, sigma=2) * 255
    io.imsave(os.path.join(img_saved_path, "%d.png" % img_counter), gaussian_img.astype(np.uint8))
    io.imsave(os.path.join(mask_saved_path, "%d.png" % img_counter), mask)
    img_counter += 1
    # 椒盐噪声
    pepper_img = util.random_noise(img, mode="pepper", amount=0.15) * 255
    io.imsave(os.path.join(img_saved_path, "%d.png" % img_counter), pepper_img.astype(np.uint8))
    io.imsave(os.path.join(mask_saved_path, "%d.png" % img_counter), mask)
    img_counter += 1
    for i in range(5):
        new_img = hsvtransform(img)
        io.imsave(os.path.join(img_saved_path, "%d.png" % img_counter), new_img)
        io.imsave(os.path.join(mask_saved_path, "%d.png" % img_counter), mask)
        img_counter += 1
    print("Image %d finished!" % img_num)
    img_num += 1
