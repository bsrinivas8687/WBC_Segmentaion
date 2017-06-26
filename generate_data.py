import os
import cv2
import numpy as np
from all_params import *

image_names = os.listdir(TRAIN_DATA_PATH)
image_names.sort()

img_num = IMG_START_NUM
for img_name in image_names:
    if 'mask' in img_name:
        continue

    img = cv2.imread(TRAIN_DATA_PATH + img_name)
    mask_img = cv2.imread(TRAIN_DATA_PATH + img_name.split('.')[0] + '-mask.jpg')
    if img.shape == (IMG_ROWS, IMG_COLS, 3):
        cv2.imwrite(NEW_TRAIN_DATA_PATH + img_name, img)
        cv2.imwrite(NEW_TRAIN_DATA_PATH + img_name.split('.')[0] + '-mask.jpg', mask_img)
        continue
    if CREATE_EXTRA_DATA == False:
        continue

    for row in range(0, img.shape[0], IMG_ROWS):
        for col in range(0, img.shape[1], IMG_COLS):
            new_img = img[row:row + IMG_ROWS, col:col + IMG_COLS, :]
            new_mask_img = mask_img[row:row + IMG_ROWS, col:col + IMG_COLS, :]
            if new_img.shape != (IMG_ROWS, IMG_COLS, 3) or np.max(new_mask_img) != 255.0:
                continue
            cv2.imwrite(NEW_TRAIN_DATA_PATH + 'train-' + str(img_num) + '.jpg', new_img)
            cv2.imwrite(NEW_TRAIN_DATA_PATH + 'train-' + str(img_num) + '-mask.jpg', new_mask_img)
            img_num = img_num + 1
