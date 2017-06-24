import os
import cv2
import numpy as np
from all_params import *

def get_train_data(path=NEW_TRAIN_DATA_PATH, augment=AUGMENT_TRAIN_DATA):
    image_names = os.listdir(path)
    image_names.sort()
    images_count = len(image_names) / 2
    
    if augment == True:
        images_count = images_count * 3
    X_train = np.ndarray((images_count, 1, IMG_ROWS, IMG_COLS), dtype=np.uint8)
    Y_train = np.ndarray((images_count, 1, IMG_ROWS, IMG_COLS), dtype=np.uint8)

    i = 0
    for img_name in image_names:
        if 'mask' in img_name:
            continue
        mask_img_name = img_name.split('.')[0] + '-mask.jpg'
        img = cv2.imread(path + img_name, cv2.IMREAD_GRAYSCALE)
        img[img <= CLEAN_THRESH] = 255
        mask_img = cv2.imread(path + mask_img_name, cv2.IMREAD_GRAYSCALE)
        X_train[i] = np.array([img])
        Y_train[i] = np.array([mask_img])
        i = i + 1

        if augment == True:
            X_train[i] = np.array([img[:, ::-1]])
            Y_train[i] = np.array([mask_img[:, ::-1]])
            i = i + 1
            X_train[i] = np.array([img[::-1, :]])
            Y_train[i] = np.array([mask_img[::-1, :]])
            i = i + 1

    X_train = X_train.transpose((0, 2, 3, 1))
    Y_train = Y_train.transpose((0, 2, 3, 1))
    return X_train, Y_train
