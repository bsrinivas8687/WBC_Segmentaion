import os
import cv2
import numpy as np
from model import get_model
from skimage.measure import label
from skimage.measure import regionprops
from all_params import *

image_names = os.listdir(TEST_DATA_PATH)
image_names.sort()
model = get_model(train=False)
model.load_weights(WEIGHTS)

for img_name in image_names:
    mask_img_name = img_name.split('.')[0] + '-mask.jpg'
    img = cv2.imread(TEST_DATA_PATH + img_name, cv2.IMREAD_GRAYSCALE)
    img[img <= CLEAN_THRESH] = 255
    mask_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    ret, thresh_img = cv2.threshold(img, THRESH, 255, cv2.THRESH_BINARY_INV)
    img_label = label(thresh_img)
    for region in regionprops(img_label):
        minr, minc, maxr, maxc = region.bbox
        if region.area < 500:
            continue
        r, c = (minr + maxr - IMG_ROWS) / 2, (minc + maxc - IMG_COLS) / 2
        if r < 0:
            r = 0
        if c < 0:
            c = 0
        if r + IMG_ROWS > img.shape[0]:
            r = img.shape[0] - IMG_ROWS
        if c + IMG_COLS > img.shape[1]:
            c = img.shape[1] - IMG_COLS
        test_img = img[r:r + IMG_ROWS, c:c + IMG_COLS]
        if test_img.shape != (IMG_ROWS, IMG_COLS):
            continue

        test_img = np.array([[test_img]], dtype=np.float32)
        test_img /= 255.0
        test_mask_img = model.predict(test_img.transpose(0, 2, 3, 1), verbose=1)
        test_mask_img = (test_mask_img * 255.0).astype(np.uint8)
        test_mask_img = test_mask_img.transpose(0, 3, 1, 2)[0][0]
        
        mask_img[r:r + IMG_ROWS, c:c + IMG_COLS] = test_mask_img
        cv2.imwrite(SUBMISSION_DATA_PATH + mask_img_name, mask_img)
