import numpy as np
from keras import backend as K
from all_params import SMOOTH

def dice_coef(y_true, y_pred):
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersection = K.sum(y_true_flat * y_pred_flat)
    return (2.0 * intersection + SMOOTH) / (K.sum(y_true_flat) + K.sum(y_pred_flat) + SMOOTH)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def np_dice_coef(y_true, y_pred):
    y_true_flat = y_true.flat[:]
    y_pred_flat = y_pred.flat[:]
    intersection = np.sum(y_true_flat * y_pred_flat)
    return (2.0 * intersection + SMOOTH) / (np.sum(y_true_flat) + np.sum(y_pred_flat) + SMOOTH)

def np_dice_coef_loss(y_true, y_pred):
    return 1.0 - np_dice_coef(y_true, y_pred)
