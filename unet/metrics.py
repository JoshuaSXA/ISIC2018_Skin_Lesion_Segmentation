from keras import backend as K
import numpy as np


smooth_default = 1

# dice_coeff and dice_loss for train and val dataset.
def dice_coef(y_true, y_pred, smooth=smooth_default):
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersec = 2. * K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) + smooth
    union = K.sum(y_true_f, axis=1, keepdims=True) + K.sum(y_pred_f, axis=1, keepdims=True) + smooth
    return K.mean(intersec / union)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# jacc_coeff and jacc_loss for train and val dataset.
def jacc_coef(y_true, y_pred, smooth = smooth_default):
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f, axis=1, keepdims=True)
    return K.mean((intersection + smooth) / (K.sum(y_true_f, axis=1, keepdims=True) + K.sum(y_pred_f, axis=1, keepdims=True) - intersection + smooth))

def jacc_loss(y_true, y_pred):
    return 1 - jacc_coef(y_true, y_pred)


# The dice_coeff and jacc_coeff for numpy version.
def test_dice_coef(mask_true, mask_pred, smooth=1):
    mask_true_f = mask_true.flatten()
    mask_pred_f = mask_pred.flatten()
    intersec = 2. * np.sum(mask_true_f * mask_pred_f) + smooth
    union = np.sum(mask_true_f) + np.sum(mask_pred_f) + smooth
    return intersec / union

def test_jacc_coef(mask_true, mask_pred, smooth = 1):
    mask_true_f = mask_true.flatten()
    mask_pred_f = mask_pred.flatten()
    intersection = np.sum(mask_true_f * mask_pred_f)
    return (intersection + smooth) / (np.sum(mask_true_f) + np.sum(mask_pred_f) - intersection + smooth)

<<<<<<< HEAD:unet/metrics.py


def bce_jacc_loss(y_true, y_pred):
    return jacc_loss(y_true, y_pred) + K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
=======
def bce_jacc_loss(y_true, y_pred):
    return jacc_loss(y_true, y_pred) + K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
>>>>>>> 14504f661ae21de4b3d2fd0affb5f0ddae239b75:metrics.py
