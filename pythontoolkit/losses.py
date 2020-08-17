import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

# Define root mean sqared error loss function.
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def dice_coef(y_true, y_pred):
    smooth=1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    #print('2',y_pred.shape)
    #print('3',y_true.shape)
    return -dice_coef(y_true, y_pred)

# def dice(im1, im2):
# 	"""
# 	From: https://gist.github.com/JDWarner/6730747
#     Computes the Dice coefficient, a measure of set similarity.
#     Parameters
#     ----------
#     im1 : array-like, bool
#         Any array of arbitrary size. If not boolean, will be converted.
#     im2 : array-like, bool
#         Any other array of identical size. If not boolean, will be converted.
#     Returns
#     -------
#     dice : float
#         Dice coefficient as a float on range [0,1].
#         Maximum similarity = 1
#         No similarity = 0
        
#     Notes
#     -----
#     The order of inputs for `dice` is irrelevant. The result will be
#     identical if `im1` and `im2` are switched.
#     """
#     im1 = np.asarray(im1).astype(np.bool)
#     im2 = np.asarray(im2).astype(np.bool)

#     if im1.shape != im2.shape:
#         raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

#     # Compute Dice coefficient
#     intersection = np.logical_and(im1, im2)

#     return 1 - (2. * intersection.sum() / (im1.sum() + im2.sum()))