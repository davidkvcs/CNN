from keras import backend as K
import numpy as np

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def dice(y_true, y_pred):
    #print(np.shape(y_pred))
    smooth = 1
    print('Calculating dice coefficient')
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    #print(np.shape(y_pred))
    #print(np.shape(y_true))
    print(K.int_shape(y_pred))
    print(K.int_shape(y_true))
    return -dice(y_true, y_pred)
