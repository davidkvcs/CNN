from keras import backend as K

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


#delineation loss
# def delineation_loss_rmse(y_true, y_pred):
# 	thres = 0.5
# 	y_pred[y_pred[...,0] > thres] = 1
# 	y_pred[y_pred[...,0] <= thres] = 0  
#     return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

#Dice loss

#Haussdorff Loss

#Absolute volume loss

#Volumetric differences loss