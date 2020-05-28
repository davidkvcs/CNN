import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

# Define root mean sqared error loss function.
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))