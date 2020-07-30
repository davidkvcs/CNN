"""
Created on Thu May 24 10:57:19 2018

@author: claesnl
"""
import tensorflow as tf
from tensorflow import keras
import warnings
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Dropout, Input
from tensorflow.keras.layers import Activation, BatchNormalization, concatenate
from tensorflow.keras import regularizers

warnings.filterwarnings('ignore')

# Define u-net structure.
def unet(X, f, dims_out):
    '''
    Inputs:
        X   inputs
        f   n_base_filters
        dims_out    output dimensions of network
    '''
    # Define convolution block:
    def conv_block(layer,fsize,dropout,downsample=True):
        for i in range(1,3):
            layer = Conv3D(fsize, kernel_size=3, kernel_regularizer=regularizers.l2(1e-1), 
                           kernel_initializer='he_normal', padding='same',strides=1)(layer)
            layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            layer = Dropout(dropout)(layer)
        if downsample:
            downsample = Conv3D(fsize*2, kernel_size=3, kernel_regularizer=regularizers.l2(1e-1), 
                                kernel_initializer='he_normal', padding='same', strides=2)(layer)
            downsample = BatchNormalization()(downsample)
            downsample = Activation('relu')(downsample)
        return layer, downsample

    # Define transposed convolution block:
    def convt_block(layer, concat, fsize):
        layer = Conv3DTranspose(fsize, kernel_size=3, kernel_regularizer=regularizers.l2(1e-1), 
                                kernel_initializer='he_normal', padding='same', strides=2)(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = concatenate([layer, concat], axis=-1) 
        return layer

    # Dropout values
    dropout = [.1,.1,.2,.3,.2,.2,.1]
    
    # ENCODING
    block1, dblock1 = conv_block(X,f,dropout[0]) 
    block2, dblock2 = conv_block(dblock1,f*2**1,dropout[1])
    block3, dblock3 = conv_block(dblock2,f*2**2,dropout[2])
    block4, _ = conv_block(dblock3,f*2**3,dropout[3],downsample=False)

    # DECODING
    block5 = convt_block(block4,block3,f*2**2) 
    block6, _ = conv_block(block5,f*2**2,dropout[4],downsample=False)

    block7 = convt_block(block6,block2,f*2**1) 
    block8, _ = conv_block(block7,f*2**1,dropout[5],downsample=False)

    block9 = convt_block(block8,block1,f)
    block10, _ = conv_block(block9,f,dropout[6],downsample=False)

    output = Conv3D(dims_out,kernel_size=3, kernel_regularizer=regularizers.l2(1e-1), 
                    kernel_initializer='he_normal', padding='same',strides=1, activation='relu')(block10)

    return output

