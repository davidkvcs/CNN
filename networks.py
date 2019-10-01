#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 10:57:19 2018

@author: claesnl
"""

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Conv3D, Conv3DTranspose, Dropout, Input
from keras.layers import Activation, BatchNormalization, concatenate
from keras import regularizers
import os

def unet(X, f, dims_out):
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

	def convt_block(layer, concat, fsize):
	    layer = Conv3DTranspose(fsize, kernel_size=3, kernel_regularizer=regularizers.l2(1e-1), 
	    						kernel_initializer='he_normal', padding='same', strides=2)(layer)
	    layer = BatchNormalization()(layer)
	    layer = Activation('relu')(layer)
	    layer = concatenate([layer, concat], axis=-1)
	    return layer

    # ENCODING
    block1, dblock1 = conv_block(X,f,.1) 
    block2, dblock2 = conv_block(dblock1,f*2**1,.1)
    block3, dblock3 = conv_block(dblock2,f*2**2,.2)
    block4, dblock4 = conv_block(dblock3,f*2**3,.2)
    block5, _ = conv_block(dblock4,f*2**4,.3,downsample=False)

    # DECODING
    block7 = convt_block(block5,block4,f*2**3) 
    block8, _ = conv_block(block7,f*2**3,.3,downsample=False)

    block9 = convt_block(block8,block3,f*2**2) 
    block10, _ = conv_block(block9,f*2**2,.2,downsample=False)

    block11 = convt_block(block10,block2,f*2**1)
    block12, _ = conv_block(block11,f*2**1,.2,downsample=False)

    block13 = convt_block(block12,block1,f)
    block14, _ = conv_block(block13,f,.1,downsample=False)

    output = Conv3D(dims_out,kernel_size=3, kernel_regularizer=regularizers.l2(1e-1), 
    				kernel_initializer='he_normal', padding='same',strides=1, activation='relu')(block14)
    return output
