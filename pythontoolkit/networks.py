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
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, UpSampling2D, Cropping2D
from tensorflow.keras.layers import Activation, BatchNormalization, concatenate
from tensorflow.keras import regularizers
from tensorflow.keras.backend import max

warnings.filterwarnings('ignore')

# Define u-net structure.
def unet(X, f, dims_out):
    '''
    Inputs:
        X   inputs - the data generated in DataGenerator
        f   n_base_filters - self.config['n_base_filters']
        dims_out    output dimensions of network - dims_out=self.config['output_channels']
    '''
    print('X.shape = ' + str(X.shape))
    print('type(X) = ' + str(type(X)))
    print('f = ' + str(f))
    print('type(f) = ' + str(type(f)))
    print('dims_out = ' + str(dims_out))
    print('type(dims_out) = ' + str(type(dims_out)))
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
                    kernel_initializer='he_normal', padding='same',strides=1, activation='sigmoid')(block10)

    return output

def unet_dgk_test(X, f, dims_out):
    '''
    Inputs:
        X   inputs
        f   n_base_filters
        dims_out    output dimensions of network
    '''
    def conv_block(layer,fsize,downsample=True):
        for i in range(1,2):
            layer = Conv3D(fsize, kernel_size=3, kernel_regularizer=regularizers.l2(1e-1), 
                           kernel_initializer='he_normal', padding='same',strides=1)(layer)
            layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
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

    # encoding
    block1, dblock1 = conv_block(X,f)
    block2, dblock2 = conv_block(dblock1,f*2**1) 
    block3, dblock3 = conv_block(dblock2,f*2**2)
    block4, _ = conv_block(dblock3,f*2**3,downsample=False)

    #decoding
    block5 = convt_block(block4,block3,f*2**2) 
    block6, _ = conv_block(block5,f*2**2,downsample=False)

    block7 = convt_block(block6,block2,f*2**1) 
    block8, _ = conv_block(block7,f*2**1,downsample=False)

    block9 = convt_block(block8,block1,f)
    block10, _ = conv_block(block9,f,downsample=False)


    output = Conv3D(dims_out,kernel_size=3, kernel_regularizer=regularizers.l2(1e-1), 
                    kernel_initializer='he_normal', padding='same',strides=1, activation='softmax')(block10)
    print('network output shape = ' + str(output.shape))
    print('network output dtype = ' + str(output.dtype))
    print('network output type = ' + str(type(output)))
    print('network output max = ' + str(max(output)))
    #print('network output min = ' + str(np.amin(output)))
    #print('network output mean = ' + str(np.mean(output)))
    return output

    ### ----define U-net architecture--------------
    #TODO: IMPLEMENT AMAILES NETWORK AND TEST IF WORKDS ON DATA"
    #https://github.com/FourierX9/wmh_ibbmTum/blob/master/train_leave_one_out.py

def unet_2D_amalie(X, img_shape, weights_file=None, custom_load_func = False):

    dim_ordering = 'channels_last'
    #inputs = Input(shape = img_shape)
    print('X.shape = ' + str(X.shape))
    concat_axis = -1
    ### the size of convolutional kernels is defined here    
    conv1 = Conv2D(64, 5, 5, activation = 'relu', padding='same', input_shape = img_shape, data_format=dim_ordering, name='conv1_1')(X)
    print('conv1.shape = ' + str(conv1.shape))
    do = Dropout(0.2)(conv1)
    print('do.shape = ' + str(do.shape))
    conv1 = Conv2D(64, 5, 5, activation = 'relu', padding='same', data_format=dim_ordering)(do)
    print('conv1.shape = ' + str(conv1.shape))
    do = Dropout(0.2)(conv1)
    print('do.shape = ' + str(do.shape))
    pool1 = MaxPool2D(pool_size=(2, 2), padding='same', data_format=dim_ordering)(do) #, data_format=dim_ordering
    print('pool1.shape = ' + str(pool1.shape))

    conv2 = Conv2D(96, 3, 3, activation = 'relu', padding='same', data_format=dim_ordering)(pool1)
    do2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(96, 3, 3, activation = 'relu', padding='same', data_format=dim_ordering)(do2)
    do2 = Dropout(0.2)(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2), padding='same', data_format=dim_ordering)(do2) #, data_format=dim_ordering
    print('pool2.shape = ' + str(pool2.shape))

    conv3 = Conv2D(128, 3, 3,activation = 'relu',  padding='same', data_format=dim_ordering)(pool2)
    do3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, 3, 3,activation = 'relu',  padding='same', data_format=dim_ordering)(do3)
    do3 = Dropout(0.2)(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2), padding='same', data_format=dim_ordering)(do3) #, data_format=dim_ordering
    print('pool3.shape = ' + str(pool3.shape))
    
    conv4 = Conv2D(256, 3, 3,activation = 'relu',  padding='same', data_format=dim_ordering)(pool3)
    do4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(256, 4, 4,activation = 'relu',  padding='same', data_format=dim_ordering)(do4)
    do4 = Dropout(0.2)(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2), padding='same', data_format=dim_ordering)(do4) #, data_format=dim_ordering
    print('pool4.shape = ' + str(pool4.shape))

    conv5 = Conv2D(512, 3, 3,activation = 'relu',  padding='same', data_format=dim_ordering)(pool4)
    do5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(512, 3, 3,activation = 'relu',  padding='same', data_format=dim_ordering)(do5)
    do5 = Dropout(0.2)(conv5)
    print('do5.shape = ' + str(do5.shape))

    up_conv5 = UpSampling2D(size=(2, 2), data_format=dim_ordering)(do5)
    ch, cw = get_crop_shape(conv4, up_conv5)
    crop_conv4 = Cropping2D(cropping=(ch,cw), data_format=dim_ordering)(conv4)
    up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)     #Amalie chaned it from merge to concatenate
    conv6 = Conv2D(256, 3, 3,activation = 'relu',  padding='same', data_format=dim_ordering)(up6)
    do6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(256, 3, 3,activation = 'relu',  padding='same', data_format=dim_ordering)(do6)
    do6 = Dropout(0.2)(conv6)

    up_conv6 = UpSampling2D(size=(2, 2), data_format=dim_ordering)(do6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping2D(cropping=(ch,cw), data_format=dim_ordering)(conv3)
    up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
    conv7 = Conv2D(128, 3, 3,activation = 'relu',  padding='same', data_format=dim_ordering)(up7)
    do7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(128, 3, 3,activation = 'relu',  padding='same', data_format=dim_ordering)(do7)
    do7 = Dropout(0.2)(conv7)

    up_conv7 = UpSampling2D(size=(2, 2), data_format=dim_ordering)(do7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = Cropping2D(cropping=(ch,cw), data_format=dim_ordering)(conv2)
    up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
    conv8 = Conv2D(96, 3, 3, activation = 'relu', padding='same', data_format=dim_ordering)(up8)
    do8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(96, 3, 3, activation = 'relu', padding='same', data_format=dim_ordering)(do8)
    do8 = Dropout(0.2)(conv8)

    up_conv8 = UpSampling2D(size=(2, 2), data_format=dim_ordering)(do8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch,cw), data_format=dim_ordering)(conv1)
    up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = Conv2D(64, 3, 3, activation = 'relu', padding='same', data_format=dim_ordering)(up9)
    do9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(64, 3, 3, activation = 'relu', padding='same', data_format=dim_ordering)(do9)
    do9 = Dropout(0.2)(conv9)

    ch, cw = get_crop_shape(X, do9)
    conv9 = ZeroPadding2D(padding=(ch, cw), data_format=dim_ordering)(conv9)
    output = Conv2D(1, 1, 1, activation='sigmoid', data_format=dim_ordering)(conv9)
    #model = Model(input=inputs, output=conv10)
    
    if not weights_file == None:
        j = 0
        i = 0
        oldlayers=[]
        if custom_load_func:
            #pass # TODO
            img_shape_old=(rows_standard, cols_standard, 3)
            model_old = get_unet(img_shape_old,weights_file,custom_load_func=False)
            for layer in model.layers:
            
               if layer.name.startswith('conv'):
                   
                   
                   for layer_old in model_old.layers:
                       if layer_old.name.startswith('conv'):
                           oldlayers.append(layer_old)
                       i += 1
                   old_weight = oldlayers[j].get_weights()
                   layer.set_weights(old_weight)
                   
                   j += 1
        else:
            model.load_weights(weights_file)
    
    #model.compile(optimizer=Adam(lr=(1e-5)), loss=dice_coef_loss, metrics=[dice_coef_for_training])
    

    return output

'''
#def get_crop_shape(target, refer):
        # width, the 3rd dimension
        print('type(target) = ' + str(type(target)))
        print('target.shape = ' + str(target.shape))
        print('refer.shape = ' + str(refer.shape))
        cw = (target.shape[3] - refer.shape[3])#.value
        print('cw = ' + str(cw))
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.shape[1] - refer.shape[1])#.value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)
'''

def unet_2D_david(X, f, dims_out):
    '''
    Unet adapted from 
    Mu, Yiping, Qi Li, and Yang Zhang. “White Matter Segmentation Algorithm for DTI Images Based on Super-Pixel Full Convolutional Network.” Journal of Medical Systems 43.9 (2019): 1–10. Web.
    Github: https://github.com/hongweilibran/wmh_ibbmTum/blob/master/train_leave_one_out.py
    Inputs:
        X   inputs - the data generated in DataGenerator
        f   n_base_filters - self.config['n_base_filters']
        dims_out    output dimensions of network - dims_out=self.config['output_channels']
    TODO: Add Cropping if dimensions not 2**X    
    '''
    print('X.shape = ' + str(X.shape))
    concat_axis = -1
    ### the size of convolutional kernels is defined here    
    def conv_bn_relu(nd, kernel_size=3, inputs=None):
        conv = Conv2D(nd, kernel_size, padding='same',activation = 'relu')(inputs) #, kernel_initializer='he_normal'
        #bn = BatchNormalization()(conv)
        #relu = Activation('relu')(conv)
        return conv
    '''
    def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (refer.shape[3] - target.shape[3])
        print(target.shape)
        print(refer.shape)
        print('cw = ' + str(cw))
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (refer.shape[2] - target.shape[2])
        print('ch = ' + str(ch))
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)
        '''

    conv1 = conv_bn_relu(f, 5, X)
    conv1 = conv_bn_relu(f, 5, conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
    
    conv2 = conv_bn_relu(f*2-2**5, 3, pool1)
    conv2 = conv_bn_relu(f*2-2**5, 3, conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
    
    conv3 = conv_bn_relu(f*2, 3, pool2)
    conv3 = conv_bn_relu(f*2, 3, conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
    
    conv4 = conv_bn_relu(f*2**2, 3, pool3)
    conv4 = conv_bn_relu(f*2**2, 3, conv4)
    pool4 = MaxPool2D(pool_size=(2, 2))(conv4)

    conv5 = conv_bn_relu(f*2**3, 3, pool4)
    conv5 = conv_bn_relu(f*2**3, 3, conv5)
    pool5 = MaxPool2D(pool_size=(2, 2))(conv5)

    up_conv5 = UpSampling2D(size=(2, 2))(conv5)
    up6 = concatenate([up_conv5, conv4], axis=concat_axis)
    conv6 = conv_bn_relu(256, 3, up6)
    conv6 = conv_bn_relu(256, 3, conv6)
    
    up_conv6 = UpSampling2D(size=(2, 2))(conv6)
    up7 = concatenate([up_conv6, conv3], axis=concat_axis)
    conv7 = conv_bn_relu(128, 3, up7)
    conv7 = conv_bn_relu(128, 3, conv7)

    up_conv7 = UpSampling2D(size=(2, 2))(conv7)
    up8 = concatenate([up_conv7, conv2], axis=concat_axis)
    conv8 = conv_bn_relu(96, 3, up8)
    conv8 = conv_bn_relu(96, 3, conv8)

    up_conv8 = UpSampling2D(size=(2, 2))(conv8)
    up9 = concatenate([up_conv8, conv1], axis=concat_axis)
    conv9 = conv_bn_relu(64, 3, up9)
    conv9 = conv_bn_relu(64, 3, conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid', padding='same')(conv9)
               
    return conv10                    

    