# coding=utf-8
"""Resnets_3D.py: Run script for inception network calling in augmentation and utilities functions"""

__author__ = "David McCoy"
__copyright__ = "Copyright 2018, Hemorrhage Detector Project @ UCSF"
__credits__ = ["Sara Dupont"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "David McCoy"
__email__ = "david.mccoy@ucsf.edu"
__status__ = "Under Construction"
# -*- coding: utf-8 -*-

import keras
from keras.layers import Input, Add, Dense, Activation, ZeroPadding3D, BatchNormalization, Flatten, \
    Conv3D, AveragePooling3D, MaxPooling3D
from keras.models import Model
from keras.initializers import glorot_uniform

import keras.backend as K

K.set_image_data_format('channels_last')
K.set_learning_phase(1)

"""
50 layer and 98 layer Resnet architecture for volumetric images
This script was originally designed for noncontrast CT all etiology hemmorhage detection
By changing the dimensions of the input tensor, this script should work for any volumetric image
However, this script is not built to take in number of layers as input and therefore, you may need to change 
the number of layers to avoid reducing the image h, w, or d to 0 in which case you will get an error such as 
cannot do x, y, z with -1 layer or something like that...
"""


def identity_block_3d(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3 of the google paper

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_D, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv3D(filters=F1, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=4, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)

    X = Conv3D(filters=F2, kernel_size=(f, f, f), strides=(1, 1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=4, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)

    X = Conv3D(filters=F3, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=4, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block_3d(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # First component of main path
    X = Conv3D(filters=F1, kernel_size=(1, 1, 1), strides=(s, s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=4, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv3D(filters=F2, kernel_size=(f, f, f), strides=(1, 1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=4, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv3D(filters=F3, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=4, name=bn_name_base + '2c')(X)

    X_shortcut = Conv3D(filters=F3, kernel_size=(1, 1, 1), strides=(s, s, s), padding='valid',
                        name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=4, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def resnet_50_3d(input_shape, classes):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding3D((3, 3, 3))(X_input)

    # Stage 1
    X = Conv3D(64, (7, 7, 7), strides=(2, 2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=4, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(X)

    # Stage 2
    X = convolutional_block_3d(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block_3d(X, f=3, filters=[64, 64, 256], stage=2, block='b')
    X = identity_block_3d(X, 3, [64, 64, 256], stage=2, block='c')

    X = convolutional_block_3d(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block_3d(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block_3d(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block_3d(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block_3d(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block_3d(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block_3d(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block_3d(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block_3d(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block_3d(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = X = convolutional_block_3d(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block_3d(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block_3d(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling3D(pool_size=(2, 2, 2), padding='same')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='sigmoid', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model


def resnet_98_3d(input_shape, classes):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding3D((3, 3, 3))(X_input)

    # Stage 1
    X = Conv3D(64, (7, 7, 7), strides=(2, 2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0),
               activation=None)(X)
    X = BatchNormalization(axis=4, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(X)

    # Stage 2
    X = convolutional_block_3d(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block_3d(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block_3d(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block_3d(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block_3d(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block_3d(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block_3d(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block_3d(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block_3d(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block_3d(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block_3d(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block_3d(X, 3, [256, 256, 1024], stage=4, block='e')

    # Stage 5
    X = convolutional_block_3d(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block_3d(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block_3d(X, 3, [512, 512, 2048], stage=5, block='c')
    X = identity_block_3d(X, 3, [512, 512, 2048], stage=5, block='d')
    X = identity_block_3d(X, 3, [512, 512, 2048], stage=5, block='e')
    X = identity_block_3d(X, 3, [512, 512, 2048], stage=5, block='f')

    # Stage 6
    X = convolutional_block_3d(X, f=3, filters=[1024, 1024, 4096], stage=6, block='a', s=2)
    X = identity_block_3d(X, 3, [1024, 1024, 4096], stage=6, block='b')
    X = identity_block_3d(X, 3, [1024, 1024, 4096], stage=6, block='c')
    X = identity_block_3d(X, 3, [1024, 1024, 4096], stage=6, block='d')
    X = identity_block_3d(X, 3, [1024, 1024, 4096], stage=6, block='e')

    # Stage 7
    X = convolutional_block_3d(X, f=3, filters=[2048, 2048, 8192], stage=7, block='a', s=2)
    X = identity_block_3d(X, 3, [2048, 2048, 8192], stage=7, block='b')
    X = identity_block_3d(X, 3, [2048, 2048, 8192], stage=7, block='c')
    X = identity_block_3d(X, 3, [2048, 2048, 8192], stage=7, block='d')

    # Stage 8
    X = convolutional_block_3d(X, f=3, filters=[4096, 4096, 16384], stage=8, block='a', s=2)
    X = identity_block_3d(X, 3, [4096, 4096, 16384], stage=8, block='b')
    X = identity_block_3d(X, 3, [4096, 4096, 16384], stage=8, block='c')

    # Stage 9
    X = convolutional_block_3d(X, f=3, filters=[8192, 8192, 32768], stage=9, block='a', s=2)
    X = identity_block_3d(X, 3, [8192, 8192, 32768], stage=9, block='b')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling3D(pool_size=(2, 2, 2), padding='same')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='sigmoid', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet98_3D')

    return model

# so our run model will call in this function, by doing so it will compile the model and use model_50_parallel and best_wts_callback
