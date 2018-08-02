"""DenseNet_3D.py: Run script for dense network calling in augmentation and utilities functions"""

__author__ = "David McCoy"
__copyright__ = "Copyright 2018, Hemorrhage Detector Project @ UCSF"
__credits__ = ["tdeboissiere"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "David McCoy"
__email__ = "david.mccoy@ucsf.edu"
__status__ = "Under Construction"

from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv3D
from keras.layers.pooling import AveragePooling3D
from keras.layers.pooling import GlobalAveragePooling3D
from keras.layers import Input, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K


def conv_block(x, concat_axis, nb_filter, dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 3x3x3 Conv3D, optional dropout
    :param x: Input keras network (some previous output layer as we go through the network)
    :param concat_axis: int -- index of contatenate axis
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: keras network with b_norm, relu and Conv3D added

    :rtype: keras network
    """
    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv3D(nb_filter, (3, 3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x


def bottle_neck(x, concat_axis, nb_filter, dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 1x1 Conv3D, optional dropout and Maxpooling3D
    :param x: keras model
    :param concat_axis: int -- index of contatenate axis
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: model

    :rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool
    """
    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv3D(nb_filter, (1, 1, 1), kernel_initializer="he_uniform", padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling3D((2, 2, 2), strides=(2, 2, 1))(x)
    return x


def dense_block(x, concat_axis, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each conv_block is fed to subsequent ones
    :param x: keras model
    :param concat_axis: int -- index of contatenate axis
    :param nb_layers: int -- the number of layers of conv_factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras model with nb_layers of conv_factory appended

    :rtype: keras model
    """
    list_feat = [x]

    for i in range(nb_layers):
        x = conv_block(x, concat_axis, growth_rate, dropout_rate, weight_decay)
        list_feat.append(x)
        x = Concatenate(axis=concat_axis)(list_feat)
        nb_filter += growth_rate
    return x, nb_filter


def denseblock_altern(x, concat_axis, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each conv_block is fed to subsequent ones. (Alternative of a above)
    :param x: keras model
    :param concat_axis: int -- index of contatenate axis
    :param nb_layers: int -- the number of layers of conv_block to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: keras model with nb_layers of conv_factory appended

    :rtype: keras model

    * The main difference between this implementation and the implementation above is that the one above
    """

    for i in range(nb_layers):
        merge_tensor = conv_block(x, concat_axis, growth_rate, dropout_rate, weight_decay)
        x = Concatenate(axis=concat_axis)([merge_tensor, x])
        nb_filter += growth_rate

    return x, nb_filter


def DenseNet(nb_classes, img_dim, depth, nb_dense_block, growth_rate, nb_filter, dropout_rate=None, weight_decay=1E-4, activation):
    """ Build the DenseNet model
    :param nb_classes: int -- number of classes
    :param img_dim: tuple -- (channels, rows, columns)
    :param depth: int -- how many layers
    :param nb_dense_block: int -- number of dense blocks to add to end
    :param growth_rate: int -- number of filters to add
    :param nb_filter: int -- number of filters
    :param dropout_rate: float -- dropout rate
    :param weight_decay: float -- weight decay

    :returns: keras model with nb_layers of conv_factory appended

    :rtype: keras model
    """

    if K.image_dim_ordering() == "th":
        concat_axis = 1
    elif K.image_dim_ordering() == "tf":
        concat_axis = -1
    model_input = Input(shape=img_dim)

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)
    # Initial convolution
    x = Conv3D(nb_filter, (3, 3, 3), kernel_initializer="he_uniform", padding="same", name="initial_conv3D", use_bias=False, kernel_regularizer=l2(weight_decay))(model_input)

    # Add dense blocks

    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = dense_block(x, concat_axis, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
        # add transition
        x = bottle_neck(x, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)

    # The last denseblock does not have a transition
    x, nb_filter = dense_block(x, concat_axis, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling3D(data_format=K.image_data_format())(x)

    x = Dense(nb_classes, activation=activation, kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)
    dense_net = Model(inputs=[model_input], outputs=[x], name="DenseNet")

    return dense_net