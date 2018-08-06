"""Main_Path_3D.py: Run script for a simple main path 3D CNN network calling"""

__author__ = "David McCoy"
__copyright__ = "Copyright 2018, Hemorrhage Detector Project @ UCSF"
__credits__ = []
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "David McCoy"
__email__ = "david.mccoy@ucsf.edu"
__status__ = "Under Construction"


from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers.pooling import AveragePooling3D
from keras.layers.pooling import GlobalAveragePooling3D
from keras.layers import Input, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K


def conv_block(x, norm_axis, growth_rate, nb_filter, dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 3x3x3 Conv3D, optional dropout
    :param x: Input keras network (some previous output layer as we go through the network)
    :param norm_axis: int -- index of contatenate axis
    :param nb_filter: int -- number of initial filters to produce from CNN
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: keras network with b_norm, relu and Conv3D added

    :rtype: keras network
    """
    x = BatchNormalization(axis=norm_axis, gamma_regularizer=l2(weight_decay),beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv3D(nb_filter, (3, 3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)


    return x, nb_filter


def Simple_3D_CNN(nb_classes, img_dim, depth, growth_rate, activation, nb_filter, dropout_rate=None, weight_decay=1E-4, nb_full_conn=1500):
    """ Build the DenseNet model
    :param nb_classes: int -- number of classes
    :param img_dim: tuple -- (channels, rows, columns)
    :param depth: int -- how many layers
    :param nb_dense_block: int -- number of dense blocks to add to end
    :param growth_rate: int -- number of filters to add
    :param activation: str -- type of activation to use in densenet
    :param nb_filter: int -- number of filters
    :param dropout_rate: float -- dropout rate
    :param weight_decay: float -- weight decay

    :returns: keras model with nb_layers of conv_factory appended

    :rtype: keras model
    """

    if K.image_dim_ordering() == "th":
        norm_axis = 1
    elif K.image_dim_ordering() == "tf":
        norm_axis = -1
    model_input = Input(shape=img_dim)

    x = Conv3D(nb_filter, (3, 3, 3), kernel_initializer="he_uniform", padding="same", name="initial_conv3D", use_bias=False, kernel_regularizer=l2(weight_decay))(model_input)

    # Add dense blocks
    for layer_idx in range(depth):
        nb_filter += growth_rate
        x, nb_filter = conv_block(x = x, norm_axis=norm_axis, nb_filter = nb_filter, growth_rate = growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(axis=norm_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling3D(data_format=K.image_data_format())(x)
    x = Dense(units=nb_classes, activation=activation, kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)
    main_path_model = Model(inputs=[model_input], outputs=[x], name="MainPathCNN")

    return main_path_model


