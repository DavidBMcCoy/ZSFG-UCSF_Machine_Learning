#!/usr/bin/env python2

"""
Model adapted from:
Implementation of Inception-Residual Network v1 [Inception Network v4 Paper](http://arxiv.org/pdf/1602.07261v1.pdf) in Keras.

Some additional details:

    Each of the A, B and C blocks have a 'scale_residual' parameter as mentioned in the paper
    Default is ON
    Simply setting 'scale=True' in the create_inception_resnet_v1() method will add scaling.

Originally built for 2d image classification, updated for 3D data

Channel class is from Training deep neural-networks using a noise adaptation layer
This creates an additional softmax layer that maps the predicted probabilities for each class
from the base model to the 'noisy labels' with initialized weights from the confusion matrix
of the baseline model with no noise adaption"""

__author__ = "David McCoy"
__copyright__ = "Copyright 2018, Hemorrhage Detector Project @ UCSF"
__credits__ = ["Sara Dupont", "Grayhem Mills"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "David McCoy"
__email__ = "david.mccoy@ucsf.edu"
__status__ = "Operational"

from keras.layers import Input, merge, Dropout, Dense, Lambda, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling3D, AveragePooling3D, Conv3D
from keras.models import Model

from keras import backend as K
from numpy import genfromtxt

import warnings

warnings.filterwarnings('ignore')


## create noise adaption softmax layer
class Channel(Dense):
    """
    Implement simple noise adaptation layer.
    References
        Goldberger & Ben-Reuven, Training deep neural-networks using a noise
        adaptation layer, ICLR 2017
        https://openreview.net/forum?id=H12GRgcxg
    # Arguments
        output_dim: int > 0
              default is input_dim which is known at build time
        See Dense layer for more arguments. There is no bias and the arguments
        `bias`, `b_regularizer`, `b_constraint` are not used.
    """

    def __init__(self, units=None, **kwargs):
        kwargs['use_bias'] = False
        if 'activation' not in kwargs:
            kwargs['activation'] = 'softmax'
        super(Channel, self).__init__(units, **kwargs)

    def build(self, input_shape):
        if self.units is None:
            self.units = input_shape[-1]
        super(Channel, self).build(input_shape)

    def call(self, x, mask=None):
        """
        :param x: the output of a baseline classifier model passed as an input
        It has a shape of (batch_size, input_dim) and
        baseline_output.sum(axis=-1) == 1
        :param mask: ignored
        :return: the baseline output corrected by a channel matrix
        """
        # convert W to the channel probability (stochastic) matrix
        # channel_matrix.sum(axis=-1) == 1
        # channel_matrix.shape == (input_dim, input_dim)
        channel_matrix = self.activation(self.kernel)

        # multiply the channel matrix with the baseline output:
        # channel_matrix[0,0] is the probability that baseline output 0 will get
        #  to channeled_output 0
        # channel_matrix[0,1] is the probability that baseline output 0 will get
        #  to channeled_output 1 ...
        # ...
        # channel_matrix[1,0] is the probability that baseline output 1 will get
        #  to channeled_output 0 ...
        #
        # we want output[b,0] = x[b,0] * channel_matrix[0,0] + \
        #                              x[b,1] * channel_matrix[1,0] + ...
        # so we do a dot product of axis -1 in x with axis 0 in channel_matrix
        return K.dot(x, channel_matrix)


def inception_resnet_stem(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input Shape is 512 x 512 x=40 x 1 (tf) or 1 x 512 x 512 x 40 (th)
    c = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', strides=(2, 2, 1))(input)
    c = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', )(c)
    c = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', )(c)
    c = MaxPooling3D(pool_size=(3, 3, 3), strides=(3, 3, 1))(c)
    c = Conv3D(filters=80, kernel_size=(1, 1, 1), activation='relu', padding='same')(c)
    c = Conv3D(filters=192, kernel_size=(3, 3, 3), activation='relu')(c)
    c = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', strides=(2, 2, 1), padding='same')(c)
    b = BatchNormalization(axis=channel_axis)(c)
    b = Activation('relu')(b)
    return b


def inception_resnet_A(input, scale_residual=True):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input is relu activation
    init = input

    ir1 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(input)

    ir2 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(input)
    ir2 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(ir2)

    ir3 = Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', padding='same')(input)
    ir3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(ir3)
    ir3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(ir3)

    ir_merge = merge([ir1, ir2, ir3], concat_axis=channel_axis, mode='concat')

    ir_conv = Conv3D(filters=256, kernel_size=(1, 1, 1), activation='linear', padding='same')(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = merge([init, ir_conv], mode='sum')
    out = BatchNormalization(axis=channel_axis)(out)
    out = Activation("relu")(out)
    return out


def inception_resnet_B(input, scale_residual=True):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input is relu activation
    init = input

    ir1 = Conv3D(filters=128, kernel_size=(1, 1, 1), activation='relu', padding='same')(input)

    ir2 = Conv3D(filters=128, kernel_size=(1, 1, 1), activation='relu', padding='same')(input)
    ir2 = Conv3D(filters=128, kernel_size=(1, 7, 7), activation='relu', padding='same')(ir2)
    ir2 = Conv3D(filters=128, kernel_size=(7, 7, 1), activation='relu', padding='same')(ir2)

    ir_merge = merge([ir1, ir2], mode='concat', concat_axis=channel_axis)

    ir_conv = Conv3D(filters=896, kernel_size=(1, 1, 1), activation='linear', padding='same')(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = merge([init, ir_conv], mode='sum')
    out = BatchNormalization(axis=channel_axis)(out)
    out = Activation("relu")(out)
    return out


def inception_resnet_C(input, scale_residual=True):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input is relu activation
    init = input

    ir1 = Conv3D(filters=128, kernel_size=(1, 1, 1), activation='relu', padding='same')(input)

    ir2 = Conv3D(filters=192, kernel_size=(1, 1, 1), activation='relu', padding='same')(input)
    ir2 = Conv3D(filters=192, kernel_size=(1, 3, 3), activation='relu', padding='same')(ir2)
    ir2 = Conv3D(filters=192, kernel_size=(3, 3, 1), activation='relu', padding='same')(ir2)

    ir_merge = merge([ir1, ir2], mode='concat', concat_axis=channel_axis)

    ir_conv = Conv3D(filters=1792, kernel_size=(1, 1, 1), activation='linear', padding='same')(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = merge([init, ir_conv], mode='sum')
    out = BatchNormalization(axis=channel_axis)(out)
    out = Activation("relu")(out)
    return out


def reduction_A(input, k=192, l=224, m=256, n=384):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    r1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 1))(input)

    r2 = Conv3D(filters=n, kernel_size=(3, 3, 3), activation='relu', strides=(2, 2, 1))(input)

    r3 = Conv3D(filters=k, kernel_size=(1, 1, 1), activation='relu', padding='same')(input)
    r3 = Conv3D(filters=l, kernel_size=(3, 3, 3), activation='relu', padding='same')(r3)
    r3 = Conv3D(filters=m, kernel_size=(3, 3, 3), activation='relu', strides=(2, 2, 1))(r3)

    m = merge([r1, r2, r3], mode='concat', concat_axis=channel_axis)
    m = BatchNormalization(axis=channel_axis)(m)
    m = Activation('relu')(m)
    return m


def reduction_resnet_B(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    r1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 1), padding='valid')(input)

    r2 = Conv3D(filters=256, kernel_size=(1, 1, 1), activation='relu', border_mode='same')(input)
    r2 = Conv3D(filters=384, kernel_size=(3, 3, 3), activation='relu', strides=(2, 2, 1))(r2)

    r3 = Conv3D(filters=256, kernel_size=(1, 1, 1), activation='relu', padding='same')(input)
    r3 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', strides=(2, 2, 1))(r3)

    r4 = Conv3D(filters=256, kernel_size=(1, 1, 1), activation='relu', border_mode='same')(input)
    r4 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', border_mode='same')(r4)
    r4 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', strides=(2, 2, 1))(r4)

    m = merge([r1, r2, r3, r4], concat_axis=channel_axis, mode='concat')
    m = BatchNormalization(axis=channel_axis)(m)
    m = Activation('relu')(m)
    return m


def create_inception_resnet(nb_classes=2, scale=True, noise_adaption=False, nlayer_b1=5, nlayer_b2=10, nlayer_b3=5):
    if noise_adaption:
        CONFUSION = genfromtxt('/media/mccoyd2/hamburger/hemorrhage_study/results/confusion_matrix.csv', delimiter=',')
        CHANNEL_WEIGHTS = CONFUSION.copy()
        CHANNEL_WEIGHTS /= CHANNEL_WEIGHTS.sum(axis=1,
                                               keepdims=True)  # row-wise division of each confusion matrix item by row sum
        # take the log of the matrix and add an offset to prevent log 0 explosion
        CHANNEL_WEIGHTS = np.log(CHANNEL_WEIGHTS + 1e-8)
    '''
    Creates a inception resnet v1 network
    :param nb_classes: number of classes.txt
    :param scale: flag to add scaling of activations
    :return: Keras Model with 1 input (299x299x3) input shape and 2 outputs (final_output, auxiliary_output)
    '''

    if K.image_dim_ordering() == 'th':
        init = Input((1, 256, 256, 40))
    else:
        init = Input((256, 256, 40, 1))

    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    x = inception_resnet_stem(init)

    # 5 x Inception Resnet A
    for i in range(nlayer_b1):
        x = inception_resnet_A(x, scale_residual=scale)

    # Reduction A - From Inception v4
    x = reduction_A(x, k=192, l=192, m=256, n=384)

    # 10 x Inception Resnet B
    for i in range(nlayer_b2):
        x = inception_resnet_B(x, scale_residual=scale)

    # Auxiliary tower
    aux_out = AveragePooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2))(x)
    aux_out = Conv3D(filters=128, kernel_size=(1, 1, 1), padding='same', activation='relu')(aux_out)
    aux_out = Conv3D(filters=768, kernel_size=(3, 3, 3), activation='relu')(aux_out)
    aux_out = Flatten()(aux_out)
    aux_out = Dense(nb_classes, activation='sigmoid')(aux_out)

    # Reduction Resnet B
    x = reduction_resnet_B(x)

    # 5 x Inception Resnet C
    for i in range(nlayer_b3):
        x = inception_resnet_C(x, scale_residual=scale)

    # Average Pooling
    x = AveragePooling3D(pool_size=(4, 4, 4))(x)

    # Dropout
    x = Dropout(DROPOUT)(x)
    x = Flatten()(x)

    # Output
    base_network_out = Dense(output_dim=nb_classes, activation='sigmoid', name='base_network_channel')(x)

    if noise_adaption:
        channeled_output = Channel(name='noise_adaption_channel', weights=[CHANNEL_WEIGHTS])(base_network_out)
        inception_model = Model(input=init, output=[channeled_output, base_network_out, aux_out], name='Inception-Resnet-v1')
    else:
        inception_model = Model(input=init, output=[base_network_out, aux_out], name='Inception-Resnet-v1')

    return inception_model
