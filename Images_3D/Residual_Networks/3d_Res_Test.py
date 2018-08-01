
# -*- coding: utf-8 -*-
"""
Written by David McCoy
50 layer and 98 layer Resnet architecture for volumetric images
This script was designed for noncontrast CT all etiology hemmorhage detection
"""
import keras
from keras.layers import Input, Add, Dense, Activation, ZeroPadding3D,ZeroPadding2D,  BatchNormalization, Flatten, Conv2D, Conv3D, AveragePooling2D, AveragePooling3D, MaxPooling3D, MaxPooling2D, GlobalMaxPooling3D
from keras.models import Model, load_model

from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint

from keras.utils import multi_gpu_model
from resnets_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

GPU = 2

def identity_block_3D(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv3D(filters=F1, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=4, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv3D(filters=F2, kernel_size=(f, f, f), strides=(1, 1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=4, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv3D(filters=F3, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=4, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    ### END CODE HERE ###

    return X



def convolutional_block_3D(X, f, filters, stage, block, s=2):
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

    ##### MAIN PATH #####
    # First component of main path
    X = Conv3D(filters=F1, kernel_size=(1, 1, 1), strides=(s, s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=4, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv3D(filters=F2, kernel_size=(f, f, f), strides=(1, 1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=4, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv3D(filters=F3, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=4, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv3D(filters=F3, kernel_size=(1, 1, 1), strides=(s, s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=4, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    ### END CODE HERE ###
    return X



def ResNet50_3D(input_shape=(512, 512, 40, 1), classes=2):
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
    X = convolutional_block_3D(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block_3D(X, f = 3, filters = [64, 64, 256], stage=2, block='b')
    X = identity_block_3D(X, 3, [64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block_3D(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block_3D(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block_3D(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block_3D(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block_3D(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block_3D(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block_3D(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block_3D(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block_3D(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block_3D(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = X = convolutional_block_3D(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block_3D(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block_3D(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling3D(pool_size=(2, 2, 2), padding='same')(X)

    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='sigmoid', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model


def ResNet98_3D(input_shape=(254, 254, 40, 1), classes=2):
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
    X = Conv3D(64, (7, 7, 7), strides=(2, 2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0), activation = None)(X)
    X = BatchNormalization(axis=4, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(X)

    # Stage 2
    X = convolutional_block_3D(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block_3D(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block_3D(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block_3D(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block_3D(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block_3D(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block_3D(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block_3D(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block_3D(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block_3D(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block_3D(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block_3D(X, 3, [256, 256, 1024], stage=4, block='e')

    # Stage 5
    X = convolutional_block_3D(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block_3D(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block_3D(X, 3, [512, 512, 2048], stage=5, block='c')
    X = identity_block_3D(X, 3, [512, 512, 2048], stage=5, block='d')
    X = identity_block_3D(X, 3, [512, 512, 2048], stage=5, block='e')
    X = identity_block_3D(X, 3, [512, 512, 2048], stage=5, block='f')

    # Stage 6
    X = convolutional_block_3D(X, f=3, filters=[1024, 1024, 4096], stage=6, block='a', s=2)
    X = identity_block_3D(X, 3, [1024, 1024, 4096], stage=6, block='b')
    X = identity_block_3D(X, 3, [1024, 1024, 4096], stage=6, block='c')
    X = identity_block_3D(X, 3, [1024, 1024, 4096], stage=6, block='d')
    X = identity_block_3D(X, 3, [1024, 1024, 4096], stage=6, block='e')

    # Stage 7
    X = convolutional_block_3D(X, f=3, filters=[2048, 2048, 8192], stage=7, block='a', s=2)
    X = identity_block_3D(X, 3, [2048, 2048, 8192], stage=7, block='b')
    X = identity_block_3D(X, 3, [2048, 2048, 8192], stage=7, block='c')
    X = identity_block_3D(X, 3, [2048, 2048, 8192], stage=7, block='d')

    # Stage 8
    X = convolutional_block_3D(X, f=3, filters=[4096, 4096, 16384], stage=8, block='a', s=2)
    X = identity_block_3D(X, 3, [4096, 4096, 16384], stage=8, block='b')
    X = identity_block_3D(X, 3, [4096, 4096, 16384], stage=8, block='c')

    # Stage 9
    X = convolutional_block_3D(X, f=3, filters=[8192, 8192, 32768], stage=9, block='a', s=2)
    X = identity_block_3D(X, 3, [8192, 8192, 32768], stage=9, block='b')


    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling3D(pool_size=(2, 2, 2), padding='same')(X)

    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='sigmoid', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet98_3D')

    return model

#compile resnet50 model
def model50_compile():
    model50 = ResNet50_3D(input_shape=(256, 256, 40, 1), classes=2)
    model50.compile(optimizer='adam', lr=0.00001, loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint_model50_orig = ModelCheckpoint(model_path+"/weights_best.hdf5", monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list_model50_orig = [checkpoint_model50_orig]

    model50_parallel = multi_gpu_model(model50, gpus=GPU)
    model50_parallel.compile(optimizer='adam', lr=0.00001, loss='binary_crossentropy', metrics=['accuracy'])
    checkpoint_model50_parallel = ModelCheckpoint(model_path+"/weights_best.hdf5", monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list_model50 = [checkpoint_model50_parallel]

    return model50, model50_parallel, callbacks_list_model50_orig, checkpoint_model50_parallel

model50, model50_parallel, callbacks_list_model50_orig, checkpoint_model50_parallel = model50_compile()

original_model = model50
parallel_model = model50_parallel

## saves at the end of every epoch (bad)
class MyCbk(keras.callbacks.Callback):

    def __init__(self, model):
         self.model_to_save = model

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save(model_path+'/model_at_epoch_%d.h5' % epoch)

class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)

# model topology instantiation above

ser_model = model50
parallel_model = ModelMGPU(ser_model, 2)

#callback to save best weights
mod_wt_path = model_path+'/best_weights.hdf5'
best_wts_callback = ModelCheckpoint(mod_wt_path, save_weights_only=False, save_best_only=True, monitor='val_acc', verbose=0, mode='max')

Adam_opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
parallel_model.compile(optimizer=Adam_opt, loss='binary_crossentropy', metrics=['accuracy'])
history_50 = parallel_model.fit(X_train, Y_train, epochs=epochs, batch_size = 15, validation_data = (X_valid, Y_valid), callbacks=[best_wts_callback])

## if the whole dataset can be loaded into memory
def load_hdf5_batch_run_model(model, hdf5_path_train, hdf5_path_valid, batch_size, full_imsize, im_size_z):
    hdf5_file_train = h5py.File(hdf5_path_train, "r")
    hdf5_file_valid = h5py.File(hdf5_path_valid, "r")

    data_num_train = hdf5_file_train["train_img"].shape[0]
    data_num_valid = hdf5_file_valid["valid_img"].shape[0]

    batches_list = list(range(int(ceil(float(data_num_train) / batch_size))))
    shuffle(batches_list)

    # loop over batches
    for epoch in epochs:
        for n, i in enumerate(batches_list):
            i_s = i * batch_size  # index of the first image in this batch
            i_e = min([(i + 1) * batch_size, data_num_train])  # index of the last image in this batch
            images_train = hdf5_file_train["train_img"][i_s:i_e, ...]
            labels_train = hdf5_file_train["train_labels"][i_s:i_e]

            images_valid = np.array(hdf5_file_valid["valid_img"][:]) # your test set features
            labels_valid = np.array(hdf5_file_valid["valid_labels"][:]) # your test set labels

            labels_train = convert_to_one_hot(labels_train, 2).T
            labels_valid = convert_to_one_hot(labels_valid, 2).T

# #compile resnet98 model
# def model98_compile():
#     model98 = ResNet98_3D(input_shape=(256, 256, 40, 1), classes=2)
#     model98_parallel = multi_gpu_model(model98, gpus=GPU)
#     model98.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     checkpoint_model98 = ModelCheckpoint(model_path+'/'+'weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#     callbacks_list_98 = [checkpoint_model98]
#
#     return model98_parallel, callbacks_list_98

#model98_parallel, callbacks_list_98 = model98_compile()

#history_50 = model50_parallel.fit(X_train, Y_train,  epochs=epochs, batch_size = 7, validation_data = (X_valid, Y_valid), callbacks= callbacks_list_model50_orig)
#history_98 = model98_parallel.fit(X_train, Y_train, batch_size=4, epochs=epochs, steps_per_epoch=len(X_train) / 20, validation_data = (X_valid, Y_valid), callbacks=callbacks_list_model98)
