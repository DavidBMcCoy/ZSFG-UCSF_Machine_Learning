#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from keras import layers, optimizers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
from keras.callbacks import ModelCheckpoint
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
from keras.utils import multi_gpu_model

from resnet_utils_no_concat import *

filepath="/media/mccoyd2/hotdog/Osteomyelitis/rerun_70/models/model_save.hdf5"
GPU = 2

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block with skip connections over three layers
    This identify block assumes that the input activation (say a[l]) has the same dimension as the output activation (say a[l+2])

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path (first two kernals are 1)
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path (features created at each layer)
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve filters from the input for use in each layer
    F1, F2, F3 = filters

    # Save the input value. In Resnets, this value will be added to the main path between batch normalization and relu
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)


    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f , f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block, the convolutional block is the same structure as the identify block but
    The CONV2D layer in the shortcut path is used to resize the input xx to a different dimension,
    so that the dimensions match up in the final addition needed to add the shortcut value back to the main path

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
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X

def ResNet50(input_shape = (64, 64, 1), classes = 2):
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
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1 
    X = Conv2D(filters = 64, kernel_size = (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2 (X, f, filters, stage, block, s = 2):
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, f = 3, filters = [64, 64, 256], stage=2, block='b')
    X = identity_block(X, f = 3, filters = [64, 64, 256], stage=2, block='c')

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, filters = [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, filters = [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, filters = [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, filters = [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, filters = [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, filters = [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, filters = [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, filters = [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, filters = [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, filters = [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use “X = AveragePooling2D(...)(X)”
    X = AveragePooling2D(pool_size=(2, 2), name = 'avg_pool')(X)

    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='sigmoid', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)


    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

model = ResNet50(input_shape = (512, 512, 1), classes = 2)

adam = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
adam_grad = optimizers.Adagrad(lr=0.001, epsilon=None, decay=0.0)

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
ser_model = model
parallel_model = ModelMGPU(ser_model, 2)
parallel_model.compile(optimizer= adam_grad, loss='binary_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

datagen = ImageDataGenerator(rescale = 1./255,
                               
                               zoom_range = 0.2,
                               rotation_range = 180,
                               horizontal_flip = True,
                               vertical_flip = True,
                               zca_whitening=False,
                               width_shift_range= 0.2,
                               height_shift_range= 0.2,
                               shear_range=0.0)
datagen.fit(X_train)
        
datagen_no_aug = ImageDataGenerator(rescale = 1./255)
datagen_no_aug.fit(X_valid)
        

##history = classifier.fit_generator(train_datagen,
#                         steps_per_epoch = 100,
#                         epochs = 100,
#                         validation_data = test_set,
#                         validation_steps = 14)

# fits the model on batches with real-time data augmentation:
history = parallel_model.fit_generator(datagen.flow(X_train, Y_train, batch_size=20),
                    steps_per_epoch=len(X_train) / 20, epochs=epochs,
                    validation_data = datagen_no_aug.flow(X_valid, Y_valid,batch_size=20),callbacks=callbacks_list)

#history = model.fit_generator(datagen.flow(X_train, Y_train), epochs = 500, validation_data = valid_datagen.flow(X_valid, Y_valid),steps_per_epoch = 10)
#history = model.fit(X_train, Y_train, epochs = 500, batch_size = 20, validation_data=(X_valid, Y_valid))

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Alc_hep_accuracy.png')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Alc_hep_loss.png')

plt.show()

# Load best model
model = load_model(filepath)
preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


