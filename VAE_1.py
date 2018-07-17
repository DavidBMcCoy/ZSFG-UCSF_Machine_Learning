#!/usr/bin/env python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
Model adapted from:
[1] Kingma, Diederik P., and Max Welling.Auto-encoding variational bayes.
https://arxiv.org/abs/1312.6114

Additional details:

    This is the first test script to adapt available keras scripts for variational autoencoders
    to convolutional 3d variational autoencoders

"""

__author__ = "David McCoy"
__copyright__ = "Copyright 2018, Hemorrhage Detector Project @ UCSF"
__credits__ = ["Sara Dupont", "Grayhem Mills"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "David McCoy"
__email__ = "david.mccoy@ucsf.edu"
__status__ = "Testing"

from keras.layers import Dense, Input, BatchNormalization
from keras.layers import Conv3D, Flatten, Lambda, MaxPooling3D, UpSampling3D
from keras.layers import Reshape, Conv3DTranspose
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
from keras import optimizers

import Vol_VAE_utils
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import h5py
import pandas as pd
import pylab
import matplotlib.colors
import shutil
import pickle

# global path variables
BASE_PATH = "/home/mccoyd2/Documents/ZSFG_ArtificialUnintellingenceToolbox/VAEs/"
DATA_BASE_PATH = "/media/mccoyd2/hamburger/hemorrhage_study/"
HDF5_PATH_TEMPLATE = DATA_BASE_PATH + 'tensors/{}_256x256x40_{}.hdf5'
VALID_LENGTH = '1685'
TEST_LENGTH = '2144'
HDF5_PATH_VALID = HDF5_PATH_TEMPLATE.format("valid", VALID_LENGTH)
HDF5_PATH_TEST = HDF5_PATH_TEMPLATE.format("test", TEST_LENGTH)
MODEL_PLOTS_PATH = BASE_PATH + 'model_plots'
MODEL_RESULTS_PATH = BASE_PATH + 'results'
MODEL_PATH = "/home/mccoyd2/Documents/ZSFG_ArtificialUnintellingenceToolbox/VAEs/models"
CHECK_POINT_NAME = '/VAE_Run1.hdf5'

# global network parameters
IMAGE_SIZE = 256
IMAGE_DEPTH = 40
INTPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH, 1)
BATCH_SIZE = 3
KERNAL_SIZE = 3
LATENT_DIM = 2
EPOCHS = 100
KEEP_PROB = 0.6


# setting up the sampling from the conditional probability distribution created by the encoder
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps

def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# after dumping the returned history from fit, this will retrieve the pickle and plot the loss over epochs and save to the results folder


def plot_result_history(model_name="Conv3DVAE_hemorrhage_test"):
    """
    plot history of the model (i.e. accuracy/loss per epoch
    from the pickled history after running the model
    """

    # i don't think this needs an rstring for the path? i could be mistaken
    history = pd.read_pickle(BASE_PATH + "history/trainHistoryDict")
    filename = os.path.join(MODEL_RESULTS_PATH, model_name)

    print(history.keys())
    # summarize history for accuracy
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss by epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(filename+'/VAE_history.png')
    plt.show()


# plotting latent variable z distribution and reconstruction from the decoder


def plot_results(models,
                 data,
                 batch_size=10,
                 model_name="Conv3DVAE_hemorrhage_test", plot=1):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data

    # display a 2D plot of the digit classes in the latent space
    z_mean_pred, z_log_var_pred, z_pred = encoder.predict(images_test, batch_size=BATCH_SIZE)

    if plot == 1:
        filename = os.path.join(MODEL_RESULTS_PATH, model_name)

        if os.path.exists(filename):
            shutil.rmtree(filename)
        os.makedirs(filename)
        plt.figure(figsize=(12, 10))
        plt.scatter(z_mean_pred[:, 0], z_mean_pred[:, 1], c=y_test[:, 1])
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.savefig(filename+'/Latent_z_scatter.png')
        plt.show()

    if plot == 2:
        filename = os.path.join(model_name, "digits_over_latent.png")
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)
        # display a 30x30 2D manifold of digits
        n = 1
        image_size = 256
        image_depth = 40
        figure = np.zeros((image_size * n, image_size * n, image_depth))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        start_range = digit_size // 2
        end_range = n * digit_size + start_range + 1
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap='Greys_r')
        plt.savefig(filename)

        plt.show()

## load the validation data used in the hemorrhage study
def load_valid_data_full():
    """
    load validation data from disk
    """
    hdf5_file_valid = h5py.File(HDF5_PATH_VALID, "r")
    data_num_valid = hdf5_file_valid["valid_img"].shape[0]
    images_valid = np.array(hdf5_file_valid["valid_img"][:])  # your test set features
    labels_valid = np.array(hdf5_file_valid["valid_labels"][:])  # your test set labels
    acns_valid = np.array(hdf5_file_valid["valid_acns"][:])
    labels_valid = Vol_VAE_utils.convert_to_one_hot(labels_valid, 2).T

    return images_valid, labels_valid, data_num_valid

## load the test data


def load_test_data_full():
    """
    load validation data from disk
    """
    hdf5_file_test = h5py.File(HDF5_PATH_TEST, "r")
    data_num_test = hdf5_file_test["test_img"].shape[0]
    images_test = np.array(hdf5_file_test["test_img"][:])  # your test set features
    labels_test = np.array(hdf5_file_test["test_labels"][:])  # your test set labels
    labels_test = Vol_VAE_utils.convert_to_one_hot(labels_test, 2).T

    return images_test, labels_test, data_num_test

## create the model


input_img = Input(shape=(256, 256, 40, 1))

x = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(input_img)
x = BatchNormalization()(x)
x = MaxPooling3D((2, 2, 2), padding='same')(x)
x = BatchNormalization()(x)
x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling3D((2, 2, 2), padding='same')(x)
x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling3D((2, 2, 2), padding='same')(x)
x = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling3D((2, 2, 2), padding='same')(x)  # this ends in shape (?, 16, 16, 3, 128)
x = Conv3D(128, (3, 3, 1), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling3D((2, 2, 1), padding='same')(x)  # this ends in shape (?, 16, 16, 3, 128)
# shape info needed to build decoder model
shape = K.int_shape(x)
# flatten output of autoencoder
x = Flatten()(x)
# connect the flattened encoder output to fully connected 16 weights
x = Dense(16, activation='relu')(x)

# output of the 16 weights connected to z_mean and z_log nodes (latent dim = 2)
z_mean = Dense(LATENT_DIM, name='z_mean')(x)
z_log_var = Dense(LATENT_DIM, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(LATENT_DIM,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(input_img, [z_mean, z_log_var, z], name='encoder')

encoder.summary()
plot_model(encoder, to_file=MODEL_PLOTS_PATH + '/vae_cnn_encoder.png', show_shapes=True)

# build decoder model
# building back up from z
latent_inputs = Input(shape=(LATENT_DIM,), name='z_sampling')
# dense connection to build back up with same dimensions as contracting fully connected layer
x = Dense(shape[1] * shape[2] * (shape[3] + 2) * shape[4], activation='relu')(
    latent_inputs)  # connect the latent 2 node layer to dense layer
x = Reshape((shape[1], shape[2], (shape[3] + 2), shape[4]))(x)  # reshape the node layer to the same as contracting path

x = Conv3DTranspose(filters=128, kernel_size=(3, 3, 3), activation='relu', strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = Conv3DTranspose(filters=64, kernel_size=(3, 3, 3), activation='relu', strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = Conv3DTranspose(filters=32, kernel_size=(3, 3, 3), activation='relu', strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = Conv3DTranspose(filters=16, kernel_size=(3, 3, 3), activation='relu', strides=(2, 2, 1), padding='same')(x)
x = BatchNormalization()(x)
x = Conv3DTranspose(filters=8, kernel_size=(3, 3, 3), activation='relu', strides=(2, 2, 1), padding='same')(x)
x = BatchNormalization()(x)

outputs = Conv3DTranspose(filters=1, kernel_size=(3, 3, 3), activation='sigmoid', padding='same',
                          name='decoder_output')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

plot_model(decoder, to_file=MODEL_PLOTS_PATH + '/vae_cnn_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(input_img)[2])

models = [encoder, decoder]

class ModelMGPU(Model):
    def __init__(self, ser_model, gpus=2):
        pmodel = multi_gpu_model(ser_model, gpus=gpus)
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


ser_model = Model(input_img, outputs)
vae = ModelMGPU(ser_model)

images_valid, labels_valid, data_num_valid = load_valid_data_full()
images_test, labels_test, data_num_test = load_test_data_full()


def vae_loss(x, x_decoded_mean):
    reconstruction_loss = mse(K.flatten(x), K.flatten(x_decoded_mean))
    reconstruction_loss *= IMAGE_SIZE * IMAGE_SIZE * IMAGE_DEPTH
    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(reconstruction_loss + kl_loss)


best_wts_callback = ModelCheckpoint(MODEL_PATH + CHECK_POINT_NAME,
                                    save_weights_only=False, save_best_only=True, monitor='val_loss', verbose=0,
                                    mode='min')

# vae.compile(optimizer='rmsprop', loss=vae_loss, )
learning_rate = 0.000001
momentum = 0.8
decay_rate = learning_rate / EPOCHS
RMS = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
vae.compile(loss=vae_loss, optimizer=RMS)
#optimizers.RMSprop(lr=0.00001, rho=0.9, epsilon=None, decay=0.0)

vae_history = vae.fit(images_valid, images_valid, shuffle=True, epochs=EPOCHS, batch_size=BATCH_SIZE,
                      validation_data=(images_test, images_test), callbacks=[best_wts_callback])

with open(BASE_PATH + 'history/trainHistoryDict', 'wb') as file_pi:
    pickle.dump(vae_history.history, file_pi)

plot_results(models, data=[images_test, labels_test], batch_size=BATCH_SIZE, model_name="Conv3DVAE_hemorrhage", plot=1)
