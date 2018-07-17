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
__credits__ = ["Sara Dupont", "Graham Mills"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "David McCoy"
__email__ = "david.mccoy@ucsf.edu"
__status__ = "Testing"

import argparse
import os

import h5py
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Dense, Input, BatchNormalization
from keras.layers import Conv3D, Flatten, Lambda, MaxPooling3D, UpSampling3D 
from keras.layers import Reshape, Conv3DTranspose
from keras.losses import mse, binary_crossentropy
from keras.models import Model
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np

import Vol_VAE_utils

# global path variables
BASE_PATH = "/home/mccoyd2/Documents/ZSFG_ArtificialUnintellingenceToolbox/VAEs/"
DATA_BASE_PATH = "/media/mccoyd2/hamburger/hemorrhage_study/"
HDF5_PATH_TEMPLATE = DATA_BASE_PATH + 'tensors/{}_256x256x40_{}.hdf5'
VALID_LENGTH = '1685'
HDF5_PATH_VALID = HDF5_PATH_TEMPLATE.format("valid", VALID_LENGTH)
MODEL_PLOTS_PATH = BASE_PATH + 'model_plots'

# global network parameters
IMAGE_SIZE = 256
IMAGE_DEPTH = 40
INTPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH, 1)
BATCH_SIZE = 10
KERNAL_SIZE = 3
LATENT_DIM = 2
EPOCHS = 30
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


## plotting latent variable z distribution and reconstruction from the decoder

def plot_results(
    models,
    data,
    batch_size=10,
    model_name="Conv3DVAE_hemorrhage"):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(
        x_test,
        batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
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


def load_valid_data_full():
    """
    load validation data from disk
    """
    hdf5_file_valid = h5py.File(HDF5_PATH_VALID, "r")
    data_num_valid = hdf5_file_valid["valid_img"].shape[0]
    images_valid = np.array(hdf5_file_valid["valid_img"][:])  # your test set features
    labels_valid = np.array(hdf5_file_valid["valid_labels"][:])  # your test set labels
    # acn = accession number. unique identifier per image
    acns_valid = np.array(hdf5_file_valid["valid_acns"][:])
    labels_valid = Vol_VAE_utils.convert_to_one_hot(labels_valid, 2).T

    return images_valid, labels_valid, data_num_valid


input_img = Input(shape=(256, 256, 40, 1))

x = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(input_img)
x = BatchNormalization()(x)
x = MaxPooling3D((2, 2, 2), padding='same')(x)
x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
x = MaxPooling3D((2, 2, 2), padding='same')(x)
x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
x = MaxPooling3D((2, 2, 2), padding='same')(x)
x = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
x = MaxPooling3D((2, 2, 2), padding='same')(x)  # this ends in shape (?, 16, 16, 3, 128)
x = Conv3D(128, (3, 3, 1), activation='relu', padding='same')(x)
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
plot_model(encoder, to_file=MODEL_PLOTS_PATH+'/vae_cnn_encoder.png', show_shapes=True)

# build decoder model
# building back up from z
latent_inputs = Input(shape=(LATENT_DIM,), name='z_sampling')
# dense connection to build back up with same dimensions as contracting fully connected layer
# connect the latent 2 node layer to dense layer
x = Dense(shape[1] * shape[2] * (shape[3]+2) * shape[4], activation='relu')(latent_inputs)  
# reshape the node layer to the same as contracting path
x = Reshape((shape[1], shape[2], shape[3]+2, shape[4]))(x)  

x = Conv3DTranspose(
    filters=128,
    kernel_size=(3, 3, 3),
    activation='relu',
    strides=2,
    padding='same')(x)
x = Conv3DTranspose(
    filters=64,
    kernel_size=(3, 3, 3),
    activation='relu',
    strides=2,
    padding='same')(x)
x = Conv3DTranspose(
    filters=32,
    kernel_size=(3, 3, 3),
    activation='relu',
    strides=2,
    padding='same')(x)
x = Conv3DTranspose(
    filters=16,
    kernel_size=(3, 3, 3),
    activation='relu',
    strides=(2, 2, 1),
    padding='same')(x)
x = Conv3DTranspose(
    filters=8,
    kernel_size=(3, 3, 3),
    activation='relu',
    strides=(2, 2, 1),
    padding='same')(x)

outputs = Conv3DTranspose(
    filters=1,
    kernel_size=(3, 3, 3),
    activation='sigmoid',
    padding='same',
    name='decoder_output')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

plot_model(decoder, to_file=MODEL_PLOTS_PATH+'/vae_cnn_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(input_img)[2])
vae = Model(input_img, outputs, name='Conv_3D_VAE')

images_valid, labels_valid, data_num_valid = load_valid_data_full()

# reconstruction_loss = mse(K.flatten(input_img), K.flatten(outputs))
# reconstruction_loss *= IMAGE_SIZE * IMAGE_SIZE
# kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
# kl_loss = K.sum(kl_loss, axis=-1)
# kl_loss *= -0.5
# vae_loss = K.mean(reconstruction_loss + kl_loss)
# vae.add_loss(vae_loss)
# vae.compile(optimizer='rmsprop')
# vae.summary()
# plot_model(vae, to_file=MODEL_PLOTS_PATH+'/vae_cnn.png', show_shapes=True)

def vae_loss(x, x_decoded_mean):
    reconstruction_loss = mse(K.flatten(x),K.flatten(x_decoded_mean))
    reconstruction_loss *= IMAGE_SIZE * IMAGE_SIZE* IMAGE_DEPTH
    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(reconstruction_loss + kl_loss)


vae.compile(optimizer='rmsprop', loss=vae_loss)

vae.fit(images_valid, images_valid, shuffle=True, epochs=EPOCHS, batch_size=5)



