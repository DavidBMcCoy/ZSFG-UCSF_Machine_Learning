#!/usr/bin/env python3
"""Run_3D_CNN_Arch.py: Run script for 3D networks calling in augmentation and utilities functions"""
from __future__ import print_function

__author__ = "David McCoy"
__copyright__ = "Copyright 2018, Hemorrhage Detector Project @ UCSF"
__credits__ = ["Sara Dupont", "Grayhem Mills"]
__license__ = "GPL"
__version__ = "2.0.0"
__maintainer__ = "David McCoy"
__email__ = "david.mccoy@ucsf.edu"
__status__ = "Under Construction"

import os
import time
import json
import argparse
import numpy as np
import keras.backend as K
from keras.utils import multi_gpu_model
import keras
import h5py

from keras.callbacks import ModelCheckpoint

import Run_Utils

# Global variables that are unlikely to change between runs
GPUs = 2
BASE_PATH = "/media/mccoyd2/hamburger/hemorrhage_study/"
OVERFLOW_PATH = "/media/mccoyd2/spaghetti/"

SUPER_BATCH_SIZE = 2000

TRAIN_LENGTH = '14186'
VALID_LENGTH = '1685'
TEST_LENGTH = '2144'

HDF5_PATH_TEMPLATE = BASE_PATH + 'tensors/{}_256x256x40_{}.hdf5'

HDF5_PATH_TRAIN = HDF5_PATH_TEMPLATE.format("train", TRAIN_LENGTH)
HDF5_PATH_VALID = HDF5_PATH_TEMPLATE.format("valid", VALID_LENGTH)
HDF5_PATH_TEST = HDF5_PATH_TEMPLATE.format("test", TEST_LENGTH)

HDF5_FILE_TRAIN = h5py.File(HDF5_PATH_TRAIN, "r")
DATA_NUM_TRAIN = HDF5_FILE_TRAIN["train_img"].shape[0]
TRAIN_INDICES = range(DATA_NUM_TRAIN)
N_STEPS_PER_EPOCH_TRAIN = int(ceil(float(SUPER_BATCH_SIZE) / BATCH_SIZE))
HDF5_FILE_TEST = h5py.File(HDF5_PATH_TEST, "r")
DATA_NUM_TEST = HDF5_FILE_TEST["test_img"].shape[0]
TEST_INDICES = range(DATA_NUM_TEST)
N_STEPS_PER_EPOCH_TEST = int(ceil(float(DATA_NUM_TEST) / BATCH_SIZE))

# we'll cache this many batches worth of augmented data in one file
# NUM_SUPER_BATCH = ceil((float(TRAIN_LENGTH) / (BATCH_SIZE * SUPER_BATCH_SIZE)) * 2)
NUM_SUPER_BATCH = 50
AUGMENTED_DATA_PATH = OVERFLOW_PATH + 'augmented_training_cache/'
ORIG_DATA_PATH = OVERFLOW_PATH + 'orig_training_cache/'

AUGMENTED_DATA_TEMPLATE = AUGMENTED_DATA_PATH + 'super_batch_{}.hdf5'
ORIG_DATA_TEMPLATE = ORIG_DATA_PATH + 'super_batch_{}.hdf5'

AUGMENTED_DATA_IMAGE_NAME = 'images'
AUGMENTED_DATA_LABEL_NAME = 'labels'

ORIG_DATA_IMAGE_NAME = 'images'
ORIG_DATA_LABEL_NAME = 'labels'
ORIG_DATA_ACN_NAME = 'acns'
ORIG_DATA_REPORTS_NAME = 'reports'
ORIG_DATA_PATHS_NAME = 'paths'

# this class makes it possible to save checkpoints while using multiple GPUS, which apparently is an issue with Keras...


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


def run_3d_cnn(model_arch, batch_size, nb_epoch, depth, nb_dense_block, nb_filter, growth_rate, dropout_rate,
               learning_rate, weight_decay, plot_architecture, model_path, check_point_name, data_aug):
    """ Run 3d cnn
    :param model_arch: int -- number indicating which architecture to use as listed in help
    :param batch_size: int -- batch size
    :param nb_epoch: int -- number of training epochs
    :param depth: int -- network depth
    :param nb_dense_block: int -- number of dense blocks
    :param nb_filter: int -- initial number of conv filter
    :param growth_rate: int -- number of new filters added by conv layers
    :param dropout_rate: float -- dropout rate
    :param learning_rate: float -- learning rate
    :param weight_decay: float -- weight decay
    :param plot_architecture: bool -- whether to plot network architecture
    :param data_aug: int -- type of data augmentation to do
    """

    images_valid, labels_valid, data_num_valid = Run_Utils.load_valid_data_full()
    images_test, labels_test, data_num_test = Run_Utils.load_test_data_full()

    img_dim = images_test.shape[1:]
    nb_classes = len(np.unique(images_test))

    if nb_classes == 2:
        activation = 'sigmoid'
        loss = 'binary_crossentropy'
    else:
        activation = 'softmax'
        loss = 'categorical_crossentropy'

    ###################
    # Construct model #
    ###################

    print("Compiling selected model")

    if model_arch == 1:
        import Main_Path_3D
    elif model_arch == 2:
        import Resnets_3D
        model = Resnets_3D.resnet_50_3d(input_shape=img_dim, classes=nb_classes)
    elif model_arch == 3:
        import Inception_Resnet
        model = Inception_Resnet.create_inception_resnet(nb_classes=2, scale=True, noise_adaption=False, nlayer_b1=5,
                                                         nlayer_b2=10, nlayer_b3=5)
    elif model_arch == 4:
        import DenseNet_3D
        model = DenseNet_3D.DenseNet(nb_classes, img_dim, depth, nb_dense_block, growth_rate, nb_filter,
                                     dropout_rate=dropout_rate, weight_decay=weight_decay, activation=activation)
    else:
        raise ValueError('Number indicated is not part of the available models')

    # Model output
    model.summary()

    if K.image_data_format() == "channels_first":
        n_channels = images_test.shape[1]
    else:
        n_channels = images_test.shape[-1]

    if GPUs >= 2:
        model_parallel = ModelMGPU(model, gpus=GPUs)
    else:
        model_parallel = model

    Adam_opt = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    best_wts_callback = ModelCheckpoint(model_path + check_point_name,
                                        save_weights_only=False, save_best_only=True, monitor='val_loss', verbose=0,
                                        mode='min')
    model_parallel.compile(optimizer=Adam_opt, loss=loss, metrics=['accuracy'])


    if plot_architecture:
        from keras.utils.visualize_util import plot
        plot(model_parallel, to_file='./figures/densenet_archi.png', show_shapes=True)

    ####################
    # Network training #
    ####################

        print("Training")

    if data_aug == 1:
        history = Run_Utils.run_real_time_generator_model(data_aug=True, train_indices=train_indices, batch_size=batch_size, model=model,
                                                          data_num_train=data_num_train, epochs=nb_epoch, images_valid=images_valid,
                                                          labels_valid=labels_valid, callback=best_wts_callback)






if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run a 3D convolutional network')
    parser.add_argument('--model_arch', type=int,
                        help='Which architecture to use [select by number]: [1] MainPath, [2] Resnet, [3] Inception, [4] Densnet')
    parser.add_argument('--batch_size', default=10, type=int, help='Batch size')
    parser.add_argument('--nb_epoch', default=200, type=int, help='Number of epochs')
    parser.add_argument('--depth', type=int, default=7, help='Network depth')
    parser.add_argument('--nb_dense_block', type=int, default=2, help='Number of dense blocks')
    parser.add_argument('--nb_filter', type=int, default=16, help='Initial number of conv filters')
    parser.add_argument('--growth_rate', type=int, default=12, help='Number of new filters added by conv layers')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=1E-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1E-4, help='L2 regularization on weights')
    parser.add_argument('--plot_architecture', type=bool, default=False, help='Save a plot of the network architecture')
    parser.add_argument('--model_path', type=str, help='path to where to save the model (no github)')
    parser.add_argument('--check_point_name', type=str,
                        help='name of the hdf5 file in which the model is saved based on validation acccuracy')
    parser.add_argument('--data_aug', type=int, default=1, help='Type of augmentation 1. data aug = True and is done in '
                                                                'real-time, 2. data aug = True but runs off of cached aug data, '
                                                                '3. data aug is set to false')
    args = parser.parse_args()

    print("Network configuration:")
    for name, value in parser.parse_args()._get_kwargs():
        print(name, value)

    list_dir = ["./log", "./figures"]

    for d in list_dir:
        if not os.path.exists(d):
            os.makedirs(d)

    run_3d_cnn(args.model_arch, args.batch_size, args.nb_epoch, args.depth, args.nb_dense_block, args.nb_filter,
               args.growth_rate, args.dropout_rate, args.learning_rate, args.weight_decay, args.plot_architecture,
               args.model_path, args.check_point_name, args.data_aug)
