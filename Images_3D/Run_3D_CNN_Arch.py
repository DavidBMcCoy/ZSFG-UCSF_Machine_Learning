#!/usr/bin/env python3
"""Run_3D_CNN_Arch.py: Run script for 3D networks calling in augmentation and utilities functions"""
from __future__ import print_function

__author__ = "David McCoy"
__copyright__ = "Copyright 2018, Hemorrhage Detector Project @ UCSF"
__license__ = "MIT"
__version__ = "3.0.0"
__maintainer__ = "David McCoy"
__email__ = "david.mccoy@ucsf.edu"
__status__ = "Under Construction"

import os
import time
import json
import argparse
import numpy as np
import keras.backend as K
import keras
import h5py

from keras.utils import multi_gpu_model
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from math import ceil

# custom imports
import Run_Utils

# Global variables that are unlikely to change between runs
GPUs = 2

# this class makes it possible to save checkpoints while using multiple GPUS, which apparently is an issue with Keras...


def run_3d_cnn(model_arch,
               batch_size,
               nb_epoch,
               depth,
               nb_dense_block,
               nb_filter,
               growth_rate,
               dropout_rate,
               learning_rate,
               weight_decay,
               plot_architecture,
               check_point_name,
               data_aug,
               base_path,
               history_filename,
               max_trans):

    """ Run 3d cnn
    :param history_filename: filename of pickle to dump classification by epoch
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

    hdf5_path = base_path+'/tensors/{}.hdf5'

    hdf5_train = hdf5_path.format("train")
    hdf5_valid = hdf5_path.format("valid")
    hdf5_test = hdf5_path.format("test")

    hdf5_file_train = h5py.File(hdf5_train, "r")
    data_num_train = hdf5_file_train["train_img"].shape[0]
    train_indices = range(data_num_train)

    if data_aug == 2:
        num_super_batch = 50
        super_batch_size = 2000
        n_steps_per_epoch_train = int(ceil(float(super_batch_size) / batch_size))
        # we'll cache this many batches worth of augmented data in one file
        # num_super_batch = ceil((float(TRAIN_LENGTH) / (BATCH_SIZE * SUPER_BATCH_SIZE)) * 2)
        augmented_data_path = overflow_path + 'augmented_training_cache/'
        orig_data_path = overflow_path + 'orig_training_cache/'
        augmented_data_template = augmented_data_path + 'super_batch_{}.hdf5'
        orig_data_template = orig_data_path + 'super_batch_{}.hdf5'
        augmented_data_image_name = 'images'
        augmented_data_label_name = 'labels'

        orig_data_image_name = 'images'
        orig_data_label_name = 'labels'
        orig_data_acn_name = 'acns'
        orig_data_reports_name = 'reports'
        orig_data_paths_name = 'paths'


    hdf_file_test = h5py.File(hdf5_test, "r")
    data_num_test = hdf_file_test["test_img"].shape[0]
    test_indices = range(data_num_test)
    n_steps_per_epoch_test = int(ceil(float(data_num_test) / batch_size))

    list_dir = [base_path +'/logs', base_path +'/figures', base_path +'/history', base_path + '/models']

    for d in list_dir:
        if not os.path.exists(d):
            os.makedirs(d)

    log_path = base_path + '/logs'
    figure_path = base_path +'/figures'
    history_path = base_path + '/history'
    model_path = base_path + '/models'

    #####################################################################
    # load in the validation and testing images which should fit in RAM #
    #####################################################################


    print('Loading validation and test sets, this will take a couple minutes')
    images_valid, labels_valid, data_num_valid = Run_Utils.load_valid_data_full(hdf5_valid)
    images_test, labels_test, data_num_test = Run_Utils.load_test_data_full(hdf5_test)

    img_dim = (256,256,40,1)
    #nb_classes = len(np.unique(images_test))

    nb_classes = 2

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

    if model_arch == 2:
        import Resnets_3D
        model = Resnets_3D.resnet_50_3d(input_shape=img_dim,
                                        classes=nb_classes)

    if model_arch == 3:
        import Inception_Resnet
        model = Inception_Resnet.create_inception_resnet(nb_classes=nb_classes,
                                                         scale=True,
                                                         noise_adaption=False,
                                                         nlayer_b1=5,
                                                         nlayer_b2=10,
                                                         nlayer_b3=5,
                                                         dropout=dropout_rate)
    if model_arch == 4:
        import DenseNet_3D
        model = DenseNet_3D.DenseNet(nb_classes=nb_classes,
                                     img_dim=img_dim,
                                     depth=depth,
                                     nb_dense_block=nb_dense_block,
                                     growth_rate=growth_rate,
                                     nb_filter=nb_filter,
                                     dropout_rate=dropout_rate,
                                     weight_decay=weight_decay,
                                     activation=activation)
    if model_arch > 5:
        raise ValueError('Number indicated is not part of the available models')

    # Model output
    model.summary()

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

    # if K.image_data_format() == "channels_first":
    #     n_channels = images_test.shape[1]
    # else:
    #     n_channels = images_test.shape[-1]

    if GPUs >= 2:
        model_parallel = ModelMGPU(model, gpus=GPUs)
    else:
        model_parallel = model

    Adam_opt = keras.optimizers.Adam(lr=learning_rate,
                                     beta_1=0.9,
                                     beta_2=0.999,
                                     epsilon=None,
                                     decay=0.0,
                                     amsgrad=True)

    best_wts_callback = ModelCheckpoint(model_path + check_point_name,
                                        save_weights_only=False,
                                        save_best_only=True,
                                        monitor='val_loss',
                                        verbose=0,
                                        mode='min')

    model_parallel.compile(optimizer=Adam_opt,
                           loss=loss,
                           metrics=['accuracy'])

    if plot_architecture:
        from keras.utils.vis_utils import plot_model
        plot_model(model_parallel, to_file=figure_path+'/densenet_arch_test.png', show_shapes=True)

        ####################
        # Network training #
        ####################

        print("Training")

    if data_aug == 1:

        history = Run_Utils.run_real_time_generator_model(data_aug=True,
                                                          indices=train_indices,
                                                          batch_size=batch_size,
                                                          model=model_parallel,
                                                          data_num_train=data_num_train,
                                                          epochs=nb_epoch,
                                                          images_valid=images_valid,
                                                          labels_valid=labels_valid,
                                                          callback=best_wts_callback,
                                                          base_path=base_path,
                                                          history_filename=history_filename,
                                                          hdf5_file_train=hdf5_file_train,
                                                          max_transformations=max_trans)
    if data_aug == 2:

        if not cached_data:
            Run_Utils.augment_training_data(indices=train_indices,
                                                      num_super_batches=num_super_batch,
                                                      max_transformations=3,
                                                      batch_size= batch_size,
                                                      super_batch_size=super_batch_size,
                                                      augmented_data_template=augmented_data_template,
                                                      allowed_transformations=(0, 1, 2, 3, 4, 5, 6, 7))
            history = Run_Utils.run_cached_aug_data_model(model,
                              noise_adaption=False,
                              n_steps_per_epoch_train=n_steps_per_epoch_train,
                              epochs=nb_epoch,
                              validation_images=images_valid,
                              validation_labels=labels_valid,
                              callback=best_wts_callback,
                              batch_size=batch_size,
                              base_path=base_path)
        #if cached_data:



    if data_aug == 3:
        history = Run_Utils.run_real_time_generator_model(data_aug=False,
                                                          train_indices=train_indices,
                                                          batch_size=batch_size,
                                                          model=model,
                                                          data_num_train=data_num_train,
                                                          epochs=nb_epoch,
                                                          images_valid=images_valid,
                                                          labels_valid=labels_valid,
                                                          callback=best_wts_callback)


### changes to pull on msi



# split_train_hdf()
# history = run_real_time_generator_model(data_aug=False)
#
# latd_generator = latd_generator(batch_size=BATCH_SIZE)
# history_inception = run_cached_aug_data_model(noise_adaption=False)
# history_inception_retrain = retrain_model_same_train()
# pred_ground_truth, Accuracy, Precision, Recall, F1_Score, cm, fpr, tpr, thresholds, roc_auc = test_model()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run a 3D convolutional network')
    parser.add_argument('--model_arch', type=int,
                        help="Which architecture to use [select by number]: [1] MainPath, [2] Resnet, "
                             "[3] Inception, [4] Densnet")
    parser.add_argument('--batch_size', default=10, type=int, help='Batch size')
    parser.add_argument('--nb_epoch', default=200, type=int, help='Number of epochs')
    parser.add_argument('--depth', type=int, default=7, help='Network depth')
    parser.add_argument('--nb_dense_block', type=int, default=10, help='Number of dense blocks')
    parser.add_argument('--nb_filter', type=int, default=16, help='Initial number of conv filters that growth rate starts from')
    parser.add_argument('--growth_rate', type=int, default=12, help='Number of new filters added by conv layers')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=1E-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1E-4, help='L2 regularization on weights')
    parser.add_argument('--plot_architecture', type=bool, default=True, help='Save a plot of the network architecture')
    parser.add_argument('--check_point_name', type=str,
                        help='name of the hdf5 file in which the model is saved based on validation acccuracy')
    parser.add_argument('--data_aug', type=int, default=1,
                        help='Type of augmentation 1. data aug = True and is done in '
                             'real-time, 2. data aug = True but runs off of cached aug data, '
                             '3. data aug is set to false')
    parser.add_argument('--base_path', type=str, help="path to directory where subfolder for data, "
                                                      "models etc. are kept")
    parser.add_argument('--history_filename', type=str, help='pickle filename to dump model training history')
    parser.add_argument('--max_trans', type=int, default=3, help='max number of 3d transformations to do on each image')
    args = parser.parse_args()

    print("Network configuration:")
    for name, value in parser.parse_args()._get_kwargs():
        print(name, value)

    if args.data_aug == 2:
        cached_data = input("Has the augmented already been augmented and saved? Please enter True or False: ")
        overflow_path = input("Please select the path to your augmented cache data") #/media/mccoyd2/spaghetti/


    run_3d_cnn(args.model_arch, args.batch_size, args.nb_epoch, args.depth, args.nb_dense_block, args.nb_filter,
               args.growth_rate, args.dropout_rate, args.learning_rate, args.weight_decay, args.plot_architecture,
               args.check_point_name, args.data_aug, args.base_path, args.history_filename, args.max_trans)
