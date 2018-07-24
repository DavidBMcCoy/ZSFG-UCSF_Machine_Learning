#!/usr/bin/env python2

"""run_3d_inception_v2.py: Run script for inception network calling in augmentation and utilities functions"""

__author__ = "David McCoy"
__copyright__ = "Copyright 2018, Hemorrhage Detector Project @ UCSF"
__credits__ = ["Sara Dupont", "Grayhem Mills"]
__license__ = "GPL"
__version__ = "2.0.0"
__maintainer__ = "David McCoy"
__email__ = "david.mccoy@ucsf.edu"
__status__ = "Operational"

import datetime
import h5py
from math import ceil
from pathlib import Path
import pickle
import numpy as np
from matplotlib import pyplot as plt

from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils import plot_model
from sklearn.metrics import roc_curve, auc

import inception_resnet_v1_noise_adapt
import vol_image_aug_v2
import vol_inception_utils

import random

BASE_PATH = "/media/mccoyd2/hamburger/hemorrhage_study/"
OVERFLOW_PATH = "/media/mccoyd2/spaghetti/"

BATCH_SIZE = 10
N_EPOCHS = 250
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

def load_valid_data_full():
    """
    load validation data from disk
    """
    hdf5_file_valid = h5py.File(HDF5_PATH_VALID, "r")
    data_num_valid = hdf5_file_valid["valid_img"].shape[0]
    images_valid = np.array(hdf5_file_valid["valid_img"][:])  # your test set features
    labels_valid = np.array(hdf5_file_valid["valid_labels"][:])  # your test set labels
    acns_valid = np.array(hdf5_file_valid["valid_acns"][:])
    labels_valid = vol_inception_utils.convert_to_one_hot(labels_valid, 2).T

    return images_valid, labels_valid, data_num_valid


def load_test_data_full():
    """
    load validation data from disk
    """
    hdf5_file_test = h5py.File(HDF5_PATH_TEST, "r")
    data_num_test = hdf5_file_test["test_img"].shape[0]
    images_test = np.array(hdf5_file_test["test_img"][:])  # your test set features
    labels_test = np.array(hdf5_file_test["test_labels"][:])  # your test set labels
    labels_test = vol_inception_utils.convert_to_one_hot(labels_test, 2).T

    return images_test, labels_test, data_num_test


def split_train_hdf(size_SB=4000):
    """
    split master hdf5 file into smaller hdf5s in order to load completely into memory with
    latd_generator which was originally created for loading the cached augmented data,
    by splitting the larger hdf5, we can load non-augmented files into memory and run the network more
    quickly than loading a batch size of 15 each time.
    :return:
    """
    hdf5_file_train = h5py.File(HDF5_PATH_TRAIN, "r")
    data_num_train = hdf5_file_train["train_img"].shape[0]
    data_num_train = range(0, data_num_train)
    random.shuffle(data_num_train)
    dt = h5py.special_dtype(vlen=str)

    for k in range(0, int(len(data_num_train)), int(size_SB)):
        image_accumulator = []
        label_accumulator = []
        acn_accumulator = []
        report_accumulator = []
        path_accumulator = []

        for i in range(0, int(size_SB), int(BATCH_SIZE)):
            i = i + k
            batch_indices = data_num_train[i:i + BATCH_SIZE]
            batch_indices.sort()
            images_train = HDF5_FILE_TRAIN["train_img"][batch_indices, ...]
            labels_train = HDF5_FILE_TRAIN["train_labels"][batch_indices]
            acns_train = HDF5_FILE_TRAIN["train_acns"][batch_indices, ...]
            reports_train = HDF5_FILE_TRAIN["train_reports"][batch_indices, ...]
            paths_train = HDF5_FILE_TRAIN["train_paths"][batch_indices, ...]

            image_accumulator.append(images_train)
            label_accumulator.append(labels_train)
            acn_accumulator.append(acns_train)
            report_accumulator.append(reports_train)
            path_accumulator.append(paths_train)

        image_accumulator = np.concatenate(image_accumulator, axis=0)
        label_accumulator = np.concatenate(label_accumulator, axis=0)
        acn_accumulator = np.concatenate(acn_accumulator, axis=0)
        report_accumulator = np.concatenate(report_accumulator, axis=0)
        path_accumulator = np.concatenate(path_accumulator, axis=0)

        filename = ORIG_DATA_TEMPLATE.format(k)
        with h5py.File(filename, mode='w') as the_file:
            # NOTE: this might be a good place to coerce the images to a specific dtype
            the_file.create_dataset(ORIG_DATA_IMAGE_NAME, data=image_accumulator)
            the_file.create_dataset(ORIG_DATA_LABEL_NAME, data=label_accumulator)
            the_file.create_dataset(ORIG_DATA_ACN_NAME, data=acn_accumulator)
            the_file.create_dataset(ORIG_DATA_REPORTS_NAME, data=report_accumulator, dtype=dt)
            the_file.create_dataset(ORIG_DATA_PATHS_NAME, data=path_accumulator, dtype=dt)


def report_counts(labels_valid, labels_test, data_num_test, data_num_valid):
    """
    show the numbers for each label in each group from the successfully loaded data
    """
    hdf5_file_train = h5py.File(HDF5_PATH_TRAIN, "r")
    data_num_train = hdf5_file_train["train_img"].shape[0]
    labels_train = np.array(hdf5_file_train["train_labels"][:])  # your test set labels

    unique_train_y, counts_train_y = np.unique(labels_train, return_counts=True)
    unique_valid_y, counts_valid_y = np.unique(labels_valid, return_counts=True)
    unique_test_y, counts_test_y = np.unique(labels_test, return_counts=True)

    print ("number of training examples = " + str(data_num_train))
    print (
        "number of training cases: " + str(counts_train_y[1]) + " | number of training controls " + str(
            counts_train_y[0]))

    print ("number of validation examples = " + str(data_num_valid))
    print ("number of validation cases: " + str(counts_valid_y[1]) + " | number of validation controls " + str(
        counts_valid_y[0]))

    print ("number of test examples = " + str(data_num_test))
    print ("number of test cases: " + str(counts_test_y[1]) + " | number of test controls " + str(counts_test_y[0]))


def generate_training_from_hdf5(
        indices,
        batch_size=BATCH_SIZE,
        image_aug=True,
        allowed_transformations=(0, 1, 2, 3, 4, 5, 6, 7, 8),
        max_transformations=3,
        verbose=True):
    """
    perform random augmentations on the training data and yield batches.
    note the input data is drawn from HDF5_FILE_TRAIN, which is loaded into memory when this module
    is imported.

    :argument
    indices -- range of the data from 0 to end for indexing
    batch_size -- number of images to generate and yield from the generator per call
    image_aug -- run the image augmentation function to random augment images
    allowed_transformation -- which augmentation functions to run from vol_image_aug script
    max_transformations -- max number of transformations to run on each image
    verbose -- if true, print the time it takes to process
    :re


    """
    while True:
        np.random.shuffle(indices)
        for i in range(0, DATA_NUM_TRAIN, batch_size):
            time_start_load_aug = datetime.datetime.now()
            # print("\n Current training index is: "+str(i)+'\n')
            # t0 = time()
            batch_indices = indices[i:i + batch_size]
            batch_indices.sort()
            # print("\n Batch indices: "+str(batch_indices))
            images_train = HDF5_FILE_TRAIN["train_img"][batch_indices, ...]
            labels_train = HDF5_FILE_TRAIN["train_labels"][batch_indices]
            acns_train = HDF5_FILE_TRAIN["train_acns"][batch_indices, ...]
            # images_valid = np.array(hdf5_file_valid["valid_img"][:]) # your test set features
            # labels_valid = np.array(hdf5_file_valid["valid_labels"][:]) # your test set labels

            labels_train = vol_inception_utils.convert_to_one_hot(labels_train, 2).T
            # labels_valid = convert_to_one_hot(labels_valid, 2).T

            if image_aug:
                images_train = vol_image_aug_v2.random_batch_augmentation(
                    images_train,
                    allowed_transformations=allowed_transformations,
                    max_transformations=max_transformations)
            if verbose:
                print('Loading and aug time: %s' % (datetime.datetime.now() - time_start_load_aug))

            yield (images_train, [labels_train, labels_train])


def augment_training_data(
        indices,
        num_super_batches,
        allowed_transformations=(0, 1, 2, 3, 4, 5, 6, 7, 8),
        max_transformations=3):
    """
    run all the augmentations on the training data and throw it on disk in chunks of 10 or so
    training batches. each batch being 13 images. the output image files will have shape
    (super_batch_size*batch_size, h, w, d, 1)
    """

    # this does the augmentation also
    training_data_generator = generate_training_from_hdf5(
        TRAIN_INDICES,
        batch_size=BATCH_SIZE,
        image_aug=True,
        allowed_transformations=allowed_transformations,
        max_transformations=max_transformations)

    for super_batch_n in range(int(num_super_batches)):
        image_accumulator = []
        label_accumulator = []
        for _, batch in zip(range(SUPER_BATCH_SIZE), training_data_generator):
            image_accumulator.append(batch[0])
            label_accumulator.append(batch[1][0])

        # we'll squash all the batches together
        image_accumulator = np.concatenate(image_accumulator, axis=0)
        label_accumulator = np.concatenate(label_accumulator, axis=0)

        # where do we write this data?
        filename = AUGMENTED_DATA_TEMPLATE.format(super_batch_n)
        with h5py.File(filename, mode='w') as the_file:
            # NOTE: this might be a good place to coerce the images to a specific dtype
            the_file.create_dataset(AUGMENTED_DATA_IMAGE_NAME, data=image_accumulator)
            the_file.create_dataset(AUGMENTED_DATA_LABEL_NAME, data=label_accumulator)


def latd_generator(batch_size=10):
    """
    continuously yield from random super batches of augmented data via batchwise steps.
    Randomly select super-batch -> randomize indices - > iterate through super-batch by batch size
    steps per epoch from keras calls generator (batch_size/total super batch volume) times 
    yield images and network to generator
    """
    while True:
        # find how many files we have
        data_path = Path(AUGMENTED_DATA_PATH)
        probably_files = list(data_path.glob('*.hdf5'))
        num_files = len(probably_files)
        file_number = np.random.randint(num_files)
        file_name = AUGMENTED_DATA_TEMPLATE.format(file_number)

        this_file = h5py.File(file_name, "r")
        data_len = this_file[AUGMENTED_DATA_IMAGE_NAME].shape[0]
        indices = range(this_file[AUGMENTED_DATA_IMAGE_NAME].shape[0])
        np.random.shuffle(indices)

        for i in range(0, data_len, batch_size):
            print file_name
            batch_indices = indices[i:i + batch_size]
            batch_indices.sort()
            images = np.array(this_file[AUGMENTED_DATA_IMAGE_NAME][batch_indices, ...])
            labels = np.array(this_file[AUGMENTED_DATA_LABEL_NAME][batch_indices])
            #labels = vol_inception_utils.convert_to_one_hot(labels, 2).T

            yield (images, [labels, labels])


def load_augmented_training_data(batch_size=13):
    """
    continuously yield from random super batches of augmented data.
    note that the size of the super batch controls "how random" the stream of training data is
    because we load an entire super batch at once and feed it in native order without mixing with
    other batches. but batches will be loaded in random order.
    """

    # find how many files we have
    data_path = Path(AUGMENTED_DATA_PATH)
    probably_files = list(data_path.glob('*.hdf5'))
    num_files = len(probably_files)

    def random_loader():
        """
        yield single training examples and matched labels from a random hdf5
        """
        file_number = np.random.randint(num_files)
        file_name = AUGMENTED_DATA_TEMPLATE.format(file_number)
        with h5py.File(file_name, 'r') as this_file:
            images = this_file[AUGMENTED_DATA_IMAGE_NAME]
            labels = this_file[AUGMENTED_DATA_LABEL_NAME]
            yield images, labels

    # load files randomly forever
    loader = random_loader()
    while True:
        # build up one batch
        batch_images = []
        batch_labels = []
        for _ in range(batch_size):
            try:
                this_image, this_label = next(loader)
            except StopIteration:
                loader = random_loader()
                this_image, this_label = next(loader)
            batch_images.append(this_image)
            batch_labels.append(this_label)

        # concatenate the accumulators
        batch_images = np.asarray(batch_images)
        batch_labels = np.asarray(batch_labels)

        # yield the batch in the exact same format as generate_training_from_hdf5 does
        yield (batch_images, [batch_labels, batch_labels])


def generate_testing_from_hdf5(indices, batch_size=15):
    """
    read test data from disk and yield
    """
    while True:
        # np.random.shuffle(indices)
        for i in range(0, data_num_test, batch_size):
            time_start_load_aug = datetime.datetime.now()
            # print("\n Current training index is: "+str(i)+'\n')
            # t0 = time()
            batch_indices = indices[i:i + batch_size]
            batch_indices.sort()
            # print("\n Batch indices: "+str(batch_indices))
            images_test = hdf5_file_test["test_img"][batch_indices, ...]
            labels_test = hdf5_file_test["test_labels"][batch_indices]

            labels_test = vol_inception_utils.convert_to_one_hot(labels_test, 2).T
            # labels_valid = convert_to_one_hot(labels_valid, 2).T

            yield (images_test, labels_test)


def run_real_time_generator_model(data_aug=False):
    """
    train network and report training time
    """
    images_valid, labels_valid, data_num_valid = load_valid_data_full()
    training_data_generator = generate_training_from_hdf5(
        TRAIN_INDICES,
        batch_size=BATCH_SIZE,
        image_aug=data_aug)

    N_STEPS_PER_EPOCH_TRAIN_RT = int(ceil(float(DATA_NUM_TRAIN) / BATCH_SIZE))

    start_time = datetime.datetime.now()

    history_inception = inception_resnet_v1_noise_adapt.parallel_model.fit_generator(
        training_data_generator,
        steps_per_epoch=N_STEPS_PER_EPOCH_TRAIN_RT,
        nb_epoch=N_EPOCHS,
        validation_data=(images_valid, [labels_valid, labels_valid]),
        callbacks=[inception_resnet_v1_noise_adapt.best_wts_callback],
        max_queue_size=10)

    end_time = datetime.datetime.now()
    print('Training time for %d epochs using batch size of %d was %s' \
          % (epochs, BATCH_SIZE, end_time - start_time))

    with open(BASE_PATH + 'history/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history_inception.history, file_pi)

    return history_inception


def run_cached_aug_data_model(noise_adaption=False):
    """
    train network and report training time
    """
    images_valid, labels_valid, data_num_valid = load_valid_data_full()

    start_time = datetime.datetime.now()

    if noise_adaption:
        history_inception = inception_resnet_v1_noise_adapt.parallel_model.fit_generator(
            latd_generator,
            steps_per_epoch=N_STEPS_PER_EPOCH_TRAIN,
            nb_epoch=N_EPOCHS,
            validation_data=(images_valid, [labels_valid, labels_valid, labels_valid]),
            callbacks=[inception_resnet_v1_noise_adapt.best_wts_callback],
            max_queue_size=10)
    else:
        history_inception = inception_resnet_v1_noise_adapt.parallel_model.fit_generator(
            latd_generator,
            steps_per_epoch=N_STEPS_PER_EPOCH_TRAIN,
            nb_epoch=N_EPOCHS,
            validation_data=(images_valid, [labels_valid, labels_valid]),
            callbacks=[inception_resnet_v1_noise_adapt.best_wts_callback],
            max_queue_size=10)

    end_time = datetime.datetime.now()
    print('Training time for %d epochs using batch size of %d was %s' % (N_EPOCHS, BATCH_SIZE, end_time - start_time))

    with open(BASE_PATH + 'history/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history_inception.history, file_pi)

    # inception_resnet_v1_noise_adapt.save(BASE_PATH+'/models/inception_aug_cach_v1.hdf5')

    return history_inception


def a_test_model(n_classes=2):
    """
    recover model and test data from disk, and test the model
    """
    images_test, labels_test, data_num_test = load_test_data_full()
    model = load_model(BASE_PATH + 'models/Inception_hemorrhage_model.hdf5')

    adam_optimizer = keras.optimizers.Adam(
        lr=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=None,
        decay=0.0,
        amsgrad=False)
    model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # score the test data
    test_data_generator = generate_testing_from_hdf5(TEST_INDICES, batch_size=BATCH_SIZE)
    scores = model.evaluate_generator(test_data_generator, steps=N_STEPS_PER_EPOCH_TEST)

    # refresh the data generator and generate predictions
    test_data_generator = generate_testing_from_hdf5(TEST_INDICES, batch_size=batch_size)
    predictions = model.predict_generator(test_data_generator, steps=N_STEPS_PER_EPOCH_TEST)
    classes = np.argmax(predictions, axis=1)

    pred_ground_truth = np.column_stack((predictions, classes, labels_test))
    pred_ground_truth = pd.DataFrame(
        pred_ground_truth,
        columns=[
            'Proba Neg',
            'Proba Pos',
            'Class Proba',
            'Neg Label',
            'Pos Label'])

    # Compute ROC curve and ROC area for each class
    fpr, tpr, thresholds = roc_curve(
        pred_ground_truth['Class Proba'],
        pred_ground_truth['Pos Label'])
    roc_auc = auc(fpr, tpr)

    accuracy, precision, recall, f1_score, cm = vol_inception_utils.calc_metrics(
        pred_ground_truth['Pos Label'],
        pred_ground_truth['Class Proba'])

    np.savetxt(BASE_PATH + 'results/confusion_matrix.csv', (cm), delimiter=',')

    return pred_ground_truth, accuracy, precision, recall, f1_score, cm, fpr, tpr, thresholds, roc_auc


def plot_ROC(fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def plot_inception_model():
    """
    what it says on the tin
    """
    plot_model(inception_resnet_v1, to_file="Inception ResNet-v1.png", show_shapes=True)


def plot_result_history():
    """
    plot history of the model (i.e. accuracy/loss per epoch 
    from the pickled history after running the model
    """

    # i don't think this needs an rstring for the path? i could be mistaken
    history = pd.read_pickle(BASE_PATH + "history/trainHistoryDict")

    print(history.keys())
    # summarize history for accuracy
    plt.plot(history['base_network_channel_loss'])
    plt.plot(history['val_base_network_channel_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(BASE_PATH + '/results/hemorrhage_accuracy.png')
    plt.show()

    # summarize history for loss
    plt.plot(history['base_network_channel_acc'])
    plt.plot(history['val_base_network_channel_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(BASE_PATH + '/results/hemorrhage_loss.png')

    plt.show()


def retrain_model_same_train():
    """
    reload the model and start training it again
    """
    images_valid, labels_valid, data_num_valid = load_valid_data_full()
    model = load_model(BASE_PATH + 'models/Inception_hemorrhage_model.hdf5')
    best_wts_callback = ModelCheckpoint(
        model_path + '/Inception_hemorrhage_model.hdf5',
        save_weights_only=False,
        save_best_only=True,
        monitor='val_loss',
        verbose=0,
        mode='min')

    adam_optimizer = keras.optimizers.Adam(
        lr=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=None,
        decay=0.0,
        amsgrad=False)
    model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    training_data_generator = generate_training_from_hdf5(
        TRAIN_INDICES,
        batch_size=7,
        image_aug=True)
    history_inception_retrain = model.fit_generator(
        training_data_generator,
        steps_per_epoch=N_STEPS_PER_EPOCH_TRAIN,
        nb_epoch=2,
        validation_data=(images_valid, labels_valid),
        callbacks=[best_wts_callback], max_queue_size=10)

    with open(BASE_PATH + 'history/retrainHistoryDict2', 'wb') as file_pi:
        pickle.dump(history_inception_retrain.history, file_pi)

    return history_inception_retrain


if __name__ == '__main__':
    #split_train_hdf()
    # history = run_real_time_generator_model(data_aug=False)
    #augment_training_data(TRAIN_INDICES, num_super_batches=NUM_SUPER_BATCH,
    #                      allowed_transformations=(0, 1, 2, 3, 4, 5, 6, 7), max_transformations=3)
    latd_generator = latd_generator(batch_size=BATCH_SIZE)
    history_inception = run_cached_aug_data_model(noise_adaption=False)
# history_inception_retrain = retrain_model_same_train()
# pred_ground_truth, Accuracy, Precision, Recall, F1_Score, cm, fpr, tpr, thresholds, roc_auc = test_model()
