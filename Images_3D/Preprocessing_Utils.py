#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Written by David McCoy
Utility functions to work with 3d_Rest Test
These functions create master lists to directories for data, converts DICOMs to NIFTI stacks, splits the data, saves to HDF5 for easier loading etc.
"""
import h5py
import os, glob, re
import nibabel as nib
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import skimage
import pandas as pd
from skimage.transform import resize
import commands
import math
import shutil
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras.utils import multi_gpu_model


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

## image globals
im_size_y = 256
im_size_x = 256
im_size_z = 40
full_imsize = 256

## run globals
batch_size = 6
epochs = 500
exclude_label = 2 
split = 0.80
valid_split = 0.20
nlabel = 2
channels = 1 
data_order = 'tf'  # 'th' for Theano, 'tf' for Tensorflow
model_type = 'resnet'

## path globals
log_path = "/media/mccoyd2/hamburger/hemorrhage_study/logs"
master_list_path = "/media/mccoyd2/hamburger/hemorrhage_study/subject_lists"
hdf5_path = "/media/mccoyd2/hamburger/hemorrhage_study/tensors"
data_paths = ["/media/mccoyd2/hamburger/hemorrhage_study/image_data","/media/mccoyd2/spaghetti/hemorrhage_study_overflow"]
NLP_path = "/media/mccoyd2/hamburger/hemorrhage_study/NLP"
model_path = "/media/mccoyd2/hamburger/hemorrhage_study/models"
## master path list global
list_subjs_master = pd.read_csv(master_list_path+"/master_subject_list.csv")


study = ['CT_BRAIN_WO_CONTRAST']
slice_thickness = ['2mm','2_mm','2_0mm']
direction = 'Axial'
organ = 'Brain'

## create master list by looping through relevant date directories, converting the DICOM stack to nifti and saving information to master DF

def create_master_list():
    study_search = [x.lower() for x in study]
    list_subjects = pd.DataFrame([])
    failed_nifti_conv_subjects = []

    r = re.compile(".*dcm")
    for path in data_paths:
        print(path)
        for group in os.listdir(path):
            if os.path.isdir(os.path.join(path, group)):
                for batch in os.listdir(os.path.join(path, group)):
                    dicom_sorted_path = os.path.join(path, group, batch, 'DICOM-SORTED')
                    if os.path.isdir(dicom_sorted_path):
                        for subj in os.listdir(dicom_sorted_path):
                            mrn = subj.split('-')[0]
                            #print(mrn)
                            if os.path.isdir(os.path.join(dicom_sorted_path, subj)):
                                for proc_perf in os.listdir(os.path.join(dicom_sorted_path, subj)):
                                    #print proc_perf
                                    for input_proc in study_search:
                                        if input_proc in proc_perf.lower():
                                            #print input_proc
                                            for proc in os.listdir(os.path.join(dicom_sorted_path, subj, proc_perf)):
                                                if direction in proc:
                                                    #print direction
                                                    for slice in slice_thickness:
                                                        if re.findall(slice.lower(), proc.lower()):
                                                            if re.findall(organ, proc):
                                                                path_study = os.path.join(dicom_sorted_path, subj, proc_perf, proc)
                                                                print(path_study)
                                                                nii_in_path = False
                                                                ACN = proc_perf.split('-')[0]
                                                                datetime = re.findall(r"(\d{14})", proc)[0]
                                                                for fname in os.listdir(path_study):
                                                                    if fname.endswith('.nii.gz'):
                                                                        #os.remove(path_study+'/'+fname)
                                                                        nifti_name = fname
                                                                        nii_in_path = True
                                                                        datetime = proc.split('-')[1]
                                                                        datetime = datetime.split('_')[0]
                                                                        list_subjects = list_subjects.append(pd.DataFrame({'Acn':[ACN], 'MRN': [mrn],'Patient_Path': [path_study+'/'+nifti_name], 'group': [group], 'Datetime': [datetime]}))
                                                                        break

                                                                if not nii_in_path:
                                                                    print(path_study)
                                                                    ACN = proc_perf.split('-')[0]
                                                                    print("Converting DICOMS for "+subj+" to NIFTI format")
                                                                    os.chdir(path_study)
                                                                    os.chdir('..')
                                                                    status, output = commands.getstatusoutput('dcm2nii '+ proc)

                                                                    if status != 0:
                                                                        failed_nifti_conv_subjects.append(subj)
                                                                    else:
                                                                        index_nifti = [i for i, s in enumerate(output) if ">" in str(s)]
                                                                        index_end = [i for i, s in enumerate(output[index_nifti[0]:]) if "\n" in str(s)]
                                                                        nifti_name = output[index_nifti[0]+1:index_nifti[0]+index_end[0]]
                                                                        list_subjects = list_subjects.append(pd.DataFrame({'Acn':[ACN],'MRN': [mrn],'Patient_Path': [path_study+'/'+nifti_name], 'group': [group], 'Datetime': [datetime]}))

    master_list = pd.DataFrame(list_subjects)
    failed_nifti_conv_subjects = pd.DataFrame(failed_nifti_conv_subjects)
    master_list.to_csv(master_list_path+"/master_subject_list.csv")
    failed_nifti_conv_subjects.to_csv(master_list_path+"/failed_nifti_converstions.csv")

    return master_list, failed_nifti_conv_subjects


def load_master_list():
    master_list = pd.read_csv(master_list_path+"/master_subject_list.csv")
    return master_list


## split the master list into training, validation and test sets - also restrict the data to only initial exams if required

def get_filenames(master_list, initial_exam = 0):

    list_subjs_master = master_list


    if initial_exam == 1:
        list_subjs_master['Datetime_Format'] = pd.to_datetime(list_subjs_master['Datetime'], format='%Y%m%d%H%M%S')
        mrn_groups = list_subjs_master.groupby(list_subjs_master['MRN'])
        list_subj_initial_CT = mrn_groups.agg(lambda x: x.loc[x.Datetime_Format.argmin()])
    else:
        list_subj_initial_CT = list_subjs_master

    ## change Acn to int for merge
    list_subj_initial_CT['Acn'] = list_subj_initial_CT['Acn'].astype(int)

    ## merge the labels from NLP
    data_from_text_DC_labeled = pd.read_excel(NLP_path+"/DC_Labeled/Rad_Labeled_Only.xlsx")
    data_from_text_ML = pd.read_csv(NLP_path+"/Reports/Hemorrhage_Reports_Batch_1_Predictions.csv")

    unique_rad_label, counts_rad_label = np.unique(data_from_text_DC_labeled['Label'], return_counts=True)
    unique_ML_label, counts_ML_label = np.unique(data_from_text_ML['Label'], return_counts=True)

    print("Radiologist labels: "+str(unique_rad_label)+" | counts of each label: "+str(counts_rad_label))
    print("ML labels: "+str(unique_ML_label)+" | counts of each label: "+str(counts_ML_label))
    data_labels_radiologist_and_ML = data_from_text_ML.append(pd.DataFrame(data = data_from_text_DC_labeled))
    data_labels_radiologist_and_ML.to_csv(NLP_path+'/merged_ML_Rad_labels_check.csv')

    merged_path_labels = pd.merge(list_subj_initial_CT, data_labels_radiologist_and_ML, on=['Acn'],how='inner')
    merged_path_labels = merged_path_labels[merged_path_labels.Label != 2]
    merged_path_labels.to_csv(NLP_path+'/merged_NLP_labels_paths_check.csv')


    unique_total_label, counts_total_label = np.unique(merged_path_labels['Label'], return_counts=True)
    print("Total labels: "+str(unique_total_label)+" | counts of each label: "+str(counts_total_label))


    ##split the data
    list_subj_train, list_subj_test, list_subj_train_labels, list_subj_test_labels, mrn_training, mrn_test, acn_training, acn_testing, reports_train, reports_test = train_test_split(merged_path_labels['Patient_Path'], merged_path_labels['Label'], merged_path_labels['MRN_x'],merged_path_labels['Acn'], merged_path_labels['Impression'], test_size=1-split, train_size=split)
    list_subj_train, list_subj_valid, list_subj_train_labels, list_subj_valid_labels, mrn_training, mrn_valid, acn_training, acn_valid, reports_train, reports_valid = train_test_split(list_subj_train, list_subj_train_labels, mrn_training, acn_training, reports_train, test_size=valid_split, train_size=1-valid_split)

    list_subj_train_labels = list_subj_train_labels.values
    list_subj_valid_labels = list_subj_valid_labels.values
    list_subj_test_labels = list_subj_test_labels.values

    ## encode the disease label
    #encoder = LabelBinarizer()
    #self.list_subj_train_labels_encode = encoder.fit_transform(self.list_subj_train_labels)
    #self.list_subj_test_labels_encode = encoder.fit_transform(self.list_subj_test_labels)
    #self.list_subj_valid_labels_encode = encoder.fit_transform(self.list_subj_valid_labels)

    #self.list_subj_train_labels_encode = self.list_subj_train_labels_encode.reshape((self.list_subj_train_labels_encode.shape[0]))
    #self.list_subj_test_labels_encode = self.list_subj_test_labels_encode.reshape((self.list_subj_test_labels_encode.shape[0]))
    #self.list_subj_valid_labels_encode = self.list_subj_valid_labels_encode.reshape((self.list_subj_valid_labels_encode.shape[0]))


    #strip whitespace from patient path data
    list_subj_train = list(list_subj_train.str.strip())
    list_subj_valid = list(list_subj_valid.str.strip())
    list_subj_test = list(list_subj_test.str.strip())

    train = pd.DataFrame({'MRN': mrn_training,'Acn': acn_training,'Paths': list_subj_train,'Report': reports_train,'Labels': list_subj_train_labels})
    valid = pd.DataFrame({'MRN': mrn_valid,'Acn': acn_valid, 'Paths': list_subj_valid,'Report': reports_valid,'Labels': list_subj_valid_labels})
    test = pd.DataFrame({'MRN': mrn_test,'Acn': acn_testing,'Paths': list_subj_test,'Report': reports_test,'Labels': list_subj_test_labels})

    # self.valid_data_df_for_review = pd.concat(self.list_subj_valid, self.list_subj_valid_labels_encode)
    # self.test_data_df_for_review = pd.concat(self.list_subj_test, self.list_subj_test_labels_encode)

    train.to_csv(master_list_path+"/"+str(full_imsize)+"x"+str(im_size_z)+"_"+str(model_type)+"_"+str(batch_size)+"_"+str(epochs)+"_training_subjects.csv")
    valid.to_csv(master_list_path+"/"+str(full_imsize)+"x"+str(im_size_z)+"_"+str(model_type)+"_"+str(batch_size)+"_"+str(epochs)+"_validation_subjects.csv")
    test.to_csv(master_list_path+"/"+str(full_imsize)+"x"+str(im_size_z)+"_"+str(model_type)+"_"+str(batch_size)+"_"+str(epochs)+"_testing_subjects.csv")

    return train, valid, test

## use the master list to load data to be used in the CNN into an hdf5 file for easier loading

def save_dataset(subject_list, group, im_size_x, im_size_y, im_size_z):
    subject_list = subject_list.reset_index(drop = True)

    y_data = []
    x_data_ = []
    x_data_failed = []

    index = 0
    x_data_paths = []

    imagePath = subject_list['Paths']
    data_set_labels = subject_list['Labels']

    for i in range(subject_list.shape[0]):

        try:
            x_nifti = nib.load(imagePath.iloc[i])
        except:
            x_nifti = nib.load(imagePath.iloc[i]+'.gz')

        try:
            x_data = x_nifti.get_data()
            x_data_paths.append(imagePath)

        except IOError:
            x_data_failed.append(imagePath[i])
            break

        if x_data.size == 0:
            x_data_failed.append(imagePath[i])
            break

        label = np.zeros(nlabel)
        label[int(data_set_labels.iloc[i])] = 1
        y_data.append(label)

        x_resized_data = skimage.transform.resize(x_data, (im_size_x, im_size_y, im_size_z), order=3, mode='reflect')

        x_data_.append(x_resized_data)

        index += 1

    x_data_ = np.asarray(x_data_, dtype=np.uint8)
    y_data = np.asarray(y_data)
    # check the order of data and chose proper data shape to save images
    if data_order == 'th':
        tensor_shape = (len(y_data), channels, full_imsize, full_imsize, im_size_z)

    elif data_order == 'tf':
        tensor_shape = (len(y_data), full_imsize, full_imsize, im_size_z,  channels)

    # open a hdf5 file and create earrays
    hdf5_path_name = os.path.join(hdf5_path, group +'_'+ str(im_size_x) +'x'+ str(im_size_y) +'x'+ str(im_size_z) +'.hdf5')
    hdf5_file = h5py.File(hdf5_path_name, mode='w')

    hdf5_file.create_dataset(group+"_img", tensor_shape, np.float32)

    hdf5_file.create_dataset(group+"_mean", tensor_shape[1:], np.float32)

    hdf5_file.create_dataset(group+"_labels", (len(y_data),), np.int8)
    hdf5_file[group+"_labels"][...] = y_data[:,1]

    mean = np.zeros(tensor_shape[1:], np.float32)

    for i in range(len(x_data_)):
        img = x_data_[i,:,:,:]
        img = np.expand_dims(img, axis = 4)
        hdf5_file[group+"_img"][i, ...] = img[None]
        mean += img / float(len(y_data))

    hdf5_file[group+"_mean"][...] = mean
    hdf5_file.close()

## if the whole dataset can be loaded into memory
def load_hdf5_total_dataset():
    print('Loading training data from hdf5 file...')
    train_dataset = h5py.File(os.path.join(hdf5_path, 'train'+'_'+str(full_imsize)+'x'+str(full_imsize)+'x'+str(im_size_z)+'.hdf5'), "r")
    train_set_x_orig = np.array(train_dataset["train_img"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_labels"][:]) # your train set labels

    print('Loading test data from hdf5 file...')
    test_dataset = h5py.File(os.path.join(hdf5_path, 'test'+'_'+str(full_imsize)+'x'+str(full_imsize)+'x'+str(im_size_z)+'.hdf5'), "r")
    test_set_x_orig = np.array(test_dataset["test_img"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_labels"][:]) # your test set labels

    print('Loading validation data from hdf5 file...')
    valid_dataset = h5py.File(os.path.join(hdf5_path, 'valid'+'_'+str(full_imsize)+'x'+str(full_imsize)+'x'+str(im_size_z)+'.hdf5'), "r")
    valid_set_x_orig = np.array(valid_dataset["valid_img"][:]) # your test set features
    valid_set_y_orig = np.array(valid_dataset["valid_labels"][:]) # your test set labels

    classes = np.array(test_dataset["test_labels"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, valid_set_x_orig, valid_set_y_orig, test_set_x_orig, test_set_y_orig


## if the whole dataset can be loaded into memory
def load_hdf5_batch_dataset(full_imsize, im_size_z):
    train_dataset = h5py.File(os.path.join(hdf5_path, 'train'+'_'+str(full_imsize)+'x'+str(full_imsize)+'x'+str(im_size_z)+'.hdf5'), "r")
    train_data_size = train_dataset["train_img"].shape[0]
    train_set_x_orig = np.array(train_dataset["train_img"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_labels"][:]) # your train set labels

    test_dataset = h5py.File(os.path.join(hdf5_path, 'test'+'_'+str(full_imsize)+'x'+str(full_imsize)+'x'+str(im_size_z)+'.hdf5'), "r")
    test_set_x_orig = np.array(test_dataset["test_img"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_labels"][:]) # your test set labels

    valid_dataset = h5py.File(os.path.join(hdf5_path, 'valid'+'_'+str(full_imsize)+'x'+str(full_imsize)+'x'+str(im_size_z)+'.hdf5'), "r")
    valid_set_x_orig = np.array(valid_dataset["valid_img"][:]) # your test set features
    valid_set_y_orig = np.array(valid_dataset["valid_labels"][:]) # your test set labels

    classes = np.array(test_dataset["test_labels"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, valid_set_x_orig, valid_set_y_orig, test_set_x_orig, test_set_y_orig

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def preprocess_data():

    # Normalize image vectors
    print('Normalizing training data...')
    X_train = train_set_x_orig/255.
    print('Normalizing validation data...')
    X_valid = valid_set_x_orig/255.
    print('Normalizing test data...')
    X_test = test_set_x_orig/255.

    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(train_set_y_orig, 2).T
    Y_valid = convert_to_one_hot(valid_set_y_orig, 2).T
    Y_test = convert_to_one_hot(test_set_y_orig, 2).T

    unique_train_y, counts_train_y = np.unique(Y_train[:,1], return_counts=True)
    unique_valid_y, counts_valid_y = np.unique(Y_valid[:,1], return_counts=True)
    unique_test_y, counts_test_y = np.unique(Y_test[:,1], return_counts=True)

    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of training cases: "+str(counts_train_y[1])+" | number of training controls "+str(counts_train_y[0]))

    print ("number of validation examples = " + str(X_valid.shape[0]))
    print ("number of validation cases: "+str(counts_valid_y[1])+" | number of validation controls "+str(counts_valid_y[0]))

    print ("number of test examples = " + str(X_test.shape[0]))
    print ("number of test cases: "+str(counts_test_y[1])+" | number of test controls "+str(counts_test_y[0]))

    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))

    print ("X_valid shape: " + str(X_valid.shape))
    print ("Y_valid shape: " + str(Y_valid.shape))

    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test


def binary_up_sample(subject_list):

    imagePath = subject_list['Paths']
    data_set_labels = subject_list['Labels']

    unique, counts = np.unique(data_set_labels, return_counts=True)
    x_upsample_list = []

    if counts[0] != counts[1]:
        factor = float(counts[0])/float(counts[1])
        factor = int(math.ceil(factor))
        for i, j in enumerate(data_set_labels): 
            if j == 1.0 :
                x_case = imagePath.iloc[i]
                x_upsample_list.append([x_case]*factor)

    flat_list_x = [item for sublist in x_upsample_list for item in sublist]
    upsample_labels = [1]*len(flat_list_x)

    upsample_x_total = np.asarray(list(imagePath) + x_upsample_list)
    upsample_labels_total = np.asarray(list(data_set_labels) + upsample_labels)
                
    m = len(upsample_labels_total)
    permutation = list(np.random.permutation(m))
    
    shuffled_x = upsample_x_total[permutation]
    shuffled_labels = upsample_labels_total[permutation]
    
    return shuffled_x, shuffled_labels

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters
    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
                                                           # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
    
    return Z3

def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [12288, 1])
    
    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction


def random_mini_batches(X, Y, mini_batch_size = 32, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if control, 1 if case), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = int(num_complete_minibatches)
    for k in range(0, int(num_complete_minibatches)):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def generator(features, labels, batch_size):

 # Create empty arrays to contain batch of features and labels#

 batch_features = np.zeros((batch_size, im_size_x, im_size_y, 1))
 batch_labels = np.zeros((batch_size,1))
 m = features.shape[0]
 while True:
   for i in range(batch_size):
     # choose random index in features
     permutation = list(np.random.permutation(m))
     shuffled_X = X[permutation,:,:,:]
     shuffled_Y = Y[permutation,:]
     batch_features[i] = some_processing(features[index])
     batch_labels[i] = labels[index]

   yield batch_features, batch_labels


## run create master
# master_list, failed_nifti_conv_subjects = create_master_list()
# master_list = load_master_list()
# train, valid, test = get_filenames(master_list, initial_exam = 0)
# ##run save dataset
# save_dataset(train, 'train',256, 256, 40)
# save_dataset(valid, 'valid',256, 256, 40)
# save_dataset(test, 'test', 256, 256, 40)
# train_set_x_orig, train_set_y_orig, valid_set_x_orig, valid_set_y_orig, test_set_x_orig, test_set_y_orig = load_hdf5_total_dataset()
# X_train, X_valid, X_test, Y_train, Y_valid, Y_test = preprocess_data()
#
#
