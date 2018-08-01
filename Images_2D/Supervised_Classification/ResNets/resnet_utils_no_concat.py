#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os, re, commands
import h5py
from utils import *
import os, glob, re
import argparse
from utils import *
import nibabel as nib
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import skimage
import pandas as pd
from skimage.transform import resize
import commands
import math
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import matplotlib.pyplot as plt

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

im_size_y = 512
im_size_x = 512
full_imsize = 512

batch_size = 20
epochs = 200
exclude_label = 2
split = 0.80
valid_split = 0.20
nlabel = 2
channels = 1
data_order = 'tf'  # 'th' for Theano, 'tf' for Tensorflow

log_path = "/home/mccoyd2/Documents/ZSFG_ArtificialUnintellingenceToolbox/Medical_Image_2D_CNN_Classification"
data_path = "/media/mccoyd2/hotdog/Osteomyelitis/rerun_70/Data"
hdf5_path = "/media/mccoyd2/hotdog/Osteomyelitis/rerun_70/Data/tensors"


data_from_text_ML = pd.read_csv('/home/mccoyd2/Documents/Osteomyelitis/rerun_70/Predictions.csv')
data_from_radiologist = pd.read_csv('/home/mccoyd2/Documents/Osteomyelitis/rerun_70/Osteomyelitis_Radiologist_Review.csv')
data_from_text_ML_FullApply = pd.read_csv('/home/mccoyd2/Documents/Osteomyelitis/rerun_70/Predictions_Full_Apply.csv')


study = ['AP_LAT_OBL','BILAT_FOOT_3_VIEWS','-FOOT_RGHT_AP_LAT_OBL', '-FOOT_LEFT_AP_LAT_OBL']
#series = ['AP-', 'AP_OBL']
series = ['AP-','AP_','Foot__AP_','AP_OBL','OBL-','Foot__OBL_','_OBLI_','Foot_AP-','Foot_OBLI-','Foot_AP-UnknownSeriesDateUnknownSeriesTime','Foot_OBLI-UnknownSeriesDateUnknownSeriesTime']



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
    shuffled_Y = Y[:,permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning

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


def load_dataset():
    train_dataset_AP = h5py.File(os.path.join(hdf5_path, 'train_AP_nomatch.hdf5'), "r")
    #train_dataset_AP = tables.open_file(os.path.join(hdf5_path, 'train_AP.hdf5'), mode = "r")
    train_set_x_orig_AP = np.array(train_dataset_AP["trainAP_img"][:]) # your train set features
    train_set_y_orig_AP = np.array(train_dataset_AP["trainAP_labels"][:]) # your train set labels
    
    train_dataset_OBL = h5py.File(os.path.join(hdf5_path, 'train_OBL_nomatch.hdf5'), "r")
    train_set_x_orig_OBL = np.array(train_dataset_OBL["trainOBL_img"][:]) # your train set features
    train_set_y_orig_OBL = np.array(train_dataset_OBL["trainOBL_labels"][:]) # your train set labels

    test_dataset_AP = h5py.File(os.path.join(hdf5_path, 'test_AP_nomatch.hdf5'), "r")
    test_set_x_orig_AP = np.array(test_dataset_AP["testAP_img"][:]) # your test set features
    test_set_y_orig_AP = np.array(test_dataset_AP["testAP_labels"][:]) # your test set labels
    
    test_dataset_OBL = h5py.File(os.path.join(hdf5_path, 'test_OBL_nomatch.hdf5'), "r")
    test_set_x_orig_OBL = np.array(test_dataset_OBL["testOBL_img"][:]) # your test set features
    test_set_y_orig_OBL = np.array(test_dataset_OBL["testOBL_labels"][:]) # your test set labels

    valid_dataset_AP = h5py.File(os.path.join(hdf5_path, 'valid_AP_nomatch.hdf5'), "r")
    valid_set_x_orig_AP = np.array(valid_dataset_AP["validAP_img"][:]) # your test set features
    valid_set_y_orig_AP = np.array(valid_dataset_AP["validAP_labels"][:]) # your test set labels
    
    valid_dataset_OBL = h5py.File(os.path.join(hdf5_path, 'valid_OBL_nomatch.hdf5'), "r")
    valid_set_x_orig_OBL = np.array(valid_dataset_OBL["validOBL_img"][:]) # your test set features
    valid_set_y_orig_OBL = np.array(valid_dataset_OBL["validOBL_labels"][:]) # your test set labels

    #classes = np.array(test_dataset["test_labels"][:]) # the list of classes

    train_set_y_orig_AP = train_set_y_orig_AP.reshape((1, train_set_y_orig_AP.shape[0]))
    train_set_y_orig_OBL = train_set_y_orig_OBL.reshape((1, train_set_y_orig_OBL.shape[0]))
    
    test_set_y_orig_AP = test_set_y_orig_AP.reshape((1, test_set_y_orig_AP.shape[0]))
    test_set_y_orig_OBL = test_set_y_orig_OBL.reshape((1, test_set_y_orig_OBL.shape[0]))
    
    valid_set_y_orig_AP = valid_set_y_orig_AP.reshape((1, valid_set_y_orig_AP.shape[0]))
    valid_set_y_orig_OBL = valid_set_y_orig_OBL.reshape((1, valid_set_y_orig_OBL.shape[0]))
    
    train_set_x_orig = np.concatenate((train_set_x_orig_AP, train_set_x_orig_OBL), axis = 0)
    train_set_y_orig = np.concatenate((train_set_y_orig_AP, train_set_y_orig_OBL), axis = 1)
    
    valid_set_x_orig = np.concatenate((valid_set_x_orig_AP, valid_set_x_orig_OBL), axis = 0)
    valid_set_y_orig = np.concatenate((valid_set_y_orig_AP, valid_set_y_orig_OBL), axis = 1)
    
    test_set_x_orig = np.concatenate((test_set_x_orig_AP, test_set_x_orig_OBL), axis = 0)
    test_set_y_orig = np.concatenate((test_set_y_orig_AP, test_set_y_orig_OBL), axis = 1)
    
    
    return train_set_x_orig, train_set_y_orig, valid_set_x_orig, valid_set_y_orig, test_set_x_orig, test_set_y_orig

#save_dataset(list_subj_train_AP, list_subj_train_OBL, list_subj_train_labels, 'train')

def save_dataset(data_set_AP, data_set_OBL, data_set_labels, group):
    
    data_set_OBL = pd.Series(data_set_OBL)
    data_set_OBL = data_set_OBL.replace('nan',np.NaN)
    
    non_na_data_set_OBL = []
    non_na_data_set_OBL_index = [] 
    
    for i, j in enumerate(data_set_OBL): 
        if len(str(j)) > 3:
            non_na_data_set_OBL.append(data_set_OBL[i])
            non_na_data_set_OBL_index.append(i)
            
    data_set_labels_OBL = data_set_labels[non_na_data_set_OBL_index]
    
    y_data_AP = np.zeros((len(range(data_set_labels.shape[0])), nlabel)) # zero-filled list for 'one hot encoding'
    y_data_OBL = np.zeros((len(range(data_set_labels_OBL.shape[0])), nlabel))

    x_data_AP = []
    x_data_OBL = []

    x_data_failed_AP = []
    x_data_failed_OBL = []

    index = 0
    list_dataset_paths_AP = []
    list_dataset_paths_OBL = []


    for i in range(data_set_labels.shape[0]):
        imagePath_AP = data_set_AP[i]
        #print(i)        
        if imagePath_AP != 'nan':
            try:
                AP_nifti = nib.load(imagePath_AP)
            except:
                AP_nifti = nib.load(imagePath_AP+'.gz')
            
            AP_data = AP_nifti.get_data()
            list_dataset_paths_AP.append(imagePath_AP)
            resized_image_AP = skimage.transform.resize(AP_data, (im_size_x, im_size_y,), order=3, mode='reflect')
            x_data_AP.append(resized_image_AP)
            
            if AP_data.size == 0:
                x_data_failed_AP.append(data_set_AP[i])
                break
            y_data_AP[index, int(data_set_labels[i])] = 1
            index += 1
    
    index = 0  
    for i in range(data_set_labels_OBL.shape[0]):
        imagePath_OBL = non_na_data_set_OBL[i]
        try:
            OBL_nifti = nib.load(imagePath_OBL)
        except:
            OBL_nifti = nib.load(imagePath_OBL+'.gz') 

        OBL_data = OBL_nifti.get_data()        
        list_dataset_paths_OBL.append(imagePath_OBL)
        resized_image_OBL = skimage.transform.resize(OBL_data, (im_size_x, im_size_y), order=3, mode='reflect')
        x_data_OBL.append(resized_image_OBL)

        if OBL_data.size == 0:
            x_data_failed_OBL.append(data_set_OBL[i])
            break
        y_data_OBL[index, int(data_set_labels_OBL[i])] = 1
        
        index += 1

        
    x_data_AP = np.asarray(x_data_AP)
    x_data_OBL = np.asarray(x_data_OBL)

    #x_data_ = np.concatenate((x_data_AP, x_data_OBL), axis =1)

    # check the order of data and chose proper data shape to save images
    if data_order == 'th':
        tensor_shape = (len(y_data), channels, full_imsize, full_imsize)

    elif data_order == 'tf':
        tensor_shape_AP = (x_data_AP.shape[0], full_imsize, full_imsize, channels)
        tensor_shape_OBL = (x_data_OBL.shape[0], full_imsize, full_imsize, channels)
    # open a hdf5 file and create earrays
    hdf5_path_AP_name = os.path.join(hdf5_path, group+'_AP_nomatch.hdf5')
    hdf5_path_OBL_name = os.path.join(hdf5_path, group+'_OBL_nomatch.hdf5')
    
    hdf5_file_AP = h5py.File(hdf5_path_AP_name, mode='w')
    hdf5_file_OBL = h5py.File(hdf5_path_OBL_name, mode='w')

    hdf5_file_AP.create_dataset(group+"AP_img", tensor_shape_AP, np.float32)
    hdf5_file_OBL.create_dataset(group+"OBL_img", tensor_shape_OBL, np.float32)

    hdf5_file_AP.create_dataset(group+"AP_mean", tensor_shape_AP[1:], np.float32)
    hdf5_file_OBL.create_dataset(group+"OBL_mean", tensor_shape_OBL[1:], np.float32)

    hdf5_file_AP.create_dataset(group+"AP_labels", (x_data_AP.shape[0],), np.int8)
    hdf5_file_OBL.create_dataset(group+"OBL_labels",(x_data_OBL.shape[0],), np.int8)

    
    hdf5_file_AP[group+"AP_labels"][...] = y_data_AP[:,1]
    hdf5_file_OBL[group+"OBL_labels"][...] = y_data_OBL[:,1]

    mean_AP = np.zeros(tensor_shape_AP[1:], np.float32)
    mean_OBL = np.zeros(tensor_shape_OBL[1:], np.float32)
    
    for i in range(x_data_AP.shape[0]):
        AP_img = x_data_AP[i,:,:,:]
        hdf5_file_AP[group+"AP_img"][i, ...] = AP_img[None]
        mean_AP += AP_img / float(len(y_data_AP))
        hdf5_file_AP[group+"AP_mean"][...] = mean_AP
    for i in range(x_data_OBL.shape[0]):
        OBL_img = x_data_OBL[i,:,:,:]
        hdf5_file_OBL[group+"OBL_img"][i, ...] = OBL_img[None]
        mean_OBL += OBL_img / float(len(y_data_OBL))
        hdf5_file_OBL[group+"OBL_mean"][...] = mean_OBL
    
    hdf5_file_AP.close()
    hdf5_file_OBL.close()


def binary_up_sample(data_set_AP, data_set_OBL, data_set_labels):
    unique, counts = np.unique(data_set_labels, return_counts=True)
    AP_upsample_list = []
    OBL_upsample_list = []

    if counts[0] != counts[1]:
        factor = float(counts[0])/float(counts[1])
        factor = int(math.floor(factor))
        for i, j in enumerate(data_set_labels):
            if j == 1.0 :
                AP_case = data_set_AP[i]
                OBL_case = data_set_OBL[i]
                AP_upsample_list.append([AP_case]*factor)
                OBL_upsample_list.append([OBL_case]*factor)
    print(factor)
    flat_list_AP = [item for sublist in AP_upsample_list for item in sublist]
    flat_list_OBL = [item for sublist in OBL_upsample_list for item in sublist]

    upsample_labels = [1]*len(flat_list_AP)

    upsample_labels_total = np.asarray(list(data_set_labels) + upsample_labels)
    upsample_AP_total = np.asarray(list(data_set_AP) + flat_list_AP)
    upsample_OBL_total = np.asarray(list(data_set_OBL) + flat_list_OBL)

    m = len(upsample_labels_total)
    permutation = list(np.random.permutation(m))

    shuffled_AP = upsample_AP_total[permutation]
    shuffled_OBL = upsample_OBL_total[permutation]
    shuffled_labels = upsample_labels_total[permutation]

    return shuffled_AP, shuffled_OBL, shuffled_labels



def create_data_sets():

    list_subjs_master = pd.read_csv(data_path+"/subject_lists/master_subject_list.csv")
    list_subjs_master = list_subjs_master.drop('View Angle',axis = 1)
    list_subjs_master = list_subjs_master.drop_duplicates()

    def format_date(date):
        expected_len = 14
        if len(str(date)) < expected_len:
            istr = str(date)+'0'*(expected_len-len(str(date)))
            new_i = int(istr)
        else:
            new_i=date

        return new_i

    list_subjs_master['Datetime'] = list_subjs_master['Datetime'].apply(lambda x: format_date(x))

    list_subjs_master['Datetime_Format'] =  pd.to_datetime(list_subjs_master['Datetime'], format='%Y%m%d%H%M%S')
    list_subjs_master['Date_Format'] = pd.to_datetime([str(date_time).split(' ')[0] for date_time in list_subjs_master['Datetime_Format']])

    x = list_subjs_master.groupby(['Acn', 'View Angle Cat']).Datetime_Format.max()
    y = list_subjs_master[list_subjs_master['Datetime_Format'].isin(x)]

    columns = y.columns.tolist()
    acn_groups = y.groupby(y['Acn'])

    datetime_match = pd.DataFrame()
    AP_Images = pd.DataFrame()
    Oblique_Images = pd.DataFrame()

    for group in acn_groups:
        group_df = pd.DataFrame(group[1])
        if group_df.shape[0] >=1:
            # lines[.][-1] is the date and lines[.][7] is the View Angle
            #if group_df['Date_Format'].iloc[0] == group_df['Date_Format'].iloc[1] and group_df['View Angle Cat'].iloc[0] != group_df['View Angle Cat'].iloc[1]:
#                    acn = group_df['Acn'].iloc[0]
            datetime_match = datetime_match.append(group_df)
            for i in range(group_df.shape[0]):
                if group_df['View Angle Cat'].iloc[i] == 'AP':
                    AP_Images = AP_Images.append(group_df.iloc[i])
                if group_df['View Angle Cat'].iloc[i] == 'OBL':
                    Oblique_Images = Oblique_Images.append(group_df.iloc[i])

    merged_path_labels_acn_by_line = pd.merge(AP_Images,Oblique_Images, on=['Acn'], how = 'left')

    
    data_labels_radiologist_and_ML = data_from_text_ML.append(pd.DataFrame(data = data_from_radiologist))
    data_labels_radiologist_and_ML_and_Apply = data_labels_radiologist_and_ML.append(pd.DataFrame(data = data_from_text_ML_FullApply))

    data_labels_radiologist_and_ML_and_Apply = data_labels_radiologist_and_ML_and_Apply.rename(index=str, columns={"Accession1": "Acn"})

    merged_path_labels = pd.merge(merged_path_labels_acn_by_line, data_labels_radiologist_and_ML_and_Apply, on=['Acn'], how = 'inner')

    merged_path_labels = merged_path_labels[merged_path_labels.Osteomyelitis != exclude_label]
    merged_path_labels = merged_path_labels[np.isfinite(merged_path_labels['Osteomyelitis'])]

    count_labels = merged_path_labels.groupby('Osteomyelitis').count()
    print(str(count_labels['Acn']))

    merged_label_groups = merged_path_labels.groupby(merged_path_labels['Acn'])
    AP_Images = pd.DataFrame()
    Oblique_Images = pd.DataFrame()

    columns = merged_path_labels.columns.tolist()

    for group in merged_label_groups:
        group = pd.DataFrame(group[1])
        for i in range(group.shape[0]):
            if group['View Angle Cat_x'].iloc[i] == 'AP':
                AP_Images = AP_Images.append(group.iloc[i])
            if group['View Angle Cat_y'].iloc[i] == 'OBL':
                Oblique_Images = Oblique_Images.append(group.iloc[i])


    merged_path_labels = merged_path_labels.reset_index(drop=True)
    ##split the data
    list_subj_train_AP, list_subj_test_AP, list_subj_train_OBL, list_subj_test_OBL, list_subj_train_labels, list_subj_test_labels, mrn_training, mrn_test, acn_training, acn_testing, reports_train, reports_test = train_test_split(merged_path_labels['Patient_Path_x'], merged_path_labels['Patient_Path_y'], merged_path_labels['Osteomyelitis'], merged_path_labels['MRN_y'], merged_path_labels['Acn'], merged_path_labels['Impression'], test_size=1-split, train_size=split)

    list_subj_train_AP, list_subj_valid_AP, list_subj_train_OBL, list_subj_valid_OBL, list_subj_train_labels, list_subj_valid_labels, mrn_training, mrn_valid, acn_training, acn_valid, reports_train, reports_valid = train_test_split(list_subj_train_AP, list_subj_train_OBL, list_subj_train_labels, mrn_training, acn_training, reports_train, test_size=valid_split ,train_size=1-valid_split)

    list_subj_train_labels = list_subj_train_labels.values
    list_subj_valid_labels = list_subj_valid_labels.values
    list_subj_test_labels = list_subj_test_labels.values

    list_subj_train_AP = list_subj_train_AP.reset_index(drop=True)
    list_subj_valid_AP = list_subj_valid_AP.reset_index(drop=True)
    list_subj_test_AP = list_subj_test_AP.reset_index(drop=True)

    list_subj_train_OBL = list_subj_train_OBL.reset_index(drop=True)
    list_subj_valid_OBL = list_subj_valid_OBL.reset_index(drop=True)
    list_subj_test_OBL = list_subj_test_OBL.reset_index(drop=True)

    reports_train = reports_train.reset_index(drop=True)
    reports_valid = reports_valid.reset_index(drop=True)
    reports_test = reports_test.reset_index(drop=True)

    mrn_training = mrn_training.reset_index(drop = True)
    mrn_valid = mrn_valid.reset_index(drop = True)
    mrn_test = mrn_test.reset_index(drop = True)

    acn_training = acn_training.reset_index(drop = True)
    acn_valid = acn_valid.reset_index(drop = True)
    acn_testing = acn_testing.reset_index(drop = True)

    train = pd.DataFrame({'MRN': mrn_training,'Acn': acn_training,'Paths_AP':list_subj_train_AP,'Paths_OBL':list_subj_train_OBL,'Report': reports_train,'Labels':list_subj_train_labels})
    valid = pd.DataFrame({'MRN': mrn_valid,'Acn': acn_valid, 'Paths_AP':list_subj_valid_AP,'Paths_OBL':list_subj_valid_OBL,'Report': reports_valid,'Labels':list_subj_valid_labels})
    test = pd.DataFrame({'MRN': mrn_test,'Acn': acn_testing,'Paths_AP':list_subj_test_AP, 'Paths_OBL':list_subj_test_OBL, 'Report': reports_test,'Labels': list_subj_test_labels})


    train.to_csv(data_path+"/subject_lists/"+str(im_size_x)+"x"+"_"+"_"+str(batch_size)+"_"+str(epochs)+"_training_subjects_nomatch.csv")
    valid.to_csv(data_path+"/subject_lists/"+str(im_size_x)+"x"+"_"+str(batch_size)+"_"+str(epochs)+"_validation_subjects_nomatch.csv")
    test.to_csv(data_path+"/subject_lists/"+str(im_size_x)+"x"+"_"+str(batch_size)+"_"+str(epochs)+"_testing_subjects_nomatch.csv")

    print("Training "+ str(np.unique(list_subj_train_labels, return_counts=True)))
    print("Validation "+ str(np.unique(list_subj_valid_labels, return_counts=True)))
    print("Testing "+ str(np.unique(list_subj_test_labels, return_counts=True)))

    return list_subj_train_AP, list_subj_train_OBL, list_subj_train_labels, list_subj_valid_AP, list_subj_valid_OBL, list_subj_valid_labels, list_subj_test_AP, list_subj_test_OBL, list_subj_test_labels

def create_nifti():
    study_search = [x.lower() for x in study]
    series_search = [x.lower() for x in series]
    list_subjects = pd.DataFrame([])
    failed_nifti_conv_subjects = []

    r = re.compile(".*dcm")

    for group in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, group)):
            for batch in os.listdir(os.path.join(data_path, group)):
                dicom_sorted_path  = os.path.join(data_path, group, batch, 'DICOM-SORTED')
                if os.path.isdir(dicom_sorted_path):
                    for subj in os.listdir(dicom_sorted_path):
                        mrn = subj.split('-')[0]
                        print subj
                        if os.path.isdir(os.path.join(dicom_sorted_path, subj)):
                            for ind_study_perf in os.listdir(os.path.join(dicom_sorted_path, subj)):
                                for ind_study_search in study_search:
                                    if re.findall(ind_study_search, ind_study_perf.lower()):
                                        for ind_series_perf in os.listdir(os.path.join(dicom_sorted_path, subj, ind_study_perf)):
                                            for ind_series_search in series_search:
                                                if re.findall(ind_series_search, ind_series_perf.lower()):
                                                    path_series = os.path.join(dicom_sorted_path, subj, ind_study_perf, ind_series_perf)
                                                    if len(filter(r.match, os.listdir(path_series))) == 1:
                                                        nii_in_path = False
                                                        ACN = ind_study_perf.split('-')[0]
                                                        try:
                                                            datetime = re.findall(r"(\d{14})",ind_series_perf)[0]
                                                        except:
                                                            datetime = re.findall(r"(\d{8})",ind_study_perf)[0]
                                                        for fname in os.listdir(path_series):
                                                            if fname.endswith('.nii.gz'):
                                                                nifti_name = fname
                                                                nii_in_path = True

                                                                list_subjects = list_subjects.append(pd.DataFrame({'Acn':[ACN], 'MRN': [mrn],'Patient_Path': [path_series+'/'+nifti_name], 'group': [group], 'Datetime': [datetime], 'View Angle': [ind_series_search]}))
                                                                break

                                                        if not nii_in_path:
                                                            ACN = ind_study_perf.split('-')[0]
                                                            print("Converting DICOMS for "+subj+" to NIFTI format")
                                                            status, output = commands.getstatusoutput('dcm2nii '+path_series)
                                                            if status != 0:
                                                                failed_nifti_conv_subjects.append(subj)
                                                            else:
                                                                index_nifti = [i for i, s in enumerate(output) if ">" in str(s)]
                                                                index_end = [i for i, s in enumerate(output[index_nifti[0]:]) if "\n" in str(s)]
                                                                nifti_name = output[index_nifti[0]+1:index_nifti[0]+index_end[0]]

                                                                list_subjects = list_subjects.append(pd.DataFrame({'Acn':[ACN],'MRN': [mrn],'Patient_Path': [path_series+'/'+nifti_name], 'group': [group], 'Datetime': [datetime],  'View Angle': [ind_series_search]}))

    list_subjects_to_DF = pd.DataFrame(list_subjects)
    list_subjects_to_DF["View Angle Cat"] = np.where(list_subjects_to_DF["View Angle"].str.contains("obl"), "OBL", "AP")
    list_subjects_to_DF.to_csv(data_path+"/subject_lists/master_subject_list.csv")


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

## Running Series

## Cycle through the relevant studies and series, create NIFTI files, and create a master subject list
#create_nifti()

## From the master list, find the accessions with both AP and OBL views that were done on the same day etc. create new lists
# For training, validation, and testing

#list_subj_train_AP, list_subj_train_OBL, list_subj_train_labels, list_subj_valid_AP, list_subj_valid_OBL, list_subj_valid_labels, list_subj_test_AP, list_subj_test_OBL, list_subj_test_labels = create_data_sets()

# from the created training, validation, and testing lists of paths, upsample the cases in each group to ensure balance
#list_subj_train_AP, list_subj_train_OBL, list_subj_train_labels = binary_up_sample(list_subj_train_AP, list_subj_train_OBL, list_subj_train_labels)
#list_subj_valid_AP, list_subj_valid_OBL, list_subj_valid_labels = binary_up_sample(list_subj_valid_AP, list_subj_valid_OBL, list_subj_valid_labels)
#list_subj_test_AP, list_subj_test_OBL, list_subj_test_labels = binary_up_sample(list_subj_test_AP, list_subj_test_OBL, list_subj_test_labels)

# save datasets as hdf5 files for easier loading of larger datasets
#save_dataset(list_subj_train_AP, list_subj_train_OBL, list_subj_train_labels, 'train')
#save_dataset(list_subj_valid_AP, list_subj_valid_OBL, list_subj_valid_labels, 'valid')
#save_dataset(list_subj_test_AP, list_subj_test_OBL, list_subj_test_labels, 'test')

## load datasets
X_train_orig, Y_train_orig, X_valid_orig, Y_valid_orig, X_test_orig, Y_test_orig = load_dataset()

# Normalize image vectors
X_train = X_train_orig#/255.
X_valid = X_valid_orig#/255.
X_test = X_test_orig#/255.

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 2).T
Y_valid = convert_to_one_hot(Y_valid_orig, 2).T
Y_test = convert_to_one_hot(Y_test_orig, 2).T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))

print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))

print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

