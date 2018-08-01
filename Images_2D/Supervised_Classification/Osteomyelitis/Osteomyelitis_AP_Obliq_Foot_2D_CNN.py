#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 09:32:32 2018

@author: mccoyd2
"""

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
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

def get_parser_classify():
    # classification parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-data",
                        help="Data to train and/or test the classification on",
                        type=str,
                        dest="path")
    parser.add_argument('--study',
                        help="list of relevant study studies",
                        dest="study",
                        default="",
                        nargs='+')
    parser.add_argument('--series',
                    help="list of image series for image analysis",
                    dest="series",
                    default="",
                    nargs='+')        
    parser.add_argument('-output-path',
                    help="Output filename for the resulting model and list of subjects for training, validation and test sets",
                    dest="output_path",
                    default="")
    
    return parser

def get_parser():
    parser_data = get_parser_classify()
    #
    parser = argparse.ArgumentParser(description="Classification function based on 2D convolutional neural networks", parents=[parser_data])
    parser.add_argument("-split",
                        help="Split ratio between train and test sets. Values should be between 0 and 1. Example: -split 0.4 would use 40 percent of the data for training and 60 percent for testing.",
                        type=restricted_float,
                        dest="split",
                        default=0.8)
    
    parser.add_argument("-valid_split",
                        help="Split ratio between validation and actual train sets within the training set. Values should be between 0 and 1. Example: -split 0.3 would use 30 percent of the training set for validation (within the model training) and 70 percent for actual training.",
                        type=restricted_float,
                        dest="valid_split",
                        default=0.2)
    
    parser.add_argument("-num_layer",
                        help="Number of layers in the  contracting path of the model",
                        type=int,
                        dest="num_layer",
                        default=4)
    
    parser.add_argument("-im-size_x",
                        help="Size of the image used in the CNN.",
                        type=int,
                        dest="im_size_x",
                        default=256)
    
    parser.add_argument("-im-size_y",
                        help="Size of the image used in the CNN.",
                        type=int,
                        dest="im_size_y",
                        default=128)

    parser.add_argument("-epochs",
                        help="Number of epochs to run the network.",
                        type=int,
                        dest="epochs",
                        default=500)
    parser.add_argument("-batch-size",
                    help="Size of batches that make up each epoch.",
                    type=int,
                    dest="batch_size",
                    default=32)
    
    parser.add_argument("-nlabel",
                    help="Number of disease labels that correspond to images.",
                    type=int,
                    dest="nlabel",
                    default=2)

    parser.add_argument("-num_neuron",
                    help="Number of neurons for the fully connected layer.",
                    type=int,
                    dest="num_neuron",
                    default=1024)

    parser.add_argument("-exclude-label",
                    help="Label that should not be used for report classification",
                    type=int,
                    dest="exclude_label",
                    default="NA")

    parser.add_argument("-outcome",
                    help="Name of column in sheet for outcome label",
                    type=str,
                    dest="outcome")
    return parser


class Osteomyelitis_Classification():
    
    def __init__(self, param):
        self.param = param 
        self.list_subjects = pd.DataFrame([])
        self.failed_nifti_conv_subjects = []
        self.batch_index_train = 0
        self.batch_index_valid = 0
        self.batch_index_test = 0
        self.list_training_subjects = []
        self.list_training_subjects_labels = []
        self.list_test_subjects = []
        self.list_test_subjects_labels = []
        K.set_image_data_format('channels_last') 
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
        self.x_ = tf.placeholder(tf.float32, shape=[None, self.param.im_size_x*self.param.im_size_x]) 
        self.x_ind = tf.placeholder(tf.float64, shape=[1, self.param.im_size_x*self.param.im_size_x]) 
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.param.nlabel]) 
        self.log_path = "/home/mccoyd2/Documents/ZSFG_ArtificialUnintellingenceToolbox/Medical_Image_2D_CNN_Classification"
        self.test_accuracy_list = []

    def run_model(self):
        summary_writer = tf.summary.FileWriter(self.param.output_path+"/logs/training", self.sess.graph)
        summary_writer2 = tf.summary.FileWriter(self.param.output_path+"/logs/validation", self.sess.graph)
        summary_writer3 = tf.summary.FileWriter(self.param.output_path+"/logs/testing", self.sess.graph)


        training_summary = tf.summary.scalar("training_accuracy", self.accuracy)
        validation_summary = tf.summary.scalar("validation_accuracy", self.accuracy)
        test_summary = tf.summary.scalar("test_accuracy", self.accuracy)

        for i in range(self.param.epochs):

            batch_train = self.get_xray_data(self.list_subj_train_AP, self.list_subj_train_OBL, self.list_subj_train_labels, self.batch_index_train)
            self.batch_index_train = batch_train[2]
            #print("Training batch %d is loaded"%(i))
            batch_validation = self.get_xray_data(self.list_subj_valid_AP, self.list_subj_valid_OBL, self.list_subj_valid_labels, self.batch_index_valid)
            self.batch_index_valid = batch_validation[2]
            #print("Validation batch %d is loaded"%(i))
            # Logging every 100th iteration in the training process.
            if i%2 == 0:
                #train_accuracy = self.accuracy.eval(feed_dict={self.x_:batch_train[0], self.y_: batch_train[1], self.keep_prob: 1.0})
                train_accuracy, train_summary = self.sess.run([self.accuracy, training_summary], feed_dict={self.x_:batch_train[0], self.y_: batch_train[1], self.keep_prob: 1.0})
                summary_writer.add_summary(train_summary, i)


                valid_accuracy, valid_summary = self.sess.run([self.accuracy, validation_summary], feed_dict={self.x_:batch_validation[0], self.y_: batch_validation[1], self.keep_prob: 1.0})
                summary_writer2.add_summary(valid_summary, i)

                print("step %d, training accuracy %g, validation accuracy %g"%(i, train_accuracy,valid_accuracy))

                if i > 50:
                    if valid_accuracy <= 0.7:
#                        prediction=tf.cast(tf.argmax(self.y_conv,1), tf.float64)
#                        for i in batch_train[0].shape[0]:
#                            image = batch_train[0][i].reshape(1,self.param.im_size_x*self.param.im_size_x)
#                            prediction.eval(feed_dict={self.x_:image})
                        print batch_validation[3]
            self.train_step.run(feed_dict={self.x_: batch_train[0], self.y_: batch_train[1], self.keep_prob: 0.5})
        
        # Evaulate our accuracy on the test data
        for i in range(len(self.list_subj_test_AP)/(self.param.batch_size)):
            testset = self.get_xray_data(self.list_subj_test_AP, self.list_subj_test_OBL, self.list_subj_test_labels, self.batch_index_test)
            test_accuracy = self.accuracy.eval(feed_dict={self.x_: testset[0], self.y_: testset[1], self.keep_prob: 1.0})
            self.batch_index_test = testset[2]
            self.test_accuracy_list.append(test_accuracy)
            print("test accuracy %g"%test_accuracy)
            #summary_writer3.add_summary(test_accuracy, i)


        self.saver = tf.train.Saver()
        self.save_path = self.saver.save(self.sess, self.param.output_path+"/models/"+str(self.param.im_size_x)+"x"+"_"+str(self.param.num_layer)+"_"+str(self.param.batch_size)+"_"+str(self.param.epochs)+"_model.ckpt")
        print("Model saved in file: %s" % self.save_path)
        self.test_accuracy_list = pd.DataFrame(self.test_accuracy_list)
        self.test_accuracy_list.to_csv(self.param.output_path+"/test_results/test_accuracy.csv")
    
    def build_vol_classifier(self):
        
        self.features = 32
        self.channels = 1 
        self.list_weight_tensors = [] 
        self.list_bias_tensors = []
        self.list_relu_tensors = []
        self.list_max_pooled_tensors = []
        self.list_features = []
        self.list_channels = []

        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        ## creates a tensor with shape = shape of constant values = 0.1
        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        
        # Convolution here: stride=1, zero-padded -> output size = input size
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') # conv2d, [1, 1, 1, 1]

        # Pooling: max pooling over 2x2 blocks
        def max_pool_2x2(x):  # tf.nn.max_pool. ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        
        input_image = tf.reshape(self.x_, [-1,self.param.im_size_x,self.param.im_size_x,1])
        #input_image = tf.reshape(x, [-1,512,512,80,1]) 

        def tensor_get_shape(tensor):
            s = tensor.get_shape()
            return tuple([s[i].value for i in range(0, len(s))])

        for i in range(self.param.num_layer):
            
            self.list_features.append(self.features)
            self.list_channels.append(self.channels)
            
            print(input_image.get_shape())
            W_conv = weight_variable([5, 5, self.channels, self.features])
            self.list_weight_tensors.append(W_conv)
            b_conv = bias_variable([self.features])
            self.list_bias_tensors.append(b_conv)
            h_conv = tf.nn.relu(conv2d(input_image, W_conv) + b_conv)  
            self.list_relu_tensors.append(h_conv)
            print(h_conv.get_shape()) 
            input_image = max_pool_2x2(h_conv) 
            self.list_max_pooled_tensors.append(input_image)
            print(input_image.get_shape()) 

            last_max_pool_dim = tensor_get_shape(self.list_max_pooled_tensors[-1])

            if i == 0:
                self.channels += self.features - 1
                print self.channels
                self.features += self.features
                print self.features
            else: 
                self.channels *= 2 
                print self.channels
                self.features *= 2 
                print self.features
            
        number_neurons = self.param.num_neuron
   
        ## Densely Connected Layer (or fully-connected layer)
        # fully-connected layer with 1024 neurons to process on the entire image
        #W_fc1 = weight_variable([(self.param.im_size / 2**(self.param.num_layer))*(self.param.im_size / 2**(self.param.num_layer))*(self.param.im_depth / 2**(self.param.num_layer))*self.features, 1024])  # [7*7*64, 1024]
        weight_dim1 = last_max_pool_dim[1]*last_max_pool_dim[2]*last_max_pool_dim[3]

        W_fc1 = weight_variable([weight_dim1, number_neurons])  # [7*7*64, 1024]

        print(W_fc1.shape)
        b_fc1 = bias_variable([number_neurons]) # [1024]]
        
        #h_pool2_flat = tf.reshape(self.list_max_pooled_tensors[-1], [-1, (self.param.im_size / 2**(self.param.num_layer))*(self.param.im_size / 2**(self.param.num_layer))*(self.param.im_depth / 2**(self.param.num_layer))*self.features])  # -> output image: [-1, 7*7*64] = 3136
        h_pool2_flat = tf.reshape(self.list_max_pooled_tensors[-1], [-1, weight_dim1])  # -> output image: [-1, 7*7*64] = 3136

        print(h_pool2_flat.get_shape)  # (?, 2621440)
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # ReLU(h_pool2_flat x weight + bias)
        print(h_fc1.get_shape) # (?, 1024)  # -> output: 1024
        
        ## Dropout (to reduce overfitting; useful when training very large neural network)
        # We will turn on dropout during training & turn off during testing
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        print(h_fc1_drop.get_shape)  # -> output: 1024
        
        ## Readout Layer
        W_fc2 = weight_variable([1024, self.param.nlabel]) # [1024, 10]
        b_fc2 = bias_variable([self.param.nlabel]) # [10]
        
        self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        print(self.y_conv.get_shape)  # -> output: 10
    
        ## Train and Evaluate the Model
        # set up for optimization (optimizer:ADAM)
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
        self.train_step = tf.train.AdamOptimizer(1e-3).minimize(self.cross_entropy)  # 1e-4
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
        self.sess.run(tf.global_variables_initializer())
        
    def get_xray_data(self, data_set_AP, data_set_OBL, data_set_labels, batch_index):
        
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                           shear_range = 0.2,
                                           zoom_range = 0.2,
                                           rotation_range = 180,
                                           horizontal_flip = True,
                                           vertical_flip = True,
                                           zca_whitening=False,
                                           width_shift_range= 0.2,
                                           height_shift_range= 0.2)
        max = len(data_set_AP)
        end = batch_index + self.param.batch_size

        begin = batch_index

        if end >= max:
            end = max
            batch_index = 0


        #x_data = np.array([], np.float32)
        y_data = np.zeros((len(range(begin, end)), self.param.nlabel)) # zero-filled list for 'one hot encoding'
        x_data_AP = []
        x_data_OBL = []
        
        x_data_AP_aug = [] 
        x_data_OBL_aug = [] 
        
        x_data_failed_AP = []
        x_data_failed_OBL = []
        
        index = 0
        list_dataset_paths_AP = []
        list_dataset_paths_OBL = []
        
        for i in range(begin, end):
            #print("Loading Image %d"%(index))
            imagePath_AP = data_set_AP[i]
            imagePath_OBL = data_set_OBL[i]
            
            try: 
                AP_nifti = nib.load(imagePath_AP)
                OBL_nifti = nib.load(imagePath_OBL)
            except: 
                AP_nifti = nib.load(imagePath_AP+'.gz')
                OBL_nifti = nib.load(imagePath_OBL+'.gz')
            
            AP_data = AP_nifti.get_data()
            OBL_data = OBL_nifti.get_data()
            
            list_dataset_paths_AP.append(imagePath_AP)
            list_dataset_paths_OBL.append(imagePath_OBL)
            # TensorShape([Dimension(256), Dimension(256), Dimension(40)])                       
            #resized_image = tf.image.resize_images(images=CT_data, size=(self.param.im_size,self.param.im_size,self.param.im_depth), method=1)
            if AP_data.size == 0:
                x_data_failed_AP.append(data_set_AP[i])
                break
            if OBL_data.size == 0: 
                x_data_failed_OBL.append(data_set_OBL[i])

            resized_image_AP = skimage.transform.resize(AP_data, (self.param.im_size_x,self.param.im_size_y,), order=3, mode='reflect')
            resized_image_OBL = skimage.transform.resize(OBL_data, (self.param.im_size_x,self.param.im_size_y), order=3, mode='reflect')
            #resized_image_stand = self.standardization(resized_image)
            #image = self.sess.run(resized_image)  # (256,256,40)
            #x_data = np.append(x_data, np.asarray(image, dtype='float32')) # (image.data, dtype='float32')
            x_data_AP.append(resized_image_AP)
            x_data_OBL.append(resized_image_OBL)
            
            y_data[index, int(data_set_labels[i])] = 1  # assign 1 to corresponding column (one hot encoding)
            #y_data.append(data_set_labels[i])
            index += 1
        
        x_data_AP = np.asarray(x_data_AP)
        x_data_OBL = np.asarray(x_data_OBL)
        
        train_datagen.fit(x_data_AP)
        train_datagen.fit(x_data_OBL)
        
        x_data_ = np.concatenate((x_data_AP, x_data_OBL), axis =2)
#        for AP_batch, y_batch in train_datagen.flow(x_data_AP, y_data, batch_size=self.param.batch_size):
#            x_data_AP_aug.append(AP_batch[i])
#            
#        for OBL_batch, y_batch in train_datagen.flow(x_data_OBL, y_data, batch_size=self.param.batch_size):
#            x_data_OBL_aug.append(OBL_batch[i])
        
#        x_data_AP_aug = np.asarray(x_data_AP_aug)
#        x_data_OBL_aug = np.asarray(x_data_OBL_aug)
#        
        
        batch_index += self.param.batch_size  # update index for the next batch


        x_data_ = x_data_.reshape(len(range(begin, end)), self.param.im_size_x * self.param.im_size_x)
        
        return x_data_, y_data, batch_index, list_dataset_paths_AP, list_dataset_paths_OBL
    

    def get_filenames(self):    
        
        try:
            self.list_subjs_master = pd.read_csv(self.param.path+"/subject_lists/master_subject_list.csv")

        except IOError:
            self.create_nifti()
            self.list_subjs_master = pd.read_csv(self.param.path+"/subject_lists/master_subject_list.csv")
        
        def format_date(date):
            expected_len = 14
            if len(str(date)) < expected_len:
                istr = str(date)+'0'*(expected_len-len(str(date)))
                new_i = int(istr)
            else:
                new_i=date
            
            return new_i    
    
        self.list_subjs_master['Datetime'] = self.list_subjs_master['Datetime'].apply(lambda x: format_date(x))
        
        self.list_subjs_master['Datetime_Format'] =  pd.to_datetime(self.list_subjs_master['Datetime'], format='%Y%m%d%H%M%S')
        self.list_subjs_master['Date_Format'] = pd.to_datetime([str(date_time).split(' ')[0] for date_time in self.list_subjs_master['Datetime_Format']])
#        except: 
#            self.list_subjs_master['Datetime_Format'] =  pd.to_datetime(self.list_subjs_master['Datetime'], format='%Y%m%d')
#            self.list_subjs_master['Date_Format'] = pd.to_datetime([str(date_time).split(' ')[0] for date_time in self.list_subjs_master['Datetime_Format']])
        ## looking for initial study, not relelvant for osteo
#        df = pd.concat([self.list_subjs_master['Patient_Path'], self.list_subjs_master['View Angle Cat'], self.list_subjs_master['Acn']], axis=1)
     
#        col_id = [str(acn)+str(date) for acn, date in zip(self.list_subjs_master['Acn'], self.list_subjs_master['Date_Format'])]
#        df = pd.concat([self.list_subjs_master, pd.DataFrame(col_id)], axis=1)
#        df2 = df.drop_duplicates(subset=[0])
#        df3 = df2.pivot(index=0, columns='View Angle Cat', values='Patient_Path')
#        
#        df = pd.concat([self.list_subjs_master['Patient_Path'], self.list_subjs_master['View Angle Cat'], self.list_subjs_master['Acn']], axis=1)
#        df.pivot(index=0, columns='View Angle Cat', values='Patient_Path')
#        pd.pivot_table(df, index='Acn', columns='View Angle Cat', values='Patient_Path')

        
        x = self.list_subjs_master.groupby(['Acn', 'View Angle Cat']).Datetime_Format.max() 
        y = self.list_subjs_master[self.list_subjs_master['Datetime_Format'].isin(x)]
       
        columns = y.columns.tolist()
        acn_groups = y.groupby(y['Acn'])
    
        datetime_match = pd.DataFrame()
        AP_Images = pd.DataFrame()
        Oblique_Images = pd.DataFrame()
        
        for group in acn_groups: 
            group_df = pd.DataFrame(group[1])   
            if group_df.shape[0] == 2: 
                # lines[.][-1] is the date and lines[.][7] is the View Angle
                if group_df['Date_Format'].iloc[0] == group_df['Date_Format'].iloc[1] and group_df['View Angle Cat'].iloc[0] != group_df['View Angle Cat'].iloc[1]:
#                    acn = group_df['Acn'].iloc[0]
                    datetime_match = datetime_match.append(group_df)
                    for i in range(group_df.shape[0]):
                        if group_df['View Angle Cat'].iloc[i] == 'AP':
                            AP_Images = AP_Images.append(group_df.iloc[i])
                        if group_df['View Angle Cat'].iloc[i] == 'OBL':
                            Oblique_Images = Oblique_Images.append(group_df.iloc[i])
                    
        self.merged_path_labels_acn_by_line = pd.merge(AP_Images,Oblique_Images, on=['Acn'], how = 'inner')
        
        self.data_from_text_ML = pd.read_csv('/home/mccoyd2/Documents/Osteomyelitis/Results/Predictions.csv')
        self.data_from_radiologist = pd.read_csv('/home/mccoyd2/Documents/Osteomyelitis/Osteomyelitis_Radiologist_Review.csv')
        self.data_from_text_ML_FullApply = pd.read_csv('/home/mccoyd2/Documents/Osteomyelitis/Results/Predictions_Full_Apply.csv')
        
        self.data_labels_radiologist_and_ML = self.data_from_text_ML.append(pd.DataFrame(data = self.data_from_radiologist))
        self.data_labels_radiologist_and_ML_and_Apply = self.data_labels_radiologist_and_ML.append(pd.DataFrame(data = self.data_from_text_ML_FullApply))
#        datetime_match = datetime_match.rename(index=str, columns={"Accession1": "Acn"})
        self.data_labels_radiologist_and_ML = self.data_labels_radiologist_and_ML.rename(index=str, columns={"Accession1": "Acn"})
        
        self.merged_path_labels = pd.merge(self.merged_path_labels_acn_by_line,self.data_labels_radiologist_and_ML, on=['Acn'], how = 'inner')
        
        self.merged_path_labels = self.merged_path_labels[self.merged_path_labels.Osteomyelitis != self.param.exclude_label]
        self.merged_path_labels = self.merged_path_labels[np.isfinite(self.merged_path_labels['Osteomyelitis'])]

        count_labels = self.merged_path_labels.groupby('Osteomyelitis').count()
        print(str(count_labels['Patient_Path']))
        
        merged_label_groups = self.merged_path_labels.groupby(self.merged_path_labels['Acn'])
        AP_Images = pd.DataFrame()
        Oblique_Images = pd.DataFrame()
        
        columns = self.merged_path_labels.columns.tolist()
        for group in merged_label_groups: 
            group = pd.DataFrame(group[1])
            for i in range(group.shape[0]):
                if group['View Angle Cat'].iloc[i] == 'AP':
                    #line = pd.DataFrame(group.iloc[i], columns=columns)
                    AP_Images = AP_Images.append(group.iloc[i])
                if group['View Angle Cat'].iloc[i] == 'OBL':
                    Oblique_Images = Oblique_Images.append(group.iloc[i])
                    
        
        
        
        self.merged_path_labels_acn_by_line = self.merged_path_labels_acn_by_line.reset_index(drop=True)
        ##split the data
        self.list_subj_train_AP, self.list_subj_test_AP, self.list_subj_train_OBL, self.list_subj_test_OBL, self.list_subj_train_labels, self.list_subj_test_labels, self.mrn_training, self.mrn_test,self.acn_training, self.acn_testing, self.reports_train, self.reports_test = train_test_split(self.merged_path_labels_acn_by_line['Patient_Path_x'], self.merged_path_labels_acn_by_line['Patient_Path_y'], self.merged_path_labels_acn_by_line['Osteomyelitis_x'], self.merged_path_labels_acn_by_line['MRN_y_x'],self.merged_path_labels_acn_by_line['Acn'], self.merged_path_labels_acn_by_line['Impression_x'], test_size=1-self.param.split, train_size=self.param.split)
                
        self.list_subj_train_AP, self.list_subj_valid_AP,self.list_subj_train_OBL,self.list_subj_valid_OBL, self.list_subj_train_labels, self.list_subj_valid_labels, self.mrn_training, self.mrn_valid, self.acn_training, self.acn_valid, self.reports_train, self.reports_valid = train_test_split(self.list_subj_train_AP, self.list_subj_train_OBL, self.list_subj_train_labels, self.mrn_training, self.acn_training,self.reports_train, test_size=self.param.valid_split ,train_size=1-self.param.valid_split)

        self.list_subj_train_labels = self.list_subj_train_labels.values
        self.list_subj_valid_labels = self.list_subj_valid_labels.values
        self.list_subj_test_labels = self.list_subj_test_labels.values
        
        self.list_subj_train_AP = self.list_subj_train_AP.reset_index(drop=True)
        self.list_subj_valid_AP = self.list_subj_valid_AP.reset_index(drop=True)
        self.list_subj_test_AP = self.list_subj_test_AP.reset_index(drop=True)
        
        self.list_subj_train_OBL = self.list_subj_train_OBL.reset_index(drop=True)
        self.list_subj_valid_OBL = self.list_subj_valid_OBL.reset_index(drop=True)
        self.list_subj_test_OBL = self.list_subj_test_OBL.reset_index(drop=True)
        
        self.reports_train = self.reports_train.reset_index(drop=True)
        self.reports_valid = self.reports_valid.reset_index(drop=True)
        self.reports_test = self.reports_test.reset_index(drop=True)
        
        self.mrn_training = self.mrn_training.reset_index(drop = True)
        self.mrn_valid = self.mrn_valid.reset_index(drop = True)
        self.mrn_test = self.mrn_test.reset_index(drop = True)
        
        self.acn_training = self.acn_training.reset_index(drop = True)
        self.acn_valid = self.acn_valid.reset_index(drop = True)
        self.acn_testing = self.acn_testing.reset_index(drop = True)
        
        train = pd.DataFrame({'MRN': self.mrn_training,'Acn': self.acn_training,'Paths_AP':self.list_subj_train_AP,'Paths_OBL':self.list_subj_train_OBL,'Report': self.reports_train,'Labels':self.list_subj_train_labels})
        valid = pd.DataFrame({'MRN': self.mrn_valid,'Acn': self.acn_valid, 'Paths_AP':self.list_subj_valid_AP,'Paths_OBL':self.list_subj_valid_OBL,'Report': self.reports_valid,'Labels':self.list_subj_valid_labels})
        test = pd.DataFrame({'MRN': self.mrn_test,'Acn': self.acn_testing,'Paths_AP':self.list_subj_test_AP, 'Paths_OBL':self.list_subj_test_OBL, 'Report': self.reports_test,'Labels': self.list_subj_test_labels})


        train.to_csv(self.param.path+"/subject_lists/"+str(self.param.im_size_x)+"x"+"_"+"_"+str(self.param.batch_size)+"_"+str(self.param.epochs)+"_training_subjects.csv")
        valid.to_csv(self.param.path+"/subject_lists/"+str(self.param.im_size_x)+"x"+"_"+str(self.param.batch_size)+"_"+str(self.param.epochs)+"_validation_subjects.csv")
        test.to_csv(self.param.path+"/subject_lists/"+str(self.param.im_size_x)+"x"+"_"+str(self.param.batch_size)+"_"+str(self.param.epochs)+"_testing_subjects.csv")
        
        print("Training "+ str(np.unique(self.list_subj_train_labels, return_counts=True)))
        print("Validation "+ str(np.unique(self.list_subj_valid_labels, return_counts=True)))
        print("Testing "+ str(np.unique(self.list_subj_test_labels, return_counts=True)))
    
    def create_nifti(self):
        self.param.study = [x.lower() for x in self.param.study]
        self.param.series = [x.lower() for x in self.param.series] 
        r = re.compile(".*dcm")
        
        for group in os.listdir(self.param.path):
            if os.path.isdir(os.path.join(self.param.path, group)):
                for batch in os.listdir(os.path.join(self.param.path, group)):
                    dicom_sorted_path  = os.path.join(self.param.path, group, batch, 'DICOM-SORTED')
                    if os.path.isdir(dicom_sorted_path):
                        for subj in os.listdir(dicom_sorted_path):
                            self.mrn = subj.split('-')[0]
                            if os.path.isdir(os.path.join(dicom_sorted_path, subj)):
                                for study in os.listdir(os.path.join(dicom_sorted_path, subj)):
                                    for param_study in self.param.study: 
                                        if re.findall(param_study, study.lower()):
                                            for series in os.listdir(os.path.join(dicom_sorted_path, subj, study)):
                                                for param_series in self.param.series: 
                                                    if re.findall(param_series, series.lower()):
                                                        path_series = os.path.join(dicom_sorted_path, subj, study, series)
                                                        if len(filter(r.match, os.listdir(path_series))) == 1: 
                                                            nii_in_path = False
                                                            ACN = study.split('-')[0]
                                                            try: 
                                                                datetime = re.findall(r"(\d{14})",series)[0]
                                                            except: 
                                                                datetime = re.findall(r"(\d{8})",study)[0]
                                                            for fname in os.listdir(path_series):    
                                                                if fname.endswith('.nii.gz'):
                                                                    nifti_name = fname
                                                                    nii_in_path = True
            
                                                                    self.list_subjects = self.list_subjects.append(pd.DataFrame({'Acn':[ACN], 'MRN': [self.mrn],'Patient_Path': [path_series+'/'+nifti_name], 'group': [group], 'Datetime': [datetime], 'View Angle': [param_series]}))
                                                                    break
            
                                                            if not nii_in_path:
                                                                ACN = study.split('-')[0]
                                                                print("Converting DICOMS for "+subj+" to NIFTI format")
                                                                status, output = commands.getstatusoutput('dcm2nii '+path_series)
                                                                if status != 0:
                                                                    self.failed_nifti_conv_subjects.append(subj)
                                                                else:
                                                                    index_nifti = [i for i, s in enumerate(output) if ">" in str(s)]
                                                                    index_end = [i for i, s in enumerate(output[index_nifti[0]:]) if "\n" in str(s)]
                                                                    nifti_name = output[index_nifti[0]+1:index_nifti[0]+index_end[0]]
            
                                                                    self.list_subjects = self.list_subjects.append(pd.DataFrame({'Acn':[ACN],'MRN': [self.mrn],'Patient_Path': [path_series+'/'+nifti_name], 'group': [group], 'Datetime': [datetime],  'View Angle': [param_series]}))
        
        list_subjects_to_DF = pd.DataFrame(self.list_subjects)
        list_subjects_to_DF["View Angle Cat"] = np.where(list_subjects_to_DF["View Angle"].str.contains("obl"), "OBL", "AP")
        list_subjects_to_DF.to_csv(self.param.path+"/subject_lists/master_subject_list.csv")
        

def main():
    parser = get_parser()
    param = parser.parse_args()
    classify = Osteomyelitis_Classification(param=param)
    classify.get_filenames()
    # classify.manual_model_test()
    classify.build_vol_classifier()
    classify.run_model()

    ##
    #tran and predict --> lot of functions to import



if __name__=="__main__":
    main()