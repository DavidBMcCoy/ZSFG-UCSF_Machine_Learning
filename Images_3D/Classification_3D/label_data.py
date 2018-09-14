import os, shutil, random, argparse
import tensorflow as tf
from keras import backend as K
import pandas as pd
import numpy as np 
from sklearn.metrics import confusion_matrix
import classification_3D_CNN as class_3D 

# import imp
# class_3D = imp.load_source('', '/home/saradupont/Documents/code/ZSFG_ArtificialUnintellingenceToolbox/Medical_Image_3D_CNN_Classification/Test_3d_Class.py' )


def print_cmd(list_paths):
	for p in list_paths:
		print '\nfsleyes %s -dr 0 100 &\n' %p



def get_metrics(true_labels, pred_labels, n_digits=2):
    (TN, FP), (FN, TP) = confusion_matrix(true_labels, pred_labels)
    #
    accuracy = round((TP+TN)/float(TN+FP+FN+TP), n_digits)
    specificity = round(TN/float(TN+FP), n_digits) # true negative rate
    sensitivity = round(TP/float(FN+TP), n_digits) # true positive rate, recall
    precision = round(TP/float(TP+FP), n_digits) # positive predictive value
    f1 = round(2/((1/sensitivity) + (1/precision)), n_digits)
    #
    return accuracy, specificity, sensitivity, precision, f1


def get_parser_label():
    parser = argparse.ArgumentParser()
    parser.add_argument("-path-data",
                        help="Path to folder with data to label. Should contain 1 folder per subject.",
                        type=str,
                        dest="path_data", 
                        required=True)
    parser.add_argument("-path-out",
                        help="Output path",
                        type=str,
                        dest="path_out",
                        default='./')
    parser.add_argument("-fname-out",
                        help="File name of the output csv file",
                        type=str,
                        dest="fname_out",
                        required=False,
                        default='labeled_data.csv')
    parser.add_argument("-contrast",
                        help="Is there a subfolder by contrast in each subject's folder ?",
                        type=str,
                        dest="contrast",
                        default='')
    parser.add_argument("-fname-im",
                        help="Is there a common file name for all data ? If none is provided, the first image in each subject's folder will be used.",
                        type=str,
                        dest="fname_im",
                        default='')
    #
    return parser


if __name__ == '__main__':
    main_parser = get_parser_label()
    main_param = main_parser.parse_args()
    #
    # path_out = '/home/saradupont/Documents/CT_classification/2018-09-12-label_data/'
    # fname_out = 'data_moffitt_label.csv'
    # path_data = '/media/saradupont/Choco/data/CT/moffitt_CT/niftis_raw'
    # contrast = 'ct'
    # fname_im = ''
    # args = '-path-out '+path_out+' -fname-out '+fname_out+' -path-data '+path_data
    # if contrast != '':
    #     args += ' -contrast '+contrast
    # if fname_im != '':
    #     args += ' -fname-im '+fname_im
    # #
    # main_param = main_parser.parse_args(args.split(' '))
    # # fname_im = '_ct_noncontrast.nii.gz'
    # # dic_fname_to_test = {mrn: os.path.join(path_data, mrn, fname_im) for mrn in os.listdir(path_data)}
    #
    #
    list_mrn = os.listdir(main_param.path_data)
    # list_path_im = [os.path.join(path_data, mrn, contrast, fname_im) for mrn in list_mrn]
    list_path_im = [os.path.join(main_param.path_data, mrn, main_param.contrast, os.listdir(os.path.join(main_param.path_data, mrn, main_param.contrast))[0]) if main_param.fname_im == '' else os.path.join(main_param.path_data, mrn, main_param.contrast, main_param.fname_im) for mrn in list_mrn]


    col_name_pat_path='patient_path'
    col_name_label='label'
    col_name_mrn='MRN'

    if not os.path.isdir(main_param.path_out):
    	os.mkdir(main_param.path_out)
    	
    data = pd.DataFrame({col_name_mrn: list_mrn, col_name_pat_path: list_path_im, col_name_label: ['']*len(list_mrn)})
    data.to_csv(os.path.join(main_param.path_out, main_param.fname_out))
    # dic_labels = {'MRN': [], 'classification_label': []}

    # path to model trained at bootstrap 0 
    # path_model = '/home/saradupont/Documents/CT_classification/2018-05-23_bootstrap/2018-05-23-boostrap0/models/'
    # fname_model = '5layers_256x80im_batchsize4_conv5_maxpool2_16features_model.ckpt.meta'

    if not os.path.isdir(os.path.join(main_param.path_out, 'models')):
        path_script = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        shutil.copytree(os.path.join(path_script, 'models'), os.path.join(main_param.path_out, 'models'))

    # get params
    class_parser = class_3D.get_parser()
    # command line from bootstrap: -fname-list-subj /home/saradupont/Documents/CT_classification/2018-05-23_bootstrap/master_list_prelim1.csv -im-size 256 -epochs 5000 -features 16 -k-conv 5 -k-max-pool 2 -training-rate 0.0001 -batch-size 4 -num-layer 5 
    str_args = '-im-size 256 -epochs 5000 -features 16 -k-conv 5 -k-max-pool 2 -training-rate 0.0001 -batch-size 4 -num-layer 5'
    str_args += ' -path-out '+main_param.path_out
    str_args += ' -fname-list-subj ' + os.path.join(main_param.path_out, main_param.fname_out)
    str_args += ' -split 0'
    class_param = class_parser.parse_args(str_args.split(' '))

    # create model instance
    classification_model = class_3D.Classification(param=class_param)
    classification_model.get_filenames(col_name_pat_path=col_name_pat_path, col_name_label=col_name_label, col_name_mrn=col_name_mrn)
    classification_model.build_vol_classifier()

    # run test 
    list_pred_labels = classification_model.test_model()

    # testset = classification_model.get_CT_data(classification_model.list_test_subjects, classification_model.list_test_subjects_labels, classification_model.batch_index_test)

    # data.shape
    # len(list_pred_labels)

    data[col_name_label] = list_pred_labels
    data.to_csv(os.path.join(main_param.path_out, main_param.fname_out))

    ###############################################
    # data = pd.read_csv(os.path.join(main_param.path_out, main_param.fname_out))

    # pos_cases = data[data[col_name_label] == 1][col_name_pat_path]

    # neg_cases = data[data[col_name_label] == 0][col_name_pat_path]
    # neg_cases_to_review = random.sample(neg_cases, 10)


    # print_cmd(pos_cases)
    # print_cmd(neg_cases_to_review)


    # data_labeled = pd.read_csv('/home/saradupont/Documents/CT_classification/2018-09-12-master_list.csv')

    # for col in data_labeled:
    # 	if 'Unnamed' in col:
    # 		data_labeled = data_labeled.drop(col, axis=1)

    # l = ['copilot' in p for p in data_labeled.patient_path]
    # data_labeled_copilot = data_labeled.iloc[l]

    # data_merged = data_labeled_copilot.merge(data, on=col_name_mrn)

    # accuracy, specificity, sensitivity, precision, f1 = get_metrics(data_merged.label_x, data_merged.label_y)

