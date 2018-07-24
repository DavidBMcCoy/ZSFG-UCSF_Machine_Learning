#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:42:41 2017

@author: davidbmccoy
"""
# Natural Language Processing to Detect Hedging in Radiology Reports 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from __future__ import division
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import operator
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import *
import pickle
import os
from time import sleep
import sys
import io
import json

def get_parser_classify():
    # classification parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-data-path",
                        help="Path to the data folder",
                        type=str,
                        dest="data_path")
    parser.add_argument("-data-name",
                        help="Name of the reports file",
                        type=str,
                        dest="data_name")    
    parser.add_argument("-outcome",
                        help="Name of column in sheet for outcome",
                        type=str,
                        dest="imaging_plane") # if you need a value for the program to work but don't have a default value, don't put an empty default value --> program should be able to work with the default values
    parser.add_argument("-impressions",
                        help="Name of column for report impressions from radiologist",
                        type=str,
                        dest="slice_thickness") # same here
    
    return parser

class Radiology_Report_NLP(): 
    
    def __init__(self, param):
        self.param = param 
        self.reports = pd.read_excel(os.path.join(self.param.data_path, self.param.data_name),header = 0)

        self.X = None # set later in report_processing
        self.y = self.reports.loc[:][self.param.outcome].values

    # these can be accessed through self.param.outcome, ne need to redefine new class attributes
    # def get_data(self):
    #     ## define input parameters 
    #     self.outcome = self.param.outcome
    #     self.impressions = self.param.impressions
    
    # this function counts the non null items, not the non nans, correct?
    def numberOfNonNans(self,reports_column):
        count = 0
        for i in reports_column:
            if not pd.isnull(i):
                count += 1
        return count 
    
    def define_apply_data(self): 
        # Not sure what this function does: why is the number of non-null items in the outcome column used ? non null items are not necessarly contiguous in the begining of the column right ?
        length_null = numberOfNonNans(self.reports[self.param.outcome])
        ## makes the reports to apply the model to 
        self.reports_apply = self.reports[length_null:-1] # why exclude last item too ?
        self.reports_apply.drop(self.reports_apply.columns[-1], axis=1, inplace=True) # tab[len(tab)-1] is equivalent to tab[-1]
        self.reports_apply = self.reports_apply.reset_index(drop=True)
    
        
        self.reports_train = reports[:length_null]
        self.reports_train = reports_train.replace('n/a',np.NaN)
        self.reports_train = reports_train[np.isfinite(reports_train[self.param.outcome])]
        self.reports_train = reports_train.reset_index(drop=True)    
    
    def report_preprocessing(self, vocab_set):
    
        ## calculate the accuracy and other metrics for bao's program 
        ## binarize the results from bao's regex program for hedging detection
        
        ##Begin preprocessing for machine learning using word to vec 
        # Cleaning the texts for word to vector
        nltk.download('stopwords')
        
        ## create a corpus of lowered, stemmed, and stop words removed
        corpus = []
        n = self.reports.shape[0]
        print("Creating word to vector dataframe after cleaning reports of stopwords etc.")
        for i in range(n):
            sys.stdout.write('\r')
            # the exact output you're looking for:
            j = (i + 1) / n
            sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
            
            sys.stdout.flush()
            sleep(0.25)
            review = re.sub('[^a-zA-Z]', ' ', self.reports[self.impressions][i])
            review = review.lower()
            review = review.split() # need to specify some character to split (like a separator for ex, or a simple space: " ")
            ps = PorterStemmer()
            review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
            review = ' '.join(review)
            corpus.append(review)
            
            if vocab_set == 0: # or len(vocab_set) == 0 to detect an empty vocabulary set ?
                cv = CountVectorizer(max_features = 1500)
            else: 
                cv = CountVectorizer(vocabulary=self.vocabulary) # was self.vocabulary defined before ? shouldn't vocab_set be used in the else statement instead of self.vocabulary ?
                
            self.X = cv.fit_transform(corpus).toarray()
            self.X = pd.DataFrame(self.X)
            self.text_features = cv.get_feature_names()
            self.vocabulary = cv.vocabulary_
            self.X.columns = self.text_features
                
            
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.30, random_state = 0)
    
    
    
    def calc_metrics(true, prediction):
        
        cm = confusion_matrix(true, prediction)
        TN, FP, FN, TP = confusion_matrix(true, prediction).ravel()
        print(classification_report(true, prediction))
        Accuracy = (TP + TN)/(TP + TN + FP + FN)
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1_Score = 2 * Precision * Recall / (Precision + Recall)
        
        return Accuracy, Precision, Recall, F1_Score, cm
        
    
    
    def bayes_classifier(self):    
        
        # Fitting Naive Bayes to the Training set
        classifier_GNB = GaussianNB()
        classifier_GNB.fit(self.X_train, self.y_train)
        
        # Predicting the Test set results
        self.bayes_prediction = classifier_GNB.predict(self.X_test) # you can't use dots in a variable name in python, it's used for attributes (here, self.bayes.prediction would be the attribute "prediction" of an onject called bayes )
        
        # Making the Confusion Matrix for the Bayesian approach
        # self.Accuracy_Bayes, self.Precision_Bayes, self.Recall_Bayes, self.F1_Score_Bayes, self.cm_Bayes = self.calc_metrics(self.y_test, self.bayes_prediction)
        # Not consistent with the next function --> either always return the metrics if they don't need to be accessed anywhere else within the class but as a result /or/ always set the metrics as attributes (I would go for the first one)
        
        Accuracy_Bayes, Precision_Bayes, Recall_Bayes, F1_Score_Bayes, cm_Bayes = self.calc_metrics(self.y_test, self.bayes_prediction)

        return Accuracy_Bayes, Precision_Bayes, Recall_Bayes, F1_Score_Bayes, cm_Bayes

    
    def random_forest_classifier(self):    
        
        classifier_RF = RandomForestClassifier(n_estimators = 10000, criterion = 'entropy', random_state = 0)
        classifier_RF.fit(self.X_train, self.y_train)
        
        # Predicting the Test set results
        self.RF_prediction = classifier_RF.predict(self.X_test)
        
        # need to define those in this function if you want to return them
        Accuracy_RF, Precision_RF, Recall_RF, F1_Score_RF, cm_RF, classifier_RF = self.calc_metrics(self.y_test, self.RF_prediction) 
        
        # Not consistent with the previous function --> either always return the metrics if they don't need to be accessed anywhere else within the class but as a result /or/ always set the metrics as attributes (I would go for the first one)
        return Accuracy_RF, Precision_RF, Recall_RF, F1_Score_RF, cm_RF, classifier_RF
        
    
    def random_forest_feature_plot(random_forest_model, X_train): 
        ## plot the importance 
        list_std = np.std([tree.feature_importances_ for tree in random_forest_model.estimators_],
                     axis=0)
        
        feats = {} # a dict to hold feature_name: feature_importance
        for feature, importance, std in zip(X_train.columns, random_forest_model.feature_importances_, list_std): # can't have the same name for the list and the "individual" std
            feats[feature] = importance, std #add the name/value pair 
        
        importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance',1:'STD'})
        importances = importances[importances > 0.01] ## remove zero important features
        importances = importances.dropna()
        importances = importances.sort_values(by='Gini-importance')
        
    #    fig = plt.figure()
    #    ax = plt.subplot(111)
        importances.plot(kind='bar', rot=45, yerr = 'STD', title = "Random forest feature importances")
        plt.savefig('RF_FeatImp.png')
    
        # Print the feature ranking
        print("Feature ranking:")
        
        for f in range(len(importances)):
            if importances['Gini-importance'][f] > 0.001: 
                print("%d. feature %s (%f) " % (f + 1, importances.index.values[f], importances['Gini-importance'][f]))
        
    #    # Plot the feature importances of the forest
    #    plt.figure()
    #    plt.title("Feature importances")
    #    plt.bar(range(X_train.shape[1]), importances[indices],
    #           color="r", yerr=std[indices], align="center")
    #    plt.xticks(range(X_train.shape[1]), indices)
    #    plt.xlim([-1, X_train.shape[1]])
    #    plt.show() 
    
    ### This needs to be in a function, it can't be like that in the class but not in a function
    def xgb_classifier(self):
        # Fitting XGB - using the fit method (haven't figured out how to extract features this way)
        classifier_XGB = XGBClassifier() 
        classifier_XGB.fit(self.X_train, self.y_train)
        
        self.XGB_prediction = classifier_XGB.predict(self.X_test)
        # cm = confusion_matrix(y_test, y_pred)
        # print(classification_report(y_test, y_pred))
        # TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
        # Accuracy_XGB = (TP + TN) / (TP + TN + FP + FN)
        # Precision_XGB = TP / (TP + FP)
        # Recall_XGB = TP / (TP + FN)
        # F1_Score_XGB = 2 * Precision_XGB * Recall_XGB / (Precision_XGB + Recall_XGB)
        Accuracy_XGB, Precision_XGB, Recall_XGB, F1_Score_XGB, cm_XGB, classifier_XGB = self.calc_metrics(self.y_test, self.XGB_prediction)
        
        # always want to plot this ? I'd to either put that in anotehr function as you did for the random forest or to use a verbose mode to choose to plot or not
        print(classifier_XGB.feature_importances_)
        plt.bar(range(len(classifier_XGB.feature_importances_)), classifier_XGB.feature_importances_)
        plt.show()
        
        fig, ax = plt.subplots(figsize=(10, 12))
        ## this function is not defined, is it imported from somewhere else ?
        plot_importance(classifier_XGB, ax= ax)

        return Accuracy_XGB, Precision_XGB, Recall_XGB, F1_Score_XGB, cm_XGB, classifier_XGB
    
    

    ####### SAME BELOW HERE : you need to define functions and use the class arguments

    ### testing extraction of fscores 
    # Create our DMatrix to make XGBoost more efficient
    
    def create_feature_map(features):
        outfile = open('xgb.fmap', 'w')
        i = 0
        for feat in features:
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))
            i = i + 1
    
        outfile.close()
        
        
    ## store data in a a DMatrix object for xgboost
    xgdmat_train = xgb.DMatrix(X_train, y_train)
    xgdmat_test = xgb.DMatrix(X_test, y_test)
    ## set some initial params
    params = {'eta': 0.01, 'seed':0, 'subsample': 0.5, 'colsample_bytree': 0.5, 
                 'objective': 'reg:linear', 'max_depth':6, 'min_child_weight':3} 
    
    num_rounds = 50000
    
    ##train the model
    mdl = xgb.train(params, xgdmat_train, num_boost_round=num_rounds)
    
    y_pred = mdl.predict(xgdmat_test)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    cm = confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    
    ### Use self.calc_metrics(y_test, y_pred) here
    ## save the metrics from cm
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    Accuracy_XGB = (TP + TN) / (TP + TN + FP + FN)
    Precision_XGB = TP / (TP + FP)
    Recall_XGB = TP / (TP + FN)
    F1_Score_XGB = 2 * Precision_XGB * Recall_XGB / (Precision_XGB + Recall_XGB)
    
    ## get the top features from the d matrix train method 
    features = [x for x in X_train.columns if x not in ['id','loss']]
    create_feature_map(features)
    
    importance = mdl.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    df_subset = df.query('fscore > 0.01')
    
    plt.figure()
    df.plot()
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.gcf().savefig('feature_importance_xgb.png')
    
    print(df.sort_values(by='fscore', ascending=False))
    
    
    ## grid search xgboost for best parameters  (to get the best accuracy)
    
    parameters_large = {'nthread':[2,3,4,5], #when use hyperthread, xgboost may become slower
                  'objective':['binary:logistic'],
                  'learning_rate': [0.01,0.02,0.05], #so called `eta` value
                  'max_depth': [2,4,6,8],
                  'min_child_weight': [3,5,7,9],
                  'silent': [1],
                  'subsample': [0.6,0.7,0.8,0.9],
                  'colsample_bytree': [0.6,0.7,0.8,0.9],
                  'n_estimators': [500,1000,10000], #number of trees, change it to 1000 for better results
                  'missing':[-999],
                  'seed': [1337]}
    
    parameters_small = {'nthread':[2,3], #when use hyperthread, xgboost may become slower
                  'objective':['binary:logistic'],
                  'learning_rate': [0.01,0.02], #so called `eta` value
                  'max_depth': [2,4,6,8],
                  'min_child_weight': [3,5,7,9],
                  'silent': [1],
                  'subsample': [0.6,0.7,0.8,0.9],
                  'colsample_bytree': [0.6,0.7],
                  'n_estimators': [100], #number of trees, change it to 1000 for better results
                  'missing':[-999],
                  'seed': [1337]}
    
    def grid_search_xg_boost(X_train, y_train, X_test, y_test, parameters):
        
        xgdmat_train = xgb.DMatrix(X_train, y_train)
        xgdmat_test = xgb.DMatrix(X_test, y_test)
        
        xgb_model = xgb.XGBClassifier()
        
        clf = GridSearchCV(xgb_model, parameters_small, n_jobs=1000, 
                           cv=StratifiedKFold(y_train, n_folds=10, shuffle=True), 
                           scoring='roc_auc',
                           verbose=10, refit=True)
        
        clf.fit(X_train, y_train)
        
        pickle.dump(clf, open("pima.pickle.dat", "wb")) ## save the model
        
        # load model from file
        loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
        
        ## exract the best scores from the grid search
        best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
        print('Raw AUC score:', score)
        
        for param_name in sorted(best_parameters.keys()):
            print("%s: %r" % (param_name, best_parameters[param_name]))
        
        predictions = clf.predict(X_test)
    
        Accuracy_XGB, Precision_XGB, Recall_XGB, F1_Score_XGB, cm_XGB = calc_metrics(y_test, predictions)
        
        num_rounds = 1000
        
        mdl_grid_best = xgb.train(best_parameters, xgdmat_train, num_boost_round=num_rounds)
        y_pred_best_grid = mdl_grid_best.predict(xgdmat_test)
        y_pred_best_grid = np.where(y_pred_best_grid > 0.5, 1, 0)
        
        Accuracy_bestXG, Precision_bestXG, Recall_bestXG, F1_Score_bestXG, cm_bestXG = calc_metrics(y_test, y_pred_best_grid)
    
        ## get the top features from the d matrix train method 
        features_best_grid = [x for x in X_train.columns if x not in ['id','loss']]
        create_feature_map_best_grid(features_best_grid)
        
        importance_bg = mdl_grid_best.get_fscore(fmap='xgb_best_grid.fmap')
        importance_bg = sorted(importance_bg.items(), key=operator.itemgetter(1))
        
        df_bg = pd.DataFrame(importance_bg, columns=['feature', 'fscore'])
        df_bg['fscore'] = df_bg['fscore'] / df_bg['fscore'].sum()
        df_bg_subset = df_bg.query('fscore > 0.01')
        
        plt.figure()
        df_bg.plot()
        df_bg.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
        plt.title('XGBoost Feature Importance')
        plt.xlabel('relative importance')
        plt.gcf().savefig('feature_importance_xgb.png')
        
        print(df.sort_values(by='fscore', ascending=False))
        
        ## plot same or subset of f > 0.01
        plt.figure()
        df_bg_subset.plot()
        df_bg_subset.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
        plt.title('XGBoost Feature Importance Subset F < 0.01')
        plt.xlabel('relative importance')
        plt.gcf().savefig('feature_importance_xgb.png')
        
        print(df_bg_subset.sort_values(by='fscore', ascending=False))
    
        
    ##get features from the best xgboost params 
    
    ## create function to save the best grid feature fscores to an fmap for extraction
    def create_feature_map_best_grid(features):
        outfile = open('xgb_best_grid.fmap', 'w')
        i = 0
        for feat in features:
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))
            i = i + 1
    
        outfile.close()
        
    
    def create_metrics_table(): 
        Accuracies =  Accuracy_Bayes, Accuracy_RF, Accuracy_XGB, Accuracy_bestXG
        Precisions = Precision_Bayes, Precision_RF, Precision_XGB, Precision_bestXG
        Recalls = Recall_Bayes, Recall_RF, Recall_XGB, Recall_bestXG
        F1_Scores = F1_Score_Bayes, F1_Score_RF, F1_Score_XGB, F1_Score_bestXG
        
        Metrics = Accuracies, Precisions, Recalls, F1_Scores
        Methods = ["Regex", "Bayesian","Random Forest", "Grid Search XG Boost"]
        Metrics = pd.DataFrame(list(Metrics),columns= Methods, index = ['Accuracy', 'Precision', 'Recall', 'F1 score'])
        
        ## save the results
        Metrics.to_csv("/home/mccoyd2/Dropbox/Uncertainty Project/Hedging_ML_Metrics.csv", sep=',') # probably need to add a backslash before the space for the path to be recognized correctly
    
    def apply_model(reports_apply, impressions, vocabulary):
        
        X_apply, corpus_apply, text_features_apply, vocabulary_apply = report_preprocessing(reports_apply, impressions, vocabulary, vocab_set = 1 )       
        xgdmat_appy = xgb.DMatrix(X_apply)
        all_data_pred_best_grid = mdl_grid_best.predict(xgdmat_appy)
        
        all_data_pred_best_grid = pd.DataFrame(all_data_pred_best_grid)
        
        all_data_pred_best_grid_cat = np.where(all_data_pred_best_grid > 0.90, 'high confidence hemorrhage', 
             (np.where(all_data_pred_best_grid < 0.20, 'high confidence no hemorrhage', 'medium confidence hemorrhage')))
    
        all_data_pred_best_grid_cat = pd.DataFrame(all_data_pred_best_grid_cat)
    
        apply_predictions_concat = pd.concat([all_data_pred_best_grid, all_data_pred_best_grid_cat], axis=1)
        
        apply_predictions_concat_reports = pd.concat([reports_apply, apply_predictions_concat], axis=1)
        list(apply_predictions_concat_reports.columns.values)
        apply_predictions_concat_reports.columns.values[17] = 'Prediction Probability'
        apply_predictions_concat_reports.columns.values[18] = 'Confidence Category'
        apply_predictions_concat_reports.to_csv("/home/mccoyd2/Documents/AI_Hemorrhage_Detection/Reports/Predictions/Hemorrhage_Reports_Batch_1_Predictions.csv", sep=',')
    
        return apply_predictions_concat_reports
    
    def get_group_list_accessions(model_applied_reports):
        hemorrhage_case_accessions = [] 
        hemorrhage_control_accessions = [] 
        hemorrhage_review_accessions = [] 
        for index in range(model_applied_reports.shape[0]):
            prediction = model_applied_reports['Confidence Category'].iloc[index]
            if prediction == 'high confidence no hemorrhage' :
                hemorrhage_control_accessions.append(model_applied_reports['Acn'].iloc[index])
            elif prediction == 'high confidence hemorrhage':
                hemorrhage_case_accessions.append(model_applied_reports['Acn'].iloc[index])
            else: 
                hemorrhage_review_accessions.append(model_applied_reports['Acn'].iloc[index])
                
        return hemorrhage_control_accessions, hemorrhage_case_accessions, hemorrhage_case_accessions
    
    hemorrhage_case_accessions = map(str, hemorrhage_case_accessions)
    hemorrhage_review_accessions = map(str, hemorrhage_review_accessions)
    hemorrhage_control_accessions = map(str, hemorrhage_control_accessions)
    
    def save_accession_sql_format(accessions,name):
        f = open(name, "w")
        
        out = ""
        for i, num in enumerate(accessions):
        	if i%1000 != 0:
        		out += "\'"+num+"\', "
        	else: 
        		out = out[:-2]
        		out += "\n"
        		f.write(out)
        		out = ""
        
        
        out = out[:-2]
        out += "\n"
        f.write(out)
        
        f.close()
    
    save_accession_sql_format(hemorrhage_case_accessions,'hemorrhage_case_accessions_batch1.txt')
    save_accession_sql_format(hemorrhage_review_accessions,'hemorrhage_review_accessions_batch1.txt')
    save_accession_sql_format(hemorrhage_control_accessions,'hemorrhage_control_accessions_batch1.txt')
    

## function maine shouldn't be in the class    
def main():
    parser = get_parser_classify()
    param = parser.parse_args()
    read = Radiology_Report_NLP(param=param)
    read.define_apply_data()
    ## reports_preprocessing only takes one input argument (vocab_set) AND you should call it only once --> the second time you call it, it will overwrite the object's attributes
    read.report_preprocessing(reports_train, impressions)    
    read.report_preprocessing(CT_reports, impressions, outcome)
    
    Accuracy_Bayes, Precision_Bayes, Recall_Bayes, F1_Score_Bayes, cm_Bayes = read.bayes_classifier() # doesn't take input args
    #Accuracy.Bayes, Precision.Bayes, Recall.Bayes, F1_Score.Bayes, cm.Bayes = read.calc_metrics(self.y_test, self.bayes.prediction) # this is already called within the function obj.bayes_classifier()
    
    ## what is the object classify ?
    # classify.manual_model_test()
    # classify.build_vol_classifier()
    # classify.run_model()

if __name__ == "__main__":
    main()