# ZSFG-UCSF Machine Learning Toolbox for Radiology
Repository for machine learning scripts being developed at Zuckerberg San Francisco General Hospital, Department of Radiology with the University of California. 

![alt text](https://github.com/DavidBMcCoy/ZSFG-UCSF_Machine_Learning/blob/master/Logo.png)

## MLT: 
MLT provides a number of scripts, written in python, utilizing high level application programming interfaces such as keras, tensorflow, and caffe to classify and segment both 2D and 3D medical images. The goal of this project is to provide relatively simple deep learning scripts to medical professionals, specifically radiologists, and researchers working in medical imaging. 

## Description:
Our department has a number of projects aimed at exploring the uses of artificial intelligence (AI) in medical imaging. This repository holds all the AI scripts we have developed alongside documentation to help those new to AI explore its use in medical imaging to improve healthcare. Furthermore, because our work-flow is comprised of first using natural language processing to classify radiology reports in order to apply labels to images, included here are scripts for applying labels to reporting text. Similarly, we have developed many other scripts for dimensional reduction/machine learning in dealing with high dimensional data, these are also provided with documentation. 

## Table of Contents: 
Layout of this repository is broken down as follows: 

### _Images_2D_ or _Images_3D_ 
* Supervised Classification
  * Binary or Multi-class outcome
    * Main-path convolutional networks
    * Residual neural networks 
    * Inception networks
    * Dense networks
* Supervised Segmentation
  * Binary or Multi-class outcome
    * U-net models
    * Auto-encoders
* Denoising 
    * Auto-encoders 
* Generative models
    * Variational auto-encoders
    * Generative adverserial networks (GANs) 
* Unsupervised Clustering
    * Variational auto-encoders -> t-SNE clustering of the latent space to 2/3D
### _Dimensional_Reduction_
* t-Distributed Stochastic Neighbor Embedding (t-SNE)
* Principle Componenent Analysis (PCA)
* Factor Analysis 
* Decision Trees (cART) 
* Backwards Feature Elimination (RFE)
### _Natural Language Processing_
* Bag of words text reduction -> ensemble machine learning methods

### _Supervised Machine Learning_
* Elastic-net regression 
* Support vector machines
* Regressions 
### _Unsupervised Machine Learning_
* K-means clustering
* Hierarchical Agglomerative Clustering

