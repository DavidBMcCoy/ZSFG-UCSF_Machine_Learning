# ZSFG-UCSF Machine Learning Toolbox for Radiology
Repository for machine learning scripts being developed at Zuckerberg San Francisco General Hospital, Department of Radiology with the University of California. 

![alt text](https://github.com/DavidBMcCoy/ZSFG-UCSF_Machine_Learning/blob/master/Logo.png)

## MLT: 
MLT provides a number of scripts, written in python, utilizing high level application programming interfaces such as keras, tensorflow, and caffe to classify and segment both 2D and 3D medical images. The goal of this project is to provide relatively simple deep learning scripts to medical professionals, specifically radiologists, and researchers working in medical imaging. 

## Description:
Our department has a number of projects aimed at exploring the uses of artificial intelligence (AI) in medical imaging. This repository holds all the AI scripts we have developed alongside documentation to help those new to AI explore its use in medical imaging to improve healthcare. Furthermore, because our work-flow is comprised of first using natural language processing to classify radiology reports in order to apply labels to images, included here are scripts for applying labels to reporting text. Similarly, we have developed many other scripts for dimensional reduction/machine learning in dealing with high dimensional data, these are also provided with documentation. 

## Citing this repository:
If you use scripts in this repository for research or clinical purposes please cite us. Scripts were developed by David McCoy, a medical data scientist with UCSF working at ZSFG; as an employee of UCSF, his work is intellectual property of UCSF and requires citation if used by others. 
```
@article{mccoyd2018,
  title={Machine Learning Toolbox: A repository of machine learning programs for medical image processing},
  author={David B. McCoy},
  journal={},
  year={2018}
}
```
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

## Using MLT:
Currently, this repository is built such that each script is stand-alone. In the future, we plan on making these sets a complete API. For the time being, simply clone this repository to your local machine to easily access the scripts for use in your own research: 

```
cd YOUR_DESIRED_PLACE_FOR_INSTALL
git clone https://github.com/DavidBMcCoy/ZSFG-UCSF_Machine_Learning.git 
```
No play data is provided. All deep learning scripts were built using diagnostic x-ray, CT, or MRI data from the Department of Public Health at ZSFG. However, detailed documentation are provided in each project folder to show how input data should be formatted. For example, scripts are provided to convert all DICOM data to NIFTI which then is loaded to HDF5 files in the tensorflow format (batch_size, x-dim, y-dim, z-dim, channels). Documentation is provided on how to format your data to work with all deep learning scripts (organizing your data so that the scripts provided and take stacks of DICOM and put them in one HDF5 file which will work with all deep learning scripts). 

## Contributions and Collaborations:
This work embodies several years of research on a diversity of projects at UCSF. If you are interested in collaborating please contact David McCoy at david.mccoy@ucsf.edu.



