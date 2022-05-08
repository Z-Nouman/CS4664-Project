# CS 4664 Brain Tumor Classification Project
## Authors: Zaid Al Nouman, Ishan Lamba, Thomas Vergeres

PyTorch implementation of CNNs and transfer learning to classify brain tumor images for the CS 4664 course at Virginia Tech. 

If you choose this option, your repo needs to have a well-written Readme file that explains, in plain English, what the repo is about and which notebooks in it show which experiments.

# Environment
All experiments were run on either glogin at Virginia Tech or through Google Colab. Many of the packages used in this project came preinstalled on these platforms, but some notable ones include: PyTorch and Torchvision; Numpy; Sklearn; Matplotlib; Pickle

# Introduction

Ishan, please add a description of this project, similar to the intros you did for the presentations in class

# Dataset

The datasets we used we obtained through Kaggle. These files should be in the same directory as our code and unzipped prior to running any experiments. 

Binary Classification Dataset: https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset

Multiclass Classification Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

# Tasks

For these tasks, we wanted to compare different state of the art convolutional neural network architectures and determine the factors most important to accurately classify a brain tumor provided an image. The different models we wanted to compare include:

- ResNet18
- GoogleNet
- VGG11 with batch normalization

The tasks we have explored are as follows:

1) Binary classification using the binary classification dataset
2) Multiclass classification using the multiclass classification dataset
3) K-nearest Neighbor classifier to act as a baseline using the binary classification dataset
4) Automated hyperparameter tuning using a hyperparameter tuning framework {learning rate, momentum, batch size}
5) Empirical experiments to compare how model performance changes with respect to some variable {amount of training data, modle architecture, model size}
6) Interpretability experiments with T-SNE and Grad-CAM
7) Bonus tasks such as grade classification of brain tumors, application of our models on other types of tumor datasets, or classifying additional brain diseases

# Binary Classification Experiment

Files for this experiment includes the binary classification dataset (unzipped) as well as the binary_classification.py. To run this experiment, simply run the python file likeso: "python3 binary_classification.py". This will run the experiment and print out the results for each of the 3 models utilized. The results of a single run of this experiment can be shown below:

Table 1: Binary Classification Experiment Results (1 run)
| Model | Size | Time to train | Validation Acc | Test Acc | F1-Score
| --- | --- | --- | --- | --- | --- |
| ResNet18 | 11 million params | 25 min | 96.08% | 95.80% | 0.9612 |
| GoogleNet | 7 million params | 39 min | 96.52% | 93.91% | 0.9426 |
| VGG11 | 133 million params | 90 min | 96.52% | 95.07% | 0.9553 |

As we can see, with a slightly bigger architecture, ResNet18 is able to achieve the highest performance, with VGG11 coming in as a close second. One theory our group had after running this experiment is that ResNet18 is able to learn the features necessary to classify a brain tumor image better than GoogleNet with it's larger architecture. However, VGG11 has too large of an architecture and learns unimportant features, causing it to come in second place. 

# Multiclass Classification Experiment

Similar to the previous experiment, the files necessary includes the multiclass classification dataset (unzipped) as well as the multiclass_classification.py. To run this experiment, simply run the python file likeso: "python3 multiclass_classification.py". This will run the experiment and print out the results for each of the 3 models utilized. The results of 5 runs of this experiment (averaged) can be shown below:

Table 2: Multiclass Classification Experiment Results (5 runs, averaged)
| Model | Size | Time to train | Validation Acc | Test Acc | F1-Score
| --- | --- | --- | --- | --- | --- |
| ResNet18 | 11 million params | 34 min | 94.07% | 93.51% | 0.9353 |
| GoogleNet | 7 million params | 46.6 min | 92.65% | 93.11% | 0.9319 |
| VGG11 | 133 million params | 121 min | 93.73% | 93.30% | 0.9330 |

As we can see, ResNet18 again achieves the highest performance, with VGG11 coming in second and GoogleNet coming in last again. 

# KNN Baseline Experiment

Ishan, please fill this out

# Hyperparameter Tuning Experiment

Thomas, please fill this out 

# Empirical Experiments

Thomas, please fill this out 

## Amount of training data

For this experiment, the files necessary includes the multiclass classification dataset (unzipped) as well as the multiclass_classification.py, multiclass_classification_50.py and multiclass_classification_20.py. To run this experiment, simply run the python file likeso: "python3 multiclass_classification_20.py" to see the experiment results with only 20% of the training data. The results of using 100% of training data can be found in Table 2, while the results for using 50% and 20% of training data can be shown below: 

Table 3: Multiclass Classification Experiment 50% of training data results (1 run)

<Insert experiment results here>
  
Table 4: Multiclass Classification Experiment 20% of training data results (1 run)

<Insert experiment results here>

# Interpretability Experiment

Thomas, please fill this out 

# Bonus Experiments

Ishan, please fill this out

# Conclusion

Overall, using state of the art CNNs and transfer learning, we were able to achieve very high performance with all 3 models on both of the datasets. While these results are still unable to challenge expert human performance (medical professionals), its results, particularly with respect to interpretibility and the empirical experiments, provide insight on vital features or aspects required to classify brain tumor images. 
  
