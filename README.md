# CS 4664 Brain Tumor Classification Project
## Authors: Zaid Al Nouman, Ishan Lamba, Thomas Vergeres

PyTorch implementation of CNNs and transfer learning to classify brain tumor images for the CS 4664 course at Virginia Tech. 

If you choose this option, your repo needs to have a well-written Readme file that explains, in plain English, what the repo is about and which notebooks in it show which experiments.

# Environment
All experiments were run on either glogin at Virginia Tech or through Google Colab. Many of the packages used in this project came preinstalled on these platforms, but some notable ones include: PyTorch and Torchvision; Numpy; Sklearn; Matplotlib; Pickle

# Introduction

Roughly 18,000 Americans die annually from [brain tumors](https://www.cancer.net/cancer-types/brain-tumor/statistics). In order to give patients a higher chance at survival, early diagnosis is key to ensuring more effective treatment. However, brain tumor diagnosis is comparatively far more challenging than other tumors. Not only is it more difficult to spot, but it is also hard to access given the fragility of the brain.

Diagnosing the tumor is also difficult because, while an initial MRI and CAT scan are done to determine the presence of a tumor, biopsies are often performed in order to determine the grade, which is the aggressiveness or severity of the tumor, as well as identify certain biomarkers in order to personalize the treatment.This is risky because the more times the brain is exposed to an invasive procedure, there is an increased chance for error and further damage could arise.

A non-invasive approach is far safer, but it’s largely done manually by certified radiologists. Radiologists identifying tumors in scans is both laborious and time-consuming. While certain countries, like the US, are fortunate to have a suitable number of radiologists per person, other countries, [like England](https://www.theguardian.com/politics/2017/dec/31/lack-of-specialist-surgeons-putting-patients-at-risk), have a shortage of radiologists, which can cause delays in image interpretation that last for months, which can literally be a life-changing difference for patients.

Classification of tumors can be an incredible asset for doctors, because it can not only provided a second opinion, but allow for a far more personalized treatment approach, not only for the type of tumor, but its location and severity. Without it, a patient is likely to get a generic treatment that might not be effective, lowering the chance of survival.

As a result, people have turned to machine learning in recent years as a way to assist radiologists in identifying and classifying tumors in order to get people treatment faster. Given that the current state-of-the-art still has flaws, we were eager to get involved as well.


# Dataset

The datasets we used we obtained through Kaggle. These files should be in the same directory as our code and unzipped prior to running any experiments. 

Binary Classification Dataset: https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset

Multiclass Classification Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Chest Tumor Dataset (Bonus): https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images

Alternative Brain Disease Dataset (Alzheimer's, Bonus): https://www.kaggle.com/datasets/yasserhessein/dataset-alzheimer

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

## Table 1: Binary Classification Experiment Results (1 run)
| Model | Size | Time to train | Validation Acc | Test Acc | F1-Score
| --- | --- | --- | --- | --- | --- |
| ResNet18 | 11 million params | 25 min | 96.08% | 95.80% | 0.9612 |
| GoogleNet | 7 million params | 39 min | 96.52% | 93.91% | 0.9426 |
| VGG11 | 133 million params | 90 min | 96.52% | 95.07% | 0.9553 |

As we can see, with a slightly bigger architecture, ResNet18 is able to achieve the highest performance, with VGG11 coming in as a close second. One theory our group had after running this experiment is that ResNet18 is able to learn the features necessary to classify a brain tumor image better than GoogleNet with it's larger architecture. However, VGG11 has too large of an architecture and learns unimportant features, causing it to come in second place. 

# Multiclass Classification Experiment

Similar to the previous experiment, the files necessary includes the multiclass classification dataset (unzipped) as well as the multiclass_classification.py. To run this experiment, simply run the python file likeso: "python3 multiclass_classification.py". This will run the experiment and print out the results for each of the 3 models utilized. The results of 5 runs of this experiment (averaged) can be shown below:

## Table 2: Multiclass Classification Experiment Results (5 runs, averaged)
| Model | Size | Time to train | Validation Acc | Test Acc | F1-Score
| --- | --- | --- | --- | --- | --- |
| ResNet18 | 11 million params | 34 min | 94.07% | 93.51% | 0.9353 |
| GoogleNet | 7 million params | 46.6 min | 92.65% | 93.11% | 0.9319 |
| VGG11 | 133 million params | 121 min | 93.73% | 93.30% | 0.9330 |

As we can see, ResNet18 again achieves the highest performance, with VGG11 coming in second and GoogleNet coming in last again. 

# KNN Baseline Experiment

The K-Nearest Neighbors algorithm was used as a non-machine learning baseline for our Binary Classification experiment. However, the sheer size of the dataset and each data point required both dimensionality reduction using PCA and use of a random 15% of our data for training and testing. The code for this experiment was included in binary_classification.py. The same instructions to run that file are applied to this experiment. The accuracy across 10 runs of this experiment (averaged) as well as the confusion matrix of our best run are shown below:
## Binary Classification KNN Baseline Results

### Accuracy (10 runs, averaged):
82.03%

### Table 3: Confusion Matrix (best run)
| | Predicted Tumor | Predicted No Tumor |
| --- | --- | --- |
| Actual Tumor | 129 | 29 |
| Actual No Tumor | 35 | 167 |

The highest accuracy came when the number of components for PCA was 128 and there were 5 neighbors (82.22%).

# Hyperparameter Tuning Experiment

Thomas, please fill this out 

# Empirical Experiments

Thomas, please fill this out 

## Amount of training data

For this experiment, the files necessary includes the multiclass classification dataset (unzipped) as well as the multiclass_classification.py, multiclass_classification_50.py and multiclass_classification_20.py. To run this experiment, simply run the python file likeso: "python3 multiclass_classification_20.py" to see the experiment results with only 20% of the training data. The results of using 100% of training data can be found in [Table 2](#table-2-multiclass-classification-experiment-results-5-runs-averaged), while the results for using 50% and 20% of training data can be shown below: 

## Table 4: Multiclass Classification Experiment 50% of training data results (1 run)

<Insert experiment results here>
  
## Table 5: Multiclass Classification Experiment 20% of training data results (1 run)

<Insert experiment results here>

# Interpretability Experiment

Thomas, please fill this out 

# Bonus Experiments

For the bonus experimensts, we analyzed two different aspects that were similar in nature to our brain tumor experiments. Classification of tumors, but in other parts of the body, and diseases of the brain other than tumors. Given the scarcity of easily accessible, clean data online that were not related to brain tumors, we focused on just chest tumors for the tumor classification bonus experiment and Alzheimer's/Dementia for the brain disease bonus experiment. Additionally, due to the uneven distribution across data classes, we believed that multi-class classification would yield a more accurate result than binary classification.

## Chest Tumor Classification Experiment

For this experiment, the files necessary include the chest tumor dataset (unzipped) as well as the image_processing_and_classification_chest_tumor.py script. To run this experiment, simply run the python file likeso: "python3 image_processing_and_classification_chest_tumor.py" to see both the KNN baseline and the performance of the 3 models utilized at identifying the type of tumor that exists within the scan. The results of two runs of the KNN baseline and a single run of the models can be shown below:
  
### KNN Baseline Accuracy (2 runs, averaged):
68.98%

### Table 6: Confusion Matrix (best run)
| | Predicted No Tumor | Predicted Adenocarcinoma | Predicted Large Cell Carcinoma | Predicted Squamous Cell Carcinoma |
| --- | --- | --- | --- | --- |
| Actual No Tumor | 14 | 2 | 1 | 0 |
| Actual Adenocarcinoma | 2 | 130 | 11 | 33 |
| Actual Large Cell Carcinoma | 1 | 16 | 42 | 8 |
| Actual Squamous Cell Carcinoma | 0 | 32 | 1 | 81 |

### Table 7: Chest Tumor Classification Experiment Results (1 run)
| Model | Size | Time to train | Validation Acc | Test Acc | F1-Score
| --- | --- | --- | --- | --- | --- |
| ResNet18 | 11 million params | ? min | ?% | ?% | ? |
| GoogleNet | 7 million params | ? min | ?% | ?% | ? |
| VGG11 | 133 million params | ? min | ?% | ?% | ? |
  
## Alzheimer's Classification Experiment
  
For this experiment, the files necessary include the Alzheimer's dataset (unzipped) as well as the image_processing_and_classification_brain_diseases.py script. To run this experiment, simply run the python file likeso: "python3 image_processing_and_classification_brain_diseases.py" to see both the KNN baseline and the performance of the 3 models utilized at identifying the level of dementia that exists within the scan. The results of two runs of the KNN baseline and a single run of the models can be shown below:
  
### KNN Baseline Accuracy (2 runs, averaged):
46.81%

### Table 8: Confusion Matrix (best run)
| | Predicted No Dementia | Predicted Very Mild Dementia | Predicted Mild Dementia | Predicted Moderate Dementia |
| --- | --- | --- | --- | --- |
  | Actual No Dementia | 101 | 64 | 28 | 0 |
| Actual Very Mild Dementia | 28 | 52 | 33 | 0 |
| Actual Mild Dementia | 10 | 24 | 19 | 0 |
| Actual Moderate Dementia | 0 | 1 | 0 | 0 |

### Table 9: Dementia Severity Classification Experiment Results (1 run)
| Model | Size | Time to train | Validation Acc | Test Acc | F1-Score
| --- | --- | --- | --- | --- | --- |
| ResNet18 | 11 million params | ? min | ?% | ?% | ? |
| GoogleNet | 7 million params | ? min | ?% | ?% | ? |
| VGG11 | 133 million params | ? min | ?% | ?% | ? |

# Conclusion

Overall, using state of the art CNNs and transfer learning, we were able to achieve very high performance with all 3 models on both of the datasets. While these results are still unable to challenge expert human performance (medical professionals), its results, particularly with respect to interpretibility and the empirical experiments, provide insight on vital features or aspects required to classify brain tumor images. 
  
