# -*- coding: utf-8 -*-
"""Brain Disease KNN.ipynb

    MAKE SURE YOU HAVE A FOLDER OF THE UNZIPPED ALZHEIMER'S DATA BEFORE YOU DO THIS!!!
"""

# File handling packages
from PIL import Image, ImageOps
import glob
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

scaler = transforms.Resize((512, 512))

to_tensor = transforms.ToTensor()
to_PIL = transforms.ToPILImage()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def process_dataset(dataset_file_name, total_num_samples, label, storage_gray):

    all_grays = storage_gray
    
    # To sanity check
    num_processed_images = 0

    # Iterate through all images in the "Brain Tumor" category (positive labels)
    for count, img in enumerate(glob.glob(dataset_file_name)):

        # Open the image
        image = Image.open(img).convert("RGB")

        # Max dimensions is height of 1427 and width of 1275
        width, height = image.size

        # If the image is smaller than 512 by 512, add padding to it
        # We choose the values 512 by 512 because more than 88% / 87.5% of these 
        # images are smaller than these values. Thus, it makes sense to choose 
        # this dimensionality because we don't add too much padding while 
        # normalizing the input size. 
        if height <= 512 and width <= 512:
            padded_width = 512 - width
            padded_height = 512 - height
          
            result = Image.new(image.mode, (width + padded_width, height + padded_height))
            result.paste(image, (padded_width // 2, (padded_height) // 2))
        else:

            # Otherwise, rescale the image to 512 by 512. This does not add 
            # padding, nor is this cropping. This merely resizes it using the 
            # transforms package from torchvision
            result = scaler(image)
        
        # This normalize function normalizes the pixel values.
        # Let's experiment to see how the model performs without this first, 
        # then add it in to see if it provides a performance boost

        # Make this 3 channels because that's how Resnet was trained
        temp = np.asarray(result)
        result = to_PIL(normalize(to_tensor(temp)))

        # Transform the image to grayscale since we the input image does 
        # not have color. This saves memory by only storing 3 channels 
        # rather than 1.
        grey_image = result.convert('L')
        grey_temp = np.asarray(grey_image)

        # Finally, we have a 2D matrix with our pixel values in a numpy array, 
        # stored as pixels
        pixels = np.array(result)

        height = pixels.shape[0]
        width = pixels.shape[1]

        # Sanity check to make sure every image is now 512 by 512
        if height == 512 and width == 512:
          num_processed_images += 1

        all_grays.append((grey_temp, label))

# Append all the images into a list with a label 0 if healthy, 1 if the patient has very mild dementia
# 2 if the patient has mild dementia, and 3 if the patient has moderate dementia
all_grays = list()
process_dataset('./AlzheimersDataSet/NonDemented/*', 3200, 0, all_grays)
process_dataset('./AlzheimersDataSet/VeryMildDemented/*', 2240, 1, all_grays)
process_dataset('./AlzheimersDataSet/MildDemented/*', 896, 2, all_grays)
process_dataset('./AlzheimersDataSet/ModerateDemented/*', 64, 3, all_grays)

# With the data in tuple form, split them up into x and y
gray_x = []

for tup in all_grays:
	gray_x.append(tup[0])
 
gray_y = []

for tup in all_grays:
	gray_y.append(tup[1])

#This splitting is done so that gray_x can be reshaped into an acceptable format for the KNN-algorithm
#https://stackoverflow.com/questions/65671880/valueerror-found-array-with-dim-3-estimator-expected-2-python was used for help
gray_x = np.array(gray_x)
gray_y = np.array(gray_y)
dimX1, dimX2, dimX3 = gray_x.shape
gray_x = np.reshape(gray_x, (dimX1, dimX2 * dimX3))

#Split the data and get just the first 360 samples of both for sets for training and testing, given the large number of data points
x_gray_train, x_gray_test, y_gray_train, y_gray_test = train_test_split(gray_x, gray_y, test_size=0.5, random_state=42)

x_gray_train = x_gray_train[0:360]
y_gray_train = y_gray_train[0:360]
x_gray_test = x_gray_test[0:360]
y_gray_test = y_gray_test[0:360]

#Reduce the dimensions and fit the KNeighborsClassifier onto the training data
from sklearn.decomposition import PCA
pca = PCA(n_components=128)
pca.fit(x_gray_train)

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=6)
knn_model.fit(x_gray_train, y_gray_train)
gray_y_pred = knn_model.predict(x_gray_test)

#Once the model is fit and makes predictions on the x_gray_test data, ouput the accuracy metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print("Accuracy Score: \n", accuracy_score(y_gray_test, gray_y_pred))
print("Classification Report: \n", classification_report(y_gray_test, gray_y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_gray_test, gray_y_pred))
