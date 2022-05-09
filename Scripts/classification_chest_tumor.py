from __future__ import print_function, division
# -*- coding: utf-8 -*-
"""Image Processing and Classification Chest Tumor.ipynb

    MAKE SURE YOU HAVE A FOLDER OF THE UNZIPPED CHEST TUMOR DATA BEFORE YOU DO THIS!!!
"""

# File handling packages
from PIL import Image, ImageOps
import glob
import pickle as pickle
import numpy as np
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

scaler = transforms.Resize((512, 512))

to_tensor = transforms.ToTensor()
to_PIL = transforms.ToPILImage()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def process_dataset(dataset_file_name, total_num_samples, label, storage):

    # To store the processed samples
    all_samples = storage

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

        # Finally, we have a 2D matrix with our pixel values in a numpy array, 
        # stored as pixels
        pixels = np.array(result)

        height = pixels.shape[0]
        width = pixels.shape[1]

        # Sanity check to make sure every image is now 512 by 512
        if height == 512 and width == 512:
          num_processed_images += 1

        all_samples.append((pixels, label))

    assert(num_processed_images == total_num_samples)

################################TO PREPROCESS DATA THE FIRST TIME##########################################
all_samples = list()
process_dataset('./ChestTumorDataSet/normal/*', 37, 0, all_samples)
process_dataset('./ChestTumorDataSet/adenocarcinoma/*', 326, 1, all_samples)
process_dataset('./ChestTumorDataSet/large.cell.carcinoma/*', 143, 2, all_samples)
process_dataset('./ChestTumorDataSet/squamous.cell.carcinoma/*', 241, 3, all_samples)

processed_dataset = {
    "Images": all_samples,
}

with open('processed_multi_dataset.pkl', 'wb') as handle:
    pickle.dump(processed_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
################################TO PREPROCESS DATA THE FIRST TIME##########################################

TRAIN_BATCH_SIZE = 4

cudnn.benchmark = True
plt.ion()   # interactive mode

device = torch.device('cuda:0')
device

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for count, ele in enumerate(dataloaders[phase]):
                inputs, labels = ele
                inputs = inputs.to(device)
                labels = labels.to(device)

                inputs = torch.reshape(inputs, (inputs.shape[0], inputs.shape[-1], 512, 512))
                inputs = torch.div(inputs, 255.0)
                
                if count % 500 == 0:
                  print("Making multiclassifier progress...", count)

                optimizer.zero_grad()


                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                preds.detach()
                labels.detach()
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

class ImageDataset(Dataset):

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        images = self.images[idx]
        labels = self.labels[idx]
        
        return images, labels

def create_data_loader(Dataset, images, labels, batch_size, num_workers=2, shuffle=True, pin_memory=True):
    ds = Dataset(
        images = images,
        labels = labels
    )

    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory)

with open("./processed_multi_dataset.pkl", 'rb') as data_file:
    all_processed_images = np.array(pickle.load(data_file)['Images'])

X = all_processed_images[:, 0]
y = all_processed_images[:, 1]

X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y)

X_validation, X_test, y_validation, y_test = train_test_split(
       X_test, y_test, test_size=0.75, random_state=42, stratify=y_test)

train_data_loader = create_data_loader(ImageDataset, X_train, y_train, TRAIN_BATCH_SIZE)
validation_data_loader = create_data_loader(ImageDataset, X_validation, y_validation, TRAIN_BATCH_SIZE)

dataloaders = {
  'train': train_data_loader,
  'val': validation_data_loader
}

dataset_sizes = {
  'train': len(X_train),
  'val': len(X_validation),
  'test': len(X_test)
}

from sklearn.metrics import f1_score, confusion_matrix
test_data_loader = create_data_loader(ImageDataset, X_test, y_test, TRAIN_BATCH_SIZE)

def test_model(model, criterion):
    since = time.time()

    best_acc = 0.0

    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    all_predictions = []
    all_labels = []
    # Iterate over data.
    for count, ele in enumerate(test_data_loader):
        inputs, labels = ele
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = torch.reshape(inputs, (inputs.shape[0], inputs.shape[-1], 512, 512))
        inputs = torch.div(inputs, 255.0)
                
        if count % 500 == 0:
          print("Making progress...", count)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        all_predictions.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu())

            # print(skipped)
    epoch_loss = running_loss / len(X_test)
    epoch_acc = float(running_corrects) / len(X_test)

    print(f'Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    print("F-Score is:", f1_score(all_labels, all_predictions, labels=[0, 1, 2, 3], average='weighted'))
    print("Confusion matrix is:", confusion_matrix(all_labels, all_predictions))
    time_elapsed = time.time() - since
    print(f'Testing complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best test Acc: {best_acc:4f}')

def train_and_return_model(model, not_vgg=True):

    model.cuda()

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                          num_epochs=25)
    
    return model, criterion

def load_model_weights(model_name, not_vgg=True):

    model = None

    # Pick a model to load
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        file_name = "best_resnet18.pt"

    elif model_name == "googlenet":
        model = models.googlenet(pretrained=True)
        file_name = "best_googlenet.pt"
    elif model_name == "vggnet":
        model = models.vgg11_bn(pretrained=True)
        file_name = "best_vggnet.pt"
    else:
        print("Invalid model name")
        return

    # Add a multi-class classification layer at the end
    if not_vgg:
        num_ftrs = model.fc.in_features

        model.fc = nn.Linear(num_ftrs, 4).cuda()
    else:
        num_ftrs = 4096
        model.classifier[6] = nn.Linear(num_ftrs, 4).cuda()

    model.cuda()

    model.load_state_dict(torch.load(file_name))

    model.eval()

    return model

###########################################################################

model_resnet18 = models.resnet18(pretrained=True)

model_resnet18, criterion = train_and_return_model(model_resnet18)

test_model(model_resnet18, criterion)

###########################################################################

model_googlenet = models.googlenet(pretrained=True)

model_googlenet, criterion = train_and_return_model(model_googlenet)

test_model(model_googlenet, criterion)

###########################################################################

model_vgg11 = models.vgg11_bn(pretrained=True)

model_vgg11, criterion = train_and_return_model(model_vgg11, False)

test_model(model_vgg11, criterion)

###########################################################################