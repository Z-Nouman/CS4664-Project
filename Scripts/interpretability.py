from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import numpy as np
import pickle as pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import datasets, models, transforms
import torch.nn as nn
import cv2
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


TRAIN_BATCH_SIZE = 1


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

    # Add a binary classification layer at the end
    if not_vgg:
        num_ftrs = model.fc.in_features

        model.fc = nn.Linear(num_ftrs, 2).cuda()
    else:
        num_ftrs = 4096
        model.classifier[6] = nn.Linear(num_ftrs, 2).cuda()

    model.cuda()

    model.load_state_dict(torch.load(file_name))

    model.eval()

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

def create_data_loader(Dataset, images, labels, batch_size, num_workers=2, shuffle=False, pin_memory=True):
    ds = Dataset(
        images = images,
        labels = labels
    )

    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory)

with open("./processed_dataset.pkl", 'rb') as data_file:
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

#Load model
model = load_model_weights("resnet18")
target_layers = [model.layer4[-1]]
input_tensor = None
skipn = 120

#Iterate through data loader to specific sample
for count, ele in enumerate(validation_data_loader):
    skipn -= 1
    if skipn >= 0:
        continue
    inputs, labels = ele

    #Skip datapoint if not tumor
    if labels.numpy()[0] != 1:
        continue
    inputs = torch.reshape(inputs, (inputs.shape[0], inputs.shape[-1], 512, 512))
    inputs = torch.div(inputs, 255.0)
    input_tensor = inputs
    break


print("IT shape: ", input_tensor.shape)
print("IT type: ", type(input_tensor))

cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

targets = None


grayscale_cam = cam(input_tensor=input_tensor, targets=targets)


grayscale_cam = grayscale_cam[0, :]
print(type(grayscale_cam))
print(grayscale_cam.shape)
rgb_img = torch.reshape(input_tensor, (input_tensor.shape[0], 512, 512, input_tensor.shape[1])).detach().numpy()[0]

print(rgb_img.shape)

#Get GradCAM visualization
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

plt.imshow(rgb_img[:, :, 0], cmap='gray')
plt.savefig('images/int/base1.png')

plt.imshow(visualization)
plt.savefig('images/int/grad1.png')

