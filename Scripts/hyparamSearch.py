from __future__ import print_function, division

import optuna


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
import plotly
from tqdm import tqdm


print("Finished Imports.")

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

        # Display the preprocessed image if you want
        # print("Preprocessed image size is:", np.asarray(image).shape)
        # display(image)

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
        # grey_image = result.convert('L')

        # Display the post-processed image
        # print(np.array(result).shape)
        # display(grey_image)
        
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
    # return all_samples


################################TO PREPROCESS DATA THE FIRST TIME##########################################
# all_samples = list()
# process_dataset('./Brain Tumor Data Set/Brain Tumor/*', 2513, 1, all_samples)
# process_dataset('./Brain Tumor Data Set/Healthy/*', 2087, 0, all_samples)

# processed_dataset = {
#     "Images": all_samples,
# }

# with open('processed_dataset.pkl', 'wb') as handle:
#     pickle.dump(processed_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
################################TO PREPROCESS DATA THE FIRST TIME##########################################

# In the future, if you want to load the processed dataset, simply do:

# with open(<PATH_NAME>, 'rb') as data_file:
#     <VARIABLE_TO_STORE_PROCESSED_DATA> = pickle.load(data_file)

TRAIN_BATCH_SIZE = 4

cudnn.benchmark = True
plt.ion()   # interactive mode

device = torch.device('cuda:0')
device


def simple_train_model(model, criterion, optimizer, scheduler, bsize, num_epochs=25):

    print("Batch size: ", bsize)

    train_data_loader = create_data_loader(ImageDataset, X_train, y_train, bsize)
    validation_data_loader = create_data_loader(ImageDataset, X_validation, y_validation, bsize)


    dataloaders = {
    'train': train_data_loader,
    'val': validation_data_loader
    }

    dataset_sizes = {
    'train': len(X_train),
    'val': len(X_validation),
    'test': len(X_test)
    }

    test_data_loader = create_data_loader(ImageDataset, X_test, y_test, bsize)



    print("Dataset Sizes: ", dataset_sizes["train"])
    print("Dataset Sizes: ", dataset_sizes["val"])

    since = time.time()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        epochLoss = 0
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # skipped = 0

            # Iterate over data.
            for count, ele in enumerate(dataloaders[phase]):
                inputs, labels = ele
                # print(inputs.shape)
                inputs = inputs.to(device)
                labels = labels.to(device)
                # if phase == 'train' and count > 20:
                #     continue
                inputs = torch.reshape(inputs, (inputs.shape[0], inputs.shape[-1], 512, 512))
                inputs = torch.div(inputs, 255.0)
                
                # if count % 300 == 0:
                #   print("Making progress...", count)

                # if type(labels) == type("string"):
                #     skipped += 1
                #     continue
                # zero the parameter gradients
                optimizer.zero_grad()


                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # print("Loss: ",loss)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # print(running_loss)
                # print(loss.item(), running_loss)
            if phase == 'train':
                scheduler.step()

            # print(skipped)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
                # best_model_wts = copy.deepcopy(model.state_dict())

            epoch_loss += running_loss

        print()
        # perfArr.append([epoch, epoch_loss])

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    return model, epoch_acc

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


dataloaders = {'train': train_data_loader,'val': validation_data_loader}

dataset_sizes = {'train': len(X_train), 'val': len(X_validation), 'test': len(X_test)}

test_data_loader = create_data_loader(ImageDataset, X_test, y_test, TRAIN_BATCH_SIZE)

from sklearn.metrics import f1_score, confusion_matrix


def objective(trial):
    batch_size_ = trial.suggest_int("batch_size_", 2, 100)

    model = models.resnet18(pretrained=True)
    not_vgg = True

    lr_ = trial.suggest_float("lr_", 1e-5, 1e-1, log=True)
    momentum_ = trial.suggest_float("momentum_", 5e-1, 1, log=True)

    if not_vgg:
        num_ftrs = model.fc.in_features

        model.fc = nn.Linear(num_ftrs, 2).cuda()

    model.cuda()

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=lr_, momentum=momentum_)


    step_size_ = trial.suggest_int("step_size_", 1, 10)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size_, gamma=0.1)

    model, accuracy = simple_train_model(model, criterion, optimizer_ft, exp_lr_scheduler, batch_size_,
                          num_epochs=10)
    
    return accuracy


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))


print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)
f = open("bestTrialParams.txt", "a")

print("  Params: ")
for key, value in trial.params.items():
    f.write("    {}: {}".format(key, value))
    print("    {}: {}".format(key, value))

f.close()

importanceFig = optuna.visualization.plot_param_importances(study)
sliceFig = optuna.visualization.plot_slice(study, params=["lr_", "momentum_", "batch_size_", "step_size_"])

importanceFig.write_image("images/hp3/importanceFig.png")
sliceFig.write_image("images/hp3/sliceFig.png")

intFig = optuna.visualization.plot_intermediate_values(study)
intFig.write_image("images/hp3/intFig.png")


