from sklearn.manifold import TSNE
import numpy as np
from numpy import reshape
import pandas as pd  
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch
import collections
from torch.utils.data import Dataset, DataLoader
# from keras.datasets import mnist
from sklearn.datasets import load_iris
from sklearn import decomposition
from functools import partial
import cv2
device = torch.device('cpu')

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

        # model.fc = nn.Linear(num_ftrs, 2).cuda()
        model.fc = nn.Linear(num_ftrs, 2)

    else:
        num_ftrs = 4096
        model.classifier[6] = nn.Linear(num_ftrs, 2).cuda()

    # model.cuda()

    model.load_state_dict(torch.load(file_name))

    model.eval()

    return model

resModel = load_model_weights("resnet18")


#from https://gist.github.com/Tushar-N/680633ec18f5cb4b47933da7d10902af
# a dictionary that keeps saving the activations as they come
activations = collections.defaultdict(list)
def save_activation(name, mod, inp, out):
	activations[name].append(out.cpu())

#Add hook to get activations
for name, m in resModel.named_modules():
	if type(m)==nn.Conv2d and name == 'layer4.1.conv2':
		# partial to assign the layer name to each hook
		m.register_forward_hook(partial(save_activation, name))



TRAIN_BATCH_SIZE = 1

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

    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=pin_memory)


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


resModel.eval()
n = 150
print("Getting activations...")
images = []
nLabels = []
for count, ele in enumerate(validation_data_loader):
    #Only get half of data points
    if count % 2 != 0:
        continue
]
    print(count)
    inputs, labels = ele
    inputs = inputs.to(device)
    labels = labels.to(device)
    images.append(inputs)
    inputs = torch.reshape(inputs, (inputs.shape[0], inputs.shape[-1], 512, 512))
    inputs = torch.div(inputs, 255.0)
    out = resModel(inputs)

    #Record Label
    nLabels.append(labels.detach().numpy())

nLabels = np.array(nLabels)
print("nLabels info: ")
print(nLabels.shape)
print(nLabels)

# concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
activations = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}


print("Activations:")
# just print out the sizes of the saved activations as a sanity check
for k,v in activations.items():
    print (k, v.size())
    print (type(k))

print("activation numbers: ", len(activations.items()))

lastLayerActivations = activations['layer4.1.conv2'].detach().numpy()


print(lastLayerActivations.shape)

#Reshape for compatibility
lla = lastLayerActivations.reshape(lastLayerActivations.shape[0], 512, 256)
lla = lla.reshape(lastLayerActivations.shape[0], -1)

dso = ImageDataset(images=X_test, labels=y_test)
dsy = dso.labels[::5]

dsy = nLabels.reshape((nLabels.shape[0],))

dsx = lla 

#Perform PCA
pca = decomposition.PCA(n_components=6)
dsx = pca.fit_transform(dsx)

#Perform TSNE
tsne = TSNE(n_components=2, verbose=1, random_state=42, perplexity=20, n_iter=2500, learning_rate=5)
z = tsne.fit_transform(dsx)


#visualizing tsne -- from: https://learnopencv.com/t-sne-for-feature-visualization/
def scale01(x):
    valueRange = np.max(x) - np.min(x)
    startsFromZero = x - np.min(x)
    return startsFromZero / valueRange

tx = z[:, 0]
ty = z[:, 1]
tx = scale01(tx)
ty = scale01(ty)

plot_size = 1200
padding = 300


tsne_plot = 255 * np.ones((plot_size + padding * 2, plot_size + padding * 2, 3), np.uint8)
print(tsne_plot.shape)

def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    # center_x = int(plot_size * x) + offset
    center_x = int(x * plot_size + padding * 1.5)

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int((1 - y) * plot_size + padding * 1.5)

    # knowing the image center,
    # compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y

def scale_image(image, sz):
    # print(type(image))
    # print(image.shape)
    # print(image)
    image = image.numpy()[0]
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(image, dsize=(sz, sz), interpolation=cv2.INTER_CUBIC)

def draw_rectangle_by_class(image, label):

    image[:, :,1] = image[:, :,0]
    image[:, :,2] = image[:, :,0]

    #Corner Coloring
    if label == 1:
        image[0:20, 0:20,0] = 255
        image[0:20, 0:20,1] = 0
        image[0:20, 0:20,2] = 0
    if label == 0:
        image[0:20, 0:20,0] = 0
        image[0:20, 0:20,1] = 0
        image[0:20, 0:20,2] = 255

    return image

#For each datapoint, draw on canvas
for i in range(len(tx)):
    im = images[i]
    label = dsy[i]
    x = tx[i]
    y = ty[i]

    im = scale_image(im, 200)

    im = draw_rectangle_by_class(im, label)
    tl_x, tl_y, br_x, br_y = compute_plot_coordinates(im, x, y, 40, 0)
    tsne_plot[tl_y:br_y, tl_x:br_x, :] = im



plt.imshow(tsne_plot)
plt.savefig('images/tsneDiagram.png', dpi=800)


df = pd.DataFrame()
df["y"] = dsy
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]

ax = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 2),
                data=df).set(title="Brain MRI T-SNE projection")

plt.savefig('images/tsneDotsDiagram.png', dpi=600)

