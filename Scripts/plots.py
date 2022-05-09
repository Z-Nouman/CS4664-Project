import pickle as pickle
import matplotlib.pyplot as plt


with open('resnet18perf.pkl', 'rb') as f:
    resnet18performance = pickle.load(f)

with open('googlenetperf.pkl', 'rb') as f:
    googlenetperformance = pickle.load(f)

with open('vgg11perf.pkl', 'rb') as f:
    vgg11performance = pickle.load(f)

print(resnet18performance)
print(googlenetperformance)
print(vgg11performance)

plt.title("ResNet-18 Performance w.r.t Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.plot([x[0] for x in resnet18performance], [x[1] for x in resnet18performance])
plt.savefig('resnet18epochgraph.png')

plt.title("GoogleNet Performance w.r.t Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.plot([x[0] for x in googlenetperformance], [x[1] for x in googlenetperformance])
plt.savefig('googlenetepochgraph.png')

plt.title("VGG-11 Performance w.r.t Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.plot([x[0] for x in googlenetperformance], [x[1] for x in googlenetperformance])
plt.savefig('vggnetepochgraph.png')