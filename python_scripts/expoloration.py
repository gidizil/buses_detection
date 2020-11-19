import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
from python_scripts.utils import BusesDataset
import os
"""Check 1 - See you can show image with anotations"""

img = np.array(Image.open('busesTrain/DSCF1013.JPG'), dtype=np.uint8)

# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image (draw the image on the ax
ax.imshow(img)

# Create a Rectangle patch
rect = patches.Rectangle((1217, 1690), 489, 201, linewidth=1.5,
                         edgecolor='chartreuse', facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)

plt.show()

"""Check 2 - See that Dataset and DataLoader works as expected"""
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dataset = BusesDataset(root, 'train', transforms=torchvision.transforms.ToTensor())

# print(dataset[0])

"""Test it's working"""
buses_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: list(zip(*x)))
images, targets = next(iter(buses_loader))


def view(images, targets, k, std=1, mean=0):
    figure = plt.figure(figsize=(30, 30))
    images=list(images)
    targets=list(targets)
    labels_dict = {1: 'green', 2: 'yellow', 3: 'white',
                   4: 'gray', 5: 'blue', 6: 'red'}
    for i in range(k):
        out = torchvision.utils.make_grid(images[i])
        inp = out.cpu().numpy().transpose((1, 2, 0))
        inp = np.array(std)*inp+np.array(mean)
        inp = np.clip(inp,0,1)
        ax = figure.add_subplot(2, 2, i + 1)
        ax.imshow(images[i].cpu().numpy().transpose((1, 2, 0)))
        bbox = targets[i]['bbox'].cpu().numpy()
        labels = targets[i]['label'].cpu().numpy()
        # l[:, 2] = l[:, 2]-l[:, 0]
        # l[:, 3]=l[:, 3]-l[:, 1]
        for j in range(len(labels)):
            ax.add_patch(patches.Rectangle((bbox[j][0], bbox[j][1]), bbox[j][2], bbox[j][3],
                                           linewidth=5, edgecolor=labels_dict[labels[j]], facecolor='none'))

    plt.show()


view(images, targets, 4)







