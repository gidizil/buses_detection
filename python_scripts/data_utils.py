"""Misc methods that can be one timers or just handful"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from torch.utils.data import Dataset, DataLoader
import ast
from PIL import Image
import torch
import torchvision
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np


class Utils:

    def __init__(self):

        self.project_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..'))

    def split_imgs_train_val(self, full_train_dir, test_rate=0.25):
        """
        Split images to train and val folders.
        add to folder a text file with corresponding
        file names in the same format as
        anotationsTrain.txt
        :param full_train_dir: str. path to full training directory
        :param test_rate: float. between 0-1.0 - split rate - 0.25 means 25% test
        :return: None. Generates train, val directories
        """
        # set path
        train_path = os.path.join(self.project_path, 'train')
        validation_path = os.path.join(self.project_path, 'validation')
        # create directories to train and val
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.exists(validation_path):
            os.makedirs(validation_path)

        # split to train and val
        # generate annotations file for each
        annots_path = os.path.join(self.project_path, 'annotationsTrain.txt')
        annots_df = pd.read_csv(annots_path,
                                sep=':', names=['img_name', 'annotations'])
        train_df, val_df = train_test_split(annots_df, test_size=test_rate)
        train_df.to_csv(os.path.join(train_path, 'labels.txt'),
                        sep=':', index=False, header=True)
        val_df.to_csv(os.path.join(validation_path, 'labels.txt'),
                      sep=':', index=False, header=True)

        # move images to train and val dirs
        train_imgs_list = train_df.img_name.tolist()
        val_imgs_list = val_df.img_name.tolist()
        full_train_path = os.path.join(self.project_path, full_train_dir)

        for img_name in train_imgs_list:
            curr_img_path = os.path.join(full_train_path, img_name)
            new_img_path = os.path.join(train_path, img_name)
            shutil.copyfile(curr_img_path, new_img_path)

        for img_name in val_imgs_list:
            curr_img_path = os.path.join(full_train_path, img_name)
            new_img_path = os.path.join(validation_path, img_name)
            shutil.copyfile(curr_img_path, new_img_path)


class BusesDataset(Dataset):
    """Dataset to load images to the model"""
    def __init__(self, root, folder='train', transforms=None, f_name='labels.txt'):
        self.transforms = []
        if transforms is not None:
            self.transforms.append(transforms)

        self.root = root
        self.folder = folder
        # TODO: Make it better later - No pandas
        xy_df = pd.read_csv(os.path.join(root, folder, f_name), sep=':')
        xy_df['annotations'] = xy_df['annotations'].apply(lambda x: ast.literal_eval('[' + x + ']'))
        xy_df['box_data'] = xy_df['annotations'].apply(lambda x: [y[0:4] for y in x])
        xy_df['labels'] = xy_df['annotations'].apply(lambda x: [y[4] for y in x])
        self.box_data = xy_df['box_data'].values
        self.labels = xy_df['labels'].values
        self.imgs = xy_df['img_name'].values

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.folder, self.imgs[idx])
        img = Image.open(img_path).convert('RGB')
        bbox = self.box_data[idx]
        label = self.labels[idx]

        for transform in self.transforms:
            img = transform(img)


        targets = {}
        targets['bbox'] = torch.tensor(bbox).double()
        targets['label'] = torch.tensor(label).type(torch.int64)

        return img.double(), targets

# u = Utils()
# u.split_imgs_train_val('busesTrain')


root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dataset = BusesDataset(root, 'train', transforms=torchvision.transforms.ToTensor())

# print(dataset[0])

"""Test it's working"""
buses_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: list(zip(*x)))
images, targets = next(iter(buses_loader))


def view(images, targets, k, std=1, mean=0):
    figure = plt.figure(figsize=(30,30))
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
        for j in range(len(labels)):
            ax.add_patch(patches.Rectangle((bbox[j][0], bbox[j][1]), bbox[j][2], bbox[j][3],
                                           linewidth=5, edgecolor=labels_dict[labels[j]], facecolor='none'))

    plt.show()


#view(images, targets,4)





