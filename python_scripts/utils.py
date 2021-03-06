"""several classes for useful methods for data manipulation, plotting etc"""
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


class DataUtils:

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

    @staticmethod
    def convert_boxes_coordinates_1(bbox_coords_list):
        """
        Converts the presentation of the bounding box
        from [x1, y1, w, h] - where (x1, y1) is upper left
        to [x1, y1, x2, y2] where (x1,y1) upper-left, (x2,y2) - lower-right
        :param bbox_coords_list: list or np. array. In the form of [x1, y1, w, h]
        :return: np.array. in the form of [x1,y1, x2, y2]. (x1,y1) upper-left, (x2,y2) - lower-right
        """

        new_bbox_coords_list = []
        for bbox_coords in bbox_coords_list:
            x1, y1 = bbox_coords[0:2]
            x2 = x1 + bbox_coords[2]
            y2 = y1 + bbox_coords[3]

            new_bbox_coords_list.append([x1, y1, x2, y2])

        return np.array(new_bbox_coords_list)

    @staticmethod
    def convert_boxes_coordinates_1_inv(bbox_coords_list):
        """
        Converts the presentation of the bounding box
        from [x1, y1, x2, y2] - where (x1, y1) is upper left
        to [x1, y1, w, h] where (x1,y1) upper-left, (x2,y2) - lower-right
        :param bbox_coords_list: list or np. array. In the form of [x1, y1, w, h]
        :return: np.array. in the form of [x1,y1, x2, y2]. (x1,y1) upper-left, (x2,y2) - lower-right
        """
        new_bbox_list = []
        for bbox_coords in bbox_coords_list:
            x1, y1, x2, y2 = bbox_coords[0:4]
            w = x2 - x1
            h = y2 - y1
            new_bbox_list.append([x1, y1, w, h])

        return new_bbox_list

    @staticmethod
    def batched_nms(targets):
        # boxes is a [batch_size, N, 4] tensor, and scores a
        # [batch_size, N] tensor.
        boxes, scores, iou_threshold = None, None, None
        batch_size, N, _ = boxes.shape
        indices = torch.arange(batch_size, device=boxes.device)
        indices = indices[:, None].expand(batch_size, N).flatten()
        boxes_flat = boxes.flatten(0, 1)
        scores_flat = scores.flatten()
        indices_flat = torchvision.ops.boxes.batched_nms(
            boxes_flat, scores_flat, indices, iou_threshold)
        # now reshape the indices as you want, maybe
        # projecting back to the [batch_size, N] space
        # I'm omitting this here
        indices = indices_flat
        return indices

    @staticmethod
    def gen_out_file(out_f_path, images_name_list, bbox_coords_list, labels_list):
        """
        Generate output file to compare to results
        :param out_f_path: str. path to save the output file
        :param images_name_list: list. list of all the images in loader name.
        :param bbox_coords_list: list of lists. list of all the bbox_coords_list for each image
        :param labels_list: list of lists. list of all the labels  for each image
        :return: None. Generates a file of the results in the format if annotationsTrain.txt
        """

        # bind bbox_coords_list and labels
        annotations = []
        for img_idx in range(len(images_name_list)):  # for each image
            img_annots = []
            labels_detects = labels_list[img_idx]
            bbox_detects = bbox_coords_list[img_idx]
            bbox_detects = DataUtils.convert_boxes_coordinates_1_inv(bbox_detects)
            for idx, bbox in enumerate(bbox_detects):  # for each detection
                bbox_cp = bbox.copy()
                bbox_cp.append(labels_detects[idx])
                img_annots.append(bbox_cp)

            annotations.append(img_annots)

        # TODO: Maybe pandas is not the fastest way of doing things
        results_df = pd.DataFrame({'img_name': images_name_list, 'annots': annotations})

        results_df.to_csv(out_f_path, index=False, sep=':', header=False)


class BusesDataset(Dataset, DataUtils):
    """Dataset to load images to the model"""
    def __init__(self, root, folder='train',
                 transforms=None, f_name='labels.txt',
                 model_type='f_rcnn', mode='train'):
        self.transforms = []
        if transforms is not None:
            self.transforms.append(transforms)
        self.model_type = model_type
        self.mode = mode

        self.root = root
        self.folder = folder
        # TODO: Make it better later - No pandas
        xy_df = pd.read_csv(os.path.join(root, folder, f_name), sep=':')
        xy_df['annotations'] = xy_df['annotations'].apply(lambda x: ast.literal_eval('[' + x + ']'))
        xy_df['box_data'] = xy_df['annotations'].apply(lambda x: [y[0:4] for y in x])
        if self.model_type == 'f_rcnn':
            xy_df['box_data'] = xy_df['box_data'].apply(lambda x:
                                                        self.convert_boxes_coordinates_1(x))
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
        img_name = self.imgs[idx]  # passing this to gen output file

        for transform in self.transforms:
            img = transform(img)

        targets = {}
        targets['boxes'] = torch.tensor(bbox).float()
        targets['labels'] = torch.tensor(label).type(torch.int64)

        if self.mode == 'train':
            return img.float(), targets
        elif self.mode == 'eval':
            return img_name, img.float(), targets
        else:
            raise ValueError("mode must be 'train' or 'eval'")




# u = DataUtils()
# u.split_imgs_train_val('busesTrain')

# TODO: Add a class of plotting utils
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dataset = BusesDataset(root, 'train', transforms=torchvision.transforms.ToTensor())

# print(dataset[0])

"""Test it's working"""
buses_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: list(zip(*x)))
images, targets = next(iter(buses_loader))


def view(images, targets, k, std=1, mean=0, model=None):
    figure = plt.figure(figsize=(30, 30))
    images=list(images)
    targets=list(targets)
    labels_dict = {1: 'green', 2: 'yellow', 3: 'white',
                   4: 'gray', 5: 'blue', 6: 'red'}
    for i in range(k):
        # out = torchvision.utils.make_grid(images[i])
        # inp = out.cpu().numpy().transpose((1, 2, 0))
        # inp = np.array(std)*inp+np.array(mean)
        # inp = np.clip(inp,0,1)
        ax = figure.add_subplot(2, 2, i + 1)
        ax.imshow(images[i].cpu().numpy().transpose((1, 2, 0)))
        bbox = targets[i]['boxes'].cpu().numpy()
        labels = targets[i]['labels'].cpu().numpy()
        for j in range(len(labels)):
            ax.add_patch(patches.Rectangle((bbox[j][0], bbox[j][1]), bbox[j][2], bbox[j][3],
                                           linewidth=5, edgecolor=labels_dict[labels[j]], facecolor='none'))

    plt.show()

#view(images, targets,4)


class MiscUtils:

    def __init__(self):
        pass

    def prepare_model_to_eval(self):
        """temporary name - all the necessary stuff to integrate with testing code"""
        pass

    @staticmethod
    def view(images, targets, k, std=1, mean=0, model_type = None):
        figure = plt.figure(figsize=(30, 30))
        images = list(images)
        targets = list(targets)
        labels_dict = {1: 'green', 2: 'yellow', 3: 'white',
                       4: 'gray', 5: 'blue', 6: 'red'}
        for i in range(k):
            out = torchvision.utils.make_grid(images[i])
            inp = out.cpu().numpy().transpose((1, 2, 0))
            inp = np.array(std) * inp + np.array(mean)
            inp = np.clip(inp, 0, 1)
            ax = figure.add_subplot(2, 2, i + 1)
            ax.imshow(images[i].cpu().numpy().transpose((1, 2, 0)))
            bbox = targets[i]['boxes'].cpu().numpy()
            if model_type == 'faster_rcnn':
                bbox = bbox.astype(int)
                bbox = DataUtils.convert_boxes_coordinates_1_inv(bbox)

            labels = targets[i]['labels'].cpu().numpy()
            for j in range(len(labels)):
                ax.add_patch(patches.Rectangle((bbox[j][0], bbox[j][1]), bbox[j][2], bbox[j][3],
                                               linewidth=5, edgecolor=labels_dict[labels[j]], facecolor='none'))
        plt.savefig('view_sample_results.jpg')
        plt.close()
    @staticmethod
    def view_from_loader(data_loader, k, std=1, mean=0):
        figure = plt.figure(figsize=(30, 30))

        images, targets = next(iter(data_loader))
        images = list(images)
        targets = list(targets)
        labels_dict = {1: 'green', 2: 'yellow', 3: 'white',
                       4: 'gray', 5: 'blue', 6: 'red'}
        for i in range(k):
            out = torchvision.utils.make_grid(images[i])
            inp = out.cpu().numpy().transpose((1, 2, 0))
            inp = np.array(std) * inp + np.array(mean)
            inp = np.clip(inp, 0, 1)
            ax = figure.add_subplot(2, 2, i + 1)
            ax.imshow(images[i].cpu().numpy().transpose((1, 2, 0)))
            bbox = targets[i]['boxes'].cpu().numpy()
            labels = targets[i]['labels'].cpu().numpy()
            for j in range(len(labels)):
                ax.add_patch(patches.Rectangle((bbox[j][0], bbox[j][1]), bbox[j][2], bbox[j][3],
                                               linewidth=5, edgecolor=labels_dict[labels[j]], facecolor='none'))
        plt.show()



