import torch
import torchvision
import numpy as np
from PIL import Image
import os
import pandas as pd
import ast
from python_scripts.utils import DataUtils

class GenRandomImage:
    """Generate images with randomly added buses"""
    def __init__(self, bg_img_path, buses_imgs_path, buses_annots_df, rand_img_out_name, rescale_buses=False):
        """

        :param img_path: str. path to image to paste on
        :param buses_imgs_path: str. path to dir with buses images
        :param buses_pos_dir: str. path to dir with buses annotations
        :param buses_pos_file: str. name of file with buses annotations
        """
        self.img_path = bg_img_path
        self.buses_path = buses_imgs_path
        self.buses_annots_df = buses_annots_df
        self.rand_img_out_name = rand_img_out_name
        if not self.rand_img_out_path.endswith('.jpg'):
            self.rand_img_out_path += '.jpg'
        self.rescale_buses = rescale_buses

        self.orig_scale = np.array([3648, 2736])
        self.root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        # all of these will be generated later-on
        self.rescaled_img = None
        self.cropped_buses_arr = None
        self.imgs_and_labels_arr = None
        self.rand_annotations = None

        # generate all shared objects
        self.scale_img()
        # self.gen_buses_df()
        self.get_cropped_buses()
        self.gen_rand_buses_locations()
        self.gen_labels_file()
        self.paste_rand_buses()

    def scale_img(self):
        """
        scales a the random new image to a given dimensions
        :return: PIL.Image.Image. The PIL image but scaled to original dimensions
        """
        img_path = self.img_path
        pil_img = Image.open(img_path)
        self.rescaled_img = pil_img.resize(self.orig_scale, resample=Image.BICUBIC)

    @staticmethod
    def scale_cropped_bus(cropped_bus, scale=None):
        """
        scales a new image to a given dimensions
        :param cropped_bus: PIL Image. the cropped bus.the size (width, height) the img would be scaled to
        :param scale: tuple of ints. The size (width, height) the img would be scaled to
        :return: PIL.Image.Image. The PIL image but scaled to original dimensions
        """
        # img_path = self.img_path
        # pil_img = Image.open(img_path)
        return cropped_bus.resize(scale, resample=Image.BICUBIC)


    @staticmethod
    def save_img(pil_img, img_folder, img_name):
        """
        Gets a PIL image and save it as jpg
        :return: None. Stores the img as jpg
        """
        save_path = os.path.join(img_folder, img_name)
        pil_img.save(save_path)

    def get_cropped_buses(self):
        """
        given the path to the images and their locations, crop the buses
        store them as pil images.
        :return: list. pil images of cropped buses
        """
        # TODO: support some labeling logic
        num_buses, scaling_factors = self.buses_coords_logic()
        buses_imgs_name = self.buses_annots_df['img_name'].values
        rand_imgs = np.random.choice(buses_imgs_name, size=num_buses)
        cropped_buses_arr = []
        self.rand_labels_arr = []

        for idx, img_name in enumerate(rand_imgs):
            # retrieve bbox coords randomly
            bboxs = self.buses_annots_df[self.buses_annots_df.img_name == img_name]['box_data'].values
            rand_bbox_loc = np.random.randint(len(bboxs[0]))  # select rand bus bbox
            rand_bbox = [bboxs[0][rand_bbox_loc]]
            rand_bbox = DataUtils.convert_boxes_coordinates_1(rand_bbox)[0]

            # Storing the corresponding img name and label - for generating output file later on
            label = self.buses_annots_df[self.buses_annots_df.img_name == img_name]['labels'].values
            rand_label = label[0][rand_bbox_loc]
            self.rand_labels_arr.append(rand_label)

            # get bus image and optionally rescale it
            bus_img = Image.open(os.path.join(self.buses_path, img_name))
            cropped_bus = bus_img.crop(rand_bbox)

            if self.rescale_buses:
                scale_factor = np.sqrt(scaling_factors[idx])
                new_w = int(scale_factor * (rand_bbox[2] - rand_bbox[0]))
                new_y = int(scale_factor * (rand_bbox[3] - rand_bbox[1]))
                cropped_bus = GenRandomImage.scale_cropped_bus(cropped_bus, scale=(new_w, new_y))

            cropped_buses_arr.append(cropped_bus)
        self.cropped_buses_arr = cropped_buses_arr

    def buses_coords_logic(self, mean=1.0, std=0.3):
        """
        logic will include:
        A. 1-6 buses
        B. one from each color
        C. Scaling
        """
        num_buses = np.random.randint(1, 7)
        # Scaling factor is now constant
        scaling_factors = np.random.normal(loc=mean, scale=std, size=num_buses)
        return num_buses, scaling_factors

    @staticmethod
    def gen_buses_df(buses_annots_path, save_df=False):

        buses_df = pd.read_csv(buses_annots_path, sep=':')
        buses_df['annotations'] = buses_df['annotations'].apply(lambda x: ast.literal_eval('[' + x + ']'))
        buses_df['box_data'] = buses_df['annotations'].apply(lambda x: [y[0:4] for y in x])
        buses_df['labels'] = buses_df['annotations'].apply(lambda x: [y[4] for y in x])

        return buses_df

        if save_df:
            pd.read_csv(buses_df, 'buses_annots_df.csv')

    def gen_rand_buses_locations(self):
        """Generate random locations to paste the buses on"""
        bg_img = self.rescaled_img.copy()
        bg_img_width, bg_img_height = bg_img.size
        prev_bboxs_list = []
        check_bbox = False

        self.rand_pos_arr = []
        for bus in self.cropped_buses_arr:
            bus_dims = bus.size
            is_iou = True
            while is_iou:
                pos_x = np.random.randint(0, bg_img_width - bus_dims[0])
                pos_y = np.random.randint(0, bg_img_height - bus_dims[1])
                curr_bbox = [pos_x, pos_y, bus_dims[0], bus_dims[1]]

                if check_bbox:
                    is_iou = GenRandomImage.is_iou(prev_bboxs_list, curr_bbox)
                else:  # Happens in first bus - first while iteration ONLY!
                    check_bbox = True
                    is_iou = False

            prev_bboxs_list.append(curr_bbox)
            self.rand_pos_arr.append((pos_x, pos_y))

    @staticmethod
    def is_iou(prev_bboxs_list, curr_bbox):
        for prev_bbox in prev_bboxs_list:
            # Check for ovarlap
            x_axis_cond = (curr_bbox[0] <= prev_bbox[0] <= curr_bbox[0] + curr_bbox[2]) or \
                          (prev_bbox[0] <= curr_bbox[0] <= prev_bbox[0] + prev_bbox[2])

            y_axis_cond = (curr_bbox[1] <= prev_bbox[1] <= curr_bbox[1] + curr_bbox[3]) or \
                          (prev_bbox[1] <= curr_bbox[1] <= prev_bbox[1] + prev_bbox[3])

            if x_axis_cond and y_axis_cond:
                return True

        return False

    # TODO: This one
    def gen_labels_file(self):
        """
        gen labels format for the new img:
        img_name: [x1, y1, w1, h2], [x2, y2, w2, h2],  ..
        :return: dict. img_name (key), buses_coords value (as a string)
        """

        # 1. Create the xywh array
        pos_arr = [list(pos) for pos in self.rand_pos_arr]
        dims_arr = [list(cropped_bus.size) for cropped_bus in self.cropped_buses_arr]
        xywh_arr = [pos + dims for pos, dims in zip(pos_arr, dims_arr)]
        self.rand_annotations = [xywh + [c] for xywh, c in zip(xywh_arr, self.rand_labels_arr)]

        # 2. Add the label to it
        self.rand_annotations = [xywh + [c] for xywh, c in zip(xywh_arr, self.rand_labels_arr)]


        # 3. add img_name and save it as a tuple (rand, img_name, rand_annots).
        # this can be extracted later on
        self.rand_img_annots = (self.rand_img_out_name, self.rand_annotations)

    def paste_rand_buses(self):
        """
        Paste the randomly selected images on the new image
        :return: None. save a jpg image with buses pasted on it
        """
        new_img = self.rescaled_img.copy()
        for bus, pos in zip(self.cropped_buses_arr, self.rand_pos_arr):
            new_img.paste(bus, pos)

        # make sure it's jpg

        rand_img_out_path = os.path.join(self.root, 'generated_images', self.rand_img_out_name)
        new_img.save(rand_img_path)


"""Test it"""
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

df_path = os.path.join(root, 'train/labels.txt')
buses_df = GenRandomImage.gen_buses_df(df_path)

rand_img_path = os.path.join(root, 'rand_images/rocket.jpg')
buses_path = os.path.join(root, 'train')
GenRandomImage(rand_img_path, buses_path, buses_df, 'rand_img_1.jpg', rescale_buses=True)
print('???????')










