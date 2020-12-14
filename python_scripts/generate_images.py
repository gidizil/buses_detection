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
    def __init__(self, img_path, buses_imgs_path, buses_annots_df):
        """

        :param img_path: str. path to image to paste on
        :param buses_imgs_path: str. path to dir with buses images
        :param buses_pos_dir: str. path to dir with buses annotations
        :param buses_pos_file: str. name of file with buses annotations
        """
        self.img_path = img_path
        self.buses_path = buses_imgs_path
        self.buses_annots_df = buses_annots_df
        self.orig_scale = np.array([3648, 2736])

        # all of this will be generated later-on
        self.rescaled_img = None
        self.cropped_buses_arr = None

        # generate all shared objects
        self.scale_img()
        # self.gen_buses_df()
        self.get_cropped_buses()
        self.gen_rand_buses_locations()
        self.paste_rand_buses()

    def scale_img(self):
        """
        scales a new image to our image dimensions
        :return: PIL.Image.Image. The PIL image but scaled to original dimensions
        """
        img_path = self.img_path
        pil_img = Image.open(img_path)
        self.rescaled_img = pil_img.resize(self.orig_scale, resample=Image.BICUBIC)

        # return self.rescaled_img

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
        # TODO: Support scaling factor
        # TODO: support some labeling logic
        num_buses = self.buses_coords_logic()
        buses_imgs_name = self.buses_annots_df['img_name'].values
        rand_imgs = np.random.choice(buses_imgs_name, size=num_buses)
        cropped_buses_arr = []
        for img in rand_imgs:
            bboxs = self.buses_annots_df[self.buses_annots_df.img_name == img]['box_data'].values
            rand_bbox_loc = np.random.randint(len(bboxs[0]))  # select rand bus bbox
            rand_bbox = [bboxs[0][rand_bbox_loc]]
            rand_bbox = DataUtils.convert_boxes_coordinates_1(rand_bbox)[0]
            bus_img = Image.open(os.path.join(self.buses_path, img))
            cropped_bus = bus_img.crop(rand_bbox)
            cropped_buses_arr.append(cropped_bus)

        self.cropped_buses_arr = cropped_buses_arr

    def buses_coords_logic(self):
        """
        logic will include:
        A. 1-6 buses
        B. one from each color
        C. Scaling
        """
        num_buses = np.random.randint(1, 7)
        # TODO: add scaling factor
        # scaling_factors = [np.random]
        return num_buses
        pass

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
        self.rand_pos_arr = []
        for bus in self.cropped_buses_arr:
            bus_dims = bus.size
            pos_x = np.random.randint(0, bg_img_width - bus_dims[0])
            pos_y = np.random.randint(0, bg_img_height - bus_dims[1])
            # TODO: Make sure buses dont overlap
            self.rand_pos_arr.append((pos_x, pos_y))

    # TODO: This one
    def gen_labels(self):
        """
        gen labels format for the new img:
        img_name: [x1, y1, w1, h2], [x2, y2, w2, h2],  ..
        :return: dict. img_name (key), buses_coords value (as a string)
        """
        pass

    def paste_rand_buses(self):
        """
        Paste the randomly selected images on the new image
        :return: None. save a jpg image with buses pasted on it
        """
        new_img = self.rescaled_img.copy()
        for bus, pos in zip(self.cropped_buses_arr, self.rand_pos_arr):
            new_img.paste(bus, pos)


        new_img.save('new_test_img.jpg')


"""Test it"""
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

df_path = os.path.join(root, 'train/labels.txt')
buses_df = GenRandomImage.gen_buses_df(df_path)

rand_img_path = os.path.join(root, 'rand_images/rocket.jpg')
buses_path = os.path.join(root, 'train')
GenRandomImage(rand_img_path, buses_path, buses_df)
print('???????')








