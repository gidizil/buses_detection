"""Misc methods that can be one timers or just handful"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from torch.utils import data


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
                        sep=':', index=False, header=False)
        val_df.to_csv(os.path.join(validation_path, 'labels.txt'),
                      sep=':', index=False, header=False)

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



class BusesDataset(data.dataset):
    """Dataset to load images to the model"""
    pass


# TODO: add stuff for config - perhaps another class
u = Utils()
u.split_imgs_train_val('busesTrain')



