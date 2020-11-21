import torchvision
from torch.utils.data import DataLoader, Dataset
from python_scripts.utils import BusesDataset, DataUtils
from models.faster_rcnn import FasterRCNNMODEL
import os
"""
This is where we'll run our model.
once we are pleased with it we'll 
integrate it to the necessary files
"""

root = root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
buses_train_dataset = BusesDataset(root, 'train',
                                   f_name='labels.txt', faster_rcnn=True,
                                   transforms=torchvision.transforms.ToTensor())

train_loader = DataLoader(buses_train_dataset,
                          batch_size=4, shuffle=True,
                          collate_fn=lambda x: list(zip(*x)))  # Understand better collate_fn

""" 1. Try Faster RCNN """
faster_rcnn_model = FasterRCNNMODEL()
faster_rcnn_model.set_model()
faster_rcnn_model.train_model(train_loader, num_epochs=7)
# TODO: ADD predict_model
# TODO: Add tensorboard support
