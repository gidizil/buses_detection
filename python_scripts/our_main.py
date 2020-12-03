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
# Set DataLoaders for train and validation
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
buses_train_dataset = BusesDataset(root, 'train',
                                   f_name='labels.txt', model_type='f_rcnn',
                                   transforms=torchvision.transforms.ToTensor(), mode='train')
buses_val_dataset = BusesDataset(root, 'validation',
                                 f_name='labels.txt', model_type='f_rcnn',
                                 transforms=torchvision.transforms.ToTensor(), mode='eval')

train_loader = DataLoader(buses_train_dataset,
                          batch_size=4, shuffle=True,
                          collate_fn=lambda x: list(zip(*x)),
                          num_workers=16)  # Understand better collate_fn

val_loader = DataLoader(buses_val_dataset,
                        batch_size=4, shuffle=True,
                        collate_fn=lambda x: list(zip(*x)),
                        num_workers=16)

""" 1. Try Faster RCNN """
faster_rcnn_model = FasterRCNNMODEL()
faster_rcnn_model.set_model()
# faster_rcnn_model.train_model(train_loader, num_epochs=7)
# TODO: ADD eval_model only

faster_rcnn_model.train_eval_model(train_loader, val_loader, num_epochs=50)
# TODO: Add tensorboard support
