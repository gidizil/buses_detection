from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import torch
import torch.nn as nn
from python_scripts.utils import MiscUtils, DataUtils
from torchvision.models.detection import FasterRCNN

class FasterRCNNMODEL:
    #TODO: Later on enable passing params params

    def __init__(self, model_params=None):
        self.params = model_params
        self.model = None
        self.optimizer = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    def set_backbone(self,backbone):
        """
        backbone is a string containing the backbone we want to use in the model. add more options
        """
        if 'vgg' in backbone.lower():
            "to somthing-check for options"
        elif 'mobilenet_v2' in backbone.lower():
            self.backbone = torchvision.models.mobilenet_v2(pretrained=True).features
            self.backbone.out_channels = 1280
        elif 'resnet50' in backbone.lower():
            self.backbone = torchvision.models.resnet50(pretrained=True).features
            self.backbone.out_channels = 256


    def set_model(self):
        """
        Set model and determine configuration
        :return: None, generate self.model to be used for training and testing
        """
        # Default values: box_score_thresh = 0.05, box_nms_thresh = 0.5
        kwargs = {'box_score_thresh': 0.3, 'box_nms_thresh': 0.3, 'box_detections_per_img': 6}
        # self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
        #                                                                   pretrained_backbone=True,
        #                                                                   **kwargs)
        self.model = FasterRCNN(self.backbone, num_classes=7, **kwargs)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        num_classes = 7
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Allow Multiple GPUs:
        # if torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(self.model)

        self.model = self.model.to(device)

        if self.params is None:
            params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            # TODO: Enable user defined model params
            pass

        self.optimizer = torch.optim.SGD(params, lr=0.01)

    def train_model(self, train_loader, num_epochs):
        """
        Train (only!) of the model
        :param train_loader: DataLoader object
        :param num_epochs: int. Number of epochs to train the model
        :return: None,
        """
        self.model.train()  # Set to training mode
        for epoch in range(num_epochs):
            for images, targets in train_loader:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Zero Gradients
                self.optimizer.zero_grad()

                # self.model = self.model.double()

                # Calculate Loss
                loss_dict = self.model(images, targets)  # what happens here?
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()

                # Update weights
                self.optimizer.step()

            print('Train Loss = {:.4f}'.format(losses.item()))

    def train_eval_model(self, train_loader, val_loader, num_epochs):
        """
        Train model and evaluate performance after each epoch
        :param train_loader: DataLoader object. Training images and targets
        :param val_loader: DataLoader object. validation images and targets
        :param num_epochs: int. Number of epochs for training and validation
        :return:
        """
        # For evaluation
        imgs_name_list = []
        bbox_list = []
        labels_list = []

        for epoch in range(num_epochs):
            train_loss = 0
            val_loss = 0
            self.model.train()  # Set to training mode
            with torch.set_grad_enabled(True):
                for images, targets in train_loader:
                    # Pass data to GPU
                    images = list(image.to(self.device) for image in images)
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                    # Zero Gradients
                    self.optimizer.zero_grad()

                    # self.model = self.model.double()

                    # Calculate Loss
                    loss_dict = self.model(images, targets)  # what happens here?
                    losses = sum(loss for loss in loss_dict.values())
                    train_loss += losses.item() * len(images)

                    # Backward Prop & Update weights
                    losses.backward()
                    self.optimizer.step()

                print('Train Loss = {:.4f}'.format(train_loss / len(train_loader.dataset)))

            # TODO: Calculate Dice and IoU loss for it

            with torch.no_grad():
                for idx, (imgs_name, images, targets) in enumerate(val_loader):
                    self.model.train()
                    images = list(image.to(self.device) for image in images)
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    val_loss += losses.item() * len(images)

                    if epoch == num_epochs - 1:
                        self.model.eval()  # Set model to evaluate performance
                        targets = self.model(images)

                        # Think of moving all this into gen_out_file - Looks nicer
                        imgs_name_list.extend(imgs_name)
                        bbox_list.extend([target['boxes'].int().cpu().tolist() for target in targets])
                        labels_list.extend([target['labels'].int().cpu().tolist() for target in targets])

                    """Optional - SEE the performance on the second last batch"""
                    if (epoch == num_epochs - 1) and idx == (len(val_loader) - 2):
                        self.model.eval()  # Set model to evaluate performance
                        targets = self.model(images)
                        MiscUtils.view(images, targets, k=len(images), model_type='faster_rcnn')

                DataUtils.gen_out_file('output_file.txt', imgs_name_list, bbox_list, labels_list)
                print('Validation Loss = {:.4f}'.format(val_loss / len(val_loader.dataset)))





