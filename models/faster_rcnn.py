from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import torch
import torch.nn as nn

class FasterRCNNMODEL:
    #TODO: Later on enable passing params params

    def __init__(self, model_params=None):
        self.params = model_params
        self.model = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def set_model(self):
        """
        Set model and determine configuration
        :return: None, generate self.model to be used for training and testing
        """

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                                          pretrained_backbone=True)

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

        for epoch in range(num_epochs):
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

                    # Backward Prop & Update weights
                    losses.backward()
                    self.optimizer.step()

                print('Train Loss = {:.4f}'.format(losses.item()))

            # TODO: Calculate Dice and IoU loss for it
            # self.model.eval() # Set model to evaluate performance
            # with torch.no_grad():
            #     for images, targets in val_loader:
            #         images = list(image.to(self.device) for image in images)
            #         targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            #
            #         loss_dict = self.model(images, targets)
            #         # losses = sum(loss for loss in loss_dict.values())
            #
            #     print('Validation Loss = {:.4f}'.format(losses.item()))





