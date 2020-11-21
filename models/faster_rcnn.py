from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import torch


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
        self.model = self.model.to(device)

        if self.params is None:
            params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            # TODO: Enable user defined model params
            pass

        self.optimizer = torch.optim.SGD(params, lr=0.01)

    def train_model(self, train_dataloader, num_epochs):
        self.model.train()  # Set to training mode
        for epoch in range(num_epochs):
            for images, targets in train_dataloader:
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

            print('Loss = {:.4f}'.format(losses.item()))





