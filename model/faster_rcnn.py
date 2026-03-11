import torch
import torch.nn as nn
# from torchvision.models import vgg16, VGG16_Weights

class FasterRCNN(nn.Module):
    """
    A brief structure of Fast R-CNN
    """
    def __init__(self, backbone, rpn, fast_rcnn):
        super().__init__
        self.backbone = backbone
        self.rpn = rpn
        self.fast_rcnn = fast_rcnn

    def forward(self, input_image):
        feature_map = self.backbone(input_image)
        cls_score, bbox_pred = self.rpn(feature_map)
        proposals = self.rpn.calProposals(cls_score, bbox_pred)

        results = self.fast_rcnn(feature_map, proposals)
        return results

class RPN(nn.Module):
    """
    input: a feature map
    output: proposals(6k * N * M)
    """
    def __init__(self, in_channels, k=9):
        """
        - kernel size = 3, padding = 1
        - Assuming the feature map has the size of N, 
        we have (N + 2 - 3) + 1 = N.  
        """
        super().__init__()
        self.midLayer = nn.Conv2d(in_channels, 256, 3, 1, 1)
        self.clsLayer = nn.Conv2d(256, 2*k, 1, 1, 0)
        self.regLayer = nn.Conv2d(256, 4*k, 1, 1, 0)

    def forward(self, input_image):
        x = torch.relu(self.midLayer(input_image))
        return self.clsLayer(x), self.regLayer(x)

    def calProposals(self, cls_score, bbox_pred):
        pass

class FastRCNN(nn.Module):
    """
    the last layer
    """
    def __init__(self, num_classes):
        super().__init__()
        # self.RoIPoolingLayer = 
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), # assume feature_map has the depth of 512
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.cls_head = nn.Linear(4096, num_classes)
        self.reg_head = nn.Linear(4096, num_classes * 4)

    def forward(self, feature_map, proposals):
        x = feature_map # x = self.RoIPoolingLayer
        x_flatten = x.view(x.size(0), -1)
        feat = self.classifier(x_flatten)
        return self.cls_head(feat), self.reg_head(feat)