import torch
import torch.nn as nn
# from torchvision.models import vgg16, VGG16_Weights
from torchvision.ops import nms

class FasterRCNN(nn.Module):
    """
    A brief structure of Fast R-CNN
    """
    def __init__(self, backbone, rpn, fast_rcnn, img_size, k):
        super().__init__
        self.backbone = backbone
        self.rpn = rpn
        self.fast_rcnn = fast_rcnn

        self.k = k
        self.img_size = img_size

    def forward(self, input_image):
        feature_map = self.backbone(input_image)
        cls_score, bbox_pred = self.rpn(feature_map)
        proposals = calProposals(cls_score, bbox_pred, self.img_size, self.k)

        results = self.fast_rcnn(feature_map, proposals)
        return results

class RPN(nn.Module):
    """
    input: a feature map
    output: proposals(6k * N * M)
    """
    def __init__(self, in_channels=512, k=9):
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
    

def calProposals(cls_scores, bbox_preds, anchors, img_size, k):
    scores = cls_scores[:, k:, :, :]

    proposals = apply_delta_to_anchors(anchors, bbox_preds)
    proposals = clip_boxes(proposals, img_size)
    
    order = scores.argsort(descending=True)
    proposals = proposals[order]

    keep_idx = nms(proposals, scores, iou_threshold=0.7)
    proposals = proposals[keep_idx[:300]]
    return proposals

def apply_delta_to_anchors():
    pass

def clip_boxes(proposals, img_size):
    for proposal in proposals:
        # [x_min, y_min, x_max, y_max]
        if proposal[0] < 0:
            proposal[0] = 0
        if proposal[1] < 0:
            proposal[1] = 0
        if proposal[2] >= img_size[0]:
            proposal[2] = img_size[0] - 1
        if proposal[3] >= img_size[1]:
            proposal[3] = img_size[1] - 1