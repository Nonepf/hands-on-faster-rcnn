from .faster_rcnn import FasterRCNN, RPN, FastRCNN
from torchvision.models import vgg16, VGG16_Weights

def faster_rcnn_vgg16(num_classes):
    """ 
    build a network using vgg16
    """
    backbone = vgg16(VGG16_Weights)
    rpn = RPN()
    fast_rcnn = FastRCNN(num_classes)

    model = FasterRCNN(backbone, rpn, fast_rcnn)
    return model
