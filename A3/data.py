import zipfile
import os

import torch
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights, detection
# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set

#data_transforms = ResNet50_Weights.IMAGENET1K_V2.transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CenterCropMainAxis:
    """Center crop the image on the smallest axis. Basicaly extract the center square."""
    def __init__(self) -> None:
        pass

    def __call__(self, image):
        if image.shape[-1] > image.shape[-2]:
            res = transforms.CenterCrop((image.shape[-2], image.shape[-2]))(image)
        else:
            res = transforms.CenterCrop((image.shape[-1], image.shape[-1]))(image)
        return res
        

class CropBird:
    """Detect the bird in a single image and crop it."""
    BIRD_LABEL = 16

    def __init__(self,device):
        self.rcnn = detection.fasterrcnn_resnet50_fpn_v2(weights=detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,box_score_thresh=0.8,trainable_backbone_layers=0)
        self.rcnn.eval() 
        self.device = device       

    def __call__(self, x):
        # bounding box detection
        with torch.no_grad():
            pred = self.rcnn.to(self.device)(x[None,:].to(self.device))[0]

        for boxe,label,_ in zip(pred["boxes"],pred["labels"],pred["scores"]):
            boxe = torch.Tensor.int(boxe)
            if label == self.BIRD_LABEL and boxe.nelement() != 0 : #Found the bird
                # I should probably extend the boxe to get more of the bird
                return transforms.functional.crop(x, boxe[1], boxe[0], boxe[3]-boxe[1], boxe[2]-boxe[0])
        return x


data_transforms = transforms.Compose([
    # data augmentation
    transforms.RandomRotation(30),
    transforms.GaussianBlur(3, sigma=(0.01, 2.0)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),

    # Detect the bird and crop it
    #CropBird(device),

    # properly crop the data to avoid stretching
    CenterCropMainAxis(),

    # Res net transforms
    transforms.Resize((232, 232)),
    transforms.CenterCrop((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
])




