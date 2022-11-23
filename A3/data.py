import zipfile
import os

import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set

#data_transforms = ResNet50_Weights.IMAGENET1K_V2.transforms

def center_crop_main_axis(image):
    """Center crop the image on the smallest axis. Basicaly extract the center square."""
    if image.shape[-1] > image.shape[-2]:
        res = transforms.CenterCrop((image.shape[-2], image.shape[-2]))(image)
    else:
        res = transforms.CenterCrop((image.shape[-1], image.shape[-1]))(image)
    return res

class CenterCropMainAxis:

    def __init__(self) -> None:
        pass

    def __call__(self, image):
        return center_crop_main_axis(image)
        

data_transforms = transforms.Compose([
    # data augmentation
    transforms.RandomRotation(30),
    transforms.GaussianBlur(3, sigma=(0.01, 2.0)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    # properly crop the data to avoid stretching
    CenterCropMainAxis(),
    # Res net transforms
    transforms.Resize((232, 232)),
    transforms.CenterCrop((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
])




