import zipfile
import os

import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set

#data_transforms = ResNet50_Weights.IMAGENET1K_V2.transforms

data_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
    transforms.Resize((232, 232)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
    transforms.randomHorizontalFlip(0.5),
])


