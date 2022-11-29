import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, detection
from torchvision import transforms

nclasses = 20 

class Net(nn.Module):

    def __init__(self,tolerance=0.3):
        super(Net,self).__init__()

        self.resnet = torchvision.models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V2")
        
        self.in_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.in_ftrs, nclasses)  
        
        for param in self.resnet.parameters():
            param.requires_grad = True

    def forward(self,x):
        x = self.resnet(x)
        return x