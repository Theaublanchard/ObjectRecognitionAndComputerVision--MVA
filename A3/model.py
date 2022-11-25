import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

nclasses = 20 

#class Net(nn.Module):
    # def __init__(self):
    #     super(Net, self).__init__()
    #     self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
    #     self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    #     self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
    #     self.fc1 = nn.Linear(320, 50)
    #     self.fc2 = nn.Linear(50, nclasses)

    # def forward(self, x):
    #     x = F.relu(F.max_pool2d(self.conv1(x), 2))
    #     x = F.relu(F.max_pool2d(self.conv2(x), 2))
    #     x = F.relu(F.max_pool2d(self.conv3(x), 2))
    #     x = x.view(-1, 320)
    #     x = F.relu(self.fc1(x))
    #     return self.fc2(x)


class EndModel(nn.Module):

    def __init__(self,in_features,out_features):
        super(EndModel,self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        #self.fc2 = nn.Linear(256, out_features)

        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        return x
        #return self.fc2(x)


# class Net(nn.Module):

#     def __init__(self,):
#         super(Net,self).__init__()

#         self.resnet = torchvision.models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V2")
#         self.num_ftrs = self.resnet.fc.out_features
#         self.endmodel = EndModel(self.num_ftrs,nclasses)

#         # freeze the resnet layers
#         #for param in self.resnet.parameters():
#         #    param.requires_grad = True
            

#     def forward(self,x):
#         x = self.resnet(x)
#         return self.endmodel(x)

class Net(nn.Module):

    def __init__(self,):
        super(Net,self).__init__()

        self.resnet = torchvision.models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V2")
        self.in_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.in_ftrs, nclasses)  

        # Freeze all resnet layers except the last two layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Unfreeze the last two layers
        for p in self.resnet.layer4[2].parameters():
            p.requires_grad = True
        for p in self.resnet.fc.parameters():
            p.requires_grad = True

    def forward(self,x):
        x = self.resnet(x)
        return x