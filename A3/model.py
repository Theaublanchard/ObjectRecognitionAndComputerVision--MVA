import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, detection
from torchvision import transforms

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


def crop_from_bb(x,pred,tolerance,BIRD_LABEL=16):
    for boxe,label,score in zip(pred["boxes"],pred["labels"],pred["scores"]):
        boxe = torch.Tensor.int(boxe)
        if label == BIRD_LABEL and boxe.nelement() != 0 : #Found the bird
            # Extend the bounding box by a tolerance
            x1 = int( boxe[0]*(1-tolerance))
            y1 = int( boxe[1]*(1-tolerance))
            x2 = int( boxe[2]*(1+tolerance))
            y2 = int( boxe[3]*(1+tolerance))
            w,h = x2-x1, y2-y1
            # Crop the image
            if w > h:
                res = transforms.functional.crop(x, y1, x1, h, h)
            else:
                res = transforms.functional.crop(x, y1, x1, w, w)
            
            return res
    # If no bird found, return the original image
    return x


class Net(nn.Module):

    def __init__(self,tolerance=0.3):
        super(Net,self).__init__()

        #self.rcnn = detection.fasterrcnn_resnet50_fpn_v2(weights=detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,box_score_thresh=0.9,trainable_backbone_layers=0)
        self.tolerance = tolerance 

        self.resnet = torchvision.models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V2")
        
        self.in_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.in_ftrs, nclasses)  
        
        #self.rcnn.eval() 
        # Freeze all resnet layers except the last two layers
        for param in self.resnet.parameters():
            param.requires_grad = True

        # Unfreeze the last two layers
        for p in self.resnet.layer4[2].parameters():
            p.requires_grad = True
        for p in self.resnet.fc.parameters():
            p.requires_grad = True

        self.resnet_transforms = transforms.Compose([
                            transforms.Resize((232, 232)),
                            transforms.CenterCrop((224, 224)),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
])

    def forward(self,x):
        # with torch.no_grad():
        #     pred_bb = self.rcnn(x)
        # bc,c,h,w = x.shape
        # cropped_x = torch.zeros(bc,c,224,224).to(x.device)
        # for k,pred in enumerate(pred_bb):
        #     cropped_x_k = crop_from_bb(x[k],pred,self.tolerance)
        #     cropped_x[k] = self.resnet_transforms(cropped_x_k)
        # x = self.resnet(cropped_x)
        x = self.resnet(x)
        return x