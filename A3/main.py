import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
import pandas as pd

# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
parser.add_argument('--save-interval', type=int, default=10, metavar='SI',
                    help='how many epoch to wait before saving the model.')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

# Data initialization and loading
from data import data_transforms

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=0)

# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import Net
model = Net()

# freeze the resnet layers
for param in model.resnet.parameters():
            param.requires_grad = False

if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), lr=args.lr,)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True)


def train(epoch):
    model.train()
    total_loss_epoch = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss_epoch += loss.data.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
    print('Training set:   Average loss: {:.4f}'.format(
        total_loss_epoch/len(train_loader.dataset)))
    return total_loss_epoch/len(train_loader.dataset)

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return validation_loss, (100. * correct / len(val_loader.dataset)).data.item()


train_loss_tab = pd.DataFrame(columns=['epoch', 'train_loss','val_loss','val_acc'])

for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    val_loss, val_acc = validation()
    scheduler.step(val_loss)
    train_loss_tab = pd.concat([train_loss_tab,pd.DataFrame.from_dict({'epoch': [epoch], 'train_loss': [train_loss], 'val_loss': [val_loss],'val_acc':[val_acc]})], ignore_index=True)
    if epoch % args.save_interval == 0 or epoch == args.epochs:
        model_file = args.experiment + '/model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_file)
        print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')
        train_loss_tab.to_csv(args.experiment + '/train_loss_tab.csv', index=False)
train_loss_tab.to_csv(args.experiment + '/train_loss_tab.csv', index=False)