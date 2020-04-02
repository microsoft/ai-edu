import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm as pbar
# from torch.utils.tensorboard import SummaryWriter
from resnet import *


def make_dataloaders(params):
    """
    Make a Pytorch dataloader object that can be used for traing and valiation
    Input:
        - params dict with key 'path' (string): path of the dataset folder
        - params dict with key 'batch_size' (int): mini-batch size
        - params dict with key 'num_workers' (int): number of workers for dataloader
    Output:
        - trainloader and testloader (pytorch dataloader object)
    """
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    transform_validation = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                    [0.2023, 0.1994, 0.2010])])

    transform_validation = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root=params['path'], train=True, transform=transform_train,download=True)
    testset = torchvision.datasets.CIFAR10(root=params['path'], train=False, transform=transform_validation,download=True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=params['batch_size'], shuffle=False, num_workers=4)
    return trainloader, testloader


def train_model(model, params):
    model = model.to(params['device'])
    optimizer = optim.Adam(model.parameters())
    total_updates = params['num_epochs'] * len(params['train_loader'])

    criterion = nn.CrossEntropyLoss()
    # best_accuracy = test_model(model, params)
    best_model = copy.deepcopy(model.state_dict())

    for epoch in pbar(range(params['num_epochs'])):
        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:

            # Loss accumulator for each epoch
            logs = {'Loss': 0.0,
                    'Accuracy': 0.0}

            # Set the model to the correct phase
            model.train() if phase == 'train' else model.eval()

            # Iterate over data
            for image, label in params[phase + '_loader']:
                image = image.to(params['device'])
                label = label.to(params['device'])

                # Zero gradient
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    # Forward pass
                    prediction = model(image)
                    loss = criterion(prediction, label)
                    accuracy = torch.sum(torch.max(prediction, 1)[1] == label.data).item()

                    # Update log
                    logs['Loss'] += image.shape[0] * loss.detach().item()
                    logs['Accuracy'] += accuracy

                    # Backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

            # Normalize and write the data to TensorBoard
            logs['Loss'] /= len(params[phase + '_loader'].dataset)
            logs['Accuracy'] /= len(params[phase + '_loader'].dataset)

if __name__ == '__main__':

    model = resnet18()

    # Train on cuda if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print("Using", device)

    data_params = {'path': './pytorch_dataset/cifar10', 'batch_size': 256}

    train_loader, validation_loader = make_dataloaders(data_params)

    train_params = {'description': 'Test',
                'num_epochs': 300,
                'check_point': 50, 'device': device,
                'train_loader': train_loader, 'validation_loader': validation_loader}

    train_model(model, train_params)

