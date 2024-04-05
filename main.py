from __future__ import print_function
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
!pip install torch-lr-finder

from albumentations.pytorch import ToTensorV2
import random


from utility import train, test, train_loader, test_loader
from policy import OneCycleLR

train = CIFAR10Dataset('./data', transform=train_transforms)
test = CIFAR10Dataset('./data', transform=test_transforms)

max_lr = 0.1
min_lr = 0.001

EPOCHS = 24
# Initialize the OneCycleLR scheduler
optimizer = optim.Adam(model.parameters(), lr=0.02)
scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=len(train_loader) * EPOCHS,
                       epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.3)

# Training loop
for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    print("Learning Rate:", scheduler.get_lr()[0])

    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()