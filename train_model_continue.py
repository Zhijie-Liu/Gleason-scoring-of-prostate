# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 17:11:03 2018

@author: 1
"""

import os
import sys
import torch
import core_lzj
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models
import pretrainedmodels.models as mymodels
from torch import nn
from utils import train
from datetime import datetime
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder


# re_im_size = 448  # make all im to the same size
# crop_im_size = 196  # im_w=96 im_h=96
# num_class = 2
# gpu = 0
#
# transform = transforms.Compose([
#     transforms.Resize(re_im_size),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomRotation(10),
#     transforms.RandomCrop(crop_im_size, padding=0),
#     transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#     ])
#
# test_transform = transforms.Compose([
#     transforms.Resize(re_im_size),
#     transforms.RandomCrop(crop_im_size, padding=0),
#     transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#     ])

re_im_size = 448  # make all im to the same size
crop_im_size = 299  # im_w=96 im_h=96
num_class = 2
gpu = 3

transform = transforms.Compose([
    # transforms.Resize(re_im_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    # transforms.RandomCrop(crop_im_size, padding=0),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

test_transform = transforms.Compose([
    # transforms.Resize(re_im_size),
    # transforms.RandomCrop(crop_im_size, padding=0),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


if __name__ == "__main__":
    base_name = 'crossvalid two class'
    train_dir = base_name + ' train'
    test_dir = base_name + ' valid'
    train_data = ImageFolder(root=train_dir, transform=transform)
    test_data = ImageFolder(root=test_dir, transform=test_transform)

    print(train_data.class_to_idx)
    print(test_data.class_to_idx)

    print(train_data.classes, len(train_data))
    print(test_data.classes, len(test_data))
    print(train_dir)
    print(train_data.classes, len(train_data))
    print(test_dir)
    print(test_data.classes, len(test_data))
    im, train_label = train_data[1]
    # im = im.numpy()
    # im1 = im[0, :, :]
    # plt.imshow(im1, cmap='gray')
    print('label', train_label)
    im_test, test_label = test_data[0]
    print('test_label', test_label)
    print('train_img_size =', im.shape)
    print('test_img_size = ', im_test.shape)

    batch_size1 = 32
    batch_size2 = 32
    train_loader = DataLoader(train_data, batch_size=batch_size1, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size2, shuffle=False, num_workers=4)

    print("n_train_batch =", len(train_loader))
    print("n_test_batch =", len(test_loader))

    select = 0
    if select == 0:
        net_path = core_lzj.get_file()
    else:
        net_path = 'date20201009221241crossvalid SRCC/ResNetparams_Adamepochs1.pkl'
        if not os.path.exists(net_path):
            print('path is not exist')
            sys.exit(555)

    # my_model = models.resnet34(num_classes=num_class)
    my_model = mymodels.inceptionresnetv2(num_classes=num_class, pretrained=False)
    params = list(my_model.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])
    print('total nubmer of trainable parameters:', nparams)
    print('')
    opt = 'Adam'
    if opt == 'Adam':
        lr = 0.001  # lr =0.05 for optim.SGD
        optimizer = torch.optim.Adam(my_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    elif opt == 'Sgd':
        lr = 0.01
        optimizer = torch.optim.SGD(my_model.parameters(), lr=lr, weight_decay=1e-3)
    else:
        optimizer = None
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400, 500], gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    print('continue to train the model:')
    num_epochs = 600
    train(net=my_model, train_data=train_loader, valid_data=test_loader, num_epochs=num_epochs, optimizer=optimizer,
          criterion=criterion, opt=opt, cuda=gpu, scheduler=scheduler, num_class=num_class, temp_name=base_name,
          net_path=net_path)
