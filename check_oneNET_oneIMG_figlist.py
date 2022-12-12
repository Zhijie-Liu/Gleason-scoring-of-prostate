# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 12:49:05 2018

@author: 1
"""
import os
import torch
import core_lzj
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import numpy as np
from datetime import datetime
import pandas as pd
import pretrainedmodels.models as newmodels

# re_im_size=448  #make all im to the same size
# crop_im_size=196  #im_w=96 im_h=96

test_transform = transforms.Compose([
    # transforms.Resize(re_im_size),
    # transforms.CenterCrop(crop_im_size),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# check_dir = './check/'
# check_dir = './three grade test/1/'
batch_size = 1
num_epochs = 1
num_class = 2
validname = 'inception_renset_v2 crossvalid1 two class'


def get_imgdata(file_dir):
    this_folder = core_lzj.MyDataset(root=file_dir + '/', transform=test_transform)
    this_data = DataLoader(this_folder, batch_size=batch_size, shuffle=False)
    this_name = this_folder.mask_img
    return this_data, this_name


def check(net, check_data, file_name, class_num):
    torch.cuda.set_device(2)
    if torch.cuda.is_available():
        net = net.cuda()
        print('check is starting')
    every_class_num = np.zeros(class_num)
    evert_class_name = []
    [evert_class_name.append([]) for _ in range(class_num)]
    fig_num = 0
    label = 999
    for im, label in check_data:
        net = net.eval()
        if torch.cuda.is_available():
            im = im.cuda()  # (bs, 3, h, w)
            label = label.cuda()  # (bs, h, w)
        output = net(im)
        _, pred_label = output.max(1)
        every_class_num[pred_label.data[0]] += 1
        evert_class_name[pred_label.data[0]].append(file_name[fig_num])
        fig_num += 1
    # label_flag = label
    class_judge_temp = every_class_num / fig_num
    return evert_class_name, class_judge_temp


def get_results(output, label):
    img_index, pred_label = output.max(1)
    # print('label ---> ',label.data[0])
    # print('pred_label = ',pred_label.data[0])
    if pred_label.data[0] == 1:
        pass
        # print ('tumor ---> ')
    else:
        pass
        # print ('normal ------>')


if __name__ == '__main__':
    check_dir = core_lzj.get_directory()
    path_dir = core_lzj.eachfile(filepath=check_dir)
    img_list, img_name = core_lzj.get_sub_directory(path_dir=path_dir)
    net_path = core_lzj.get_file()
    # my_model = models.resnext50_32x4d(num_classes=3)
    my_model = newmodels.inceptionresnetv2(num_classes=num_class, pretrained=False)
    my_model.load_state_dict(torch.load(net_path))
    oneNET_judge = []
    for img_dir in img_list:
        check_data, file_name = get_imgdata(file_dir=img_dir)
        judge_temp, acc = check(net=my_model, check_data=check_data, file_name=file_name, class_num=num_class)
        print(img_dir,acc)
        # oneNET_judge.append(judge_temp)

    everyNETdata_tumor = pd.DataFrame(data=judge_temp, index=list(range(num_class)))
    everyNETdata_tumor.to_csv(check_dir + validname + '_' + 'symbol' + core_lzj.get_time() + '_acc_result.csv')





