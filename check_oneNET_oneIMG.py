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
num_class = 3
validname = 'resnext crossvalid1 three class'
gpu = 0


def get_imgdata(file_dir):
    this_folder = core_lzj.MyTestDataset(root=file_dir + '/', transform=test_transform)
    this_data = DataLoader(this_folder, batch_size=batch_size, shuffle=False)
    this_name = this_folder.mask_img
    return this_data, this_name


def check(net, check_data, file_name, class_num, row_num, col_num, cuda):
    if torch.cuda.is_available():
        net.to(cuda)
        print('check is starting')
    every_class_num = np.zeros(class_num)
    img_mat = np.zeros([row_num, col_num], dtype=int) - 1
    # evert_class_name = []
    # [evert_class_name.append([]) for _ in range(class_num)]
    fig_num = 0
    # label = 999
    for im, im_name in zip(check_data, file_name):
        net.eval()
        if torch.cuda.is_available():
            im = im.to(cuda)  # (bs, 3, h, w)
            # label = label.cuda()  # (bs, h, w)
        output = net(im)
        _, pred_label = output.max(1)
        every_class_num[pred_label.data[0]] += 1
        im_row_col = im_name.split('.')[0]
        im_row = int(im_row_col[-5:-3]) - 1
        im_col = int(im_row_col[-2:]) - 1
        img_mat[im_row, im_col] += pred_label.data[0] + 1
        # evert_class_name[pred_label.data[0]].append(file_name[fig_num])
        fig_num += 1
    # label_flag = label
    class_num_temp = every_class_num / fig_num
    return every_class_num, class_num_temp, img_mat


if __name__ == '__main__':
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    select = 0
    if select:
        check_dir = core_lzj.get_directory()
        net_path = core_lzj.get_file()
    else:
        check_dir = 'check/0/145x_12_12'
        net_path = 'resnext crossvalid1 three class/ResNetparams_Adam_bj0.4_r10_1999.pkl'
    cal_row_col = check_dir.split('_')
    row = int(cal_row_col[-2])
    col = int(cal_row_col[-1])
    my_model = models.resnext50_32x4d(num_classes=3)
    my_model.load_state_dict(torch.load(net_path))
    result = []
    device, init_flag = core_lzj.cuda_init(gpu)
    data, file = get_imgdata(file_dir=check_dir)
    every_num, every_per, img = check(net=my_model, check_data=data, file_name=file, class_num=num_class, row_num=row,
                                      col_num=col, cuda=device)
    print(check_dir)

    class_data = pd.DataFrame(data=[every_num, every_per], columns=list(range(num_class)))
    class_data.to_csv(check_dir + validname + '_' + 'oneNET' + core_lzj.get_time() + '_classnum_result.csv',
                      index=False)
    img_data = pd.DataFrame(data=img)
    img_data.to_csv(check_dir + validname + '_' + 'oneNET' + core_lzj.get_time() + '_imgmat_result.csv',
                    header=False, index=False)
    core_lzj.cuda_empty_cache(init_flag)





