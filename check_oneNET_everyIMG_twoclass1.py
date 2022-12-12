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
import pretrainedmodels.models as mymodels
import numpy as np
from datetime import datetime
import pandas as pd
import re
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
validname = 'without collagen crossvalid 20210906'
gpu = 3


def get_imgdata(file_dir):
    this_folder = core_lzj.MyDataset(root=file_dir + '/', transform=test_transform)
    this_data = DataLoader(this_folder, batch_size=batch_size, shuffle=False)
    this_name = this_folder.mask_img
    return this_data, this_name


def check(net, check_data, file_name, class_num, cuda):
    if torch.cuda.is_available():
        net.to(cuda)
        print('check is starting')
    every_class_num = np.zeros(class_num)
    evert_class_name = []
    [evert_class_name.append([]) for _ in range(class_num)]
    fig_num = 0
    label = 999
    for im, label in check_data:
        net.eval()
        if torch.cuda.is_available():
            im, label = im.to(cuda), label.to(cuda)  # (bs, 3, h, w)
        output = net(im)
        _, pred_label = output.max(1)
        every_class_num[pred_label.data[0]] += 1
        evert_class_name[pred_label.data[0]].append(file_name[fig_num])
        fig_num += 1
    # label_flag = label
    class_judge_temp = every_class_num / fig_num
    return class_judge_temp


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
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    select = 1
    if select == 0:
        check_dir = core_lzj.get_directory()
        net_path = core_lzj.get_file()
    else:
        check_dir = '20211123twoclass test/1'
        net_path = 'date20211123000413crossvalid 20211122twoclass/InceptionResNetV2params_Adamepochs160.pkl'
    path_dir = core_lzj.each_dir(filepath=check_dir)
    img_list, img_name = core_lzj.get_sub_directory(path_dir=path_dir)
    my_model = mymodels.inceptionresnetv2(num_classes=num_class, pretrained=False)
    my_model.load_state_dict(torch.load(net_path, map_location='cuda:' + gpu.__str__())['model'])
    oneNET_judge = []
    device, init_flag = core_lzj.cuda_init(gpu)
    for img_dir in img_list:
        data, file = get_imgdata(file_dir=img_dir)
        print(img_dir)
        judge_temp = check(net=my_model, check_data=data, file_name=file, class_num=num_class, cuda=device)

        oneNET_judge.append(judge_temp)

    everyNETdata_tumor = pd.DataFrame(data=oneNET_judge, index=img_name, columns=list(range(num_class)))
    everyNETdata_tumor.to_csv(check_dir + validname + '_' + 'everyNET' + core_lzj.get_time() + '_acc_result.csv')
    core_lzj.cuda_empty_cache(init_flag)




