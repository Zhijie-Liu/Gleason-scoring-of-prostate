# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 12:09:30 2018

@author: 1
"""
import os
import random
import shutil
import numpy as np

path = "./"

normal_dir = './normal450/'
test_normal_path = "test/normal/"
train_normal_path = "train/normal/"

tumor_dir = './tumor450/'
test_tumor_path = "test/tumor/"
train_tumor_path = "train/tumor/"
trash_path = './trash/'

def eachFile(filepath):
    img_list = []
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        img_list.append(child)
    return img_list


class choose_img():
    def __init__(self, filepath=normal_dir, test_path=test_normal_path, \
                 dest_path=path):
        self.test_path = test_path
        print(test_path)
        self.dest_path = dest_path
        self.filepath = filepath

    """
    def __call__(self):
        return self.num_list()
    """

    def num_list(self, ratio):
        files = eachFile(self.filepath)
        # print(len(files))  #all files in tht filepath
        num_fig = []
        for file in files:
            if os.path.isdir(file):
                file_list = os.listdir(file)
                dirname, basename = os.path.splitdrive(file)
                print('num_fig of folder', basename, ' = ', len(file_list))
                num_fig.append(int(len(file_list) * ratio))
        print('fig_list =', num_fig)
        print('')
        return num_fig

    def random_select(self, num_list):
        files = eachFile(self.filepath)
        dest_path = self.dest_path + self.test_path

        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        print('num_list =', num_list)
        len_list = len(num_list)
        print('len_list =', len_list)
        print('dest_path =', dest_path)
        inum = 0
        for file in files:
            if os.path.isdir(file):
                # print(file)              
                dirname, basename = os.path.splitdrive(file)
                file_list = os.listdir(file)
                print('num_fig of folder', basename, ' = ', len(file_list))

                if inum < len_list:
                    slice_list = random.sample(file_list, num_list[inum])
                    inum += 1
                    # print(slice_list)
                    for sli in slice_list:
                        name = str(sli)
                        path = basename + '/' + name
                        # print('dest_path',dest_path)
                        # print('path',path)
                        shutil.move(path, dest_path)

        num_fig = os.listdir(dest_path)
        print('num_fig of floder ', self.test_path, ' = ', len(num_fig))

        return None


if __name__ == '__main__':
    """
    date preparing:  train/normal  test/normal
    """
    # file = "./train"
    # if os.path.exists(file):
    # shutil.rmtree(file)

    # file = "./test"
    # if os.path.exists(file):
    # shutil.rmtree(file)

    check_on = True  # True for first time
    # train/normal ... test_path=train_normal_path

    ch_img = choose_img(filepath=normal_dir, test_path=train_normal_path, \
                        dest_path=path)
    train_normal_list = ch_img.num_list(ratio=0.78)
    # num_list=[350,300,250,550,1050,150,450,250,200,250,350,800,300,\
    #           250,200,250,150,450,200,550,200]   #you own num_list

    if not check_on:
        ch_img.random_select(num_list=train_normal_list)

    # test/normal ... test_path=test_normal_path
    ch_img = choose_img(filepath=normal_dir, test_path=test_normal_path, \
                        dest_path=path)
    # num_list=ch_img.num_list(ratio=0.0)
    np_list = 0.25 * np.array(train_normal_list)  # you own num_list
    test_normal_list = np_list.astype(np.int32)
    if not check_on:
        ch_img.random_select(num_list=test_normal_list)

    """
    date preparing:   train/tumor   test/tumor
    """

    # train/tumor
    ch_img = choose_img(filepath=tumor_dir, test_path=train_tumor_path, \
                        dest_path=path)
    train_tumor_list = ch_img.num_list(ratio=0.78)
    # num_list=[300,300,300,500,300,300,300,300,400,100,400,300,\
    #           50,450,400,300,300,200,450,500,400,300,150,200]  #you own num_list
    if not check_on:
        ch_img.random_select(num_list=train_tumor_list)

    # test/tumor ...
    ch_img = choose_img(filepath=tumor_dir, test_path=test_tumor_path, \
                        dest_path=path)
    # num_list=ch_img.num_list(ratio=0.0)
    # num_list=[1,2,3]  #you own num_list
    np_list = 0.25 * np.array(train_tumor_list)
    test_tumor_list = np_list.astype(np.int32)
    if not check_on:
        ch_img.random_select(num_list=test_tumor_list)

    equal_on = False

    if equal_on:
        train_normal_file = eachFile(train_normal_path)
        test_normal_file = eachFile(test_normal_path)
        train_tumor_file = eachFile(train_tumor_path)
        test_tumor_file = eachFile(test_normal_path)
        if not os.path.exists(trash_path):
            os.mkdir(trash_path)
        else:
            shutil.rmtree(trash_path)
            os.mkdir(trash_path)
        if sum(train_normal_list) > sum(train_tumor_list):
            slice_train_list = random.sample( \
                train_normal_file, \
                sum(train_normal_list) - sum(train_tumor_list))
            for sli in slice_train_list:
                shutil.move(str(sli), trash_path)
        elif sum(train_normal_list) < sum(train_tumor_list):
            slice_train_list = random.sample( \
                train_tumor_file, \
                sum(train_tumor_list) - sum(train_normal_list))
            for sli in slice_train_list:
                shutil.move(str(sli), trash_path)
        if sum(test_normal_list) > sum(test_tumor_list):
            slice_test_list = random.sample( \
                test_normal_file, \
                sum(test_normal_list) - sum(test_tumor_list))
            for sli in slice_test_list:
                shutil.move(str(sli), trash_path)
        elif sum(test_normal_list) < sum(test_tumor_list):
            slice_test_list = random.sample( \
                test_tumor_file, \
                sum(test_tumor_list) - sum(test_normal_list))
            for sli in slice_test_list:
                shutil.move(str(sli), trash_path)