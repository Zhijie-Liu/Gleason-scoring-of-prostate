import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

img_size = 50
net_img_size = 300
step = int(net_img_size/img_size)

def get_probability_matrix(nrow, ncol, blue_channel, yellow_channel, orange_channel, red_channel, gray_channel):
    matrix_0, matrix_1, matrix_2, matrix_3, matrix_no = np.zeros((nrow, ncol)), np.zeros((nrow, ncol)), np.zeros((nrow, ncol)), np.zeros((nrow, ncol)), np.zeros((nrow, ncol))
    for i in range(0, nrow - step + 1):
        for j in range(0, ncol - step + 1):
            blue_sub = blue_channel[i * img_size: i * img_size + net_img_size, j * img_size: j * img_size + net_img_size]
            yellow_sub = yellow_channel[i * img_size: i * img_size + net_img_size, j * img_size: j * img_size + net_img_size]
            orange_sub = orange_channel[i * img_size: i * img_size + net_img_size, j * img_size: j * img_size + net_img_size]
            red_sub = red_channel[i * img_size: i * img_size + net_img_size, j * img_size: j * img_size + net_img_size]
            gray_sub = gray_channel[i * img_size: i * img_size + net_img_size, j * img_size: j * img_size + net_img_size]
            sub_sum = [blue_sub.sum(), yellow_sub.sum(), orange_sub.sum(), red_sub.sum(), gray_sub.sum()]
            output_max = np.argmax(sub_sum)
            if output_max == 0:
                matrix_0[i:i + step, j:j + step] += 1
            elif output_max == 1:
                matrix_1[i:i + step, j:j + step] += 1
            elif output_max == 2:
                matrix_2[i:i + step, j:j + step] += 1
            elif output_max == 3:
                matrix_3[i:i + step, j:j + step] += 1
            else:
                matrix_no[i:i + step, j:j + step] += 1

            print(i, j)

    return matrix_0, matrix_1, matrix_2, matrix_3, matrix_no


if __name__ == '__main__':
    gray = cv2.imread('musk/gray.tif',cv2.IMREAD_GRAYSCALE)
    ret1, gray_bin = cv2.threshold(gray,128,1,cv2.THRESH_BINARY)
    blue = cv2.imread('musk/blue.tif', cv2.IMREAD_GRAYSCALE)
    ret2, blue_bin = cv2.threshold(blue, 128, 1, cv2.THRESH_BINARY)
    yellow = cv2.imread('musk/yellow.tif', cv2.IMREAD_GRAYSCALE)
    ret3, yellow_bin = cv2.threshold(yellow, 128, 1, cv2.THRESH_BINARY)
    orange = cv2.imread('musk/orange.tif', cv2.IMREAD_GRAYSCALE)
    ret4, orange_bin = cv2.threshold(orange, 128, 1, cv2.THRESH_BINARY)
    red = cv2.imread('musk/red.tif', cv2.IMREAD_GRAYSCALE)
    ret5, red_bin = cv2.threshold(red, 128, 1, cv2.THRESH_BINARY)


    img_nrow, img_ncol = int(gray.shape[0] / img_size), int(gray.shape[1] / img_size)
    dim_row = np.concatenate((np.arange(1, step + 1, 1), np.ones(img_nrow - step * 2) * step, np.arange(step, 0, -1)))
    dim_col = np.concatenate((np.arange(1, step + 1, 1), np.ones(img_ncol - step * 2) * step, np.arange(step, 0, -1)))
    adjust_matrix = dim_row.reshape(-1, 1) * dim_col.reshape(1, -1)

    prob_matrix_0, prob_matrix_1, prob_matrix_2, prob_matrix_3, prob_matrix_no = get_probability_matrix(nrow=img_nrow,
                                                                                                        ncol=img_ncol,
                                                                                                        blue_channel=blue_bin,
                                                                                                        yellow_channel=yellow_bin,
                                                                                                        orange_channel=orange_bin,
                                                                                                        red_channel=red_bin,
                                                                                                        gray_channel=gray_bin)

    pd_lut_0 = pd.read_csv('lut/blue.csv', header=None)
    pd_lut_1 = pd.read_csv('lut/yellow.csv', header=None)
    pd_lut_2 = pd.read_csv('lut/orange.csv', header=None)
    pd_lut_3 = pd.read_csv('lut/red.csv', header=None)
    pd_lut_no = pd.read_csv('lut/gray.csv', header=None)

    np_lut0 = np.array(pd_lut_0).astype(np.uint8)
    np_lut1 = np.array(pd_lut_1).astype(np.uint8)
    np_lut2 = np.array(pd_lut_2).astype(np.uint8)
    np_lut3 = np.array(pd_lut_3).astype(np.uint8)
    np_lutno = np.array(pd_lut_no).astype(np.uint8)

    np_cv2_lut0 = np.flip(np_lut0, 1)
    np_cv2_lut1 = np.flip(np_lut1, 1)
    np_cv2_lut2 = np.flip(np_lut2, 1)
    np_cv2_lut3 = np.flip(np_lut3, 1)
    np_cv2_lutno = np.flip(np_lutno, 1)

    lut0 = np.expand_dims(np_cv2_lut0, axis=0)
    lut1 = np.expand_dims(np_cv2_lut1, axis=0)
    lut2 = np.expand_dims(np_cv2_lut2, axis=0)
    lut3 = np.expand_dims(np_cv2_lut3, axis=0)
    lutno = np.expand_dims(np_cv2_lutno, axis=0)

    adjusted_matrix_0 = np.round(prob_matrix_0 / adjust_matrix * 255).astype(np.uint8)
    adjusted_matrix_1 = np.round(prob_matrix_1 / adjust_matrix * 255).astype(np.uint8)
    adjusted_matrix_2 = np.round(prob_matrix_2 / adjust_matrix * 255).astype(np.uint8)
    adjusted_matrix_3 = np.round(prob_matrix_3 / adjust_matrix * 255).astype(np.uint8)
    adjusted_matrix_no = np.round(prob_matrix_no / adjust_matrix * 255).astype(np.uint8)

    adjusted_matrix_0_raw = np.zeros([img_nrow * img_size, img_ncol * img_size]).astype(np.uint8)
    adjusted_matrix_1_raw = np.zeros([img_nrow * img_size, img_ncol * img_size]).astype(np.uint8)
    adjusted_matrix_2_raw = np.zeros([img_nrow * img_size, img_ncol * img_size]).astype(np.uint8)
    adjusted_matrix_3_raw = np.zeros([img_nrow * img_size, img_ncol * img_size]).astype(np.uint8)
    adjusted_matrix_no_raw = np.zeros([img_nrow * img_size, img_ncol * img_size]).astype(np.uint8)

    for i in range(img_nrow):
        for j in range(img_ncol):
            adjusted_matrix_0_raw[i * img_size: (i + 1) * img_size, j * img_size: (j + 1) * img_size] = \
                adjusted_matrix_0[i, j]
            adjusted_matrix_1_raw[i * img_size: (i + 1) * img_size, j * img_size: (j + 1) * img_size] = \
                adjusted_matrix_1[i, j]
            adjusted_matrix_2_raw[i * img_size: (i + 1) * img_size, j * img_size: (j + 1) * img_size] = \
                adjusted_matrix_2[i, j]
            adjusted_matrix_3_raw[i * img_size: (i + 1) * img_size, j * img_size: (j + 1) * img_size] = \
                adjusted_matrix_3[i, j]
            adjusted_matrix_no_raw[i * img_size: (i + 1) * img_size, j * img_size: (j + 1) * img_size] = \
                adjusted_matrix_no[i, j]

    prob_0 = cv2.LUT(np.dstack([adjusted_matrix_0_raw] * 3), lut0)
    prob_1 = cv2.LUT(np.dstack([adjusted_matrix_1_raw] * 3), lut1)
    prob_2 = cv2.LUT(np.dstack([adjusted_matrix_2_raw] * 3), lut2)
    prob_3 = cv2.LUT(np.dstack([adjusted_matrix_3_raw] * 3), lut3)
    prob_no = cv2.LUT(np.dstack([adjusted_matrix_no_raw] * 3), lutno)

    prob = prob_0.astype(np.int) + prob_1.astype(np.int) + prob_2.astype(np.int) + prob_3.astype(
        np.int) + prob_no.astype(np.int)
    prob[np.where(prob > 255)] = 255
    prob_all = prob.astype(np.uint8)


    cv2.imwrite('musk/ChenXingLong_groundtruth-step10.png', prob_all)

    data0 = pd.DataFrame(data=adjusted_matrix_0_raw)
    data0.to_csv('musk/ChenXingLong_groundtruth_0.csv', header=False, index=False)

    data0_img = cv2.applyColorMap(adjusted_matrix_0_raw, cv2.COLORMAP_JET)
    cv2.imwrite('musk/ChenXingLong_groundtruth_0.png', data0_img)

    data1 = pd.DataFrame(data=adjusted_matrix_1_raw)
    data1.to_csv('musk/ChenXingLong_groundtruth_1.csv', header=False, index=False)

    data1_img = cv2.applyColorMap(adjusted_matrix_1_raw, cv2.COLORMAP_JET)
    cv2.imwrite('musk/ChenXingLong_groundtruth_1.png', data1_img)

    data2 = pd.DataFrame(data=adjusted_matrix_2_raw)
    data2.to_csv('musk/ChenXingLong_groundtruth_2.csv', header=False, index=False)

    data2_img = cv2.applyColorMap(adjusted_matrix_2_raw, cv2.COLORMAP_JET)
    cv2.imwrite('musk/ChenXingLong_groundtruth_2.png', data2_img)

    data3 = pd.DataFrame(data=adjusted_matrix_3_raw)
    data3.to_csv('musk/ChenXingLong_groundtruth_3.csv', header=False, index=False)

    data3_img = cv2.applyColorMap(adjusted_matrix_3_raw, cv2.COLORMAP_JET)
    cv2.imwrite('musk/ChenXingLong_groundtruth_3.png', data3_img)

    datano = pd.DataFrame(data=adjusted_matrix_no_raw)
    datano.to_csv('musk/ChenXingLong_groundtruth_no.csv', header=False, index=False)

    datano_img = cv2.applyColorMap(adjusted_matrix_no_raw, cv2.COLORMAP_JET)
    cv2.imwrite('musk/ChenXingLong_groundtruth_no.png', datano_img)
