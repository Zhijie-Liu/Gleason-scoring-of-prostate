import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image


linewidth = 5
img_size = 300


if __name__ == '__main__':
    path0 = 'score fix/0.csv'
    path1 = 'score fix/1.csv'
    path2 = 'score fix/2.csv'
    path3 = 'score fix/3.csv'
    pathno = 'score fix/no.csv'
    matrix0 = pd.read_csv(path0, header=None)
    matrix1 = pd.read_csv(path1, header=None)
    matrix2 = pd.read_csv(path2, header=None)
    matrix3 = pd.read_csv(path3, header=None)
    matrixno = pd.read_csv(pathno, header=None)

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

    prob_0 = cv2.LUT(np.dstack([matrix0.astype(np.uint8)] * 3), lut0)
    prob_1 = cv2.LUT(np.dstack([matrix1.astype(np.uint8)] * 3), lut1)
    prob_2 = cv2.LUT(np.dstack([matrix2.astype(np.uint8)] * 3), lut2)
    prob_3 = cv2.LUT(np.dstack([matrix3.astype(np.uint8)] * 3), lut3)
    prob_no = cv2.LUT(np.dstack([matrixno.astype(np.uint8)] * 3), lutno)

    prob = prob_0.astype(np.int) + prob_1.astype(np.int) + prob_2.astype(np.int) + prob_3.astype(np.int) + prob_no.astype(np.int)
    prob[np.where(prob > 255)] = 255
    prob_all = prob.astype(np.uint8)



    cv2.imwrite('score fix/0.png', prob_0)
    cv2.imwrite('score fix/1.png', prob_1)
    cv2.imwrite('score fix/2.png', prob_2)
    cv2.imwrite('score fix/3.png', prob_3)
    cv2.imwrite('score fix/no.png', prob_no)
    cv2.imwrite('score fix/all.png', prob_all)
    # pd_label = pd.read_csv(path, header=None)
    # np_label = np.array(pd_label)
    # width = np_label.shape[0]
    # length = np_label.shape[1]
    # img_zero = Image.new('RGB', (img_size - 2 * linewidth, img_size - 2 * linewidth), (190, 190, 190))
    # img_one = Image.new('RGB', (img_size - 2 * linewidth, img_size - 2 * linewidth), (93, 134, 183))
    # img_two = Image.new('RGB', (img_size - 2 * linewidth, img_size - 2 * linewidth), (228, 128, 100))
    # img_three = Image.new('RGB', (img_size - 2 * linewidth, img_size - 2 * linewidth), (128, 228, 100))
    # img_prediction = Image.new('RGB', (length * img_size, width * img_size), (0, 0, 0))
    # for i in range(width):
    #     for j in range(length):
    #         print(i, j)
    #         if np_label[i, j] == 0:
    #             img_prediction.paste(img_zero, (j * img_size + linewidth, i * img_size + linewidth))
    #         elif np_label[i, j] == 1:
    #             img_prediction.paste(img_one, (j * img_size + linewidth, i * img_size + linewidth))
    #         elif np_label[i, j] == 2:
    #             img_prediction.paste(img_two, (j * img_size + linewidth, i * img_size + linewidth))
    #         elif np_label[i, j] == 3:
    #             img_prediction.paste(img_three, (j * img_size + linewidth, i * img_size + linewidth))
    #
    # save_path = os.path.dirname(path)
    # file_name = os.path.basename(path).split('_')[0] + '.png'
    # img_prediction.save(os.path.join(save_path, file_name))


    # probability = cv2.applyColorMap(np_data256, cv2.COLORMAP_JET)
    # probability_299 = cv2.resize(probability, (5400, 4950))
    # cv2.imwrite('probability/probability_299.png', probability_299)
    # a = 1
