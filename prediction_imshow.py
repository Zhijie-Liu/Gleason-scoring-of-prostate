import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image


linewidth = 5
img_size = 300


if __name__ == '__main__':
    path = 'score/zhuchenyun A-2_matrix.csv'
    pd_label = pd.read_csv(path, header=None)
    np_label = np.array(pd_label)
    width = np_label.shape[0]
    length = np_label.shape[1]
    img_zero = Image.new('RGB', (img_size - 2 * linewidth, img_size - 2 * linewidth), (190, 190, 190))
    img_one = Image.new('RGB', (img_size - 2 * linewidth, img_size - 2 * linewidth), (93, 134, 183))
    img_two = Image.new('RGB', (img_size - 2 * linewidth, img_size - 2 * linewidth), (228, 128, 100))
    img_three = Image.new('RGB', (img_size - 2 * linewidth, img_size - 2 * linewidth), (128, 228, 100))
    img_prediction = Image.new('RGB', (length * img_size, width * img_size), (0, 0, 0))
    for i in range(width):
        for j in range(length):
            print(i, j)
            if np_label[i, j] == 0:
                img_prediction.paste(img_zero, (j * img_size + linewidth, i * img_size + linewidth))
            elif np_label[i, j] == 1:
                img_prediction.paste(img_one, (j * img_size + linewidth, i * img_size + linewidth))
            elif np_label[i, j] == 2:
                img_prediction.paste(img_two, (j * img_size + linewidth, i * img_size + linewidth))
            elif np_label[i, j] == 3:
                img_prediction.paste(img_three, (j * img_size + linewidth, i * img_size + linewidth))

    save_path = os.path.dirname(path)
    file_name = os.path.basename(path).split('_')[0] + '.png'
    img_prediction.save(os.path.join(save_path, file_name))


    # probability = cv2.applyColorMap(np_data256, cv2.COLORMAP_JET)
    # probability_299 = cv2.resize(probability, (5400, 4950))
    # cv2.imwrite('probability/probability_299.png', probability_299)
    # a = 1
