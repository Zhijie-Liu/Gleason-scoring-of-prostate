import os
import cv2
import shutil
import core_lzj


def collagen_move(img_path, new_path):
    sub_path = os.listdir(img_path)
    for path in sub_path:
        path_temp = os.path.join(img_path, path)
        print(path_temp)
        if os.path.isdir(path_temp):
            collagen_move(path_temp, new_path)
        else:
            img = cv2.imread(path_temp)
            if img[:, :, (0, 2)].sum() * 0.63 > img[:, :, 1].sum():
                shutil.move(path_temp, new_path)


if __name__ == '__main__':
    image_path = 'four class new'
    collagen_path = 'four class new collaggen'
    core_lzj.check_folder_existence(collagen_path)
    # aaa = cv2.imread('check/Fith time-10_9-FRL-A-1_4_9_01_03.png')
    collagen_move(image_path, collagen_path)