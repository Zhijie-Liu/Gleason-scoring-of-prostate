
from torch.nn import functional as F
import pretrainedmodels.models as mymodels
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import core_lzj
import numpy as np
import torch
import cv2
import os


num_class1 = 3
num_class2 = 4
gpu = 3
img_size = 50
net_img_size = 300
# thershold = 0.2
step = int(net_img_size/img_size)

preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# preprocess1 = transforms.Compose([
#     # transforms.ToTensor(),
#     transforms.Resize((299, 299))
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# preprocess2 = transforms.Compose([
#     # transforms.Resize((299, 299)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])


# def get_one(img, h_location=0, w_location=0):
#     # img = Image.new('RGB', (net_img_size, net_img_size))
#     for i in range(0, step):
#         for j in range(0, step):
#             x1, y1 = (w_location + j)*50, (h_location + i)*50
#             x2, y2 = (w_location + j)*50 + 300, (h_location + i)*50 + 300
#             img_temp = img.crop(j*50, i*50, j*50 + 300, i*50 + 300)
#             # img_path = path + '/' + basename + '_' + str(h_location + i + 1) + '_' + str(w_location + j + 1) + '.png'
#             # img.paste(Image.open(img_path), (j*50, i*50))
#     return preprocess(img_temp).unsqueeze(0)


def get_probability_matrix(net1, net2, cuda, nrow, ncol, img_mosaic):
    matrix_0, matrix_1, matrix_2, matrix_3, matrix_no = np.zeros((nrow, ncol)), np.zeros((nrow, ncol)), np.zeros((nrow, ncol)), np.zeros((nrow, ncol)), np.zeros((nrow, ncol))
    # zero, one, two, three = 0, 0, 0, 0
    for i in range(0, nrow - step + 1):
        for j in range(0, ncol - step + 1):
            # img_temp = img_mosaic.crop((j * img_size, i * img_size, j * img_size + net_img_size, i * img_size + net_img_size))
            img_temp = img_mosaic[i * img_size: i * img_size + net_img_size, j * img_size: j * img_size + net_img_size]
            img = preprocess(Image.fromarray(cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB))).unsqueeze(0)
            if torch.cuda.is_available():
                img = img.to(cuda)
            output = net1(img)
            output_prob = F.softmax(output, dim=1).cpu().detach().squeeze().numpy()
            print(output_prob)
            output_max = np.argmax(output_prob)
            # _, pred_label = output.max(1)
            # temp_matrix = np.zeros((6, 6))
            if output_max == 0:
                # matrix_0[i:i + step, j:j + step] += 1
                # zero += 1
                matrix_0[i:i + step, j:j + step] += 1
            elif output_max == 1:
                output2 = net2(img)
                output_prob2 = F.softmax(output2, dim=1).cpu().detach().squeeze().numpy()
                print('subtype:', output_prob2)
                output_max2 = np.argmax(output_prob2)
                if output_max2 == 0:
                    matrix_no[i:i + step, j:j + step] += 1
                elif output_max2 == 1:
                    matrix_1[i:i + step, j:j + step] += 1
                elif output_max2 == 2:
                    matrix_2[i:i + step, j:j + step] += 1
                elif output_max2 == 3:
                    matrix_3[i:i + step, j:j + step] += 1
            else:
                matrix_no[i:i + step, j:j + step] += 1

            print(i, j)

    return matrix_0, matrix_1, matrix_2, matrix_3, matrix_no


if __name__ == '__main__':
    select = 0
    if select == 0:
        img_path = '20210902 Score/'
        net_path1 = 'date20210906162244crossvalid 20210906 two class/InceptionResNetV2params_Adamepochs120.pkl'
        net_path2 = 'date20210911163451crossvalid 20210910 three class/InceptionResNetV2params_Adamepochs100.pkl'
    elif select == 1:
        img_path = core_lzj.get_file()
        net_path = core_lzj.get_file()
    elif select == 2:
        img_path = core_lzj.get_file()
        net_path = 'date20200905213524crossvalid1 two classclass/InceptionResNetV2params_Adamepochs600.pkl'
    elif select == 3:
        img_path = 'check/027h-2_18_21'
        net_path = core_lzj.get_file()
    else:
        img_path = []
        net_path = []
        core_lzj.exit_program()

    device, init_flag = core_lzj.cuda_init(gpu)
    models1 = mymodels.inceptionresnetv2(num_classes=num_class1, pretrained=False)
    models2 = mymodels.inceptionresnetv2(num_classes=num_class2, pretrained=False)

    if torch.cuda.is_available():
        models1.to(device)
        models2.to(device)

    # img = get_one(h_location=0, w_location=0, path='probability/194w_99_108', basename='194w_99_108')
    models1.load_state_dict(torch.load(net_path1, map_location='cuda:' + gpu.__str__())['model'])
    models1.eval()
    models2.load_state_dict(torch.load(net_path2, map_location='cuda:' + gpu.__str__())['model'])
    models2.eval()

    pd_lut_0 = pd.read_csv('score20210104/lut0.csv', header=None)
    pd_lut_1 = pd.read_csv('score20210104/lut1.csv', header=None)
    pd_lut_2 = pd.read_csv('score20210104/lut2.csv', header=None)
    pd_lut_3 = pd.read_csv('score20210104/lut3.csv', header=None)
    pd_lut_no = pd.read_csv('score20210104/lutno.csv', header=None)

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

    img_list = core_lzj.each_img(img_path)

    for img_dir in img_list:
        img_raw = cv2.imread(img_dir)
        img_nrow, img_ncol = int(img_raw.shape[0] / img_size), int(img_raw.shape[1] / img_size)

        prob_matrix_0, prob_matrix_1, prob_matrix_2, prob_matrix_3, prob_matrix_no = get_probability_matrix(net1=models1, net2=models2, cuda=device, nrow=img_nrow, ncol=img_ncol, img_mosaic=img_raw)
        dim_row = np.concatenate((np.arange(1, step + 1, 1), np.ones(img_nrow - step * 2) * step, np.arange(step, 0, -1)))
        dim_col = np.concatenate((np.arange(1, step + 1, 1), np.ones(img_ncol - step * 2) * step, np.arange(step, 0, -1)))
        adjust_matrix = dim_row.reshape(-1, 1) * dim_col.reshape(1, -1)
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



        # prob_0 = cv2.LUT(np.dstack([adjusted_matrix_0] * 3), np.flip(lut0, 1))
        # prob_1 = cv2.LUT(np.dstack([adjusted_matrix_1] * 3), np.flip(lut1, 1))
        # prob_2 = cv2.LUT(np.dstack([adjusted_matrix_2] * 3), np.flip(lut2, 1))
        # prob_3 = cv2.LUT(np.dstack([adjusted_matrix_3] * 3), np.flip(lut3, 1))
        # prob_0 = cv2.LUT(np.dstack([adjusted_matrix_0] * 3), lut0)
        # prob_1 = cv2.LUT(np.dstack([adjusted_matrix_1] * 3), lut1)
        # prob_2 = cv2.LUT(np.dstack([adjusted_matrix_2] * 3), lut2)
        # prob_3 = cv2.LUT(np.dstack([adjusted_matrix_3] * 3), lut3)

        prob_0 = cv2.LUT(np.dstack([adjusted_matrix_0_raw] * 3), lut0)
        prob_1 = cv2.LUT(np.dstack([adjusted_matrix_1_raw] * 3), lut1)
        prob_2 = cv2.LUT(np.dstack([adjusted_matrix_2_raw] * 3), lut2)
        prob_3 = cv2.LUT(np.dstack([adjusted_matrix_3_raw] * 3), lut3)
        prob_no = cv2.LUT(np.dstack([adjusted_matrix_no_raw] * 3), lutno)

        prob = prob_0 + prob_1 + prob_2 + prob_3 + prob_no

        dir_path = os.path.dirname(img_dir)
        file_name = os.path.basename(img_dir).split('.')[0]

        cv2.imwrite(os.path.join(dir_path, file_name) + '_50 pixels per step two net.png', prob)




