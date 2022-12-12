
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


num_class = 5
gpu = 1
img_size = 100
net_img_size = 300
# thershold = 0.2
step = int(net_img_size/img_size)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
classification = 0
features_blobs = []


def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


def returnCAM(feature_conv, weight_softmax, idx):
    size_upsample = (net_img_size, net_img_size)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    # for idx in class_idx:
    cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
    features_blobs = []
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    # cam_img = np.uint8(255 * cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


preprocess = transforms.Compose([
    # transforms.Resize((299, 299)),
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


def get_probability_matrix(net, cuda, nrow, ncol, img_mosaic):
    matrix = np.zeros((nrow*img_size, ncol*img_size))
    # zero, one, two, three = 0, 0, 0, 0
    for i in range(0, nrow - step + 1):
        for j in range(0, ncol - step + 1):
            # img_temp = img_mosaic.crop((j * img_size, i * img_size, j * img_size + net_img_size, i * img_size + net_img_size))
            img_temp = img_mosaic[i * img_size: i * img_size + net_img_size, j * img_size: j * img_size + net_img_size]
            img = preprocess(Image.fromarray(cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB))).unsqueeze(0)
            if torch.cuda.is_available():
                img = img.to(cuda)
            global features_blobs
            features_blobs = []
            output = net(img)
            CAMs = returnCAM(features_blobs[0], weight_softmax, classification)


            # heatmap = cv2.applyColorMap(CAMs[0], cv2.COLORMAP_JET)
            matrix[i * img_size: i * img_size + net_img_size, j * img_size: j * img_size + net_img_size] += CAMs[0]
            # output_prob  = F.softmax(output, dim=1).cpu().detach().squeeze().numpy()
            # print(output_prob)
            # output_max = np.argmax(output_prob)
            # _, pred_label = output.max(1)
            # temp_matrix = np.zeros((6, 6))
            # if output_max == 0:
            #     # matrix_0[i:i + step, j:j + step] += 1
            #     # zero += 1
            #     matrix_0[i:i + step, j:j + step] += 1
            # elif output_max == 1:
            #     # matrix_1[i:i + step, j:j + step] += 1
            #     # one += 1
            #     matrix_1[i:i + step, j:j + step] += 1
            # elif output_max == 2:
            #     # matrix_no[i:i + step, j:j + step] += 1
            #     # two += 1
            #     matrix_2[i:i + step, j:j + step] += 1
            # elif output_max == 3:
            #     matrix_3[i:i + step, j:j + step] += 1
            #     # three += 1
            # else:
            #     matrix_no[i:i + step, j:j + step] += 1

            print(i, j)

    return matrix


if __name__ == '__main__':
    select = 0
    if select == 0:
        img_path = 'active/BPH'
        net_path = 'last 20210610 four class/date20210614223143crossvalid 20210610/InceptionResNetV2params_Adamepochs100.pkl'
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
    models = mymodels.inceptionresnetv2(num_classes=num_class, pretrained=False)

    if torch.cuda.is_available():
        models.to(device)

    models.load_state_dict(torch.load(net_path, map_location='cuda:' + gpu.__str__())['model'])
    models.eval()

    models._modules.get('conv2d_7b').register_forward_hook(hook_feature)

    params = list(models.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())
    # pd_lut_0 = pd.read_csv('lut/blue.csv', header=None)
    # pd_lut_1 = pd.read_csv('lut/yellow.csv', header=None)
    # pd_lut_2 = pd.read_csv('lut/orange.csv', header=None)
    # pd_lut_3 = pd.read_csv('lut/red.csv', header=None)
    # pd_lut_no = pd.read_csv('lut/gray.csv', header=None)
    #
    # np_lut0 = np.array(pd_lut_0).astype(np.uint8)
    # np_lut1 = np.array(pd_lut_1).astype(np.uint8)
    # np_lut2 = np.array(pd_lut_2).astype(np.uint8)
    # np_lut3 = np.array(pd_lut_3).astype(np.uint8)
    # np_lutno = np.array(pd_lut_no).astype(np.uint8)
    #
    # np_cv2_lut0 = np.flip(np_lut0, 1)
    # np_cv2_lut1 = np.flip(np_lut1, 1)
    # np_cv2_lut2 = np.flip(np_lut2, 1)
    # np_cv2_lut3 = np.flip(np_lut3, 1)
    # np_cv2_lutno = np.flip(np_lutno, 1)
    #
    # lut0 = np.expand_dims(np_cv2_lut0, axis=0)
    # lut1 = np.expand_dims(np_cv2_lut1, axis=0)
    # lut2 = np.expand_dims(np_cv2_lut2, axis=0)
    # lut3 = np.expand_dims(np_cv2_lut3, axis=0)
    # lutno = np.expand_dims(np_cv2_lutno, axis=0)

    img_list = core_lzj.each_img(img_path)

    for img_dir in img_list:
        img_raw = cv2.imread(img_dir)
        img_nrow, img_ncol = int(img_raw.shape[0] / img_size), int(img_raw.shape[1] / img_size)
        print(img_dir)

        prob_matrix = get_probability_matrix(net=models, cuda=device, nrow=img_nrow, ncol=img_ncol, img_mosaic=img_raw)
        dim_row = np.repeat(np.concatenate((np.arange(1, step + 1, 1), np.ones(img_nrow - step * 2) * step, np.arange(step, 0, -1))), img_size)
        dim_col = np.repeat(np.concatenate((np.arange(1, step + 1, 1), np.ones(img_ncol - step * 2) * step, np.arange(step, 0, -1))), img_size)
        adjust_matrix = dim_row.reshape(-1, 1) * dim_col.reshape(1, -1)
        matrix = prob_matrix / adjust_matrix
        matrix_uint8 = np.uint8(255 * matrix)
        heatmap = cv2.applyColorMap(matrix_uint8, cv2.COLORMAP_JET)
        # img_name = img_dir.split('/')[-1].split('.')[0]
        img_name = os.path.basename(img_dir).split('.')[0] + '_heatmap.png'
        cv2.imwrite(os.path.join(os.path.dirname(img_dir), img_name), heatmap)
        data = pd.DataFrame(data=matrix)
        data_name = os.path.basename(img_dir).split('.')[0] + '_heatmap.csv'
        data.to_csv(os.path.join(os.path.dirname(img_dir), data_name), header=False, index=False)
        # adjusted_matrix_0 = np.round(prob_matrix_0 / adjust_matrix * 255).astype(np.uint8)
        # adjusted_matrix_1 = np.round(prob_matrix_1 / adjust_matrix * 255).astype(np.uint8)
        # adjusted_matrix_2 = np.round(prob_matrix_2 / adjust_matrix * 255).astype(np.uint8)
        # adjusted_matrix_3 = np.round(prob_matrix_3 / adjust_matrix * 255).astype(np.uint8)
        # adjusted_matrix_no = np.round(prob_matrix_no / adjust_matrix * 255).astype(np.uint8)
        #
        # adjusted_matrix_0_raw = np.zeros([img_nrow * img_size, img_ncol * img_size]).astype(np.uint8)
        # adjusted_matrix_1_raw = np.zeros([img_nrow * img_size, img_ncol * img_size]).astype(np.uint8)
        # adjusted_matrix_2_raw = np.zeros([img_nrow * img_size, img_ncol * img_size]).astype(np.uint8)
        # adjusted_matrix_3_raw = np.zeros([img_nrow * img_size, img_ncol * img_size]).astype(np.uint8)
        # adjusted_matrix_no_raw = np.zeros([img_nrow * img_size, img_ncol * img_size]).astype(np.uint8)

        # for i in range(img_nrow):
        #     for j in range(img_ncol):
        #         adjusted_matrix_0_raw[i * img_size: (i + 1) * img_size, j * img_size: (j + 1) * img_size] = \
        #         adjusted_matrix_0[i, j]
        #         adjusted_matrix_1_raw[i * img_size: (i + 1) * img_size, j * img_size: (j + 1) * img_size] = \
        #         adjusted_matrix_1[i, j]
        #         adjusted_matrix_2_raw[i * img_size: (i + 1) * img_size, j * img_size: (j + 1) * img_size] = \
        #         adjusted_matrix_2[i, j]
        #         adjusted_matrix_3_raw[i * img_size: (i + 1) * img_size, j * img_size: (j + 1) * img_size] = \
        #         adjusted_matrix_3[i, j]
        #         adjusted_matrix_no_raw[i * img_size: (i + 1) * img_size, j * img_size: (j + 1) * img_size] = \
        #         adjusted_matrix_no[i, j]



        # prob_0 = cv2.LUT(np.dstack([adjusted_matrix_0] * 3), np.flip(lut0, 1))
        # prob_1 = cv2.LUT(np.dstack([adjusted_matrix_1] * 3), np.flip(lut1, 1))
        # prob_2 = cv2.LUT(np.dstack([adjusted_matrix_2] * 3), np.flip(lut2, 1))
        # prob_3 = cv2.LUT(np.dstack([adjusted_matrix_3] * 3), np.flip(lut3, 1))
        # prob_0 = cv2.LUT(np.dstack([adjusted_matrix_0] * 3), lut0)
        # prob_1 = cv2.LUT(np.dstack([adjusted_matrix_1] * 3), lut1)
        # prob_2 = cv2.LUT(np.dstack([adjusted_matrix_2] * 3), lut2)
        # prob_3 = cv2.LUT(np.dstack([adjusted_matrix_3] * 3), lut3)

        # prob_0 = cv2.LUT(np.dstack([adjusted_matrix_0_raw] * 3), lut0)
        # prob_1 = cv2.LUT(np.dstack([adjusted_matrix_1_raw] * 3), lut1)
        # prob_2 = cv2.LUT(np.dstack([adjusted_matrix_2_raw] * 3), lut2)
        # prob_3 = cv2.LUT(np.dstack([adjusted_matrix_3_raw] * 3), lut3)
        # prob_no = cv2.LUT(np.dstack([adjusted_matrix_no_raw] * 3), lutno)
        #
        # prob = prob_0.astype(np.int) + prob_1.astype(np.int) + prob_2.astype(np.int) + prob_3.astype(np.int) + prob_no.astype(np.int)
        # prob[np.where(prob > 255)] = 255
        # prob_all = prob.astype(np.uint8)
        # dir_path = os.path.dirname(img_dir)
        # file_name = os.path.basename(img_dir).split('.')[0]
        #
        # cv2.imwrite(os.path.join(dir_path, file_name) + '_50 pixels per step.png', prob_all)
        #
        # data0 = pd.DataFrame(data=adjusted_matrix_0_raw)
        # data0.to_csv(os.path.join(dir_path, file_name) + '_0.csv', header=False, index=False)
        # data1 = pd.DataFrame(data=adjusted_matrix_1_raw)
        # data1.to_csv(os.path.join(dir_path, file_name) + '_1.csv', header=False, index=False)
        # data2 = pd.DataFrame(data=adjusted_matrix_2_raw)
        # data2.to_csv(os.path.join(dir_path, file_name) + '_2.csv', header=False, index=False)
        # data3 = pd.DataFrame(data=adjusted_matrix_3_raw)
        # data3.to_csv(os.path.join(dir_path, file_name) + '_3.csv', header=False, index=False)
        # datano = pd.DataFrame(data=adjusted_matrix_no_raw)
        # datano.to_csv(os.path.join(dir_path, file_name) + '_no.csv', header=False, index=False)




