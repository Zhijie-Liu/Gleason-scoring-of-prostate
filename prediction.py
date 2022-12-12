
from torch.nn import functional as F
import pretrainedmodels.models as mymodels
from torchvision import transforms
from PIL import Image
import pandas as pd
import core_lzj
import numpy as np
import torch
import cv2
import os


num_class = 5
gpu = 1
img_size = 300
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


def get_probability_matrix(net, cuda, nrow, ncol, img_mosaic):
    matrix, matrix_0, matrix_1, matrix_no = np.zeros((nrow, ncol)), np.zeros((nrow, ncol)), np.zeros((nrow, ncol)), np.zeros((nrow, ncol))
    zero, one, two, three = 0, 0, 0, 0
    for i in range(0, nrow - step + 1):
        for j in range(0, ncol - step + 1):
            img_temp = img_mosaic.crop((j * img_size, i * img_size, j * img_size + net_img_size, i * img_size + net_img_size))
            img = preprocess(img_temp).unsqueeze(0)
            if torch.cuda.is_available():
                img = img.to(cuda)
            output = net(img)
            output_prob = F.softmax(output, dim=1).cpu().detach().squeeze().numpy()
            print(output_prob)
            output_max = np.argmax(output_prob)
            # _, pred_label = output.max(1)
            # temp_matrix = np.zeros((6, 6))
            if output_max == 0:
                # matrix_0[i:i + step, j:j + step] += 1
                zero += 1
                matrix[i:i + step, j:j + step] = 0
            elif output_max == 1:
                # matrix_1[i:i + step, j:j + step] += 1
                one += 1
                matrix[i:i + step, j:j + step] = 1
            elif output_max == 2:
                # matrix_no[i:i + step, j:j + step] += 1
                two += 1
                matrix[i:i + step, j:j + step] = 2
            elif output_max == 3:
                matrix[i:i + step, j:j + step] = 3
                three += 1
            else:
                matrix[i:i + step, j:j + step] = -1

            print(i, j)

    return matrix, [zero, one, two, three]


if __name__ == '__main__':
    select = 0
    if select == 0:
        img_dir = 'score/12_10-A-JQB.tif (RGB).tif'
        net_path = 'score/InceptionResNetV2params_Adamepochs300.pkl'
    elif select == 1:
        img_dir = core_lzj.get_file()
        net_path = core_lzj.get_file()
    elif select == 2:
        img_dir = core_lzj.get_file()
        net_path = 'date20200905213524crossvalid1 two classclass/InceptionResNetV2params_Adamepochs600.pkl'
    elif select == 3:
        img_dir = 'check/027h-2_18_21'
        net_path = core_lzj.get_file()
    else:
        img_dir = []
        net_path = []
        core_lzj.exit_program()

    img_raw = Image.open(img_dir)
    img_nrow, img_ncol = int(img_raw.height/img_size), int(img_raw.width/img_size)
    # img_nrow, img_ncol = int(img_dir.split('_')[-2]), int(img_dir.split('_')[-1])
    device, init_flag = core_lzj.cuda_init(gpu)
    # img_basename = img_dir.split('/')[-1]
    models = mymodels.inceptionresnetv2(num_classes=num_class, pretrained=False)

    if torch.cuda.is_available():
        models.to(device)

    # img = get_one(h_location=0, w_location=0, path='probability/194w_99_108', basename='194w_99_108')
    models.load_state_dict(torch.load(net_path, map_location='cuda:' + gpu.__str__())['model'])
    models.eval()
    # prob_matrix_0, prob_matrix_1, prob_matrix_no = get_probability_matrix(net=models, cuda=device, nrow=img_nrow,
    #                                                                       ncol=img_ncol, img_mosaic=img_raw)
    prob_matrix, percentage = get_probability_matrix(net=models, cuda=device, nrow=img_nrow, ncol=img_ncol, img_mosaic=img_raw)
    print(percentage)
    file = os.path.dirname(img_dir) + '/' + os.path.basename(img_dir).split('.')[0] + '_percentage.dat'
    f = open(file, 'w')
    print(percentage, file=f, flush=True)
    print(np.array(percentage) / (img_nrow * img_ncol), file=f, flush=True)
    f.close()
    matrix_data = pd.DataFrame(data=prob_matrix)
    matrix_data.to_csv(os.path.dirname(img_dir) + '/' + os.path.basename(img_dir).split('.')[0] + '_matrix.csv', header=False, index=False)

    # dim_row = np.concatenate((np.arange(1, step + 1, 1), np.ones(img_nrow - step * 2) * step, np.arange(step, 0, -1)))
    # dim_col = np.concatenate((np.arange(1, step + 1, 1), np.ones(img_ncol - step * 2) * step, np.arange(step, 0, -1)))
    # adjust_matrix = dim_row.reshape(-1, 1) * dim_col.reshape(1, -1)
    # adjusted_matrix_0 = np.round(prob_matrix_0 / adjust_matrix * 255).astype(np.uint8)
    # adjusted_matrix_1 = np.round(prob_matrix_1 / adjust_matrix * 255).astype(np.uint8)
    # adjusted_matrix_no = np.round(prob_matrix_no / adjust_matrix * 255).astype(np.uint8)
    # adjusted_data = pd.DataFrame(data=adjusted_matrix)
    # adjusted_data256 = pd.DataFrame(data=adjusted_matrix256)
    # adjusted_data.to_csv(os.path.dirname(img_dir) + '/' + os.path.basename(img_dir) + '_' + myfunctions.get_time() +
    #                      '_probability_adjusted.csv', header=False, index=False)
    # adjusted_data256.to_csv(os.path.dirname(img_dir) + '/' + os.path.basename(img_dir) + '_' + myfunctions.get_time() +
    #                         '_probability_adjusted256.csv', header=False, index=False)
    # prob_0 = cv2.applyColorMap(adjusted_matrix_0, cv2.COLORMAP_JET)
    # prob_0_resize = cv2.resize(prob_0, (img_raw.width, img_raw.height))
    #
    # prob_1 = cv2.applyColorMap(adjusted_matrix_1, cv2.COLORMAP_JET)
    # prob_1_resize = cv2.resize(prob_1, (img_raw.width, img_raw.height))
    #
    # prob_no = cv2.applyColorMap(adjusted_matrix_no, cv2.COLORMAP_JET)
    # prob_no_resize = cv2.resize(prob_no, (img_raw.width, img_raw.height))
    #
    # cv2.imwrite('probability/probability128_0.png', prob_0_resize)
    # cv2.imwrite('probability/probability128_1.png', prob_1_resize)
    # cv2.imwrite('probability/probability128_no.png', prob_no_resize)

    # heatmap = cv2.applyColorMap(CAMs[0], cv2.COLORMAP_JET)
    # img_list = myfunctions.eachfile(img_dir)



