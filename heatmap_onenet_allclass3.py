
import pretrainedmodels.models as mymodels
from torchvision import models, transforms
import pandas as pd
from PIL import Image
import numpy as np
import core_lzj
import prediction
import torch
import cv2
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
num_class = 5
classification = 3
features_blobs = []
gpu = 1
img_size = 300
net_img_size = 300
step = int(net_img_size/img_size)


def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


def returnCAM(feature_conv, weight_softmax, idx):
    size_upsample = (300, 300)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    # for idx in class_idx:
    cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam



if __name__ == '__main__':
    select = 2
    if select == 0:
        # img_dir = 'normal to tumor test/tumor'
        # net_path = 'normal to tumor test/InceptionResNetV2params_Adamepochs600.pkl'
        img_dir = '20210910 three class test/2'
        net_path = 'date20210911163451crossvalid 20210910 three class/InceptionResNetV2params_Adamepochs100.pkl'
    elif select == 1:
        img_dir = core_lzj.get_file()
        net_path = core_lzj.get_file()
    elif select == 2:
        img_dir = 'last 20210610 four class/20210610test/3'
        net_path = 'last 20210610 four class/date20210614223143crossvalid 20210610/InceptionResNetV2params_Adamepochs100.pkl'
    elif select == 3:
        img_dir = 'check/027h-2_18_21'
        net_path = core_lzj.get_file()
    else:
        img_dir = []
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

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        # transforms.Resize((299, 299)),
        transforms.ToTensor(),
        normalize
    ])

    path_dir = core_lzj.each_dir(filepath=img_dir)
    for dir in path_dir:

        img_path = core_lzj.each_img(filepath=dir)
        save_path = dir + '_heatmap'
        # core_lzj.check_folder_existence(save_path)
        # img_no = Image.new('RGB', (img_size - 2 * linewidth, img_size - 2 * linewidth), (190, 190, 190))
        # img_normal = Image.new('RGB', (img_size - 2 * linewidth, img_size - 2 * linewidth), (93, 134, 183))
        # img_tumor = Image.new('RGB', (img_size - 2 * linewidth, img_size - 2 * linewidth), (228, 128, 100))

        for path in img_path:
            print(path)
            heatmap_path = os.path.join(dir, 'heatmap')
            # heatmap_path = os.path.join(dir, 'heatmap', os.path.basename(path).split('.')[0])
            core_lzj.check_folder_existence(heatmap_path)
            img_raw = Image.open(path)
            # img_nrow, img_ncol = int(img_raw.height / img_size), int(img_raw.width / img_size)
            # for i in range(0, img_nrow - step + 1):
            #     for j in range(0, img_ncol - step + 1):
            #         features_blobs = []
            #         img_temp = img_raw.crop((j * img_size, i * img_size, j * img_size + net_img_size, i * img_size + net_img_size))
            img = preprocess(img_raw).unsqueeze(0)
            if torch.cuda.is_available():
                img = img.to(gpu)
            output = models(img)
            CAMs = returnCAM(features_blobs[0], weight_softmax, classification)
            heatmap = cv2.applyColorMap(CAMs[0], cv2.COLORMAP_JET)
            # raw_name = os.path.basename(path).split('.')[0] + '_' + str(i + 1) + '_' + str(j + 1) + '.png'
            heatmap_name = os.path.join(os.path.basename(path).split('.')[0] + '_heatmap.png')
            data = pd.DataFrame(data=CAMs[0])
            data_name = os.path.basename(path).split('.')[0] + '_heatmap.csv'
            data.to_csv(os.path.join(heatmap_path, data_name), header=False, index=False)
            # img_temp.save(os.path.join(heatmap_path, raw_name))
            cv2.imwrite(os.path.join(heatmap_path, heatmap_name), heatmap)
            features_blobs = []

    core_lzj.cuda_empty_cache(init_flag)




