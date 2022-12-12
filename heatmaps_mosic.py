from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import pretrainedmodels.models as mymodels
import core_lzj
import numpy as np
import torch
import cv2
import os


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
num_class = 2
features_blobs = []
gpu = 0


def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


def returnCAM(feature_conv, weight_softmax, idx):
    size_upsample = (299, 299)
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
    net = mymodels.inceptionresnetv2(num_classes=num_class, pretrained=False)
    select = 0
    if select == 0:
        img_dir = 'probability/194w300_16_18'
        net_path = 'probability/InceptionResNetV2params_Adamepochs600.pkl'
    elif select == 1:
        img_dir = core_lzj.get_directory()
        net_path = core_lzj.get_file()
    elif select == 2:
        img_dir = core_lzj.get_directory()
        net_path = 'date20200905213524crossvalid1 two classclass/InceptionResNetV2params_Adamepochs600.pkl'
    elif select == 3:
        img_dir = 'check/027h-2_18_21'
        net_path = core_lzj.get_file()
    else:
        img_dir = []
        net_path = []
        core_lzj.exit_program()

    CAM_dir = img_dir + '/' + 'CAM_' + img_dir.split('/')[-1]
    RawaddCAM_dir = img_dir + '/' + 'RawaddCAM_' + img_dir.split('/')[-1]
    core_lzj.check_folder_existence(CAM_dir)
    core_lzj.check_folder_existence(RawaddCAM_dir)
    net.load_state_dict(torch.load(net_path, map_location='cuda:0'))
    device, init_flag = core_lzj.cuda_init(gpu)
    if torch.cuda.is_available():
        net.to(device,)
    net.eval()
    # label = 1
    net._modules.get('conv2d_7b').register_forward_hook(hook_feature)

    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        normalize
    ])
    img_list = core_lzj.each_img(img_dir)
    for path in img_list:
        print(path)
        features_blobs = []
        img_pil = Image.open(path)
        # img_pil.save(CAM_dir + '/Raw_' + os.path.basename(path))

        img_tensor = preprocess(img_pil)
        img_variable = Variable(img_tensor.unsqueeze(0))
        if torch.cuda.is_available():
            img_variable = img_variable.to(device)
        logit = net(img_variable)

        h_x = F.softmax(logit, dim=1).data.squeeze()
        # print(h_x)
        probs, idx = h_x.sort(0, True)
        # probs = probs.numpy()
        idx = idx.cpu().numpy()
        print(h_x.cpu().numpy())

        # for i in range(0,2):
        #     print('{:.3f} -> {}'.format(probs[i], idx[i]))

        CAMs = returnCAM(features_blobs[0], weight_softmax, 0)

        # print('output CAM for the top1 prediction: %s' % idx[0])
        img = cv2.imread(path)

        heatmap = cv2.applyColorMap(CAMs[0], cv2.COLORMAP_JET)
        result = heatmap * 0.5 + img * 0.5
        cv2.imwrite(CAM_dir + '/CAM_' + os.path.basename(path), heatmap)
        cv2.imwrite(RawaddCAM_dir + '/Raw+CAM_' + os.path.basename(path), result)

    core_lzj.cuda_empty_cache(init_flag)