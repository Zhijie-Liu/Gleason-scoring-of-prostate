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
gpu = 1

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (299, 299)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
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
        img_path = 'check/027h-2_18_21/027h-2_18_21_03_14.png'
        net_path = 'date20200905213524crossvalid1 two classclass/InceptionResNetV2params_Adamepochs600.pkl'
    elif select == 1:
        img_path = core_lzj.get_file()
        net_path = core_lzj.get_file()
    elif select == 2:
        img_path = core_lzj.get_file()
        net_path = 'date20200905213524crossvalid1 two classclass/InceptionResNetV2params_Adamepochs600.pkl'
    elif select == 3:
        img_path = 'check/027h-2_18_21/027h-2_18_21_03_14.png'
        net_path = core_lzj.get_file()
    else:
        img_path = []
        net_path = []
        core_lzj.exit_program()

    net.load_state_dict(torch.load(net_path))
    init_flag = core_lzj.cuda_init(gpu)
    if torch.cuda.is_available():
        net = net.cuda()
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
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    img_pil = Image.open(img_path)


    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    if torch.cuda.is_available():
        img_variable = img_variable.cuda()
    logit = net(img_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.cpu().numpy()
    idx = idx.cpu().numpy()

    for i in range(0,2):
        print('{:.3f} -> {}'.format(probs[i], idx[i]))

    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
    heatmap = cv2.applyColorMap(CAMs[0], cv2.COLORMAP_JET)
    print('output CAM for the top1 prediction: %s' % idx[0])
    img_pil.save('heatmaps/test.jpg')
    img = cv2.imread('heatmaps/test.jpg')


    result = heatmap * 0.5 + img * 0.5
    cv2.imwrite('heatmaps/test2.jpg', result)
    cv2.imwrite('heatmaps/test3.jpg', heatmap)
    core_lzj.cuda_empty_cache(init_flag)