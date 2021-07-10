from models.dect import PANnet
from models.reco import DenseNet_with_TCN_big
from models.connect_layer_new_one_batch_hwdb import Connect
import torch
from torch import nn
import time


class Model(nn.Module):
    def __init__(self, line_height=32, num_classes=3000,is_english=False,is_TCN=True,is_transformer=True):
        super().__init__()
        PANnet_config = {
            'backbone': 'resnet34',
            'fpem_repeat': 4,  # fpem模块重复的次数
            'pretrained': False,  # backbone 是否使用imagesnet的预训练模型
            'result_num': 7,
            'segmentation_head': 'FPEM_FFM'  # 分割头，FPN or FPEM_FFM
        }
        self.PAN_layer = PANnet(model_config=PANnet_config)
        self.Connect_layer = Connect(line_height)
        self.DenseNet_layer = DenseNet_with_TCN_big(num_classes=num_classes,is_english=is_english,
                                                    is_TCN=is_TCN,
                                                    is_transformer=is_transformer)

    def forward(self, img_tensers, gt_boxes=None, is_train=True):
        assert (is_train == True and gt_boxes != None) or is_train == False
        t1 = time.time()
        kernels, features = self.PAN_layer(img_tensers)
        # print(kernels.shape)
        t2 = time.time()
        if gt_boxes is not None:
            text_kernel_features, sub_img_nums = self.Connect_layer(kernels, features, gt_boxes, is_train)
            out_chars = self.DenseNet_layer(text_kernel_features)
            return kernels, out_chars, sub_img_nums
        else:
            text_kernel_features, sub_img_nums, line_top_lefts, line_contours = self.Connect_layer(kernels, features, gt_boxes, is_train)
            out_chars = self.DenseNet_layer(text_kernel_features)
            return kernels, out_chars, sub_img_nums, line_top_lefts, line_contours
        # t3 = time.time()
        # out_chars = self.DenseNet_layer(text_kernel_features)
        # t4 = time.time()
        # print('PAN_layer',t2-t1)
        # print('Connect_layer',t3-t2)
        # print('DenseNet_layer',t4-t3)


if __name__ == '__main__':
    device = torch.device('cuda')

    x = torch.zeros(1, 3, 1200, 1200).to(device)
    model = Model().to(device)
    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    for i in range(1):
        model(x, torch.tensor([[[[100, 100], [400, 100], [400, 80], [100, 80]],
                                 ]]),
              True)
