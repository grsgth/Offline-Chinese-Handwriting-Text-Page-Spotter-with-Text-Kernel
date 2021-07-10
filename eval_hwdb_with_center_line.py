import torch
import numpy as np
from utils.hwdb2_0_chars import char_set
from torchvision import transforms
from utils.get_dgrl_data import get_pred_data
from utils.pred_utils import get_ar_cr, get_pred_str, polygon_IOU, normal_leven
from models.model_with_TCN_big_new_one_batch_hwdb import Model

from tqdm import tqdm
import os


def predict(model, pred_iter):
    with torch.no_grad():
        img_np, img_tensor, boxes, page_label = next(pred_iter)
        boxes = boxes[0]
        imgs = img_tensor.to(device)

        kernel, out_chars, sub_img_nums, line_top_lefts, line_contours = model(imgs, None, is_train=False)

        line_contours = line_contours[0]
        prediction_char = out_chars
        prediction_char = prediction_char.log_softmax(-1)
        pred_strs = get_pred_str(prediction_char, char_set)

        pred_str_group = ['' for _ in range(len(page_label))]
        not_in_char = ''
        TP = 0
        FP = 0
        FN = 0

        for pred_i in range(len(pred_strs)):
            pred_str_poly = line_contours[pred_i]
            pred_str_poly = np.squeeze(pred_str_poly, 1)
            find_flag = 0
            for label_i in range(len(boxes)):

                label_box = boxes[label_i] / 4
                pred_iou = polygon_IOU(pred_str_poly, label_box)
                if pred_iou > 0.9:
                    pred_str_group[label_i] += pred_strs[pred_i]
                    find_flag = 1
                    break
            if find_flag == 0:
                FP += 1
                not_in_char += pred_strs[pred_i]

        for i in range(len(pred_str_group)):
            if len(pred_str_group[i]) / len(page_label[i]):
                TP += 1
            else:
                FN += 1

        pred_strs_s = ''.join(pred_str_group) + not_in_char
        # CR, AR, All = get_ar_cr(pred_strs_s, ''.join(page_label))
        CR, AR, All = 0, 0, 0
        char_c = len(''.join(page_label))
        edit_d = normal_leven(pred_strs_s, ''.join(page_label))
        for sub_p, sub_l in zip(pred_str_group, page_label):
            sub_cr, sub_ar, sub_all = get_ar_cr(sub_p, sub_l)
            CR += sub_cr
            AR += sub_ar
            All += sub_all
        AR -= len(not_in_char)

    return CR, AR, All, edit_d, char_c, TP, FP, FN


if __name__ == '__main__':

    device = torch.device('cuda')
    img_transform = transforms.ToTensor()
    model = Model(num_classes=3000, line_height=32, is_transformer=True, is_TCN=True).to(device)
    model.load_state_dict(torch.load(
        r'./output/hwdb2'
        r'/model.pth'))
    model.eval()
    file_paths = []

    for root_path in [r'./data/hwdb2/HWDB2.0Test/dgrl',
                      r'./data/hwdb2/HWDB2.1Test/dgrl',
                      r'./data/hwdb2/HWDB2.2Test/dgrl']:
        for file_path in os.listdir(root_path):
            if file_path.endswith('dgrl'):
                file_paths.append(os.path.join(root_path, file_path))

    CR_all, AR_all, All_all = 0, 0, 0
    EDIT_DISTANCE_ALL, CHAR_COUNT_ALL = 0, 0
    TP_all, FP_all, FN_all = 0, 0, 0
    pbar = tqdm(total=len(file_paths))
    pred_iter = iter(get_pred_data(file_paths, 1600))

    for i in range(len(file_paths)):
        cr, ar, all, edit_d, char_c, TP, FP, FN = predict(model, pred_iter)
        CR_all += cr
        AR_all += ar
        All_all += all
        EDIT_DISTANCE_ALL += edit_d
        CHAR_COUNT_ALL += char_c
        TP_all += TP
        FP_all += FP
        FN_all += FN

        Precision = TP_all / (TP_all + FP_all)
        Recall = TP_all / (TP_all + FN_all)
        F1 = 2 / (1 / Precision + 1 / Recall)

        pbar.display('CR:{:.6f} AR:{:.6f} edid_d:{:.6f} Precision:{:.6f} Recall:{:.6f} F1:{:.6f}\n'.format(
            CR_all / All_all, AR_all / All_all, (CHAR_COUNT_ALL - EDIT_DISTANCE_ALL) / CHAR_COUNT_ALL,
            Precision, Recall, F1))
        pbar.update(1)
