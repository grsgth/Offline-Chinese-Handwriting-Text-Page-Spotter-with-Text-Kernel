import torch
import numpy as np
import cv2
from utils.hwdb2_0_chars import char_set
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from utils.get_dgrl_data import get_pred_data
from utils.pred_utils import get_ar_cr, get_pred_str, normal_leven
from torch.cuda.amp import autocast as autocast
from models.model_with_TCN_big_new_one_batch_hwdb import Model

def predict(model, pred_iter, show=False):
    with torch.no_grad():
        img_np, img_tensor, boxes, page_label = next(pred_iter)
        label_np = np.ones_like(img_np, dtype=np.uint8) * 255
        boxes = boxes[0]
        imgs = img_tensor.to(device)
        # with autocast():
        kernel, out_chars, sub_img_nums = model(imgs, [boxes], is_train=False)

        prediction_char = out_chars
        prediction_char = prediction_char.log_softmax(-1)
        pred_strs = get_pred_str(prediction_char, char_set)

        pred_str_group = pred_strs

        # CR, AR, All = get_ar_cr(''.join(pred_str_group), ''.join(page_label))
        CR, AR, All = 0, 0, 0
        char_c = len(''.join(page_label))
        edit_d = normal_leven(''.join(pred_str_group), ''.join(page_label))
        for sub_p, sub_l in zip(pred_str_group, page_label):
            sub_cr, sub_ar, sub_all = get_ar_cr(sub_p, sub_l)
            CR += sub_cr
            AR += sub_ar
            All += sub_all

        if show:
            for box in boxes:
                box = np.int_(box)
                cv2.polylines(img_np, [box], True, 128, 1)
            char_size = int(label_np.shape[1] / len(page_label) / 5)
            if isinstance(label_np, np.ndarray):
                label_np = Image.fromarray(cv2.cvtColor(label_np, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(label_np)

            fontText = ImageFont.truetype('simfang.ttf', char_size, encoding="utf-8")
            draw.text((0, 0), 'CR:{:.6f} AR:{:.6f}'.format(CR / All, AR / All), (0, 0, 0), font=fontText)
            for i in range(len(pred_str_group)):
                left = boxes[i][0][0]
                top = boxes[i][0][1]
                draw.text((left, top), 'label:' + page_label[i], (0, 0, 0), font=fontText)
                draw.text((left, top + char_size), 'preds:' + pred_str_group[i], (0, 0, 0), font=fontText)

            label_np = cv2.cvtColor(np.asarray(label_np), cv2.COLOR_RGB2BGR)

            # show_np = np.hstack([img_np, label_np])
            # show_np = cv2.resize(show_np, None, fx=0.7, fy=0.7)
            if img_np.shape[1] > 1600:
                scale = 1600 / img_np.shape[1]
                img_np = cv2.resize(img_np, None, fx=scale, fy=scale)
                label_np = cv2.resize(label_np, None, fx=scale, fy=scale)

            cv2.imshow('1', img_np)
            cv2.imshow('label', label_np)
            cv2.waitKey()
    return CR, AR, All, edit_d, char_c


if __name__ == '__main__':
    from tqdm import tqdm
    import os

    device = torch.device('cuda')
    img_transform = transforms.ToTensor()

    model = Model(num_classes=3000, line_height=32, is_transformer=True, is_TCN=True).to(device)
    model.load_state_dict(torch.load(
        r'./output/hwdb2'
        r'/model.pth'))

    model.eval()

    file_paths = []
    for root_path in [r'D:\git\OCR\handwritind_dect_reco\data/hwdb2/HWDB2.0Test/dgrl',
                      r'D:\git\OCR\handwritind_dect_reco\data/hwdb2/HWDB2.1Test/dgrl',
                      r'D:\git\OCR\handwritind_dect_reco\data/hwdb2/HWDB2.2Test/dgrl']:
        for file_path in os.listdir(root_path):
            if file_path.endswith('dgrl'):
                file_paths.append(os.path.join(root_path, file_path))

    CR_all, AR_all, All_all = 0, 0, 0
    edit_d_a, char_c_a = 0, 0
    TP_all, FP_all, FN_all = 0, 0, 0
    pbar = tqdm(total=len(file_paths))
    pred_iter = iter(get_pred_data(file_paths, 1600))

    for i in range(len(file_paths)):
        cr, ar, all, edit_d, char_c = predict(model, pred_iter, False)
        CR_all += cr
        AR_all += ar
        All_all += all
        edit_d_a += edit_d
        char_c_a += char_c
        pbar.display('CR:{:.6f} AR:{:.6f} edit_d:{:.6f}\n'.format(
            CR_all / All_all, AR_all / All_all, (char_c_a - edit_d_a) / char_c_a))
        pbar.update(1)
