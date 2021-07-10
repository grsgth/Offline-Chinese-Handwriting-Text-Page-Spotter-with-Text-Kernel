import os
import random
import pathlib
import pyclipper
from torch.utils import data
import glob
import numpy as np
import cv2
from dataset.augment import DataAugment
from utils.utils import draw_bbox

import torch

data_aug = DataAugment()


def check_and_validate_polys(polys, xxx_todo_changeme):
    '''
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    '''
    (h, w) = xxx_todo_changeme
    if polys.shape[0] == 0:
        return polys
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)  # x coord not max w-1, and not min 0
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)  # y coord not max h-1, and not min 0

    validated_polys = []
    for poly in polys:
        p_area = cv2.contourArea(poly)
        if abs(p_area) < 1:
            continue
        validated_polys.append(poly)
    return np.array(validated_polys)


def check_shrinked_poly(box):
    if min(((box[0, 1] - box[1, 1]) ** 2 + (box[0, 0] - box[1, 0]) ** 2),
           ((box[1, 1] - box[2, 1]) ** 2 + (box[1, 0] - box[2, 0]) ** 2),
           ((box[2, 1] - box[3, 1]) ** 2 + (box[2, 0] - box[3, 0]) ** 2),
           ((box[3, 1] - box[1, 1]) ** 2 + (box[3, 0] - box[1, 0]) ** 2),
           ) == 0:
        return False
    return True


def generate_rbox(im_size, text_polys, text_labels, training_mask, i, n, m):
    """
    生成mask图，白色部分是文本，黑色是北京
    :param im_size: 图像的h,w
    :param text_polys: 框的坐标
    :param text_tags: 标注文本框是否参与训练
    :return: 生成的mask图
    """
    h, w = im_size
    score_map = np.zeros((h, w), dtype=np.uint8)
    new_text_polys = []
    for poly, text_label in zip(text_polys, text_labels):
        poly = poly.astype(np.int)
        r_i = 1 - (1 - m) * (n - i) / (n - 1)
        d_i = cv2.contourArea(poly) * (1 - r_i * r_i) / cv2.arcLength(poly, True)
        pco = pyclipper.PyclipperOffset()
        # pco.AddPath(pyclipper.scale_to_clipper(poly), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        # shrinked_poly = np.floor(np.array(pyclipper.scale_from_clipper(pco.Execute(-d_i)))).astype(np.int)
        pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrinked_poly = np.array(pco.Execute(-d_i))
        cv2.fillPoly(score_map, shrinked_poly, 1)
        cv2.fillPoly(training_mask, shrinked_poly, 1)
        if len(shrinked_poly[0]) != 4 or len(shrinked_poly[0]) == 4 and check_shrinked_poly(shrinked_poly[0]):
            tmp_box = cv2.minAreaRect(shrinked_poly[0])
            tmp_box = cv2.boxPoints(tmp_box)
            shrinked_poly = [tmp_box]
        bbox = sorted(shrinked_poly[0], key=lambda x: x[0])
        new_bbox = []
        new_bbox += sorted(bbox[:2], key=lambda x: x[1])
        new_bbox += sorted(bbox[2:], key=lambda x: -x[1])
        bbox = [new_bbox[0], new_bbox[3], new_bbox[2], new_bbox[1]]
        bbox = np.int0(bbox)
        new_text_polys.append(bbox)

        # 制作mask
        # rect = cv2.minAreaRect(shrinked_poly)
        # poly_h, poly_w = rect[1]

        # if min(poly_h, poly_w) < 10:
        #     cv2.fillPoly(training_mask, shrinked_poly, 0)

        # 闭运算填充内部小框
        # kernel = np.ones((3, 3), np.uint8)
        # score_map = cv2.morphologyEx(score_map, cv2.MORPH_CLOSE, kernel)

    return score_map, training_mask, new_text_polys


def augmentation(im: np.ndarray, text_polys: np.ndarray, text_label, text_lengths, label_strs, epoch_count) -> tuple:
    # the images are rescaled with ratio {0.5, 1.0, 2.0, 3.0} randomly
    # im, text_polys = data_aug.random_scale(im, text_polys, scales)
    # the images are horizontally fliped and rotated in range [−10◦, 10◦] randomly
    # if random.random() < 0.5:
    #     im, text_polys = data_aug.horizontal_flip(im, text_polys)
    # if random.random() < 1:
    #     im, text_polys = data_aug.random_rotate_img_bbox(im, text_polys, degrees)

    # 640 × 640 random samples are cropped from the transformed images
    # im, text_polys = data_aug.random_crop_img_bboxes(im, text_polys)

    # im, text_polys = data_aug.resize(im, text_polys, input_size, keep_ratio=False)
    # im, text_polys = data_aug.random_crop_image_pse(im, text_polys, input_size)

    left = max(int(np.min(text_polys[:, :, 0])) - 64, 0)
    right = min(int(np.max(text_polys[:, :, 0])) + 64, im.shape[1])
    top = max(int(np.min(text_polys[:, :, 1])) - 64, 0)
    bottom = min(int(np.max(text_polys[:, :, 1])) + 64, im.shape[0])

    im = im[top:bottom, left:right]
    text_polys[:, :, 1] -= top
    text_polys[:, :, 0] -= left

    mid1 = int(text_polys.shape[0] / 2 + 0.5)
    mid2 = int(text_polys.shape[0] / 2 + 0.5)
    if len(text_polys) >= 4:
        choice_pos = random.random()
        # if epoch_count % 2 == 0:
        if choice_pos<0.5:
            fill_box = np.int0([text_polys[mid1 - 1, 3], text_polys[mid1 - 1, 2], [im.shape[1], im.shape[0]], [0, im.shape[0]]])
            cv2.fillPoly(im, [fill_box], 255)

            bottom = int(np.max(text_polys[mid1 - 1, :, 1]))

            im = im[:bottom, :, :]
            text_polys = text_polys[:mid1]
            text_label = text_label[:mid1]
            text_lengths = text_lengths[:mid1]
            label_strs = label_strs[:mid1]
        else:
            # cv2.fillPoly(im, [text_polys[mid2-1].astype(np.int)], 255)
            fill_box = np.int0([[0, 0], [im.shape[1], 0], text_polys[mid2, 1], text_polys[mid2, 0]])

            cv2.fillPoly(im, [fill_box], 255)
            top = int(np.min(text_polys[mid2, :, 1]))

            im = im[top:, :, :]
            text_polys = text_polys[mid2:]

            text_polys[:, :, 1] -= top
            text_label = text_label[mid2:]
            text_lengths = text_lengths[mid2:]
            label_strs = label_strs[mid2:]

    return im, text_polys, text_label, text_lengths, label_strs


def augmentation1(im: np.ndarray, text_polys: np.ndarray, text_label, text_lengths, label_strs, epoch_count) -> tuple:

    # the images are rescaled with ratio {0.5, 1.0, 2.0, 3.0} randomly
    # im, text_polys = data_aug.random_scale(im, text_polys, scales)
    # the images are horizontally fliped and rotated in range [−10◦, 10◦] randomly
    # if random.random() < 0.5:
    #     im, text_polys = data_aug.horizontal_flip(im, text_polys)
    # if random.random() < 1:
    #     im, text_polys = data_aug.random_rotate_img_bbox(im, text_polys, degrees)

    # 640 × 640 random samples are cropped from the transformed images
    # im, text_polys = data_aug.random_crop_img_bboxes(im, text_polys)

    # im, text_polys = data_aug.resize(im, text_polys, input_size, keep_ratio=False)
    # im, text_polys = data_aug.random_crop_image_pse(im, text_polys, input_size)
    # print(im.shape)
    left = max(int(np.min(text_polys[:, :, 0])) - 64, 0)
    right = min(int(np.max(text_polys[:, :, 0])) + 64, im.shape[1])
    top = max(int(np.min(text_polys[:, :, 1])) - 64, 0)
    bottom = min(int(np.max(text_polys[:, :, 1])) + 64, im.shape[0])

    im = im[top:bottom, left:right]
    text_polys[:, :, 1] -= top
    text_polys[:, :, 0] -= left

    mid1 = int(text_polys.shape[0] / 3 + 0.5)
    mid2 = int(text_polys.shape[0] / 3 * 2 + 0.5)
    if len(text_polys) >= 3:
        choice_pos = random.random()
        if epoch_count%3==0:
            fill_box = np.int0([text_polys[mid1 - 1, 3], text_polys[mid1 - 1, 2], [im.shape[1], im.shape[0]], [0, im.shape[0]]])
            cv2.fillPoly(im, [fill_box], 255)

            bottom = int(np.max(text_polys[mid1 - 1, :, 1]))

            im = im[:bottom, :, :]
            text_polys = text_polys[:mid1]
            text_label = text_label[:mid1]
            text_lengths = text_lengths[:mid1]
            label_strs = label_strs[:mid1]
        elif epoch_count%3==1:

            # cv2.fillPoly(im, text_polys[:mid1].astype(np.int), 255)
            # cv2.fillPoly(im, text_polys[mid2:].astype(np.int), 255)
            fill_box1 = np.int0([[0, 0], [im.shape[1], 0], text_polys[mid1, 1], text_polys[mid1, 0]])
            fill_box2 = np.int0([text_polys[mid2 - 1, 3], text_polys[mid2 - 1, 2], [im.shape[1], im.shape[0]], [0, im.shape[0]]])
            cv2.fillPoly(im, [fill_box1, fill_box2], 255)
            top = int(np.min(text_polys[mid1, :, 1]))
            bottom = int(np.max(text_polys[mid2 - 1, :, 1]))
            im = im[top:bottom, :, :]
            text_polys = text_polys[mid1:mid2]

            text_polys[:, :, 1] -= top
            text_label = text_label[mid1:mid2]
            text_lengths = text_lengths[mid1:mid2]
            label_strs = label_strs[mid1:mid2]
        else:
            # cv2.fillPoly(im, [text_polys[mid2-1].astype(np.int)], 255)
            fill_box = np.int0([[0, 0], [im.shape[1], 0], text_polys[mid2, 1], text_polys[mid2, 0]])

            cv2.fillPoly(im, [fill_box], 255)
            top = int(np.min(text_polys[mid2, :, 1]))

            im = im[top:, :, :]
            text_polys = text_polys[mid2:]

            text_polys[:, :, 1] -= top
            text_label = text_label[mid2:]
            text_lengths = text_lengths[mid2:]
            label_strs = label_strs[mid2:]

    return im, text_polys, text_label, text_lengths, label_strs


def PerspectiveTransform(img_np, text_polys, trans_rate_w=0.2, trans_rate_h=0.2):
    img_h, img_w = img_np.shape[0], img_np.shape[1]
    tmp_polys = np.ones((len(text_polys), 4, 3), dtype="float32")

    tmp_polys[:, :, :2] = text_polys
    tmp_polys = np.transpose(tmp_polys, (2, 0, 1))
    origin_box = np.array([[0, 0], [0, img_h], [img_w, img_h], [img_w, 0]], dtype="float32")
    trans_w_change = int(trans_rate_w * img_w)
    trans_h_change = int(trans_rate_h * img_h)
    if random.random() < 0.5:
        trans_box = np.array([
            [random.randint(0, trans_w_change), random.randint(0, trans_h_change)],
            [random.randint(0, trans_w_change), random.randint(img_h - trans_h_change, img_h)],
            [random.randint(img_w - trans_w_change, img_w), random.randint(img_h - trans_h_change, img_h)],
            [random.randint(img_w - trans_w_change, img_w), random.randint(0, trans_h_change)]
        ], dtype="float32")
    else:
        randomx1 = random.randint(0, trans_w_change * 2)
        randomx2 = random.randint(img_w - trans_w_change * 2, img_w)
        trans_box = np.array([
            [randomx1, random.randint(0, trans_h_change)],
            [randomx1, random.randint(img_h - trans_h_change, img_h)],
            [randomx2, random.randint(img_h - trans_h_change, img_h)],
            [randomx2, random.randint(0, trans_h_change)]
        ], dtype="float32")
    M = cv2.getPerspectiveTransform(origin_box, trans_box)
    tmp_polys = np.reshape(tmp_polys, (3, -1))
    tmp_polys = np.dot(M, tmp_polys)
    tmp_polys = np.reshape(tmp_polys, (3, len(text_polys), -1))
    tmp_polys = np.transpose(tmp_polys, (1, 2, 0))
    text_polys[:, :, 0] = tmp_polys[:, :, 0] / tmp_polys[:, :, 2]
    text_polys[:, :, 1] = tmp_polys[:, :, 1] / tmp_polys[:, :, 2]

    trans_img = cv2.warpPerspective(img_np, M, (img_w, img_h), borderValue=(255, 255, 255))
    return trans_img, text_polys


def image_label(im_fn: str, text_polys: np.ndarray, text_label: list, text_lengths, label_strs, n: int, m: float, input_size: int,
                defrees: int = 10,
                scales: np.ndarray = np.array([0.5, 1.5]), is_train=True, epoch_count=0) -> tuple:
    '''
    get image's corresponding matrix and ground truth
    return
    images [512, 512, 3]
    score  [128, 128, 1]
    geo    [128, 128, 5]
    mask   [128, 128, 1]
    '''

    im = cv2.imread(im_fn)

    # print(im_fn)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    h, w, _ = im.shape
    # 检查越界

    text_polys = check_and_validate_polys(text_polys, (h, w))


    h, w, _ = im.shape
    short_edge = max(h, w)
    # short_edge=w
    if short_edge > input_size:
        # 保证短边 >= inputsize
        scale = input_size / short_edge
        im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
        text_polys *= scale
    h, w, _ = im.shape
    if h > 1200 and is_train:
        # 保证短边 >= inputsize
        scale = 1200 / h
        im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
        text_polys *= scale

    if is_train and len(label_strs)>=14 and 'HWDB' not in im_fn:
        im, text_polys, text_label, text_lengths, label_strs = augmentation(im, text_polys,
                                                                            text_label,
                                                                            text_lengths,
                                                                            label_strs,
                                                                            epoch_count=epoch_count)
    if random.random() < 0.5 and is_train:
        im, text_polys = PerspectiveTransform(im, text_polys)
    elif random.random() < 0.7 and is_train:
        scale = random.uniform(0.7, 1)
        im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
        text_polys *= scale

    h, w, _ = im.shape
    training_mask = np.ones((h, w), dtype=np.uint8)
    score_maps = []
    for i in range(1, n + 1):
        # s1->sn,由小到大
        score_map, training_mask, new_text_polys = generate_rbox((h, w), text_polys, text_label, training_mask, i, n, m)
        score_maps.append(score_map)
        if i == n:
            text_polys = new_text_polys
    score_maps = np.array(score_maps, dtype=np.float32)

    imgs = data_aug.random_crop_author([im, score_maps.transpose((1, 2, 0)), training_mask], (input_size, input_size))
    return imgs[0], imgs[1].transpose((2, 0, 1)), imgs[2], text_polys, text_label, text_lengths, label_strs  # im,score_maps,training_mask#


class MyDataset(data.Dataset):
    def __init__(self, data_dirs, char_dict, data_shape: int = 640, n=6, m=0.5, transform=None, target_transform=None, max_text_length=80,
                 is_train=True, epoch_count=2):
        self.char_dict = char_dict
        self.data_shape = data_shape
        self.transform = transform
        self.target_transform = target_transform
        self.max_text_length = max_text_length
        self.n = n
        self.m = m
        self.is_train = is_train
        self.data_list = self.load_data(data_dirs)
        self.epoch_count = epoch_count

    def __getitem__(self, index):
        # print(self.image_list[index])
        img_path, text_polys, label_strs = self.data_list[index]
        label_tensors = []
        text_lengths = []

        for label in label_strs:

            label_tensor = torch.zeros(self.max_text_length, dtype=torch.long)
            for i, c_label in enumerate(label):
                label_tensor[i] = int(self.char_dict[c_label])
            label_tensors.append(label_tensor)
            text_lengths.append(len(label))
        label_tensors = torch.cat([t.unsqueeze(0) for t in label_tensors], 0)
        text_lengths = torch.tensor(text_lengths, dtype=torch.long)

        img, score_maps, training_mask, text_polys, label_tensors, text_lengths, label_strs = image_label(img_path,
                                                                                                          text_polys,
                                                                                                          label_tensors,
                                                                                                          text_lengths,
                                                                                                          label_strs,
                                                                                                          input_size=self.data_shape,
                                                                                                          n=self.n,
                                                                                                          m=self.m, is_train=self.is_train,
                                                                                                          epoch_count=self.epoch_count)
        # img = draw_bbox(img,text_polys)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            score_maps = self.target_transform(score_maps)
            training_mask = self.target_transform(training_mask)
        return img, torch.tensor(score_maps), training_mask, text_polys, label_tensors, text_lengths, label_strs

    def load_data(self, data_dirs: list) -> list:
        data_list = []
        for data_dir in data_dirs:
            for x in glob.glob(data_dir + '/page_imgs/*.png', recursive=True):

                d = pathlib.Path(x)
                label_path = os.path.join(data_dir, 'gt', ('gt_' + str(d.stem) + '.txt'))
                bboxs, label_strs = self._get_annotation(label_path)

                if len(bboxs) > 0:
                    data_list.append((x, bboxs, label_strs))
                else:
                    print('there is no suit bbox on {}'.format(label_path))

        return data_list

    def _get_annotation(self, label_path: str) -> tuple:
        boxes = []
        label_tensors = []
        text_length = []
        label_strs = []
        with open(label_path, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                params = line.strip('\n').split(' ')
                # try:
                label = params[8]
                if len(label) == 0:
                    continue

                # label_tensor = torch.zeros(self.max_text_length, dtype=torch.long)
                # for i, c_label in enumerate(label):
                #     label_tensor[i] = int(self.char_dict[c_label])

                # if label == '*' or label == '###':
                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, params[:8]))

                box = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int)
                box = sorted(box, key=lambda x: x[0])
                new_bbox = []
                new_bbox += sorted(box[:2], key=lambda x: x[1])
                new_bbox += sorted(box[2:], key=lambda x: -x[1])
                box = [new_bbox[0], new_bbox[3], new_bbox[2], new_bbox[1]]
                # print(box)
                if cv2.contourArea(np.int0(box)) < 2000:
                    continue
                # label_tensors.append(label_tensor)
                boxes.append(box)
                # print(label)
                label_strs.append(label)
                # text_length.append(len(label))
                # print(cv2.contourArea(box),len(label))
                # except:
                #     print('load label failed on {}'.format(label_path))

        # label_tensors = torch.cat([t.unsqueeze(0) for t in label_tensors], 0)
        # text_length = torch.tensor(text_length, dtype=torch.long)
        return np.array(boxes, dtype=np.float32), label_strs

    def __len__(self):
        return len(self.data_list)

    def save_label(self, img_path, label):
        save_path = img_path.replace('img', 'save')
        if not os.path.exists(os.path.split(save_path)[0]):
            os.makedirs(os.path.split(save_path)[0])
        img = draw_bbox(img_path, label)
        cv2.imwrite(save_path, img)
        return img


class AlignCollate(object):

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)

        imgs, score_maps, training_masks, text_polys, label_tensors, text_lengths, label_strs = zip(*batch)
        batch_size = len(imgs)
        img_channel = imgs[0].shape[0]
        max_h, max_w = 0, 0
        for img in imgs:
            img_h, img_w = img.shape[1:]
            max_h = max(img_h, max_h)
            max_w = max(img_w, max_w)
        imgs_tensor = torch.zeros((batch_size, img_channel, max_h, max_w), dtype=imgs[0].dtype)
        score_maps_tensor = torch.zeros((batch_size, max_h, max_w), dtype=score_maps[0].dtype)
        for batch_size_i in range(batch_size):
            img = imgs[batch_size_i]
            score_map = score_maps[batch_size_i][0]
            img_h, img_w = img.shape[1:]
            imgs_tensor[batch_size_i, :, :img_h, :img_w] = img
            score_maps_tensor[batch_size_i, :img_h, :img_w] = score_map

        return imgs_tensor, score_maps_tensor, text_polys, label_tensors, text_lengths, label_strs


if __name__ == '__main__':
    import torch

    from utils.utils import show_img
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from dataset.hwdb1_chars_dict import char_dict

    train_data = MyDataset(
        [
            # 'D:/git/OCR/handwritind_dect_reco/data/hwdb2/img_test',
            # 'D:/git/OCR/handwritind_dect_reco/data/hwdb2/HWDB2.0Test',
            # 'D:/git/OCR/handwritind_dect_reco/data/hwdb2/HWDB2.0Train',
            # 'D:/git/OCR/handwritind_dect_reco/data/hwdb2/HWDB2.1Test',
            # 'D:/git/OCR/handwritind_dect_reco/data/hwdb2/HWDB2.1Train',
            # 'D:/git/OCR/handwritind_dect_reco/data/hwdb2/HWDB2.2Test',
            # 'D:/git/OCR/handwritind_dect_reco/data/hwdb2/HWDB2.2Train',
            'D:/git/OCR/text_line_gen_un/page_gen_with_box/gen_for_icdar',
            # 'D:/git/OCR/handwritind_dect_reco/data/iam/iam_train',
            # r'F:\dataset\iam\iam_train_c'
            # r'F:\dataset\iam'
        ],
        char_dict,
        data_shape=1600, n=2, m=0.6,
        transform=transforms.ToTensor(), max_text_length=200, is_train=True)
    train_loader = DataLoader(dataset=train_data, collate_fn=AlignCollate(), batch_size=1, shuffle=True, num_workers=0)

    pbar = tqdm(total=len(train_loader))
    for i, (img, label, text_polys, label_tensors, text_lengths, label_str) in enumerate(train_loader):
        print(i)
        print(label.shape)
        print(img.shape)
        print(label_str)
        show_img((img[0] * label[0].to(torch.float)).numpy().transpose(1, 2, 0), color=True)
        show_img(label[0])
        img_cv2 = np.array(img[0] * 255, dtype=np.uint8)[0]

        cv2.imshow('1', img_cv2)
        for text_poly in text_polys[0]:
            text_poly = np.array(text_poly, dtype=np.int)

            cv2.drawContours(img_cv2, [text_poly], -1, 128, 1)
        # print(label_tensors)

        cv2.imshow('2', img_cv2)
        cv2.waitKey()

        plt.show()
        # break

    pbar.close()
