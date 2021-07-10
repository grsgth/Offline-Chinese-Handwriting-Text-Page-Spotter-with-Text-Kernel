import difflib

import torch
from skimage.draw import polygon

# from dataset.hwdb1_chars_dict import char_set
import numpy as np

def remove_blank(labels, blank=0):
    new_labels = []

    # combine duplicate
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l

    # remove blank
    new_labels = [l for l in new_labels if l != blank]

    return new_labels

def get_pred_str(prediction_char,char_set):
    blank = 0
    label_pred = []

    preds_softmax = torch.argmax(prediction_char, -1)

    for i in range(len(preds_softmax)):
        pred_softmax = preds_softmax[i]
        pred_softmax = pred_softmax.cpu().numpy()
        pred_s = remove_blank(pred_softmax)
        pred_str = ''
        for ci in pred_s:
            if ci != blank and ci < len(char_set):
                pred_str += char_set[ci]
        label_pred.append(pred_str)


    return label_pred


def get_ar_cr(pred_str, label_str):

    pred_str = pred_str.replace('”', '\"')
    pred_str = pred_str.replace('“', '\"')
    pred_str = pred_str.replace('‘', '\'')
    pred_str = pred_str.replace('’', '\'')
    pred_str = pred_str.replace('—', '-')
    pred_str = pred_str.replace('―', '-')
    pred_str = pred_str.replace('`', '\'')
    pred_str = pred_str.replace('，', ',')

    label_str = label_str.replace('”', '\"')
    label_str = label_str.replace('“', '\"')
    label_str = label_str.replace('‘', '\'')
    label_str = label_str.replace('’', '\'')
    label_str = label_str.replace('—', '-')
    label_str = label_str.replace('―', '-')
    label_str = label_str.replace('`', '\'')
    label_str = label_str.replace('，', ',')
    CR_correct_char = max(len(label_str), len(pred_str))
    AR_correct_char = max(len(label_str), len(pred_str))
    All_char = max(len(label_str), len(pred_str))

    for block in difflib.SequenceMatcher(None, label_str, pred_str).get_opcodes():
        label_m = block[2] - block[1]
        pred_m = block[4] - block[3]
        if block[0] == 'delete':
            CR_correct_char -= max(label_m, pred_m)
            AR_correct_char -= max(label_m, pred_m)

        elif block[0] == 'insert':
            AR_correct_char -= max(label_m, pred_m)

        elif block[0] == 'replace':

            CR_correct_char -= label_m

            AR_correct_char -= max(pred_m, label_m)
    return CR_correct_char, AR_correct_char, All_char

def normal_leven(str1, str2):
    len_str1 = len(str1) + 1
    len_str2 = len(str2) + 1
    # 创建矩阵
    matrix = [0 for n in range(len_str1 * len_str2)]
    # 矩阵的第一行
    for i in range(len_str1):
        matrix[i] = i
    # 矩阵的第一列
    for j in range(0, len(matrix), len_str1):
        if j % len_str1 == 0:
            matrix[j] = j // len_str1
    # 根据状态转移方程逐步得到编辑距离
    for i in range(1, len_str1):
        for j in range(1, len_str2):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            matrix[j * len_str1 + i] = min(matrix[(j - 1) * len_str1 + i] + 1,
                                           matrix[j * len_str1 + (i - 1)] + 1,
                                           matrix[(j - 1) * len_str1 + (i - 1)] + cost)

    return matrix[-1]  # 返回矩阵的最后一个值，也就是编辑距离

def polygon_IOU(polygon_1, polygon_2):
    """
    计算两个多边形的IOU
    :param polygon_1: [[row1, col1], [row2, col2], ...]
    :param polygon_2: 同上
    :return:
    """
    rr1, cc1 = polygon(polygon_2[:, 0], polygon_2[:, 1])
    rr2, cc2 = polygon(polygon_1[:, 0], polygon_1[:, 1])

    try:
        r_max = max(rr1.max(), rr2.max()) + 1
        c_max = max(cc1.max(), cc2.max()) + 1
    except:
        return 0

    canvas = np.zeros((r_max, c_max))

    canvas[rr2, cc2] += 1
    union = np.sum(canvas > 0)
    canvas[rr1, cc1] += 1

    if union == 0:
        return 0
    intersection = np.sum(canvas == 2)

import difflib

import torch
from skimage.draw import polygon

# from dataset.hwdb1_chars_dict import char_set
import numpy as np

def remove_blank(labels, blank=0):
    new_labels = []

    # combine duplicate
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l

    # remove blank
    new_labels = [l for l in new_labels if l != blank]

    return new_labels

def get_pred_str(prediction_char,char_set):
    blank = 0
    label_pred = []

    preds_softmax = torch.argmax(prediction_char, -1)

    for i in range(len(preds_softmax)):
        pred_softmax = preds_softmax[i]
        pred_softmax = pred_softmax.cpu().numpy()
        pred_s = remove_blank(pred_softmax)
        pred_str = ''
        for ci in pred_s:
            if ci != blank and ci < len(char_set):
                pred_str += char_set[ci]
        label_pred.append(pred_str)


    return label_pred


def get_ar_cr(pred_str, label_str):

    pred_str = pred_str.replace('”', '\"')
    pred_str = pred_str.replace('“', '\"')
    pred_str = pred_str.replace('‘', '\'')
    pred_str = pred_str.replace('’', '\'')
    pred_str = pred_str.replace('—', '-')
    pred_str = pred_str.replace('―', '-')
    pred_str = pred_str.replace('`', '\'')
    pred_str = pred_str.replace('，', ',')

    label_str = label_str.replace('”', '\"')
    label_str = label_str.replace('“', '\"')
    label_str = label_str.replace('‘', '\'')
    label_str = label_str.replace('’', '\'')
    label_str = label_str.replace('—', '-')
    label_str = label_str.replace('―', '-')
    label_str = label_str.replace('`', '\'')
    label_str = label_str.replace('，', ',')
    CR_correct_char = max(len(label_str), len(pred_str))
    AR_correct_char = max(len(label_str), len(pred_str))
    All_char = max(len(label_str), len(pred_str))

    for block in difflib.SequenceMatcher(None, label_str, pred_str).get_opcodes():
        label_m = block[2] - block[1]
        pred_m = block[4] - block[3]
        if block[0] == 'delete':
            CR_correct_char -= max(label_m, pred_m)
            AR_correct_char -= max(label_m, pred_m)

        elif block[0] == 'insert':
            AR_correct_char -= max(label_m, pred_m)

        elif block[0] == 'replace':

            CR_correct_char -= label_m

            AR_correct_char -= max(pred_m, label_m)
    return CR_correct_char, AR_correct_char, All_char

def normal_leven(str1, str2):
    len_str1 = len(str1) + 1
    len_str2 = len(str2) + 1
    # 创建矩阵
    matrix = [0 for n in range(len_str1 * len_str2)]
    # 矩阵的第一行
    for i in range(len_str1):
        matrix[i] = i
    # 矩阵的第一列
    for j in range(0, len(matrix), len_str1):
        if j % len_str1 == 0:
            matrix[j] = j // len_str1
    # 根据状态转移方程逐步得到编辑距离
    for i in range(1, len_str1):
        for j in range(1, len_str2):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            matrix[j * len_str1 + i] = min(matrix[(j - 1) * len_str1 + i] + 1,
                                           matrix[j * len_str1 + (i - 1)] + 1,
                                           matrix[(j - 1) * len_str1 + (i - 1)] + cost)

    return matrix[-1]  # 返回矩阵的最后一个值，也就是编辑距离

def polygon_IOU(polygon_1, polygon_2):
    """
    计算两个多边形的IOU
    :param polygon_1: [[row1, col1], [row2, col2], ...]
    :param polygon_2: 同上
    :return:
    """
    rr1, cc1 = polygon(polygon_2[:, 0], polygon_2[:, 1])
    rr2, cc2 = polygon(polygon_1[:, 0], polygon_1[:, 1])

    try:
        r_max = max(rr1.max(), rr2.max()) + 1
        c_max = max(cc1.max(), cc2.max()) + 1
    except:
        return 0

    canvas = np.zeros((r_max, c_max))

    canvas[rr2, cc2] += 1
    union = np.sum(canvas > 0)
    canvas[rr1, cc1] += 1

    if union == 0:
        return 0
    intersection = np.sum(canvas == 2)

    return intersection / union