import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision.transforms.functional import perspective
import random


class Connect(nn.Module):
    def __init__(self, line_height):
        super(Connect, self).__init__()
        self.Line_Height = line_height
        self.tps = TPS()

    def forward(self, kernels: torch.Tensor, features: torch.Tensor, gt_boxes=None, is_train=True):
        """
        :param kernels: batch_size*H*W
        :param features: batch_size*C*H*W
        :param gt_boxes: List[List[List[int,int]]]
        :return text_kernel_features: List[N*C*line_height*max_text_length]
                text_lengths: List[List[int]]
        """
        assert (is_train == True and gt_boxes != None) or is_train == False

        sub_img_nums=[]
        text_kernel_features = []
        text_kernel_features_length = []
        line_top_lefts = []
        line_contours = []
        if gt_boxes is not None:
            sub_text_kernel_features = []
            max_text_kernel_length = 64
            for batch_size_i in range(len(features)):

                feature = features[batch_size_i]
                sub_img_nums.append(len(gt_boxes[batch_size_i]))

                _, H, W = feature.shape
                for gt_box in gt_boxes[batch_size_i]:
                    # print(gt_box)
                    gt_box = gt_box / 4
                    gt_box = np.array(gt_box)
                    trans_h = self.Line_Height

                    gt_box_w = max(((gt_box[0, 1] - gt_box[1, 1]) ** 2 + (gt_box[0, 0] - gt_box[1, 0]) ** 2) ** 0.5,
                                   ((gt_box[2, 1] - gt_box[3, 1]) ** 2 + (gt_box[2, 0] - gt_box[3, 0]) ** 2) ** 0.5)
                    gt_box_h = max(((gt_box[0, 1] - gt_box[3, 1]) ** 2 + (gt_box[0, 0] - gt_box[3, 0]) ** 2) ** 0.5,
                                   ((gt_box[2, 1] - gt_box[1, 1]) ** 2 + (gt_box[2, 0] - gt_box[1, 0]) ** 2) ** 0.5)
                    if gt_box_w <= 4 or gt_box_w <= 4:
                        sub_text_kernel_feature = torch.zeros(
                            (feature.shape[0], self.Line_Height, self.Line_Height),
                            dtype=feature.dtype, device=feature.device)
                    else:
                        origin_box = gt_box
                        if random.random() < 0.5 and gt_box_w > 6 * gt_box_h and is_train:
                            origin_box[0, 0] += gt_box_h * random.uniform(-2, 2)
                            origin_box[1, 0] += gt_box_h * random.uniform(-2, 2)
                            origin_box[2, 0] += gt_box_h * random.uniform(-2, 2)
                            origin_box[3, 0] += gt_box_h * random.uniform(-2, 2)

                            origin_box[0, 1] += gt_box_h * random.uniform(-0.2, 0.2)
                            origin_box[1, 1] += gt_box_h * random.uniform(-0.2, 0.2)
                            origin_box[2, 1] += gt_box_h * random.uniform(-0.2, 0.2)
                            origin_box[3, 1] += gt_box_h * random.uniform(-0.2, 0.2)

                        trans_box = np.array([[0, 0], [gt_box_w, 0], [gt_box_w, gt_box_h], [0, gt_box_h]])

                        if random.random() < 0.5 and gt_box_w > 6 * gt_box_h and is_train:
                            trans_w_change = 0.1 * gt_box_w
                            trans_h_change = 0.2 * gt_box_h
                            if random.random() < 0.5:
                                trans_box = np.array([
                                    [random.uniform(0, trans_w_change), random.uniform(0, trans_h_change)],
                                    [random.uniform(gt_box_w - trans_w_change, gt_box_w), random.uniform(0, trans_h_change)],
                                    [random.uniform(gt_box_w - trans_w_change, gt_box_w),
                                     random.uniform(gt_box_h - trans_h_change, gt_box_h)],

                                    [random.uniform(0, trans_w_change), random.uniform(gt_box_h - trans_h_change, gt_box_h)]
                                ], dtype="float32")
                            else:
                                randomx1 = random.uniform(0, trans_w_change * 2)
                                randomx2 = random.uniform(gt_box_w - trans_w_change * 2, gt_box_w)
                                trans_box = np.array([
                                    [randomx1, random.uniform(0, trans_h_change)],
                                    [randomx2, random.uniform(0, trans_h_change)],
                                    [randomx2, random.uniform(gt_box_h - trans_h_change, gt_box_h)],
                                    [randomx1, random.uniform(gt_box_h - trans_h_change, gt_box_h)]
                                ], dtype="float32")
                        # trans_box[:,  0] = np.clip(trans_box[:, 0], 0, W - 1)
                        # trans_box[:,  1] = np.clip(trans_box[:, 1], 0, H - 1)

                        sub_text_kernel_feature = perspective(feature, origin_box, trans_box)[:, :int(gt_box_h), :int(gt_box_w)]
                        if is_train:

                            interpolate_w = int(sub_text_kernel_feature.shape[-1] * trans_h / gt_box_h * random.uniform(0.8, 1.2)) + 2
                        else:
                            interpolate_w = int(sub_text_kernel_feature.shape[-1] * trans_h / gt_box_h)

                        interpolate_w = min(1200, interpolate_w)
                        sub_text_kernel_feature = F.interpolate(sub_text_kernel_feature.unsqueeze(0),
                                                                (trans_h, interpolate_w),
                                                                mode='bilinear',
                                                                align_corners=True)
                        # sub_text_kernel_feature = F.upsample_bilinear(sub_text_kernel_feature.unsqueeze(0), (trans_h, interpolate_w))
                        max_text_kernel_length = max(max_text_kernel_length, interpolate_w)
                    # text_kernel_features_length.append(interpolate_w // 4)
                    sub_text_kernel_features.append(torch.squeeze(sub_text_kernel_feature, 0))
            text_kernel_tensor_features = torch.zeros(
                (sum(sub_img_nums), sub_text_kernel_features[0].shape[0], self.Line_Height, max_text_kernel_length+32),
                dtype=features[0].dtype, device=features[0].device)
            for sub_i,sub_text_kernel_tensor_feature in enumerate(sub_text_kernel_features):
                text_kernel_tensor_features[sub_i,:,:,16:sub_text_kernel_tensor_feature.shape[-1]+16]=sub_text_kernel_tensor_feature
            return text_kernel_tensor_features,sub_img_nums


        else:
            max_text_kernel_length = 0
            line_trans_list = []
            sub_img_nums=[]
            for kernel, feature in zip(kernels, features):
                kernel = torch.sigmoid(kernel)
                kernel = kernel.detach().cpu().numpy()
                kernel = (kernel * 255).astype(np.uint8)

                _, binary = cv2.threshold(kernel, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contours = sorted(contours, key=lambda x: np.min(x[:, :, 0]))

                line_count = 0


                line_top_left = []
                sub_line_contour = []
                for contour in contours:
                    contour_area = cv2.contourArea(contour)
                    contour = np.array(contour)

                    if contour_area > 30:
                        contour_top = np.min(contour[:, :, 0])
                        contour_left = np.min(contour[:, :, 1])

                        line_top_left.append([contour_top, contour_left])

                        trans_points, trans_widths, trans_heights = get_trans_points(contour)
                        approx = cv2.approxPolyDP(contour, max(trans_heights) * 0.2, True)
                        sub_line_contour.append(approx)
                        try:
                            sub_line_trans = trans_line(trans_points, trans_widths, trans_heights, feature, self.tps, self.Line_Height)

                        except:
                            sub_line_trans = torch.zeros((1, 512, self.Line_Height, self.Line_Height), dtype=feature.dtype,
                                                         device=feature.device)

                        line_trans_list.append(sub_line_trans)
                        max_text_kernel_length = max(max_text_kernel_length, sub_line_trans.shape[3])

                        line_count += 1

                # line_tensors = torch.zeros((line_count, 512, self.Line_Height, max_width + 64), dtype=feature.dtype, device=feature.device)
                # for i in range(line_count):
                #     sub_line_trans = line_trans_list[i]
                #
                #     line_tensors[i, :, :, 32:sub_line_trans.shape[-1] + 32] = sub_line_trans

                # text_kernel_features.append(line_tensors)
                sub_img_nums.append(len(sub_line_contour))
                line_top_lefts.append(line_top_left)
                line_contours.append(sub_line_contour)
            text_kernel_tensor_features = torch.zeros(
                (sum(sub_img_nums), line_trans_list[0].shape[1], self.Line_Height, max_text_kernel_length+32),
                dtype=features[0].dtype, device=features[0].device)
            for sub_i,sub_line_trans in enumerate(line_trans_list):
                # print(text_kernel_tensor_features.shape,sub_line_trans.shape)
                text_kernel_tensor_features[sub_i,:,:,16:sub_line_trans.shape[-1]+16]=sub_line_trans

            return text_kernel_tensor_features,sub_img_nums, line_top_lefts, line_contours


class TPS(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X, Y, w, h):
        """ 计算grid"""
        device = X.device
        grid = torch.ones(1, h, w, 2, device=device)
        grid[:, :, :, 0] = torch.linspace(-1, 1, w)
        grid[:, :, :, 1] = torch.linspace(-1, 1, h)[..., None]
        grid = grid.view(-1, h * w, 2)

        """ 计算W, A"""
        n, k = X.shape[:2]
        device = X.device

        Z = torch.zeros(1, k + 3, 2, device=device)
        P = torch.ones(n, k, 3, device=device)
        L = torch.zeros(n, k + 3, k + 3, device=device)

        eps = 1e-9
        D2 = torch.pow(X[:, :, None, :] - X[:, None, :, :], 2).sum(-1)
        K = D2 * torch.log(D2 + eps)

        P[:, :, 1:] = X
        Z[:, :k, :] = Y
        L[:, :k, :k] = K
        L[:, :k, k:] = P
        L[:, k:, :k] = P.permute(0, 2, 1)

        Q = torch.solve(Z, L)[0]
        W, A = Q[:, :k], Q[:, k:]

        """ 计算U """
        eps = 1e-9
        D2 = torch.pow(grid[:, :, None, :] - X[:, None, :, :], 2).sum(-1)
        U = D2 * torch.log(D2 + eps)

        """ 计算P """
        n, k = grid.shape[:2]
        device = grid.device
        P = torch.ones(n, k, 3, device=device)
        P[:, :, 1:] = grid

        # grid = P @ A + U @ W
        grid = torch.matmul(P, A) + torch.matmul(U, W)
        return grid.view(-1, h, w, 2)


def point_norm(points_int, width, height, device):
    """
    将像素点坐标归一化至 -1 ~ 1
    """
    points_int_clone = torch.tensor(points_int, device=device)
    x = ((points_int_clone * 2)[..., 0] / (width - 1) - 1)
    y = ((points_int_clone * 2)[..., 1] / (height - 1) - 1)
    return torch.stack([x, y], dim=-1).contiguous().view(-1, 2)


def trans_line(kernel_points, box_widths, box_heights, feature, tps, out_height):

    img_h, img_w = feature.shape[1:]
    kernel_points = np.array(kernel_points)
    kernel_points_c = np.reshape(kernel_points.copy(), (-1, 1, 2))
    box_heights = np.array(box_heights)
    max_box_height = int(max(box_heights))
    rect_box = cv2.boundingRect(kernel_points_c)

    margin_x, margin_y, margin_w, margin_h = rect_box
    margin_x = max(0, margin_x)
    margin_y = max(0, margin_y)
    margin_w = min(margin_w, img_w - margin_x)
    margin_h = min(margin_h, img_h - margin_y)

    line_feature = feature[:, margin_y:margin_y + margin_h, margin_x:margin_x + margin_w].clone()
    if int(sum(box_widths) - margin_w) > 0:
        line_feature = torch.dstack([line_feature,
                                     torch.zeros((line_feature.shape[0],
                                                  line_feature.shape[1],
                                                  int(sum(box_widths) - margin_w)),
                                                 dtype=line_feature.dtype,
                                                 device=line_feature.device)])

    kernel_points[:, :, 0] -= margin_x
    kernel_points[:, :, 1] -= margin_y

    point_source = []
    point_target = []

    line_h, line_w = line_feature.shape[1:]
    sum_w = 0

    for i in range(len(kernel_points)):
        point1 = kernel_points[i, 0]
        point2 = kernel_points[i, 1]
        if i > 0:
            sum_w += box_widths[i - 1]
        box_height = box_heights[i]

        sub_margin = int((margin_h - box_height) // 2)
        point_source.append(point1)
        point_source.append(point2)
        point_target.append([int(sum_w), 0])
        point_target.append([int(sum_w), max_box_height - 1])
    point_source = np.array([point_source])
    point_target = np.array([point_target])
    point_source = point_norm(point_source, line_w, line_h, device=feature.device)
    point_target = point_norm(point_target, line_w, line_h, device=feature.device)
    warped_grid = tps(point_target[None, ...], point_source[None, ...], line_w, line_h)  # 这个输入的位置需要归一化，所以用norm

    ten_wrp = F.grid_sample(line_feature[None, ...], warped_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    max_marign = int((margin_h - max_box_height) / 2)
    ten_wrp = ten_wrp[:, :, :max_box_height, :]
    size_h = out_height
    size_w = int(line_w * out_height / max_box_height*1)

    ten_wrp = F.interpolate(ten_wrp, size=(size_h, size_w),mode='bilinear',align_corners=True)
    return ten_wrp


def get_trans_points(contour):
    contour_area = cv2.contourArea(contour)
    epsilon = 0.2 * contour_area / cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)
    center_points = get_center_points(contour)
    trans_points = []
    trans_widths = []
    trans_heights = []
    pre_point_left1 = None
    pre_point_left2 = None
    if center_points[0][0] > center_points[-1][0]:
        center_points = center_points[::-1]
    for point_i, kernel_point in enumerate(center_points):
        if point_i < len(center_points) - 1:
            kernel_point_next = center_points[point_i + 1]
            if point_i > 0:
                _, _, pre_r = center_points[point_i - 1]
            else:
                pre_r = 0
            x_left, y_left, left_r = kernel_point
            # print(kernel_point_next)
            x_right, y_right, right_r = kernel_point_next

            center_x, center_y = (x_right + x_left) / 2, (y_left + y_right) / 2
            box_width = ((x_left - x_right) ** 2 + (y_right - y_left) ** 2) ** 0.5
            if x_left==x_right:
                angle=0
            else:
                angle = np.arctan((y_right - y_left) / (x_right - x_left)) * 180 / np.pi
            box_height = max(left_r, right_r, pre_r)

            rect_box = ((center_x, center_y), (box_width, box_height), angle)
            point_box = cv2.boxPoints(rect_box)
            point_box = np.int0(point_box)
            if point_i > 0:
                point_box[1][0] = int((point_box[1][0] + pre_point_left1[0]) / 2)
                point_box[1][1] = int((point_box[1][1] + pre_point_left1[1]) / 2)
                point_box[0][0] = int((point_box[0][0] + pre_point_left2[0]) / 2)
                point_box[0][1] = int((point_box[0][1] + pre_point_left2[1]) / 2)
            trans_points.append([point_box[1], point_box[0]])
            trans_widths.append(box_width)
            trans_heights.append(box_height)

            if point_i == len(center_points) - 2:
                trans_points.append([point_box[2], point_box[3]])

                trans_heights.append(box_height)
            pre_point_left1 = point_box[2]
            pre_point_left2 = point_box[3]
    return trans_points, trans_widths, trans_heights


def get_center_points(contour):
    center_points = []
    contour_area = cv2.contourArea(contour)
    contour_length = cv2.arcLength(contour, True)
    min_height = contour_area / contour_length
    x_min, x_max, y_min, y_max = np.min(contour[:, :, 0]), np.max(contour[:, :, 0]), np.min(contour[:, :, 1]), np.max(contour[:, :, 1])
    raw_dist = np.zeros((y_max - y_min, x_max - x_min), dtype=np.float32)

    for i in range(y_min, y_max, 1):
        for j in range(x_min, x_max, 1):
            raw_dist[i - y_min, j - x_min] = cv2.pointPolygonTest(contour, (j, i), True)

    minVal, maxVal, _, maxDistPt = cv2.minMaxLoc(raw_dist)

    while maxVal > min_height * 0.5:
        # print(maxVal, min_height)
        point_x = (maxDistPt[0] + x_min)
        point_y = (maxDistPt[1] + y_min)

        if len(center_points) < 2:
            center_points.append([point_x, point_y, max(maxVal, min_height) * 4])

        else:
            left_x, left_y, _ = center_points[0]
            right_x, right_y, _ = center_points[-1]
            left_w = ((point_x - left_x) ** 2 + (point_y - left_y) ** 2) ** 0.5
            right_w = ((point_x - right_x) ** 2 + (point_y - right_y) ** 2) ** 0.5
            left_right_w = ((left_x - right_x) ** 2 + (left_y - right_y) ** 2) ** 0.5

            if right_w > left_right_w and right_w > left_w:

                center_points.insert(0, [point_x, point_y, max(maxVal, min_height) * 4])
            elif left_w > left_right_w and left_w > right_w:

                center_points.insert(len(center_points), [point_x, point_y, max(maxVal, min_height) * 4])

            else:
                min_sum = 99999
                min_i = 0
                for center_point_i, center_point in enumerate(center_points[:-1]):
                    left_x, left_y, _ = center_point
                    right_x, right_y, _ = center_points[center_point_i + 1]
                    left_w = ((point_x - left_x) ** 2 + (point_y - left_y) ** 2) ** 0.5
                    right_w = ((point_x - right_x) ** 2 + (point_y - right_y) ** 2) ** 0.5
                    left_right_w = ((left_x - right_x) ** 2 + (left_y - right_y) ** 2) ** 0.5
                    sum_w = left_w + right_w
                    if sum_w < min_sum and left_w < left_right_w and right_w < left_right_w:
                        min_sum = sum_w
                        min_i = center_point_i
                if min_sum > 4 * min_height:
                    center_points.insert(min_i + 1, [point_x, point_y, max(maxVal, min_height) * 4])

        circle_x, circle_y = maxDistPt
        pad_y_min = max(int(circle_y - 4 * min_height), 0)
        pad_y_max = min(int(circle_y + 4 * min_height), raw_dist.shape[0])

        pad_x_min = max(int(circle_x - 4 * min_height), 0)
        pad_x_max = min(int(circle_x + 4 * min_height), raw_dist.shape[1])
        raw_dist[pad_y_min:pad_y_max, pad_x_min:pad_x_max] = 0
        minVal, maxVal, _, maxDistPt = cv2.minMaxLoc(raw_dist)
        #
        # cv2.imshow('raw_dist', kernel_big)
        # cv2.waitKey()
    if len(center_points) == 1:
        point_x, point_y, point_r = center_points[0]
        point_left_x, point_left_y, point_left_r = point_x - point_r / 2, point_y, point_r
        point_right_x, point_right_y, point_right_r = point_x + point_r / 2, point_y, point_r

        point_left_x = int(max(0, point_left_x))
        point_left_y = int(max(0, point_left_y))
        point_right_x = int(min(x_max, point_right_x))
        point_right_y = int(min(y_max, point_right_y))
        center_points.insert(0, [point_left_x, point_left_y, point_left_r])
        center_points.append([point_right_x, point_right_y, point_right_r])

    left_x1, left_y1, left_r1 = center_points[0]
    left_x2, left_y2, left_r2 = center_points[1]
    left_margin = cv2.pointPolygonTest(contour, (left_x1, left_y1), True)
    if left_margin > 0:
        point_out = get_out_point(left_x1, left_y1, left_x2, left_y2, contour, left_r1)
        center_points.insert(0, point_out)
    right_x1, right_y1, right_r1 = center_points[-1]
    right_x2, right_y2, right_r2 = center_points[-2]
    right_margin = cv2.pointPolygonTest(contour, (right_x1, right_y1), True)
    if right_margin > 0:
        point_out = get_out_point(right_x1, right_y1, right_x2, right_y2, contour, right_r1)
        center_points.append(point_out)

    return center_points


def get_out_point(x1, y1, x2, y2, contour, r):
    point_w = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    margin = 0.1 * r
    while True:
        out_rate = (margin) / point_w

        out_x = int((out_rate + 1) * x1 - out_rate * x2)
        out_y = int((out_rate + 1) * y1 - out_rate * y2)

        margin_now = cv2.pointPolygonTest(contour, (out_x, out_y), True)

        if margin_now < 0:
            break
        margin += 0.1 * r
    out_rate = (margin + r / 2) / point_w
    out_x = int((out_rate + 1) * x1 - out_rate * x2)
    out_y = int((out_rate + 1) * y1 - out_rate * y2)
    return [int(out_x), int(out_y), r]
# if __name__ == '__main__':
#     import numpy as np
#     import cv2
#
#     connect_layer = Connect(32)
#     img = cv2.imread('../data/hwdb2/HWDB2.1Test/page_imgs/1.png')[:, :, 0]
#     print(img.shape)
#     kernels = torch.tensor(img).unsqueeze(0)
#     features = torch.tensor(img).unsqueeze(0).unsqueeze(0) / 255.0
#     boxes = torch.tensor([[[[242, 25], [1726, 16], [1727, 125], [242, 135]]]]) * 4
#     while True:
#         text_kernel_features = connect_layer(kernels, features, boxes, True)
#         pt_kernel = text_kernel_features[0][0, 0]
#         pt_kernel = np.array(pt_kernel * 255, dtype=np.uint8)
#         print(pt_kernel.shape)
#         cv2.imshow('1', img)
#         cv2.imshow('2', pt_kernel)
#         cv2.waitKey()
