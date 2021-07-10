<<<<<<< HEAD
import numpy as np
import cv2
import struct
import random
from torchvision.transforms import ToTensor, ToPILImage
img_transform = ToTensor()
def get_pred_data(file_paths,width=1600):

    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            header_size = np.fromfile(f, dtype='uint32', count=1)[0]
            header = np.fromfile(f, dtype='uint8', count=header_size - 4)
            formatcode = "".join([chr(c) for c in header[:8]])
            Illustration_size = header_size - 36
            Illustration = "".join([chr(c) for c in header[8:Illustration_size + 8]])
            Code_type = "".join([chr(c) for c in header[Illustration_size + 8:Illustration_size + 28]])
            Code_length = header[Illustration_size + 28] + header[Illustration_size + 29] << 4
            Bits_per_pixel = header[Illustration_size + 30] + header[Illustration_size + 31] << 4
            # print(header_size, formatcode, Illustration)
            # print(Code_type, Code_length, Bits_per_pixel)
            # print()
            Image_height = np.fromfile(f, dtype='uint32', count=1)[0]
            Image_width = np.fromfile(f, dtype='uint32', count=1)[0]
            Line_number = np.fromfile(f, dtype='uint32', count=1)[0]
            page_np = np.ones((Image_height * 4, Image_width), dtype=np.uint8) * 255
            page_label = []
            boxes = []
            Y1 = 0
            Y2 = 0
            margin = 0
            for ln in range(Line_number):
                Char_number = np.fromfile(f, dtype='uint32', count=1)[0]
                Label = np.fromfile(f, dtype='uint16', count=Char_number)
                # print(Label)
                Label_str = "".join([struct.pack('H', c).decode('GBK', errors='ignore') for c in Label])
                # print(Label_str, Char_number)
                Top_left = np.fromfile(f, dtype='uint32', count=2)
                Top, Left = Top_left[0], Top_left[1]

                Height = np.fromfile(f, dtype='uint32', count=1)[0]


                # Top+=ln*Image_height//Line_number//8
                Width = np.fromfile(f, dtype='uint32', count=1)[0]
                Bitmap = np.fromfile(f, dtype='uint8', count=Height * Width).reshape([Height, Width])
                contours, hierarchy = cv2.findContours(
                    255 - Bitmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # if random.random()<0.5:
                #     Top+=random.uniform(-0.2,0.2)*Height
                #     Top = int(Top)
                all_contours = []
                for contour in contours:
                    for points in contour:
                        all_contours.append(points)
                all_contours = np.array(all_contours)
                rect = cv2.minAreaRect(all_contours)
                rect_w = max(rect[1])
                rect_h = min(rect[1])


                # Top-=int(ln*Image_height//Line_number//10)
                if rect_w < Image_width * 0.25:
                    x1, y1, x2, y2 = cv2.boundingRect(all_contours)
                    bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                else:
                    bbox = cv2.boxPoints(rect)
                bbox = sorted(bbox, key=lambda x: x[0])
                new_bbox = []
                new_bbox += sorted(bbox[:2], key=lambda x: x[1])
                new_bbox += sorted(bbox[2:], key=lambda x: -x[1])
                bbox = [new_bbox[0], new_bbox[3], new_bbox[2], new_bbox[1]]

                # left_w = random.uniform(-1, 1) * rect_h
                # right_w = random.uniform(-1, 1) * rect_h
                # bbox[0][0] += left_w
                # bbox[1][0] += right_w
                # bbox[2][0] += right_w
                # bbox[3][0] += left_w
                # top_h=random.uniform(-0.2, 0.2) * rect_h
                # bottom_h = random.uniform(-0.2, 0.2) * rect_h
                # bbox[0][1] += top_h
                # bbox[1][1] += top_h
                # bbox[2][1] += bottom_h
                # bbox[3][1] += bottom_h

                bbox = np.int0(bbox)

                bbox[:, 0] += Left
                bbox[:, 1] += Top

                origin_sub = page_np[Top:Top + Height, Left:Left + Width]
                page_np[Top:Top + Height, Left:Left + Width] = (origin_sub > Bitmap) * Bitmap + (origin_sub <= Bitmap) * origin_sub
                if ln == 0:
                    Y1 = max(Top - 64, 0)
                if ln == Line_number - 1:
                    Y2 = Top + Height
                # cv2.drawContours(page_np, [bbox], -1, 128, 2)
                # cv2.imshow('1', cv2.resize(page_np[Y1:, :],dsize=None,fx=0.5,fy=0.5))
                # cv2.waitKey()
                bbox[:, 1] -= Y1
                boxes.append(bbox)
                Label_str = Label_str.replace('\x00', '')
                Label_str = Label_str.replace('〔', '(')
                Label_str = Label_str.replace('〕', ')')
                Label_str = Label_str.replace('＂', '"')
                Label_str = Label_str.replace('％', '%')
                Label_str = Label_str.replace('（', '(')
                Label_str = Label_str.replace('）', ')')
                Label_str = Label_str.replace('，', ',')
                Label_str = Label_str.replace('－', '-')
                Label_str = Label_str.replace('．', '.')
                Label_str = Label_str.replace('／', '/')
                Label_str = Label_str.replace('０', '0')
                Label_str = Label_str.replace('１', '1')
                Label_str = Label_str.replace('２', '2')
                Label_str = Label_str.replace('３', '3')
                Label_str = Label_str.replace('４', '4')
                Label_str = Label_str.replace('５', '5')
                Label_str = Label_str.replace('６', '6')
                Label_str = Label_str.replace('７', '7')
                Label_str = Label_str.replace('８', '8')
                Label_str = Label_str.replace('９', '9')
                Label_str = Label_str.replace('：', ':')
                Label_str = Label_str.replace('；', ';')
                Label_str = Label_str.replace('？', '?')
                Label_str = Label_str.replace('Ａ', 'A')
                Label_str = Label_str.replace('Ｂ', 'B')
                Label_str = Label_str.replace('Ｃ', 'C')
                Label_str = Label_str.replace('Ｆ', 'F')
                Label_str = Label_str.replace('Ｇ', 'G')
                Label_str = Label_str.replace('Ｈ', 'H')
                Label_str = Label_str.replace('Ｍ', 'M')
                Label_str = Label_str.replace('Ｎ', 'N')
                Label_str = Label_str.replace('Ｏ', 'O')
                Label_str = Label_str.replace('Ｐ', 'P')
                Label_str = Label_str.replace('Ｒ', 'R')
                Label_str = Label_str.replace('Ｓ', 'S')
                Label_str = Label_str.replace('Ｖ', 'V')
                Label_str = Label_str.replace('Ｗ', 'W')
                Label_str = Label_str.replace('ａ', 'a')
                Label_str = Label_str.replace('ｄ', 'd')
                Label_str = Label_str.replace('ｅ', 'e')
                Label_str = Label_str.replace('ｈ', 'h')
                Label_str = Label_str.replace('ｉ', 'i')
                Label_str = Label_str.replace('ｌ', 'l')
                Label_str = Label_str.replace('ｍ', 'm')
                Label_str = Label_str.replace('ｎ', 'n')
                Label_str = Label_str.replace('ｏ', 'o')
                Label_str = Label_str.replace('ｐ', 'p')
                Label_str = Label_str.replace('ｒ', 'r')
                Label_str = Label_str.replace('ｓ', 's')
                Label_str = Label_str.replace('ｔ', 't')
                Label_str = Label_str.replace('ｕ', 'u')
                Label_str = Label_str.replace('ｙ', 'y')
                page_label.append(Label_str)
        Y2 = min(Image_height * 4, Y2 + 64)
        img_np = page_np[Y1:Y2, :]
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        boxes = np.array(boxes, dtype=np.float)
        h, w, _ = img_np.shape
        short_edge = max(h,w)
        if short_edge > width:
            # 保证短边 >= inputsize
            scale = width / short_edge
            img_np = cv2.resize(img_np, dsize=None, fx=scale, fy=scale)
            boxes *= scale
        img_tensor = img_transform(img_np).unsqueeze(0)
=======
import numpy as np
import cv2
import struct
import random
from torchvision.transforms import ToTensor, ToPILImage
img_transform = ToTensor()
def get_pred_data(file_paths,width=1600):

    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            header_size = np.fromfile(f, dtype='uint32', count=1)[0]
            header = np.fromfile(f, dtype='uint8', count=header_size - 4)
            formatcode = "".join([chr(c) for c in header[:8]])
            Illustration_size = header_size - 36
            Illustration = "".join([chr(c) for c in header[8:Illustration_size + 8]])
            Code_type = "".join([chr(c) for c in header[Illustration_size + 8:Illustration_size + 28]])
            Code_length = header[Illustration_size + 28] + header[Illustration_size + 29] << 4
            Bits_per_pixel = header[Illustration_size + 30] + header[Illustration_size + 31] << 4
            # print(header_size, formatcode, Illustration)
            # print(Code_type, Code_length, Bits_per_pixel)
            # print()
            Image_height = np.fromfile(f, dtype='uint32', count=1)[0]
            Image_width = np.fromfile(f, dtype='uint32', count=1)[0]
            Line_number = np.fromfile(f, dtype='uint32', count=1)[0]
            page_np = np.ones((Image_height * 4, Image_width), dtype=np.uint8) * 255
            page_label = []
            boxes = []
            Y1 = 0
            Y2 = 0
            margin = 0
            for ln in range(Line_number):
                Char_number = np.fromfile(f, dtype='uint32', count=1)[0]
                Label = np.fromfile(f, dtype='uint16', count=Char_number)
                # print(Label)
                Label_str = "".join([struct.pack('H', c).decode('GBK', errors='ignore') for c in Label])
                # print(Label_str, Char_number)
                Top_left = np.fromfile(f, dtype='uint32', count=2)
                Top, Left = Top_left[0], Top_left[1]

                Height = np.fromfile(f, dtype='uint32', count=1)[0]


                # Top+=ln*Image_height//Line_number//8
                Width = np.fromfile(f, dtype='uint32', count=1)[0]
                Bitmap = np.fromfile(f, dtype='uint8', count=Height * Width).reshape([Height, Width])
                contours, hierarchy = cv2.findContours(
                    255 - Bitmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # if random.random()<0.5:
                #     Top+=random.uniform(-0.2,0.2)*Height
                #     Top = int(Top)
                all_contours = []
                for contour in contours:
                    for points in contour:
                        all_contours.append(points)
                all_contours = np.array(all_contours)
                rect = cv2.minAreaRect(all_contours)
                rect_w = max(rect[1])
                rect_h = min(rect[1])


                # Top-=int(ln*Image_height//Line_number//10)
                if rect_w < Image_width * 0.25:
                    x1, y1, x2, y2 = cv2.boundingRect(all_contours)
                    bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                else:
                    bbox = cv2.boxPoints(rect)
                bbox = sorted(bbox, key=lambda x: x[0])
                new_bbox = []
                new_bbox += sorted(bbox[:2], key=lambda x: x[1])
                new_bbox += sorted(bbox[2:], key=lambda x: -x[1])
                bbox = [new_bbox[0], new_bbox[3], new_bbox[2], new_bbox[1]]

                # left_w = random.uniform(-1, 1) * rect_h
                # right_w = random.uniform(-1, 1) * rect_h
                # bbox[0][0] += left_w
                # bbox[1][0] += right_w
                # bbox[2][0] += right_w
                # bbox[3][0] += left_w
                # top_h=random.uniform(-0.2, 0.2) * rect_h
                # bottom_h = random.uniform(-0.2, 0.2) * rect_h
                # bbox[0][1] += top_h
                # bbox[1][1] += top_h
                # bbox[2][1] += bottom_h
                # bbox[3][1] += bottom_h

                bbox = np.int0(bbox)

                bbox[:, 0] += Left
                bbox[:, 1] += Top

                origin_sub = page_np[Top:Top + Height, Left:Left + Width]
                page_np[Top:Top + Height, Left:Left + Width] = (origin_sub > Bitmap) * Bitmap + (origin_sub <= Bitmap) * origin_sub
                if ln == 0:
                    Y1 = max(Top - 64, 0)
                if ln == Line_number - 1:
                    Y2 = Top + Height
                # cv2.drawContours(page_np, [bbox], -1, 128, 2)
                # cv2.imshow('1', cv2.resize(page_np[Y1:, :],dsize=None,fx=0.5,fy=0.5))
                # cv2.waitKey()
                bbox[:, 1] -= Y1
                boxes.append(bbox)
                Label_str = Label_str.replace('\x00', '')
                Label_str = Label_str.replace('〔', '(')
                Label_str = Label_str.replace('〕', ')')
                Label_str = Label_str.replace('＂', '"')
                Label_str = Label_str.replace('％', '%')
                Label_str = Label_str.replace('（', '(')
                Label_str = Label_str.replace('）', ')')
                Label_str = Label_str.replace('，', ',')
                Label_str = Label_str.replace('－', '-')
                Label_str = Label_str.replace('．', '.')
                Label_str = Label_str.replace('／', '/')
                Label_str = Label_str.replace('０', '0')
                Label_str = Label_str.replace('１', '1')
                Label_str = Label_str.replace('２', '2')
                Label_str = Label_str.replace('３', '3')
                Label_str = Label_str.replace('４', '4')
                Label_str = Label_str.replace('５', '5')
                Label_str = Label_str.replace('６', '6')
                Label_str = Label_str.replace('７', '7')
                Label_str = Label_str.replace('８', '8')
                Label_str = Label_str.replace('９', '9')
                Label_str = Label_str.replace('：', ':')
                Label_str = Label_str.replace('；', ';')
                Label_str = Label_str.replace('？', '?')
                Label_str = Label_str.replace('Ａ', 'A')
                Label_str = Label_str.replace('Ｂ', 'B')
                Label_str = Label_str.replace('Ｃ', 'C')
                Label_str = Label_str.replace('Ｆ', 'F')
                Label_str = Label_str.replace('Ｇ', 'G')
                Label_str = Label_str.replace('Ｈ', 'H')
                Label_str = Label_str.replace('Ｍ', 'M')
                Label_str = Label_str.replace('Ｎ', 'N')
                Label_str = Label_str.replace('Ｏ', 'O')
                Label_str = Label_str.replace('Ｐ', 'P')
                Label_str = Label_str.replace('Ｒ', 'R')
                Label_str = Label_str.replace('Ｓ', 'S')
                Label_str = Label_str.replace('Ｖ', 'V')
                Label_str = Label_str.replace('Ｗ', 'W')
                Label_str = Label_str.replace('ａ', 'a')
                Label_str = Label_str.replace('ｄ', 'd')
                Label_str = Label_str.replace('ｅ', 'e')
                Label_str = Label_str.replace('ｈ', 'h')
                Label_str = Label_str.replace('ｉ', 'i')
                Label_str = Label_str.replace('ｌ', 'l')
                Label_str = Label_str.replace('ｍ', 'm')
                Label_str = Label_str.replace('ｎ', 'n')
                Label_str = Label_str.replace('ｏ', 'o')
                Label_str = Label_str.replace('ｐ', 'p')
                Label_str = Label_str.replace('ｒ', 'r')
                Label_str = Label_str.replace('ｓ', 's')
                Label_str = Label_str.replace('ｔ', 't')
                Label_str = Label_str.replace('ｕ', 'u')
                Label_str = Label_str.replace('ｙ', 'y')
                page_label.append(Label_str)
        Y2 = min(Image_height * 4, Y2 + 64)
        img_np = page_np[Y1:Y2, :]
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        boxes = np.array(boxes, dtype=np.float)
        h, w, _ = img_np.shape
        short_edge = max(h,w)
        if short_edge > width:
            # 保证短边 >= inputsize
            scale = width / short_edge
            img_np = cv2.resize(img_np, dsize=None, fx=scale, fy=scale)
            boxes *= scale
        img_tensor = img_transform(img_np).unsqueeze(0)
>>>>>>> 150fd5bc2e07ecd3476ca5859adbd3c12b95a406
        yield img_np, img_tensor, [boxes], page_label