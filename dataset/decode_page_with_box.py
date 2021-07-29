import os
import numpy as np
import struct

import cv2
from tqdm import tqdm
data_dir = './dgrl'
count = 0
def read_from_gnt_dir(gnt_dir):
    global count
    def one_file(f):

        laber_writer = open('./gt/gt_{}.txt'.format(count), 'w', encoding='utf-8')
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
        page_np = np.ones((Image_height, Image_width), dtype=np.uint8) * 255
        page_label = ''
        # print(Image_height, Image_width, Line_number)
        Y1 = 0
        Y2 = 0
        for ln in range(Line_number):
            Char_number = np.fromfile(f, dtype='uint32', count=1)[0]
            Label = np.fromfile(f, dtype='uint16', count=Char_number)
            Label_str = "".join([struct.pack('H', c).decode('GBK', errors='ignore') for c in Label])
            # print(Label_str, Char_number)
            page_label += Label_str
            Top_left = np.fromfile(f, dtype='uint32', count=2)
            Top, Left = Top_left[0], Top_left[1]
            if Left > Image_width:
                Left = 16
            Height = np.fromfile(f, dtype='uint32', count=1)[0]
            Width = np.fromfile(f, dtype='uint32', count=1)[0]
            Bitmap = np.fromfile(f, dtype='uint8', count=Height * Width).reshape([Height, Width])
            contours, hierarchy = cv2.findContours(
                255-Bitmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            all_contours = []
            for contour in contours:
                for points in contour:
                    all_contours.append(points)

            all_contours = np.array(all_contours)
            rect = cv2.minAreaRect(all_contours)
            bbox = cv2.boxPoints(rect)
            bbox = sorted(bbox, key=lambda x: x[0])
            new_bbox = []
            new_bbox += sorted(bbox[:2], key=lambda x: x[1])
            new_bbox += sorted(bbox[2:], key=lambda x: -x[1])
            bbox = [new_bbox[0], new_bbox[3], new_bbox[2], new_bbox[1]]
            bbox = np.int0(bbox)

            bbox[:,0]+=Left
            bbox[:,1]+=Top

            origin_sub = page_np[Top:Top + Height, Left:Left + Width]

            page_np[Top:Top + Height, Left:Left + Width] = (origin_sub >Bitmap) * Bitmap + (origin_sub <=Bitmap) * origin_sub

            if ln == 0:
                Y1 = max(Top-16,0)
            if ln == Line_number - 1:
                Y2 = Top + Height

            # cv2.drawContours(page_np, [bbox], -1, 128, 2)
            bbox[:, 1] -= Y1
            # cv2.imshow('1', page_np[Y1:,:])
            # cv2.waitKey()
            laber_writer.write('{} {} {} {} {} {} {} {} {}\n'.format(bbox[0][0], bbox[0][1],
                                                                     bbox[1][0], bbox[1][1],
                                                                     bbox[2][0], bbox[2][1],
                                                                     bbox[3][0], bbox[3][1],
                                                                     Label_str.replace('\x00', '')))
        Y2 = min(Image_height,Y2+16)
        cv2.imwrite('./page_imgs/{}.png'.format(count), page_np[Y1:Y2, :])


        # print(Top_left,Height,Width)

    pbar = tqdm(total=len(os.listdir(gnt_dir)))
    for file_i,file_name in enumerate(os.listdir(gnt_dir)):
        pbar.update(1)

        if file_name.endswith('.dgrl'):
            file_path = os.path.join(gnt_dir, file_name)
            with open(file_path, 'rb') as f:
                one_file(f)
            count = count + 1
    pbar.close()
if __name__ == '__main__':
    read_from_gnt_dir('./dgrl')