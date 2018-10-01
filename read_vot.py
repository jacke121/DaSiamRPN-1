import cv2
import numpy as np
import os

data = {}
seq = r"D:\data\vot2017/bag/"
img_list = sorted([p for p in os.listdir(seq) if os.path.splitext(p)[1] == '.jpg'])
gt = np.loadtxt(seq + '/groundtruth.txt', delimiter=',')

if gt.shape[1] == 8:
    x_min = np.min(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
    y_min = np.min(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
    x_max = np.max(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
    y_max = np.max(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
    gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)

data[seq] = {'images': img_list, 'gt': gt}

for i, path in enumerate(img_list):
    img = cv2.imread(seq + path)
    pos_x, pos_y, target_w, target_h = gt[i]
    # pos_x = pos_x * 0.85
    # pos_y = pos_y * 0.8
    cv2.rectangle(img, (int(pos_x), int(pos_y)), (int(pos_x + target_w), int(pos_y + target_h)), (0, 255, 0), 2)

    cv2.imshow("asadf", img)
    cv2.waitKeyEx()

print(data)
