# --------------------------------------------------------
# DaSiamRPN
#!/usr/bin/python
import time

import vot
from vot import Rectangle, Point, Polygon, convert_region, parse_region
import os, sys
import cv2  # imread
import torch
import numpy as np
from os.path import realpath, dirname, join

from net import SiamRPNBIG
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect

# load net
net_file = join(realpath(dirname(__file__)), 'SiamRPNBIG.model')
net = SiamRPNBIG()
net.load_state_dict(torch.load(net_file))
net.train().cuda()

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area =    torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                    torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

# warm up
for i in range(10):
    net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda())
    net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())

# start to track
data_root =r"D:\data\vot2016/"
dirs = os.listdir(data_root)
for dir_name in dirs:
    # if (dir_name != "conduction1"):
       # continue
    print(dir_name)
    if os.path.isfile(data_root + dir_name):
        continue
    files = os.listdir(data_root + dir_name)
    files = sorted(files)
    images = [data_root + dir_name + "/" + file_name for file_name in files if (file_name[-3:] == "jpg")]

    gt_txt = data_root + dir_name + "/" + "groundtruth.txt"
    # with open(gt_txt, 'r') as f:
    #     gt_box_str = f.readline()
    # gt_box = map(float,list(gt_box_str.split(',')))

    boxs=open(gt_txt, 'r').readlines()
    region_type = "polygon"
    region = convert_region(parse_region(boxs[0]), region_type)
    #Polygon = region
    cx, cy, w, h = get_axis_aligned_bbox(region)

    frame = 0
    image_file = images[frame]
    if not image_file:
        sys.exit(0)

    target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
    im = cv2.imread(image_file)  # HxWxC
    cv2.imshow("Tracking", im)
    cv2.waitKey(1)
    print("SiamRPN init")
    state = SiamRPN_init(im, target_pos, target_sz, net)  # init tracker
    print("begin tracking")
    while True:
        frame +=1
        if frame >= len(images):
            break
        image_file = images[frame]
        if not image_file:
            break
        im = cv2.imread(image_file)  # HxWxC

        region = convert_region(parse_region(boxs[frame]), region_type)
        # Polygon = region
        cx, cy, w, h = get_axis_aligned_bbox(region)

        start=time.time()
        state = SiamRPN_track(state, im)  # track
        res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        print(frame,time.time()-start)
        a = torch.FloatTensor(np.concatenate((state['target_pos'], state['target_sz']))).view(1, 4)
        b = torch.FloatTensor([cx, cy, w, h]).view(1, 4)
        #
        iou=bbox_iou(a, b,False)
        print(iou)
        img = im.copy()
        # cv2.rectangle(img, (int(res[0]), int(res[1])), (int(res[0] + res[2]), int(res[1] + res[3])), (0, 255, 0), 2)
        cv2.rectangle(img, (int(cx), int(cy)), (int(cx+w), int(cy+h)), (255, 0, 0), 2)
        cv2.imshow("Tracking", img)
        cv2.waitKey()
cv2.destroyAllWindows()

