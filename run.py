# --------------------------------------------------------
# DaSiamRPN
#!/usr/bin/python
import vot
from vot import Rectangle, Point, Polygon
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
net.eval().cuda()

# warm up
for i in range(10):
    net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda())
    net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())

# start to track
data_root = "/home/ivlab/data/vot2017/"
dirs = os.listdir(data_root)
for dir_name in dirs:
    if (dir_name != "ants1"):
        continue
    print(dir_name)
    if os.path.isfile(data_root + dir_name):
        continue
    files = os.listdir(data_root + dir_name)
    files = sorted(files)
    images = [data_root + dir_name + "/" + file_name for file_name in files if (file_name[-3:] == "jpg")]

    gt_txt = data_root + dir_name + "/" + "groundtruth.txt"
    with open(gt_txt, 'r') as f:
        gt_box_str = f.readline()
    gt_box = map(float,list(gt_box_str.split(',')))

    region_type = "polygon"
    if region_type == 'polygon':
        region = Polygon([Point(gt_box[i],gt_box[i+1]) for i in xrange(0,len(gt_box),2)])
    else:
        region = Rectangle(gt_box.x, gt_box.y, gt_box.width, gt_box.height)
    Polygon = region
    cx, cy, w, h = get_axis_aligned_bbox(Polygon)

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
        frame = frame + 1
        if frame >= len(images):
            break
        print(frame)
        image_file = images[frame]
        if not image_file:
            break
        im = cv2.imread(image_file)  # HxWxC
        state = SiamRPN_track(state, im)  # track
        res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])

        img = im.copy()
        cv2.rectangle(img, (int(res[0]), int(res[1])), (int(res[0] + res[2]), int(res[1] + res[3])), (0, 255, 0), 2)
        cv2.imshow("Tracking", img)
        cv2.waitKey(1)


"""
handle = vot.VOT("polygon")
Polygon = handle.region()
cx, cy, w, h = get_axis_aligned_bbox(Polygon)

image_file = handle.frame()
if not image_file:
    sys.exit(0)

target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
im = cv2.imread(image_file)  # HxWxC
state = SiamRPN_init(im, target_pos, target_sz, net)  # init tracker
while True:
    image_file = handle.frame()
    if not image_file:
        break
    im = cv2.imread(image_file)  # HxWxC
    state = SiamRPN_track(state, im)  # track
    res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])

    handle.report(Rectangle(res[0], res[1], res[2], res[3]))
    """

