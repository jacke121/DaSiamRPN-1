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
net.eval().cuda()

# warm up
for i in range(10):
    net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda())
    net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())

# start to track

    # if (dir_name != "conduction1"):
       # continue
dir_name=r"E:\git_track\MBMD\model\train\mouse_error\JPEGImages"
files = os.listdir(dir_name)
files = sorted(files)
images = [ dir_name + "/" + file_name for file_name in files if (file_name[-3:] == "jpg")]

gt_txt =  dir_name + "/" + "groundtruth.txt"
# with open(gt_txt, 'r') as f:
#     gt_box_str = f.readline()
# gt_box = map(float,list(gt_box_str.split(',')))

region_type = "polygon"
region = convert_region(parse_region(open(gt_txt, 'r').readline()), region_type)
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
    frame = frame + 1
    if frame >= len(images):
        break
    image_file = images[frame]
    if not image_file:
        break
    im = cv2.imread(image_file)  # HxWxC
    start=time.time()
    state = SiamRPN_track(state, im)  # track
    res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
    print(frame,time.time()-start)
    img = im.copy()
    cv2.rectangle(img, (int(res[0]), int(res[1])), (int(res[0] + res[2]), int(res[1] + res[3])), (0, 255, 0), 2)
    cv2.imshow("Tracking", img)
    cv2.waitKeyEx()
cv2.destroyAllWindows()

