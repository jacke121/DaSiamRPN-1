# DaSiamRPN
#!/usr/bin/python
import datetime
import time
import torch.nn as nn
import torch.optim as optim
import vot
from vot import Rectangle, Point, Polygon, convert_region, parse_region
import os, sys
import cv2  # imread
import torch
import numpy as np
from os.path import realpath, dirname, join
import random
from net import SiamRPNBIG
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect

TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained":"", #  set empty to disable
        # "backbone_pretrained":"../weights/mobilenetv2_weights.pth", #  set empty to disable
    },
    "yolo": {
        "anchors": "16,24, 23,39, 25,84, 31,66, 42,54, 46,38, 56,81, 59,121, 74,236",
        "classes": 1,
    },
    "lr": {
        "other_lr": 0.01,
        "freeze_backbone": False,   #  freeze backbone wegiths to finetune
        "decay_gamma": 0.5,
        "decay_step": 15,           #  decay lr in every ? epochs
    },
    "optimizer": {
        "type": "adam",
        "weight_decay": 4e-05,
    },
    "batch_size": 30,
    "train_path": r"\\192.168.55.73\team-CV\dataset\origin_all_datas\_2train",
    "epochs": 200001,
    "img_h": 352,
    "img_w": 352,
    # "parallels": [0,1,2,3],                         #  config GPU device
    "parallels": [0],                         #  config GPU device
    # "pretrain_snapshot": "",
    # "pretrain_snapshot": r"F:\Team-CV\checkpoints\shuffle_v2\0.8204_0167.weights",                        #  load checkpoint
    "pretrain_snapshot": r"D:\Team-CV\checkpoints\torch_yolov09_5\0.9783_0090.weights",
}
class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        print('1')

    def forward(self, pred, truth):
        truth = truth.repeat(2).view(-1,2)
        truth = truth.view(-1,2)
        pred  = pred.repeat(2).view(-1,2)
        return  torch.mean(torch.mean((pred-truth)**2,1),0)


def _get_optimizer(config, net):
    optimizer = None

    # Assign different lr for each layer
    params = None
    base_params = list(
        map(id, net.parameters())
    )
    # logits_params = filter(lambda p: id(p) not in base_params, net.parameters())

    if not config["lr"]["freeze_backbone"]:
        params = [
            {"params": net.parameters(), "lr": config["lr"]["other_lr"]},
        ]
    else:
        for p in net.backbone.parameters():
            p.requires_grad = False
        params = [
            {"params": base_params, "lr": config["lr"]["other_lr"]},
        ]
    print("Using " + config["optimizer"]["type"] + " optimizer.")
    # Initialize optimizer class
    if config["optimizer"]["type"] == "adam":
        optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"])
    elif config["optimizer"]["type"] == "amsgrad":
        optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"],
                               amsgrad=True)
    elif config["optimizer"]["type"] == "rmsprop":
        optimizer = optim.RMSprop(params, weight_decay=config["optimizer"]["weight_decay"])
    else:
        # Default to sgd
        optimizer = optim.SGD(params, momentum=0.9,
                              weight_decay=config["optimizer"]["weight_decay"],
                              nesterov=(config["optimizer"]["type"] == "nesterov"))

    return optimizer

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

# load net
net_file = join(realpath(dirname(__file__)), 'SiamRPNBIG.model')
net = SiamRPNBIG()

net.train()

config = TRAINING_PARAMS
optimizer = _get_optimizer(config, net)

net=net.cuda()
net.load_state_dict(torch.load(net_file))
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)

# warm up
for i in range(10):
    net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda())
    net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())

# start to track
data_root =r"D:\data\vot2016/"
checkpoint_dir="checkpoint_dir"
os.makedirs(checkpoint_dir,exist_ok=True)
dirs = os.listdir(data_root)
myloss=MyLoss()

target=torch.ones(1,requires_grad=True)
for dir_name in dirs:
    if os.path.isfile(data_root + dir_name):
        continue
    files = os.listdir(data_root + dir_name)
    files = sorted(files)
    images = [data_root + dir_name + "/" + file_name for file_name in files if (file_name[-3:] == "jpg")]
    gt_txt = data_root + dir_name + "/" + "groundtruth.txt"
    gt = np.loadtxt(gt_txt, delimiter=',')
    if gt.shape[1] == 8:
        x_min = np.min(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_min = np.min(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        x_max = np.max(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_max = np.max(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)

    #Polygon = region
    cx, cy, w, h = gt[0]

    frame = 0
    image_file = images[frame]
    if not image_file:
        sys.exit(0)
    target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
    im = cv2.imread(image_file)  # HxWxC
    # cv2.imshow("Tracking", im)
    # cv2.waitKey(1)
    print("SiamRPN init")
    state = SiamRPN_init(im, target_pos, target_sz, net)  # init tracker
    print("begin tracking")
    dataloader=images
    best_acc = 0.2
    m_index = [i for i in range(len(dataloader))]
    for epoch in range(config["epochs"]):
        recall=0
        frame=0

        random.shuffle(m_index)
        while True:
            frame +=1
            if frame >= len(images):
                break

            aaaaa=    m_index[frame]
            image_file = images[aaaaa]
            if not image_file:
                break
            im = cv2.imread(image_file)  # HxWxC
            cx, cy, w, h = gt[aaaaa]

            start=time.time()
            state = SiamRPN_track(state, im)  # track
            res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])

            net.temple(zs)
            reg_output, cls_output = net.forward(xs)  # of shape (50, 4*5, 17, 17), (50, 2*5, 17, 17)
            feat_h, feat_w = cls_output.size(-2), cls_output.size(-1)

            assert zs.shape[0] == xs.shape[0] == gt_boxes.shape[0]
            total_loss = total_cls_loss = total_reg_loss = 0.0
            for i in range(zs.shape[0]):
                rpn_labels, \
                rpn_bbox_targets, \
                rpn_bbox_inside_weights, \
                rpn_bbox_outside_weights, \
                A \
                    = gen_anchor_target(cls_output[i].shape[-2:], xs[i].shape[-2:], gt_boxes[i][np.newaxis, :])

                # reg_loss_fn = torch.nn.SmoothL1Loss(reduce=False, size_average=False)
                reg_loss_fn = smooth_l1_loss
                reg_loss = reg_loss_fn(reg_output[i], torch.from_numpy(rpn_bbox_targets).to(DEVICE),
                                       torch.from_numpy(rpn_bbox_inside_weights).to(DEVICE),
                                       torch.from_numpy(rpn_bbox_outside_weights).to(DEVICE))

                cls_loss_fn = torch.nn.CrossEntropyLoss(reduce=False, size_average=False)

                rpn_labels = rpn_labels.reshape(A, feat_h, feat_w)  # from (1, 1, A*17, 17) to (A, 17, 17)
                logits = cls_output[i].view(A, 2, feat_h, feat_w)  # from (2*A, 17, 17) to (A, 2, 17, 17)
                cls_loss = cls_loss_fn(logits, torch.from_numpy(rpn_labels).to(DEVICE).long())  # (A, 17, 17)

                mask = np.ones_like(rpn_labels)
                mask[np.where(rpn_labels == -1)] = 0  # mask where we 'don't care'
                mask = torch.from_numpy(mask).to(DEVICE)
                cls_loss = torch.sum(cls_loss * mask) / torch.sum(mask)

                # import pdb
                # pdb.set_trace()
                # print("{} + l * {} = {}".format(cls_loss, reg_loss, cls_loss+cfg.TRAIN.LAMBDA*reg_loss))

                total_cls_loss += cls_loss
                total_reg_loss += reg_loss
                total_loss += cls_loss + cfg.TRAIN.LAMBDA * reg_loss

            total_loss /= batch_size
            total_reg_loss /= batch_size
            total_cls_loss /= batch_size
            print(
                f"Epoch{i_ep} Iter{i_iter} --- total_loss: {total_loss:.4f}, cls_loss: {total_cls_loss:.4f}, reg_loss: {total_reg_loss:.4f}")
            total_loss.backward()
            optimizer.step()
       #-----------

            _loss = loss.item()
            # example_per_second = config["batch_size"] / duration
            lr = optimizer.param_groups[0]['lr']
            #
            strftime = datetime.datetime.now().strftime("%H:%M:%S")
            # # if (losses[7] / 3 >= recall / (step + 1)):#mini_batch为0走这里
            # print(
            #     '%s [Epoch %d/%d,batch %03d/%d loss:%.5f,y %.5f,w %.5f,h %.5f,conf %.5f,cls %.5f,total %.5f,rec %.3f,avrec %.3f %.3f]' %
            #     (strftime, epoch, config["epochs"], step, dataload_len,
            #      losses[1], losses[2], losses[3],
            #      losses[4], losses[5], losses[6],
            #      _loss, current_recall, recall / (step + 1), lr))

            # if recall / len(dataloader) > best_acc:
        if recall / len(dataloader) > best_acc:
            best_acc = recall / len(dataloader)
            last_recall = best_acc
            torch.save(net.state_dict(), '%s/%.4f_%04d.weights' % (checkpoint_dir, recall / len(dataloader), epoch))

        lr_scheduler.step()
            # img = im.copy()
            # cv2.rectangle(img, (int(res[0]), int(res[1])), (int(res[0] + res[2]), int(res[1] + res[3])), (0, 255, 0), 2)
            # cv2.rectangle(img, (int(cx), int(cy)), (int(cx+w), int(cy+h)), (255, 0, 0), 2)
            # cv2.imshow("Tracking", img)
            # cv2.waitKey()
cv2.destroyAllWindows()

