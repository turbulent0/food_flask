import time
import os
import sys
import cv2
import numpy as np
import pandas as pd
import random
from PIL import Image, ImageDraw, ImageFile
# fix bugs with loading png files
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset, RandomSampler
from torch.hub import load_state_dict_from_url
from torchvision.ops.boxes import batched_nms
from torchsummary import summary
import xmltodict
import pprint
import json
import ast
from math import sqrt
import itertools
import io


class ResNet50Backbone(nn.Module):

    def __init__(self, freeze=True):
        super(ResNet50Backbone, self).__init__()

        model = models.resnet50(pretrained=True)
        if freeze:
            model.requires_grad_(False)
        else:
            # do not unfreeze first layers as they share low-level distribution
            # which is very hard to train and it same for most of the imagesqa1 IOK9NB HHHHH
            for name, parameter in model.named_parameters():
                if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                    parameter.requires_grad_(False)
        self.x = nn.Sequential(*list(model.children())[:7])
        # our neck
        # output channels for each additional features block
        self.out_channels = [1024, 512, 512, 256, 256, 256]
        # remove strides to get correct feature spatial size
        conv4_block1 = self.x[-1][0]
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        return self.x(x)

class SSD(nn.Module):

    def __init__(self, backbone, classes=21, freeze=True):
        super(SSD, self).__init__()

        self.base = backbone

        self.classes = classes  
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        self.additional_blocks = []
        self.loc = []
        self.conf = []

        # adding additional blocks before loc and conf heads
        self.build_additional_features(self.base.out_channels)

        # creating loc and conf heads
        for nboxes, channels in zip(self.num_defaults, self.base.out_channels):
            self.loc.append(nn.Conv2d(channels, nboxes * 4, kernel_size=3, padding=1))
            self.conf.append(nn.Conv2d(channels, nboxes * self.classes, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)

        self._init_weights()

    def build_additional_features(self, input_size):
        for i, (input_size, output_size, channels) in enumerate(zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])):
            # we dont use bias because we use batchnorm that should center data
            # here we first reduce number of channels and then expand, like in autoencoder
            if i < 3:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )

            self.additional_blocks.append(layer)

        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    def _init_weights(self):
        # initialize weight for convolutions
        layers = [self.additional_blocks, self.loc, self.conf]
        for layer in layers:
            for param in layer.parameters():
                # check for convolution layer
                if param.dim() > 1: nn.init.xavier_uniform_(param)

    def bbox_view(self, src):
        # Apply heads and reshape
        ret = []
        for s, l, c in zip(src, self.loc, self.conf):
            ret.append((l(s).view(s.size(0), 4, -1), c(s).view(s.size(0), self.classes, -1)))

        locs, confs = list(zip(*ret))
        # .contiguous() used cause some operation not creating new tensor 
        # but creating some memory mapping of current. And .contiguous() basicly apply
        # this mapping and create new tensor
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs

    def forward(self, x):
        x = self.base(x)
        
        # creating different feature levels
        detection_feed = [x]
        for i,l in enumerate(self.additional_blocks):
            x = l(x)
            detection_feed.append(x)

        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
        locs, confs = self.bbox_view(detection_feed)

        # For SSD 300, shall return nbatch x 8732 x {classes, locations} results
        return locs, confs

class DefaultBoxes(object):
    def __init__(self, fig_size, feat_size, aspect_ratios, \
                       scale_xy=0.1, scale_wh=0.2, scale_min=0.07, scale_max=0.9):

        self.feat_size = feat_size
        self.fig_size = fig_size
        self.scale_min = scale_min
        self.scale_max = scale_max

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        self.aspect_ratios = aspect_ratios

        self.default_boxes = []

        for idx, sfeat in enumerate(self.feat_size):
            sk = scale_min + ((scale_max - scale_min)/(len(self.feat_size)-1))*(idx)
            sk_next = scale_min + ((scale_max - scale_min)/(len(self.feat_size)-1))*(idx+1)
            
            all_sizes = []
            for alpha in self.aspect_ratios[idx]:
                w, h = sk*sqrt(alpha), sk/sqrt(alpha)
                all_sizes.append((w,h))
            # for aspect 1 adding additional box
            sk = sqrt(sk*sk_next)
            w, h = sk*sqrt(1.), sk/sqrt(1.)
            all_sizes.append((w,h))
            
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j+0.5)/sfeat, (i+0.5)/sfeat
                    self.default_boxes.append((cx, cy, w, h))

        self.dboxes = torch.tensor(self.default_boxes)
        # make values in interval [0,1]
        self.dboxes.clamp_(min=0, max=1)
        # For IoU calculation
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]
        self.dboxes_ltrb.clamp_(min=0, max=1)
        # LTRB boxes added clamp, cause new values can be out of interval [0,1]

    @property
    def scale_xy(self):
        # also called variance
        return self.scale_xy_

    @property
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order="ltrb"):
        if order == "ltrb": return self.dboxes_ltrb
        if order == "xywh": return self.dboxes

def default_boxes_300():
    # we use default values
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    # length matched with num_defaults parametr of SSD model (dont forget about + 1 for aspect ration 1)
    aspect_ratios = [[1.,2.,0.5], [1.,2.,0.5,3.,1./3], [1.,2.,0.5,3.,1./3], [1.,2.,0.5,3.,1./3], [1.,2.,0.5], [1.,2.,0.5]]
    scale_min=0.07
    scale_max=0.9
    dboxes = DefaultBoxes(figsize, feat_size, aspect_ratios, scale_min=scale_min, scale_max=scale_max)
    return dboxes

def calc_iou_tensor(box1, box2):
    """ param: box1  tensor (N, 4) (x1,y1,x2,y2)
        param: box2  tensor (M, 4) (x1,y1,x2,y2)
        output: tensor (N, M)
    """
    N = box1.size(0)
    M = box2.size(0)

    box1 = box1.unsqueeze(1).expand(-1, M, -1)
    box2 = box2.unsqueeze(0).expand(N, -1, -1)

    # Left Top & Right Bottom
    lt = torch.max(box1[:,:,:2], box2[:,:,:2])
    rb = torch.min(box1[:,:,2:], box2[:,:,2:])

    delta = rb - lt
    delta[delta < 0] = 0
    intersect = delta[:,:,0]*delta[:,:,1]

    delta1 = box1[:,:,2:] - box1[:,:,:2]
    area1 = delta1[:,:,0]*delta1[:,:,1]
    delta2 = box2[:,:,2:] - box2[:,:,:2]
    area2 = delta2[:,:,0]*delta2[:,:,1]

    iou = intersect/(area1 + area2 - intersect)
    return iou

class BoxUtils:
    """
        Util to encode/decode target/result boxes

        param: dboxes - DefaultBoxes instance
    """

    def __init__(self, dboxes):
        self.dboxes = dboxes(order="ltrb")
        self.dboxes_xywh = dboxes(order="xywh")
        self.nboxes = self.dboxes.size(0)

        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh

    def encode(self, bboxes, labels_in, criteria = 0.5, num_classes=21):

        ious = calc_iou_tensor(bboxes, self.dboxes) # [N_bboxes, num_default_boxes]
        best_dbox_ious, best_dbox_idx = ious.max(dim=0) # [num_default_boxes], [num_default_boxes]
        best_bbox_ious, best_bbox_idx = ious.max(dim=1) # [N_bboxes], [N_bboxes]

        # set best ious 2.0
        # this needed to not filter out this bboxes on next step
        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)

        # filter out dboxes with IoU <= criteria
        masks = best_dbox_ious > criteria
        # setting all labels of filtered out dboxes to zero(background)
        labels_out = torch.zeros(self.nboxes, dtype=torch.long)
        labels_out[masks] = labels_in[best_dbox_idx[masks]]

        # setting ghound trouth boxes on place of best matched default boxes
        # and convert (x1,y1,x2,y2) format to (xc,yc,w,h)
        bboxes_out = self.dboxes.clone()
        bboxes_out[masks, :] = bboxes[best_dbox_idx[masks], :]
        # Transform format to xywh format
        x, y, w, h = 0.5*(bboxes_out[:, 0] + bboxes_out[:, 2]), \
                     0.5*(bboxes_out[:, 1] + bboxes_out[:, 3]), \
                     -bboxes_out[:, 0] + bboxes_out[:, 2], \
                     -bboxes_out[:, 1] + bboxes_out[:, 3]

        # make coordinates to be offset to default boxes and encode "variance" for xy and wh
        bboxes_out[:, 0] = (x - self.dboxes_xywh[:, 0])/(self.scale_xy*self.dboxes_xywh[:, 2])
        bboxes_out[:, 1] = (y - self.dboxes_xywh[:, 1])/(self.scale_xy*self.dboxes_xywh[:, 3])
        # i recommend always check base of logariphm when use some new libraries
        # cause log can be sometimes log by 2 and not by e
        bboxes_out[:, 2] = torch.log(w/self.dboxes_xywh[:, 2])/self.scale_wh
        bboxes_out[:, 3] = torch.log(h/self.dboxes_xywh[:, 3])/self.scale_wh
        return bboxes_out, labels_out

    def decode(self, bboxes):
        
        xy = self.dboxes_xywh[:,:2]
        wh = self.dboxes_xywh[:,2:]

        # don't forget that we need to remove "variance" from output 
        _xy = (bboxes[:,:2]*self.scale_xy*wh) + xy 
        _wh2 = (torch.exp(bboxes[:,2:]*self.scale_wh)*wh)/2 
        xy1 = (_xy - _wh2)
        xy2 = (_xy + _wh2)
        boxes = torch.cat([xy1,xy2],dim=-1)

        return boxes

def get_model():
	model_path = 'ssd300.pth'
	state = torch.load('ssd300.pth', map_location='cpu')
	model = SSD(ResNet50Backbone(), classes=5).cpu()
	model.load_state_dict(state)
	model.eval()
	return model

def transform_image(image_bytes):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_ts = transforms.Compose([
        transforms.Resize((300,300)),
        transforms.ToTensor(), # converts to [0,1] interval
        transforms.Normalize(mean=mean, std=std)
    ])
    image = Image.open(io.BytesIO(image_bytes))
    image.save('templates/uploaded_image.jpg')
    return img_ts(image).unsqueeze(0)

def get_predictions(tensor, model):
	tensor = tensor.detach().cpu()
	rboxes, rlabels = model(tensor)
	def denorm(x):
		mean = np.array([0.485, 0.456, 0.406])
		std = np.array([0.229, 0.224, 0.225])
		return (x * std) + mean
	rboxes, rlabels = rboxes[0].transpose(1,0), rlabels[0]
	tensor = (denorm(tensor[0].numpy().transpose(1,2,0).copy())*255).astype(np.uint8)
	w,h = tensor.shape[:2]
	dboxes = default_boxes_300()
	bu = BoxUtils(dboxes)
	bboxes = bu.decode(rboxes)
	scores, labels = F.softmax(rlabels, dim=0).max(dim=0)
	thresh = 0.7
	iou_threshold = 0.35
	mask = (labels > 0)
	labels = labels[mask]
	scores = scores[mask]
	bboxes = bboxes[mask]
	mask = (scores > thresh)
	labels = labels[mask]
	scores = scores[mask]
	bboxes = bboxes[mask]
	print('Boxes:',bboxes.shape[0])
	idx = batched_nms(bboxes, scores, labels, iou_threshold=iou_threshold,).numpy().astype(np.uint64)
	labels = labels.detach().numpy().astype(np.uint8)
	scores = scores.detach().numpy()
	print(scores)
	print(labels)
	bboxes = bboxes.detach().numpy()
	size = 300
	classes = ["bg","A", "B", "C", "D"]
	for i in idx:
		b = bboxes[i]
		img = cv2.rectangle(tensor, (int(b[0]*size), int(b[1]*size)), (int(b[2]*size), int(b[3]*size)), (0, 255, 0), 2)
		print(int(b[0]*size), int(b[1]*size), int(b[2]*size), int(b[3]*size), classes[labels[i]], scores[i])
		im = Image.fromarray(img)
		im.save("static/pred.jpeg")
	return bboxes[idx], labels[idx], scores[idx]







