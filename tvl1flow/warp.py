import numpy as np
import os

TAG_FLOAT = 202021.25

def read_flow(file):

	assert type(file) is str, "file is not str %r" % str(file)
	assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
	assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
	f = open(file,'rb')
	flo_number = np.fromfile(f, np.float32, count=1)[0]
	assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
	w = np.fromfile(f, np.int32, count=1)
	h = np.fromfile(f, np.int32, count=1)
	data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
	#if error try: data = np.fromfile(f, np.float32, count=2*w*h)
	# Reshape data into 3D array (columns, rows, bands)
	flow = np.resize(data, (int(h), int(w), 2))
	f.close()

	return flow

import torch.nn as nn
import torch

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow.
    Code heavily inspired by
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    grid = grid
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0*vgrid[:, 0, :, :]/max(W-1, 1)-1.0
    vgrid[:, 1, :, :] = 2.0*vgrid[:, 1, :, :]/max(H-1, 1)-1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid)
    # Define a first mask containing pixels that wasn't properly interpolated
    mask = torch.autograd.Variable(torch.ones(x.size()))
    mask = nn.functional.grid_sample(mask, vgrid)
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output, mask


## pytorch
# import imageio
#
# file = 'of/tvl1_target.flo'
# img_file = 'image/target/im1.png'
# img_np = imageio.imread(img_file)/255.0
#
# img2_file = 'image/target/im2.png'
# img2_np = imageio.imread(img2_file)/255.0
#
# flow = read_flow(file)
# flow = torch.tensor(flow).unsqueeze(0).permute(0,3,1,2)
# print(flow.shape)
#
# target = torch.tensor(img_np).unsqueeze(0).permute(0,3,1,2).type(torch.FloatTensor)
# print(target.shape)
#
# warp_result, mask = warp(target, flow)
#
# mask_w = warp_result * mask
#
# warp_result = warp_result[0].permute(1,2,0).numpy()
# mask_w = mask_w[0].permute(1,2,0).numpy()
# print(np.abs(warp_result - img_np).mean())
#
# print(np.abs(img2_np - img_np).mean())
#
# from matplotlib import pyplot as plt
# plt.subplot(221)
# plt.imshow(img_np)
# plt.subplot(222)
# plt.imshow(warp_result)
# plt.subplot(223)
# plt.imshow(img2_np)
# plt.subplot(224)
# plt.imshow(mask_w)
# plt.show()

# import imageio
# import cv2
# import numpy as np
# img_file = 'image/target/im1.png'
# prev_frame =  cv2.imread(img_file)
#
# prvs = cv2.cvtColor(prev_frame,cv2.COLOR_RGB2GRAY)
# #
# img2_file = 'image/target/im2.png'
# next = cv2.imread(img2_file)
# next = cv2.cvtColor(next,cv2.COLOR_RGB2GRAY)
#
# print(prvs.shape)
# print(next.shape)
#
# flow = cv2.calcOpticalFlowFarneback(prvs,next,None, 0.5, 3, 15, 3, 5, 1.2, 0)
#
# print(flow.shape)
#
# avg_u = np.mean(flow[:, :, 0])
# avg_v = np.mean(flow[:, :, 1])
#
# height = flow.shape[0]
# width = flow.shape[1]
# R2 = np.dstack(np.meshgrid(np.arange(width), np.arange(height)))
# pixel_map = R2 + flow
# print(pixel_map.shape)
# new_frame = cv2.remap(prev_frame, pixel_map, None, cv2.INTER_LINEAR)


import cv2
import numpy as np
from imageio import imread
img_file = '/home/lrh/git/FRVD-pytorch/tvl1flow/image/target/im3.png'
img2_file = '/home/lrh/git/FRVD-pytorch/tvl1flow/image/target/im4.png'

prev_frame =  cv2.imread(img_file)
next_frame = cv2.imread(img2_file)
prvs = cv2.cvtColor(prev_frame,cv2.COLOR_RGB2GRAY)
next = cv2.cvtColor(next_frame,cv2.COLOR_RGB2GRAY)
# im1 = np.asarray(imread(img_file))
# im2 = np.asarray(imread(img2_file))
# im1 = np.float64(im1 / 255)
# im2 = np.float64(im2 / 255)

flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

hsv = draw_hsv(flow)
im2w = warp_flow(prev_frame, flow)
cv2.imwrite("of/flow.jpg",hsv)
cv2.imwrite("of/im1.jpg", prev_frame)
cv2.imwrite("of/im2.jpg", next_frame)
cv2.imwrite("of/im2w.jpg", im2w)
