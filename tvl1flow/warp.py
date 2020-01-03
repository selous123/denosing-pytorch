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


import imageio

file = 'of/tvl1_target.flo'
img_file = 'image/target/im1.png'
img_np = imageio.imread(img_file)/255.0

img2_file = 'image/target/im2.png'
img2_np = imageio.imread(img2_file)/255.0

flow = read_flow(file)

flow = torch.tensor(flow).unsqueeze(0).permute(0,3,1,2)


target = torch.tensor(img_np).unsqueeze(0).permute(0,3,1,2).type(torch.FloatTensor)
#print(target)

print(flow)

warp_result, mask = warp(target, flow)

warp_result = warp_result[0].permute(1,2,0).numpy()
print(np.abs(warp_result - img_np).mean())

print(np.abs(img2_np - img_np).mean())

from matplotlib import pyplot as plt
plt.subplot(131)
plt.imshow(img_np)
plt.subplot(132)
plt.imshow(warp_result)
plt.subplot(133)
plt.imshow(img2_np)
plt.show()
