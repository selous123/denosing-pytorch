import torch
import torch.nn as nn


class GradImgLoss(nn.Module):
    def __init__(self):
        super(GradImgLoss,self).__init__()

    def forward(self, x, y):
        assert x.shape == y.shape
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        ##grad_x
        h_grad_x = torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:])
        h_grad_y = torch.abs(y[:,:,1:,:]-y[:,:,:h_x-1,:])
        ##grad_y
        w_grad_x = torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1])
        w_grad_y = torch.abs(y[:,:,:,1:]-y[:,:,:,:w_x-1])
        #print("h,w:", h_tv, w_tv)
        grad_loss = torch.pow((h_grad_x - h_grad_y), 2).sum() + torch.pow((w_grad_x - w_grad_y), 2).sum()
        return grad_loss

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
