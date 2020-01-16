import unittest
import torch
import torch.nn as nn
import torch.nn.functional as func
from torchvision.models import vgg16
import sys
sys.path.append('..')
from option import args
#from .flow import flownets

from model import common
## Frame-Recurrent Video Denosing without any image alignment
def make_model(args, parent=False):
    _model = FRVD(args)
    return _model

class DNet(nn.Module):
    def __init__(self, args, in_dim=6, conv=common.default_conv):
        super(DNet, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)

        #self.sub_mean = common.MeanShift(args.rgb_range)
        #self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(in_dim, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        #x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        #x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


class FRVD(nn.Module):
    def __init__(self, args):
        super(FRVD, self).__init__()
        self.args = args
        self.device = torch.device('cpu' if args.cpu else 'cuda')

        self.dnet = DNet(args).to(self.device)  # 3 is channel number


    # make sure to call this before every batch train.
    def init_hidden(self, height=None, width=None):
        if self.training:
            width = self.args.patch_size
            height = self.args.patch_size
            self.batch_size = self.args.batch_size
        else:
            ## keep the shape of testing data. We should specific the width and
            ## height through parameters
            self.batch_size = self.args.test_batch_size

            if self.args.test_patch_size == -1:
                if height is None or width is None:
                    raise ValueError('Test patch size should be setting in parameters before model defined')
            else:
                width = self.args.test_patch_size
                height = self.args.test_patch_size


        #self.lastNoiseImg = torch.zeros([self.batch_size, 3, height, width]).to(self.device)
        #print("shape is :", self.lastNoiseImg.shape)
        self.EstTargetImg = torch.zeros([self.batch_size, 3, height, width]).to(self.device)
        # height_gap = 2 / (self.height * self.SRFactor - 1)
        # width_gap = 2 / (self.width * self.SRFactor - 1)
        # height, width = torch.meshgrid([torch.range(-1, 1, height_gap), torch.range(-1, 1, width_gap)])
        # self.hr_identity = torch.stack([width, height]).to(device)

    # x is a 4-d tensor of shape N×C×H×W
    def forward(self, input):
        # Apply DNet
        dnInput = torch.cat((input, self.EstTargetImg), dim=1)
        #print(dnInput.shape)
        estImg = self.dnet(dnInput)
        #self.lastNoiseImg = input
        #print(self.lastNoiseImg.shape)
        self.EstTargetImg = estImg
        #print(self.EstTargetImg.shape)
        self.EstTargetImg.retain_grad()
        return self.EstTargetImg#, self.lastNoiseImg


class TestFRVD(unittest.TestCase):

    def testDNet(self):
        block = DNet(args)
        input = torch.rand(2, 6, 32, 56)
        output = block(input)

        #print("DNet shape:", output.shape)
        self.assertEqual(output.shape, torch.empty(2, 3, 32, 56).shape)

    def testFRVD(self):
        block = FRVD(args)
        #block.eval()
        if block.training:
            H = args.patch_size
            W = args.patch_size
            b = args.batch_size
        else:
            H = args.test_patch_size
            W = args.test_patch_size
            b = args.test_batch_size
        input = torch.rand(7, b, 3, H, W).to(torch.device("cuda"))
        #"cuda:0" if torch.cuda.is_available() else
        #block.init_hidden()
        for batch_frames in input:
            output1 = block(batch_frames)
            self.assertEqual(output1.shape, torch.empty(b, 3, H, W).shape)
            #self.assertEqual(output2.shape, torch.empty(b, 3, H, W).shape)


if __name__ == '__main__':
    unittest.main()
