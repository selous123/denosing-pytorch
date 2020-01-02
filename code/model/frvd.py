import unittest
import torch
import torch.nn as nn
import torch.nn.functional as func
from torchvision.models import vgg16
import sys
sys.path.append('..')
from option import args

from model import common
## Frame-Recurrent Video Denosing
def make_model(args, parent=False):
    _model = FRVD(args)
    return _model


class ConvLeaky(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConvLeaky, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim,
                               kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        out = self.conv1(input)
        out = func.leaky_relu(out, 0.2)
        out = self.conv2(out)
        out = func.leaky_relu(out, 0.2)
        return out

class FNetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, typ):
        super(FNetBlock, self).__init__()
        self.convleaky = ConvLeaky(in_dim, out_dim)
        if typ == "maxpool":
            self.final = lambda x: func.max_pool2d(x, kernel_size=2)
        elif typ == "bilinear":
            self.final = lambda x: func.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        else:
            raise Exception('Type does not match any of maxpool or bilinear')

    def forward(self, input):
        out = self.convleaky(input)
        out = self.final(out)
        return out

class FNet(nn.Module):
    def __init__(self, in_dim=6):
        super(FNet, self).__init__()
        self.convPool1 = FNetBlock(in_dim, 32, typ="maxpool")
        self.convPool2 = FNetBlock(32, 64, typ="maxpool")
        self.convPool3 = FNetBlock(64, 128, typ="maxpool")
        self.convBinl1 = FNetBlock(128, 256, typ="bilinear")
        self.convBinl2 = FNetBlock(256, 128, typ="bilinear")
        self.convBinl3 = FNetBlock(128, 64, typ="bilinear")
        self.seq = nn.Sequential(self.convPool1, self.convPool2, self.convPool3,
                                 self.convBinl1, self.convBinl2, self.convBinl3)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        out = self.seq(input)
        out = self.conv1(out)
        out = func.leaky_relu(out, 0.2)
        out = self.conv2(out)
        self.out = torch.tanh(out)
        self.out.retain_grad()
        return self.out

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

        self.fnet = FNet().to(self.device)
        self.dnet = DNet(args).to(self.device)  # 3 is channel number

        self.ofmap = None
        self.afterWarp = None

    # make sure to call this before every batch train.
    def init_hidden(self, height=None, width=None):
        if self.training:
            width = args.patch_size
            height = args.patch_size
            self.batch_size = args.batch_size
        else:
            ## keep the shape of testing data. We should specific the width and
            ## height through parameters
            self.batch_size = args.test_batch_size

            if args.test_patch_size == -1:
                if height is None or width is None:
                    raise ValueError('Test patch size should be setting in parameters before model defined')
            else:
                width = args.test_patch_size
                height = args.test_patch_size


        self.lastNoiseImg = torch.zeros([self.batch_size, 3, height, width]).to(self.device)
        #print("shape is :", self.lastNoiseImg.shape)
        self.EstTargetImg = torch.zeros([self.batch_size, 3, height, width]).to(self.device)
        height_gap = 2 / (height - 1)
        width_gap = 2 / (width - 1)
        #print("::::::", height, width, height_gap, width_gap)
        height, width = torch.meshgrid([torch.range(-1, 1, height_gap), torch.range(-1, 1, width_gap)])
        self.identity = torch.stack([width, height]).to(self.device)

        # height_gap = 2 / (self.height * self.SRFactor - 1)
        # width_gap = 2 / (self.width * self.SRFactor - 1)
        # height, width = torch.meshgrid([torch.range(-1, 1, height_gap), torch.range(-1, 1, width_gap)])
        # self.hr_identity = torch.stack([width, height]).to(device)

    # x is a 4-d tensor of shape N×C×H×W
    def forward(self, input):
        # Apply FNet
        # print(f'input.shape is {input.shape}, lastImg shape is {self.lastLrImg.shape}')
        # print(input.shape)
        # print(self.lastNoiseImg.shape)
        preflow = torch.cat((input, self.lastNoiseImg), dim=1)
        flow = self.fnet(preflow)

        self.ofmap = flow
        #print("arch of fnet:", self.fnet)
        #print("f shape:", flow.shape)
        #print("lr identity:", self.lr_identity)
        relative_place = flow + self.identity
        #print(torch.max(relative_place), torch.min(relative_place))
        ## For calculate loss
        relative_placeNWHC = relative_place.permute(0, 2, 3, 1)
        self.EstNoiseImg = func.grid_sample(self.lastNoiseImg, relative_placeNWHC)
        # print(self.EstNoiseImg)
        afterWarp = func.grid_sample(self.EstTargetImg, relative_placeNWHC)
        self.afterWarp = afterWarp  # for debugging, should be removed later.
        #depthImg = self.todepth(afterWarp)

        # Apply SRNet
        dnInput = torch.cat((input, afterWarp), dim=1)
        estImg = self.dnet(dnInput)
        self.lastNoiseImg = input
        self.EstTargetImg = estImg
        self.EstTargetImg.retain_grad()
        return self.EstTargetImg, self.EstNoiseImg

    def get_opticalflow_map(self):
        return self.ofmap

    def get_warpresult(self):
        return self.afterWarp

class TestFRVSR(unittest.TestCase):
    def testFNet(self):
        block = FNet()
        input = torch.rand(2, 6, 32, 56)
        output = block(input)

        #print(output.shape)
        self.assertEqual(output.shape, torch.empty(2, 2, 32, 56).shape)

    def testDNet(self):
        block = DNet(args)
        input = torch.rand(2, 6, 32, 56)
        output = block(input)

        #print("DNet shape:", output.shape)
        self.assertEqual(output.shape, torch.empty(2, 3, 32, 56).shape)

    def testFRVD(self):
        block = FRVD(args)
        block.eval()
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
        block.init_hidden()
        for batch_frames in input:
            output1, output2 = block(batch_frames)
            self.assertEqual(output1.shape, torch.empty(b, 3, H, W).shape)
            self.assertEqual(output2.shape, torch.empty(b, 3, H, W).shape)


if __name__ == '__main__':
    unittest.main()
