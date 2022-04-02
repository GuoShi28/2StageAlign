''' network architecture for backbone '''
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import models.archs.arch_util as arch_util
import numpy as np
import math
import pdb
from torch.nn.modules.utils import _pair
from models.archs.BaseBlocks import ACNSeBlock, SimpleBlock
from models.archs.DAlign_Block import Easy_PCD, Self_Easy_PCD
from .arch_util import flow_warp
from utils.util import discard_module_prefix
# import matplotlib.pyplot as plt
from torch.autograd import Variable

class BiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, wInter):
        super(BiGRU, self).__init__()
        self.forward_net = ConvGRU(hidden_dim=hidden_dim,input_dim=input_dim, wInter=wInter)
        self.conv_1x1 = nn.Conv2d(2*input_dim, input_dim, 1, 1, bias=True)
        self.wInter = wInter

    def forward(self, x):
        center_frame = x.shape[1]//2
        reversed_idx = list(reversed(range(x.shape[1])))
        x_rev = x[:, reversed_idx, ...]
        if self.wInter and self.training:
            out_fwd, inter_fwd = self.forward_net(x)
            out_rev, inter_rev = self.forward_net(x_rev)
        else:
            out_fwd = self.forward_net(x)
            out_rev = self.forward_net(x_rev)

        rev_rev = out_rev[:, reversed_idx, ...]
        B, N, C, H, W = out_fwd.size()
        # center_frame = N // 2
        result = torch.cat((out_fwd, rev_rev), dim=2)
        # result = result.view(B*N,-1,H,W)
        result = result[:,center_frame,...]
        result = self.conv_1x1(result)
          
        if self.wInter and self.training:
            inter_result = torch.cat((inter_fwd, inter_rev), dim=1)
            inter_result = self.conv_1x1(inter_result)
            result = torch.cat((result, inter_result), dim=0)
            result.view(2*B, -1, C, H, W)
        else:
            result.view(B, -1, C, H, W) 

        return result  

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=64, input_dim=64, wInter=True):
        super(ConvGRU, self).__init__()
        self.hidden_dim=hidden_dim
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.wInter = wInter

    def forward(self, x, hidden_state=None):
        # x in size: B, N, C, H, W
        ## init hidden state
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            tensor_size = (x.size(3),x.size(4))
            hidden_state = self._init_hidden(batch_size=x.size(0),tensor_size=tensor_size)

        layer_output_list = []
        last_state_list = []

        seq_len = x.size(1)
        output_inner = []
        
        for t in range(seq_len):
            if self.wInter and t==seq_len//2 and self.training:
                inter_inner = hidden_state
            in_tensor = x[:, t, :, :, :].clone() 
            hx = torch.cat([hidden_state, in_tensor], dim=1)
            hid_z = torch.sigmoid(self.convz(hx))
            hid_r = torch.sigmoid(self.convr(hx))
            # pdb.set_trace()
            hid_q = torch.tanh(self.convq(torch.cat([hid_r*hidden_state, in_tensor], dim=1)))
            hidden_state = (1-hid_z) * hidden_state + hid_z * hid_q
            output_inner.append(hidden_state)

        layer_output = torch.stack(output_inner, dim=1)
        # pdb.set_trace()
        if self.wInter and self.training:
            return layer_output, inter_inner
        else:
            return layer_output

    def _init_hidden(self, batch_size, tensor_size):
        height, width = tensor_size
        return Variable(torch.zeros(batch_size, self.hidden_dim, height, width)).cuda()


class JDDB_BiGRU(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, in_channel=1, output_channel=1, center=None, wInter=True):
        super(JDDB_BiGRU, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.nframes = nframes
        self.wInter = wInter
        ## feature extract and feature demosaic
        input_scaleing = 4*2
        self.feature_extract = SimpleBlock(depth=5, n_channels=nf, input_channels=in_channel*input_scaleing, \
            output_channel=nf, kernel_size=3) # 64, H, W
        self.feature_extract_acse1 = ACNSeBlock(res_num=2, n_channels=nf, input_channels=nf, \
            output_channel=nf, kernel_size=3) # 64, H, W
        self.feature_extract_acse2 = ACNSeBlock(res_num=2, n_channels=nf, input_channels=nf, \
            output_channel=nf, kernel_size=3) # 64, H, W
        self.feature_extract_acse3 = ACNSeBlock(res_num=2, n_channels=nf, input_channels=nf, \
            output_channel=nf, kernel_size=3) # 64, H*2, W*2
        self.feature_extract_acse4 = ACNSeBlock(res_num=2, n_channels=nf, input_channels=nf, \
            output_channel=nf, kernel_size=3) # 64, H*2, W*2
        # pdb.set_trace()
        ## EDVR part: refined alignment
        self.pcd_align = Self_Easy_PCD(nf=nf, groups=groups)
        ## GRU merge part
        self.ConvBGRU = BiGRU(input_dim=nf, hidden_dim=nf, wInter=self.wInter)
        
        # self.merge = nn.Conv2d(nf*nframes, nf, 3, 1, 1, bias=True)

        self.feature_up = nn.ConvTranspose2d(in_channels=nf, out_channels=nf,\
            kernel_size=2, stride=2, padding=0, bias=True) # 64, H*2, W*2

        # encoder
        self.conv_block_s1 = SimpleBlock(depth=2, n_channels=nf, input_channels=nf, \
            output_channel=nf, kernel_size=3) # 64, H*2, W*2
        self.acse_block_s1 = ACNSeBlock(res_num=2, n_channels=nf, input_channels=nf, \
            output_channel=nf, kernel_size=3) # 64, H*2, W*2
        self.pool1 = nn.Conv2d(nf, 2*nf, 3, 2, 1, bias=True) # 128 
        
        self.conv_block_s2 = SimpleBlock(depth=2, n_channels=2*nf, input_channels=2*nf, \
            output_channel=2*nf, kernel_size=3) # 128, H, W
        self.acse_block_s2 = ACNSeBlock(res_num=2, n_channels=2*nf, input_channels=2*nf, \
            output_channel=2*nf, kernel_size=3) # 128, H, W
        self.pool2 = nn.Conv2d(2*nf, 4*nf, 3, 2, 1, bias=True) # 256

        self.conv_block_s3 = SimpleBlock(depth=2, n_channels=4*nf, input_channels=4*nf, \
            output_channel=4*nf, kernel_size=3) # 256, H//2, W//2
        self.acse_block_s3 = ACNSeBlock(res_num=2, n_channels=4*nf, input_channels=4*nf, \
            output_channel=4*nf, kernel_size=3) # 256, H//2, W//2
        self.conv_block_s3_2 = SimpleBlock(depth=2, n_channels=4*nf, input_channels=4*nf, \
            output_channel=4*nf, kernel_size=3) # 256, H//2, W//2
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # decoder
        self.up1 = nn.ConvTranspose2d(in_channels=4*nf, out_channels=2*nf,\
            kernel_size=2, stride=2, padding=0, bias=True) # 128, H, W
        ### With SkipConnection
        # cat with conv_block_s4 # 128, H, W
        self.conv_block_s4 = SimpleBlock(depth=2, n_channels=2*nf, input_channels=4*nf, \
            output_channel=2*nf, kernel_size=3) # 128, H, W
        self.acse_block_s4 = ACNSeBlock(res_num=2, n_channels=2*nf, input_channels=2*nf, \
            output_channel=2*nf, kernel_size=3) # 128, H, W
        
        self.up2 = nn.ConvTranspose2d(in_channels=2*nf, out_channels=nf,\
            kernel_size=2, stride=2, padding=0, bias=True) # 64, H*2, W*2
        # cat with conv_block_s3 # 64, H*2, W*2
        self.conv_block_s5 = SimpleBlock(depth=3, n_channels=nf, input_channels=2*nf, \
            output_channel=output_channel, kernel_size=3) # 64, H*2, W*2
        

    def forward(self, x, nmap):
        B, N, C, H, W = nmap.size()  # N video frames, C is 4 response to RGGB channel
        center_idx = N//2
        x_temp = x.view(-1, C, H, W)
        x_nm_temp = nmap.view(-1, C, H, W)
        temp = torch.cat([x_temp, x_nm_temp], dim=1)

        x_s1 = self.feature_extract(temp)          # B*N, fea_C, H, W
        # print(x_s1.shape)
        x_s1 = self.feature_extract_acse1(x_s1)
        x_s1 = self.feature_extract_acse2(x_s1)
        x_s1 = self.feature_extract_acse3(x_s1) # B*N, fea_C, H, W
        x_s1 = self.feature_extract_acse4(x_s1) # B*N, fea_C, H, W
        
        # ----- align multi frame -----
        x_s1 = x_s1.view(-1, N, self.nf, H, W)
        ## refine alignment using DConv
        x_s1 = self.pcd_align(x_s1)
        # ------ merge aligned features using Bi-GRU
        x_s1 = self.ConvBGRU(x_s1)# [B*N, fea, H, W] -> [B, N, fea, H, W]
        x_s1 = x_s1.view(-1, self.nf, H, W)

        #x_s1 = x_s1.view(-1, self.nf*N, H, W)
        #x_s1 = self.merge(x_s1)

        # refine net: encoder -- decoder
        x_s1 = self.feature_up(x_s1) # B, fea_C, H*2, W*2
        ###
        x_s1 = self.conv_block_s1(x_s1)       # 64, H*2, W*2
        x_s1 = self.acse_block_s1(x_s1)
        ###
        L1_temp = x_s1.clone()
        ###
        x_s2 = self.pool1(x_s1)               # 128, H, W
        x_s2 = self.conv_block_s2(x_s2)       # 128, H, W
        x_s2 = self.acse_block_s2(x_s2)       # 128, H, W
        ###
        L2_temp = x_s2.clone()
        ###
        x_s3 = self.pool2(x_s2)               # 256, H//2, W//2
        x_s3 = self.conv_block_s3(x_s3)       # 256, H//2, W//2
        x_s3 = self.acse_block_s3(x_s3)       # 256, H//2, W//2
        x_s3 = self.conv_block_s3_2(x_s3)       # 256, H//2, W//2
        
        # decoder
        out = self.up1(x_s3)                 # 128, H, W
        out = torch.cat((out, L2_temp), 1)      # 256, H, W
        out = self.conv_block_s4(out)        # 128, H, W
        out = self.acse_block_s4(out)        # 128, H, W

        out = self.up2(out)                  # 64, H*2, W*2
        out = torch.cat((out, L1_temp), 1)      # 128, H*2, W*2
        out = self.conv_block_s5(out)        # out_ch, H, W
        # pdb.set_trace()
        #if self.training==False:
        #    out = out[:, :, 0:2*H_ori, 0:2*W_ori]

        # pdb.set_trace()
        if self.training:
            return out
        else:
            return out





