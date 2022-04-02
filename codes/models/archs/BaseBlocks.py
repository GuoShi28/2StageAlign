import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class SimpleBlock(nn.Module):
    def __init__(self, depth=3, n_channels=64, input_channels=3, output_channel=64, kernel_size=3):
        super(SimpleBlock, self).__init__()
        padding = 1
        layers = []
        layers.append(nn.Conv2d(in_channels=input_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            # layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            # layers.append(nn.ReLU(inplace=True))
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=output_channel, kernel_size=kernel_size, padding=padding, bias=False))
        self.simple_block = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        out = self.simple_block(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

class ACNSeBlock(nn.Module):
    def __init__(self, res_num=5, n_channels=64, input_channels=3, output_channel=64, kernel_size=3):
        super(ACNSeBlock, self).__init__()
        padding = 1
        self.res_num = res_num
        self.square_conv = nn.Conv2d(in_channels=input_channels, out_channels=n_channels, \
            kernel_size=(kernel_size, kernel_size), padding=(padding, padding), bias=False)

        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.extract_conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, \
            kernel_size=kernel_size, padding=padding, bias=True)
      
        self.res_block1 = SimpleBlock(depth=2, n_channels=n_channels, input_channels=n_channels, \
                output_channel=n_channels, kernel_size=3)  # 64, H, W
        self.res_block2 = SimpleBlock(depth=2, n_channels=n_channels, input_channels=n_channels, \
                output_channel=n_channels, kernel_size=3) # 64, H, W

        self.down = nn.Conv2d(in_channels=n_channels, out_channels=int(n_channels/2), kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=int(n_channels/2), out_channels=n_channels, kernel_size=1, stride=1, bias=True)
        self.spatial_att = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=7, stride=1, padding=3,bias=True)

        self._initialize_weights()

    def forward(self, x):
        # a Asymmetric Convolution Blocks
        x_temp = self.square_conv(x)
        x_temp = self.relu(x_temp)
        x_temp = self.extract_conv(x_temp)
        x_temp = x + x_temp
        # pdb.set_trace()
        '''
        for i in range(self.res_num):
            pdb.set_trace()
            x_temp2 = self.res_block[i](x_temp)
            x_temp = x_temp + x_temp2
        '''
        x_temp2 = self.res_block1(x_temp)
        x_temp = x_temp + x_temp2
        x_temp2 = self.res_block2(x_temp)
        x_temp = x_temp + x_temp2

        # channel attention
        x_se = F.avg_pool2d(x_temp, kernel_size=(x_temp.size(2), x_temp.size(3)))
        x_se = self.down(x_se)
        x_se = self.relu(x_se)
        x_se = self.up(x_se)
        x_se = F.sigmoid(x_se)
        x_se = x_se.repeat(1, 1, x_temp.size(2), x_temp.size(3))
        # spatial attention
        x_sp = F.sigmoid(self.spatial_att(x_temp))
        
        x_temp = x_temp + x_temp * x_se + x_temp * x_sp 
        
        return x_temp

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
