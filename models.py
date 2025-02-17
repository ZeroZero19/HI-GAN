#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F


# Denoise DND and NAM
class UNet_ND(nn.Module):

    def __init__(self):
        super(UNet_ND, self).__init__()
        self.main = MainNet(in_nc=3, out_nc=12)
        self.main2 = MainNet(in_nc=15, out_nc=24)
        self.out = nn.Conv2d(24, 3, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        out1 = self.main(x)
        out1[:,:3,:,:] = out1[:,:3,:,:] + x
        out2 = self.main2(torch.cat([x, out1], dim=1))
        out2[:, :3, :, :] = out2[:, :3, :, :] + x
        out2[:, 12:, :, :] = out2[:, 12:, :, :] + out1

        return self.out(out2) + x

class UNet_D(nn.Module):

    def __init__(self):
        super(UNet_D, self).__init__()
        self.main = MainNet(in_nc=3, out_nc=12)
        self.main2 = MainNet(in_nc=15, out_nc=24)
        self.out = nn.Conv2d(24, 3, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        out1 = self.main(x)
        out1[:,:3,:,:] = out1[:,:3,:,:] + x
        out2 = self.main2(torch.cat([x, out1], dim=1))
        out2[:, :3, :, :] = out2[:, :3, :, :] + x
        out2[:, 12:, :, :] = out2[:, 12:, :, :] + out1

        return self.out(out2) + x

class HI_GAN(nn.Module):
    def __init__(self, ):
        super(HI_GAN, self).__init__()
        self.main = MainNet(in_nc=6, out_nc=6)
        self.main2 = MainNet(in_nc=12, out_nc=12)
        self.main3 = MainNet(in_nc=24, out_nc=24)
        self.out = nn.Conv2d(24, 3, kernel_size=3, padding=1, bias=True)
    def forward(self, unet_nd_dn, unet_d_dn):
        concat_img1 = torch.cat([unet_nd_dn, unet_d_dn], dim=1)
        out1 = 0.2*self.main(concat_img1) + concat_img1

        concat_img2 = torch.cat([concat_img1, out1], dim=1)
        out2 = 0.2*self.main2(concat_img2) + concat_img2

        concat_img3 = torch.cat([concat_img2, out2], dim=1)
        out3 = 0.2*self.main3(concat_img3) + concat_img3

        out = 0.2 * self.out(out3) + 0.5 * unet_nd_dn + 0.5 * unet_d_dn
        return out

# Denoise Cell
class UNet_ND_cell(nn.Module):

    def __init__(self):
        super(UNet_ND_cell, self).__init__()
        self.main = MainNet(in_nc=1, out_nc=1)
        self.main2 = MainNet(in_nc=2, out_nc=2)
        self.out = nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        out1 = self.main(x) + x
        cat1 = torch.cat([x, out1], dim=1)
        out2 = self.main2(cat1) + cat1
        cat2 = torch.cat([x,out1, out2], dim=1)
        return self.out(cat2) + x

class UNet_D_cell(nn.Module):

    def __init__(self):
        super(UNet_D_cell, self).__init__()
        self.main = MainNet(in_nc=1, out_nc=1)
        self.main2 = MainNet(in_nc=2, out_nc=2)
        self.out = nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        out1 = self.main(x) + x
        cat1 = torch.cat([x, out1], dim=1)
        out2 = self.main2(cat1) + cat1
        cat2 = torch.cat([x,out1, out2], dim=1)
        return self.out(cat2) + x

class HI_GAN_cell(nn.Module):
    def __init__(self, ):
        super(HI_GAN_cell, self).__init__()
        self.main = MainNet(in_nc=2, out_nc=2)
        self.main2 = MainNet(in_nc=4, out_nc=4)
        self.main3 = MainNet(in_nc=8, out_nc=8)
        self.out = nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=True)
    def forward(self, unet_nd_dn, unet_d_dn):
        cat1 = torch.cat([unet_nd_dn, unet_d_dn], dim=1)
        out1 = 0.2*self.main(cat1) + cat1

        cat2 = torch.cat([cat1, out1], dim=1)
        out2 = 0.2*self.main2(cat2) + cat2

        cat3 = torch.cat([cat2, out2], dim=1)
        out3 = 0.2*self.main3(cat3) + cat3

        out = 0.2 * self.out(out3) + 0.5 * unet_nd_dn + 0.5 * unet_d_dn
        return out


## Sub classes
class MainNet(nn.Module):
    """B-DenseUNets"""
    def __init__(self, in_nc=12, out_nc=12):
        super(MainNet, self).__init__()
        self.inc = nn.Sequential(
            single_conv(in_nc, 64),
            single_conv(64, 64),
        )
        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            single_conv(64, 128),
            RDB(128, 4, 32),
        )
        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            single_conv(128, 256),
            RDB(256, 10, 32),
        )
        self.up1 = up(256)
        self.conv3 = nn.Sequential(
            RDB(128, 6, 32),
        )
        self.up2 = up(128)
        self.conv4 = nn.Sequential(
            RDB(64, 4, 32),
        )
        self.outc = outconv(64, out_nc)
    def forward(self, x):
        inx = self.inc(x)

        down1 = self.down1(inx)
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)

        up1 = self.up1(conv2, conv1)
        conv3 = self.conv3(up1)

        up2 = self.up2(conv3, inx)
        conv4 = self.conv4(up2)

        out = self.outc(conv4)
        return out

class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = x2 + x1
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out

class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

