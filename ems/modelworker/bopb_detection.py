# The scratch detector of "Bringing Old Photos Back to Life" (Wan et al., 2020), as Microsoft
# publishes it: microsoft/Bringing-Old-Photos-Back-to-Life at 33875eccf4ebcd3665cf38cc56f3a0ce563d3a9c,
# Global/detection_models/networks.py (UNet and its blocks) and antialiasing.py (Downsample),
# unchanged below this header but for what inference does not reach: the other networks of
# networks.py, the synchronised batch norm import, and its wrapper, which UNet assigns to a local
# name and so never applies. Used by defects.py.
#
# MIT License
#
# Copyright (c) Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample(nn.Module):
    # https://github.com/adobe/antialiased-cnns

    def __init__(self, pad_type="reflect", filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels

        # print('Filter size [%i]'%filt_size)
        if self.filt_size == 1:
            a = np.array([1.0,])
        elif self.filt_size == 2:
            a = np.array([1.0, 1.0])
        elif self.filt_size == 3:
            a = np.array([1.0, 2.0, 1.0])
        elif self.filt_size == 4:
            a = np.array([1.0, 3.0, 3.0, 1.0])
        elif self.filt_size == 5:
            a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif self.filt_size == 6:
            a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif self.filt_size == 7:
            a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])

        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer("filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, :: self.stride, :: self.stride]
            else:
                return self.pad(inp)[:, :, :: self.stride, :: self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer(pad_type):
    if pad_type in ["refl", "reflect"]:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ["repl", "replicate"]:
        PadLayer = nn.ReplicationPad2d
    elif pad_type == "zero":
        PadLayer = nn.ZeroPad2d
    else:
        print("Pad type [%s] not recognized" % pad_type)
    return PadLayer


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        depth=5,
        conv_num=2,
        wf=6,
        padding=True,
        batch_norm=True,
        up_mode="upsample",
        with_tanh=False,
        sync_bn=True,
        antialiasing=True,
    ):
        """
		Implementation of
		U-Net: Convolutional Networks for Biomedical Image Segmentation
		(Ronneberger et al., 2015)
		https://arxiv.org/abs/1505.04597
		Using the default arguments will yield the exact version used
		in the original paper
		Args:
			in_channels (int): number of input channels
			out_channels (int): number of output channels
			depth (int): depth of the network
			wf (int): number of filters in the first layer is 2**wf
			padding (bool): if True, apply padding such that the input shape
							is the same as the output.
							This may introduce artifacts
			batch_norm (bool): Use BatchNorm after layers with an
							   activation function
			up_mode (str): one of 'upconv' or 'upsample'.
						   'upconv' will use transposed convolutions for
						   learned upsampling.
						   'upsample' will use bilinear upsampling.
		"""
        super().__init__()
        assert up_mode in ("upconv", "upsample")
        self.padding = padding
        self.depth = depth - 1
        prev_channels = in_channels

        self.first = nn.Sequential(
            *[nn.ReflectionPad2d(3), nn.Conv2d(in_channels, 2 ** wf, kernel_size=7), nn.LeakyReLU(0.2, True)]
        )
        prev_channels = 2 ** wf

        self.down_path = nn.ModuleList()
        self.down_sample = nn.ModuleList()
        for i in range(depth):
            if antialiasing and depth > 0:
                self.down_sample.append(
                    nn.Sequential(
                        *[
                            nn.ReflectionPad2d(1),
                            nn.Conv2d(prev_channels, prev_channels, kernel_size=3, stride=1, padding=0),
                            nn.BatchNorm2d(prev_channels),
                            nn.LeakyReLU(0.2, True),
                            Downsample(channels=prev_channels, stride=2),
                        ]
                    )
                )
            else:
                self.down_sample.append(
                    nn.Sequential(
                        *[
                            nn.ReflectionPad2d(1),
                            nn.Conv2d(prev_channels, prev_channels, kernel_size=4, stride=2, padding=0),
                            nn.BatchNorm2d(prev_channels),
                            nn.LeakyReLU(0.2, True),
                        ]
                    )
                )
            self.down_path.append(
                UNetConvBlock(conv_num, prev_channels, 2 ** (wf + i + 1), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i + 1)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up_path.append(
                UNetUpBlock(conv_num, prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        if with_tanh:
            self.last = nn.Sequential(
                *[nn.ReflectionPad2d(1), nn.Conv2d(prev_channels, out_channels, kernel_size=3), nn.Tanh()]
            )
        else:
            self.last = nn.Sequential(
                *[nn.ReflectionPad2d(1), nn.Conv2d(prev_channels, out_channels, kernel_size=3)]
            )


    def forward(self, x):
        x = self.first(x)

        blocks = []
        for i, down_block in enumerate(self.down_path):
            blocks.append(x)
            x = self.down_sample[i](x)
            x = down_block(x)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, conv_num, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        for _ in range(conv_num):
            block.append(nn.ReflectionPad2d(padding=int(padding)))
            block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=0))
            if batch_norm:
                block.append(nn.BatchNorm2d(out_size))
            block.append(nn.LeakyReLU(0.2, True))
            in_size = out_size

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, conv_num, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == "upsample":
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2, align_corners=False),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_size, out_size, kernel_size=3, padding=0),
            )

        self.conv_block = UNetConvBlock(conv_num, in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


