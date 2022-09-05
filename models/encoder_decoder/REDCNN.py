# https://github.com/SSinyu/RED-CNN
import os
import numpy as np
import torch.nn as nn


class RED_CNN(nn.Module):
    def __init__(self, out_ch=96):
        super(RED_CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(2, 2), padding=0)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(2, 2), padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.tconv5 = nn.ConvTranspose2d(out_ch, 3, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))

        self.relu = nn.ReLU()

    def forward(self, x):
        # # encoder
        residual_1 = x
        print(x.shape)
        out0 = self.relu(self.conv1(x))
        out1 = self.relu(self.conv2(out0))
        residual_2 = out1
        out2 = self.relu(self.conv3(out1))
        out3 = self.relu(self.conv4(out2))
        residual_3 = out3
        out4 = self.relu(self.conv5(out3))
        print(out0.shape, out1.shape, out2.shape, out3.shape, out4.shape)
        # decoder
        out5 = self.tconv1(out4)
        out5 += residual_3
        out6 = self.tconv2(self.relu(out5))
        out7 = self.tconv3(self.relu(out6))
        print("out5", out5.shape, out6.shape, out7.shape)
        out7 += residual_2
        print(out4.shape, out5.shape, out6.shape, out7.shape)

        out8 = self.tconv4(self.relu(out7))
        out9 = self.tconv5(self.relu(out8))
        out9 += residual_1
        out10 = self.relu(out9)

        print(out8.shape, out9.shape, out10.shape)
        return out10


def RED_CNN_init():
    return RED_CNN(out_ch=96)
