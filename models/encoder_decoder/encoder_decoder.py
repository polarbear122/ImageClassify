import os
import numpy as np
import torch.nn as nn


class RED_CNN(nn.Module):
    def __init__(self, out_ch=96):
        super(RED_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))

        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(2, 2), padding=0)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(2, 2), padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.tconv5 = nn.ConvTranspose2d(out_ch, 3, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))

        self.relu = nn.ReLU()
