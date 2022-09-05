from models.encoder_decoder.DCGAN.layer import DECNR2d, Deconv2d

import torch
import torch.nn as nn
import torch.nn.functional as F


class DCGAN(nn.Module):
    def __init__(self, nch_in=100, nch_out=3, nch_ker=64, norm='bnorm'):
        super(DCGAN, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        self.dec5 = DECNR2d(1 * self.nch_in, 8 * self.nch_ker, kernel_size=4, stride=1, padding=0, norm=self.norm,
                            relu=0.0, drop=[])
        self.dec4 = DECNR2d(8 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm,
                            relu=0.0, drop=[])
        self.dec3 = DECNR2d(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm,
                            relu=0.0, drop=[])
        self.dec2 = DECNR2d(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm,
                            relu=0.0, drop=[])
        self.dec1 = Deconv2d(1 * self.nch_ker, 1 * self.nch_out, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):

        x = self.dec5(x)
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)

        x = torch.tanh(x)

        return x


if __name__ == '__main__':
    nch_in = 100
    nch_out = 3
    nch_ker = 64
    ny_in = nx_in = 1
    input = torch.randn(10, nch_in, ny_in, nx_in)
    netG = DCGAN()
    output = netG(input)
