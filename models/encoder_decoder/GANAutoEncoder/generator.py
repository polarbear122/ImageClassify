import torch
from torch import nn

from models.encoder_decoder.DCGAN.layer import DECNR2d, Deconv2d


class DCGAN(nn.Module):
    def __init__(self, nch_in=512, nch_out=3, nch_ker=64, norm='bnorm'):
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

        self.t1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.t2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.t3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.t4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.t5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.t6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(2, 2)),
            # bottleneck
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

    def forward(self, x):
        print("gen 00", x.shape)
        x = self.t1(x)
        print("gen 01", x.shape)
        x = self.t2(x)
        print("gen 02", x.shape)
        x = self.t3(x)
        print("gen 03", x.shape)
        x = self.t4(x)
        print("gen 04", x.shape)
        x = self.t5(x)
        print("gen 05", x.shape)
        x = self.t6(x)
        print("gen 06", x.shape)
        print("x shape", x.shape)
        x = self.dec5(x)
        print("dec5 shape", x.shape)
        x = self.dec4(x)
        print("dec4 shape", x.shape)
        x = self.dec3(x)
        print("dec3 shape", x.shape)
        x = self.dec2(x)
        print("dec2 shape", x.shape)
        x = self.dec1(x)
        print("dec1 shape", x.shape)
        x = torch.tanh(x)
        print("tach shape", x.shape)
        return x


def DCGAN_init():
    return DCGAN()
