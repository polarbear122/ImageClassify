import torch
from torch import nn


class Generator(nn.Module):

    # Generator model
    def __init__(self):
        super(Generator, self).__init__()

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
            nn.Conv2d(512, 4000, kernel_size=(2, 2)),
            # bottleneck
            nn.BatchNorm2d(4000),
            nn.ReLU(),
        )
        self.t7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4000, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.t8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.t9 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.t10 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Tanh()
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
        x = self.t7(x)
        print("gen 07", x.shape)
        x = self.t8(x)
        print("gen 08", x.shape)
        x = self.t9(x)
        print("gen 09", x.shape)
        x = self.t10(x)
        print("gen 10", x.shape)
        return x  # output of generator
