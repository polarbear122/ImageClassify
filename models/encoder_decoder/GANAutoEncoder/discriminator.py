import torch
from torch import nn


class Discriminator(nn.Module):

    # discriminator model
    def __init__(self):
        super(Discriminator, self).__init__()

        self.t1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.t2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.t3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.t4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.t5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(4, 4), stride=(1, 1), padding=0),
            nn.Sigmoid()
        )
        # self.linear = nn.Linear(4, 1)
        # self.squeeze = torch.squeeze

    def forward(self, x):
        print("0", x.shape)
        x = self.t1(x)
        print("1", x.shape)
        x = self.t2(x)
        print("2", x.shape)
        x = self.t3(x)
        print("3", x.shape)
        x = self.t4(x)
        print("4", x.shape)
        x = self.t5(x)
        print("5", x.shape)
        # x = self.squeeze(x)
        # print("6", x.shape)
        return x  # output of discriminator
