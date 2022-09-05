# https://zhuanlan.zhihu.com/p/355731441
import argparse
import os
import torch.nn as nn

import torch.optim as optim
import torch.utils.data

import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from config import config_dataset_root
from generate_my_dataset import MyDataset
from models.encoder_decoder.GANAutoEncoder.discriminator import Discriminator
from models.encoder_decoder.GANAutoEncoder.generator import DCGAN_init


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


if __name__ == "__main__":
    epochs = 100
    BatchSize = 20
    lr = 0.0002
    beta1 = 0.5
    over = 4
    try:
        os.makedirs("result/train/real")
    except OSError:
        pass

    root = config_dataset_root  # 调用图像
    transform_method = transforms.ToTensor()
    train_data = MyDataset(txt=root + 'train.txt', transform=transform_method, pose_arr_numpy=[])
    assert train_data
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=BatchSize,
                                             shuffle=True, num_workers=2)
    wtl2 = 0.999

    netG = DCGAN_init()
    netG.apply(weights_init)

    netD = Discriminator()
    netD.apply(weights_init)

    criterion = nn.CrossEntropyLoss()
    criterionMSE = nn.MSELoss()

    real_label = 1
    fake_label = 0

    netD.cuda()
    netG.cuda()
    criterion.cuda()
    criterionMSE.cuda()

    optimD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for epoch in range(0, epochs):
        netG.train()
        netD.train()

        loss_G_train = []
        loss_D_real_train = []
        loss_D_fake_train = []
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            input_real, target = inputs.to(device), targets.to(device)
            print("input_real shape", input_real.shape)
            # start
            output = netG(input_real)
            # backward netD
            set_requires_grad(netD, True)
            optimD.zero_grad()

            print("output shape", output.shape)
            pred_real = netD(input_real)
            pred_fake = netD(output.detach())
            print("torch.ones_like(pred_real)", torch.ones_like(pred_real).shape)
            print("pred_real", pred_real.shape)
            loss_D_real = criterion(pred_real, torch.ones_like(pred_real))
            loss_D_fake = criterion(pred_fake, torch.zeros_like(pred_fake))
            loss_D = 0.5 * (loss_D_real + loss_D_fake)

            loss_D.backward()
            optimD.step()

            # backward netG
            set_requires_grad(netD, False)
            optimG.zero_grad()
            print("output shape", output.shape)
            pred_fake = netD(output)

            loss_G = criterion(pred_fake, torch.ones_like(pred_fake))

            loss_G.backward()
            optimG.step()

            # get losses
            loss_G_train += [loss_G.item()]
            loss_D_real_train += [loss_D_real.item()]
            loss_D_fake_train += [loss_D_fake.item()]

            vutils.save_image(input_real, 'result/train/real/real_samples_epoch_%03d.png' % epoch)
