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
from models.encoder_decoder.GANAutoEncoder.generator import Generator


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == "__main__":
    epochs = 100
    Batch_Size = 12
    lr = 0.0002
    beta1 = 0.5
    over = 4
    parser = argparse.ArgumentParser()

    opt = parser.parse_args()
    try:
        os.makedirs("result/train/cropped")
        os.makedirs("result/train/real")
        os.makedirs("result/train/recon")
        os.makedirs("model")
    except OSError:
        pass

    root = config_dataset_root  # 调用图像
    transform_method = transforms.ToTensor()
    train_data = MyDataset(txt=root + 'train.txt', transform=transform_method, pose_arr_numpy=[])
    assert train_data
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=Batch_Size,
                                             shuffle=True, num_workers=2)
    wtl2 = 0.999

    netG = Generator()
    netG.apply(weights_init)

    netD = Discriminator()
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    criterionMSE = nn.MSELoss()

    input_real = torch.FloatTensor(Batch_Size, 3, 50, 50)
    label = torch.FloatTensor(Batch_Size)
    real_label = 1
    fake_label = 0

    netD.cuda()
    netG.cuda()
    criterion.cuda()
    criterionMSE.cuda()
    input_real, label = input_real.cuda(), label.cuda()

    input_real = Variable(input_real)

    label = Variable(label)

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    for epoch in range(0, epochs):
        for i, data in enumerate(dataloader, 0):
            real_cpu, _ = data

            batch_size = real_cpu.size(0)
            with torch.no_grad():
                input_real.resize_(real_cpu.size()).copy_(real_cpu)

            # start the discriminator by training with real data---
            netD.zero_grad()
            with torch.no_grad():
                label.resize_(batch_size).fill_(real_label)

            output = netD(input_real)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.data.mean()

            # train the discriminator with fake data---
            fake = netG(input_real)
            label.data.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()

            # train the generator now---
            netG.zero_grad()
            label.data.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG_D = criterion(output, label)

            wtl2Matrix = input_real.clone()
            wtl2Matrix.data.fill_(wtl2 * 10)
            wtl2Matrix.data[:, :, int(over):int(128 / 2 - over), int(over):int(128 / 2 - over)] = wtl2

            errG_l2 = (fake - input_real).pow(2)
            errG_l2 = errG_l2 * wtl2Matrix
            errG_l2 = errG_l2.mean()

            errG = (1 - wtl2) * errG_D + wtl2 * errG_l2

            errG.backward()

            D_G_z2 = output.data.mean()
            optimizerG.step()

            print('[%d / %d][%d / %d] Loss_D: %.4f Loss_G: %.4f / %.4f l_D(x): %.4f l_D(G(z)): %.4f'
                  % (epoch, epochs, i, len(dataloader),
                     errD.data, errG_D.data, errG_l2.data, D_x, D_G_z1,))

            if i % 100 == 0:
                vutils.save_image(real_cpu,
                                  'result/train/real/real_samples_epoch_%03d.png' % (epoch))
