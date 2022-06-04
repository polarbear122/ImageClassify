"""Train CIFAR10 with PyTorch."""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from log_config.log import logger as Log
from models import *
from utils import progress_bar
from dataset.generate_my_dataset import generate_dataset


# Training
def train(__epoch, __train_loader, __net):
    print('\nEpoch: %d' % __epoch)
    __net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(__train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = __net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(__train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(__epoch, __test_loader, __net):
    global best_acc
    __net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(__test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = __net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(__test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        Log.info("Epoch: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)" %
                 (__epoch, test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    # Save checkpoint.:q:i
    print('Saving..')
    acc = 100. * correct / total
    state = {
        'net'  : __net.state_dict(),
        'acc'  : acc,
        'epoch': __epoch,
    }
    if not os.path.isdir(ck_path):
        os.mkdir(ck_path)
    torch.save(state, ck_path + 'ckpt_update.pth')
    if acc > best_acc:
        torch.save(state, ck_path + 'ckpt' + str(__epoch) + '.pth')
        best_acc = acc


if __name__ == "__main__":
    ck_path = "checkpoint/ResNet18/"
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    # parser.add_argument('--resume', '-r', action='store_true',
    #                     help='resume from checkpoint')
    # parser.add_argument('--restart', action='restart', help='restart train')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')

    # 此处修改训练数据集
    train_loader, test_loader = generate_dataset()

    # Model
    print('==> Building model..')
    # net = VGG('VGG19')
    net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    # net = SimpleDLA()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    need_restart = True
    if need_restart is False:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(ck_path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(ck_path + 'ckpt_update.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(start_epoch, start_epoch + 200):
        train(epoch, train_loader, net)
        test(epoch, test_loader, net)
        scheduler.step()
