"""Train CIFAR10 with PyTorch."""
import argparse
import os

import torch.backends.cudnn as cudnn
import torch.optim
from torch import optim

from generate_my_dataset import generate_dataset
from log_config.log import logger as Log
from models import *
from utils import progress_bar


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
    Log.info("TrainEpoch: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)" %
             (__epoch, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


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
        print(test_loss)
        Log.info("TestEpoch : %d | Loss: %.3f | Acc: %.3f%% (%d/%d)" %
                 (__epoch, test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    # Save checkpoint.
    print('Saving..')
    acc = 100. * correct / total
    state = {
        'net': __net.state_dict(),
        'acc': acc,
        'epoch': __epoch,
    }
    if not os.path.isdir(ck_path):
        os.mkdir(ck_path)
    torch.save(state, ck_path + 'ckpt_update.pth')
    if acc > best_acc:
        print("本轮训练精度优于最佳精度, epoch: %d ,acc: %0.3f%% ,pre_best_acc: %.3f%%," % (__epoch, acc, best_acc))
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
    best_acc = 0  # best test.py accuracy
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
    print("net:", net)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    need_restart = True  # 是否重新开始训练
    if need_restart is False:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(ck_path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(ck_path + 'ckpt_update.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1,
                          momentum=0.95, weight_decay=1e-3)
    # optimizer = optim.SGD(net.parameters(), lr=0.1)
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.2, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400)

    for epoch in range(start_epoch, start_epoch + 400):
        train(epoch, train_loader, net)
        test(epoch, test_loader, net)
        scheduler.step()
