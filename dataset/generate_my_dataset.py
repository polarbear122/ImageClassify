# 创建自己的数据集
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
from log_config.log import logger as Log
import cv2
import json


def cv2PIL(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


def PIL2cv(img_pil):
    return cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)


def read_json(json_path):
    json_data = open(json_path)
    json_string = json_data.read()
    j = json.loads(json_string)
    return j


# 定义读取文件的格式
def default_loader(path):
    # img = Image.open(path).convert('RGB')
    img = cv2.imread(path)
    path = path.split('/')
    video_name = path[5]
    video_id = int(video_name[6:10])
    img_name = path[6]
    img_id = int(img_name.split('.')[0])
    img = cv2.resize(img, (10, 25))
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    alpha_pose_path = "E:/CodeResp/pycode/DataSet/pose_result" + video_name + "/alphapose-results.json"
    alpha_pose = read_json(alpha_pose_path)
    for pose in alpha_pose:
        if pose["score"] < 1:
            continue
        if pose["image_id"] == img_name:
            pose_box = [pose["box"][0], pose["box"][1], pose["box"][0] + pose["box"][2],
                        pose["box"][1] + pose["box"][3]]
            tl_width_height_box = pose["box"]  # 获取注释中的box，(左上角点，宽高)格式
        #     true_box = [xtl, ytl, xbr, ybr]
        #     iou_val = box_iou(pose_box, true_box)
        #     if iou_val > max_iou:
        #         x_keypoints_proposal = get_key_points(pose["keypoints"])
        #         max_pose_box = tl_width_height_box
        #         plot_max_box = pose_box
        #         img_frame_id = int(annotation["@frame"])
        #         max_iou = iou_val
        # elif pose["image_id"] == str(int(annotation["@frame"]) + 1) + ".jpg":
        #     break

    return img


# 首先继承上面的dataset类。然后在__init__()方法中得到图像的路径，然后将图像路径组成一个数组，这样在__getitim__()中就可以直接读取：
class MyDataset(Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()  # 对继承自父类的属性进行初始化
        fh = open(txt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []
        for line in fh:  # 迭代该列表#按行循环txt文本中的内
            line = line.strip('\n')
            line = line.rstrip('\n')  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()  # 用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
            imgs.append((words[0], int(words[1])))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = self.loader(fn)  # 按照路径读取图片
        if self.transform is not None:
            img = self.transform(img)  # 数据标签转换为Tensor
        return img, label  # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


def generate_dataset():
    # torch.cuda.set_device(gpu_id)#使用GPU
    learning_rate = 0.0001

    # 数据集的设置**************************************************************************
    root = "dataset/txt_init_img/video30/"  # 调用图像

    # 根据自己定义的那个MyDataset来创建数据集！注意是数据集！而不是loader迭代器
    # *********************************************数据集读取完毕***************************
    # 图像的初始化操作
    # train_transforms = transforms.Compose([
    #     transforms.RandomResizedCrop((227, 227)),
    #     transforms.ToTensor(),
    # ])
    # text_transforms = transforms.Compose([
    #     transforms.RandomResizedCrop((227, 227)),
    #     transforms.ToTensor(),
    # ])

    # 数据集加载方式设置
    train_data = MyDataset(txt=root + 'train.txt', transform=transforms.ToTensor())
    test_data = MyDataset(txt=root + 'test.txt', transform=transforms.ToTensor())
    # 然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=False, num_workers=4)
    print('num_of_trainData:', len(train_data))
    print('num_of_testData:', len(test_data))
    Log.info('num_of_trainData:%d' % (len(train_data)))
    Log.info('num_of_testData:%d' % (len(test_data)))
    return train_loader, test_loader


if __name__ == "__main__":
    generate_dataset()
