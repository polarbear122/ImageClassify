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
import pandas as pd
import os
from log_config.log import logger as Log
import cv2
import json

pose_arr_list = []  # 以list形式保存pose arr数组，不存在的数组用0数组代替
pose_arr_position = [0]  # 记录每个视频pose的长度位置，取video_id位置的数据pose_arr_list[video_id-1,video_id]
pose_arr_numpy = np.zeros((1, 1))  # 以numpy数组形式保存pose arr数组，读取时需结合pose arr position使用


def init_read_pose_annotation():
    data_path = "D:/CodeResp/IRBOPP/train/halpe26_data/data_by_video/all_single/"
    pose_arr = pd.read_csv(data_path + "data1.csv", header=None, sep=',', encoding='utf-8').values
    pose_arr_list.append(pose_arr)
    for __video_id in range(2, 347):
        try:
            csv_data_name = data_path + "data" + str(__video_id) + ".csv"  # 拼接字符串成csv数据的地址
            pose_arr = pd.read_csv(csv_data_name, header=None, sep=',', encoding='utf-8').values
        except OSError:
            print("data ", __video_id, "is not exist")
            pose_arr = np.zeros((1, 1))
        else:
            print("data has been load ", __video_id)
        pose_arr_list.append(pose_arr)


def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


def pil_to_cv(img_pil):
    return cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)


def read_json(json_path):
    json_data = open(json_path)
    json_string = json_data.read()
    j = json.loads(json_string)
    return j


# 整数转偶数
def int_to_even(number: int):
    return int(number // 2 * 2)


# 修正特征点的位置
def rectify_keypoints(points_float, xtl, ytl, img_width, img_height):
    points_rectify = []
    offset = 2
    for __i in range(len(points_float)):
        if __i % 3 == 0:
            x_rectify = round(points_float[__i] - xtl)  # 需要防止越界
            if x_rectify >= img_width:
                x_rectify = img_width - offset
            elif x_rectify <= 0:
                x_rectify = offset
            points_rectify.append(x_rectify)
        elif __i % 3 == 1:
            y_rectify = round(points_float[__i] - ytl)
            if y_rectify >= img_height:
                y_rectify = img_height - offset
            elif y_rectify <= 0:
                y_rectify = offset
            points_rectify.append(y_rectify)
        else:
            points_rectify.append(points_float[__i])
    return points_rectify


# 画出姿势的连线,传入一个和原图像大小一致的空白图像，传入特征点的位置
def draw_pose(img_blank, human_keypoints):
    # kp_num == 26
    l_pair = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),  # Body
        (17, 18), (18, 19), (19, 11), (19, 12),
        (11, 13), (12, 14), (13, 15), (14, 16),
        (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25), ]  # Foot

    # 点的颜色
    p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
               # Nose, LEye, REye, LEar, REar
               (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
               # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
               (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),
               # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
               (77, 255, 255), (0, 255, 255), (77, 204, 255),  # head, neck, shoulder
               (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)]  # foot
    # 线的颜色
    line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                  (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                  (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
                  (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36),
                  (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)]

    img = img_blank.copy()
    part_line = {}
    kp_preds = np.array(human_keypoints).reshape(-1, 3)

    # Draw keypoints
    for n in range(kp_preds.shape[0]):
        cor_x, cor_y = round(kp_preds[n, 0]), round(kp_preds[n, 1])
        part_line[n] = (cor_x, cor_y)
        if n < len(p_color):
            cv2.circle(img, (cor_x, cor_y), 3, p_color[n], -1)
        else:
            cv2.circle(img, (cor_x, cor_y), 1, (255, 255, 255), 2)

    # Draw limbs
    for i, (start_p, end_p) in enumerate(l_pair):
        if start_p in part_line and end_p in part_line:
            start_xy = part_line[start_p]
            end_xy = part_line[end_p]
            cv2.line(img, start_xy, end_xy, line_color[i])
    return img


# 首先继承上面的dataset类。然后在__init__()方法中得到图像的路径，然后将图像路径组成一个数组，这样在__getitim__()中就可以直接读取：
class MyDataset(Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, txt, transform=None, target_transform=None):  # 初始化一些需要传入的参数
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
        self.pose_arr_list = pose_arr_list
        self.loader = self.default_loader

    # 定义读取文件的格式
    def default_loader(self, path):
        path_split = path.split('/')
        video_name = path_split[7]
        img_name = path_split[8]
        video_id = int(video_name.split('_')[1])
        img_id = int(img_name.split('.')[0])
        alpha_pose = self.pose_arr_list[video_id - 1]  # video id从1开始，而list从0开始
        pose = alpha_pose[img_id]

        xtl, ytl, width, height = round(pose[80]), round(pose[81]), round(pose[82]), round(pose[83])
        # xbr, ybr = xtl + width, ytl + height
        points_float = pose[2:80]
        raw_image = cv2.imread(path)
        img_height, img_width, img_shape = raw_image.shape
        points_limbs_blank = np.zeros((img_height, img_width, 3))  # 初始化一个0矩阵,存储特征点和肢体连线，彩色图像
        prints_rectify = rectify_keypoints(points_float, xtl, ytl, img_width, img_height)
        img_points_limbs = draw_pose(points_limbs_blank, prints_rectify)

        raw_img_numpy = np.concatenate((raw_image, img_points_limbs), axis=2)
        # 如果使用ndarray.resize扩展形状大小，空白部分用第一个元素补全，如果使用numpy.resize()
        # 扩展形状大小，空白部分依次用原数据的从头到尾的顺序填充。

        # print("raw_img_numpy:", raw_img_numpy.shape)
        raw_img_resize = np.resize(raw_img_numpy, (200, 200, 4)).astype(np.float32)
        # print(type(raw_img_resize))
        # print(raw_img_resize.shape)
        return raw_img_resize

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = self.loader(fn)  # 按照路径读取图片
        # print("type(img) ", type(img))
        if self.transform is not None:
            img = self.transform(img)  # 数据标签转换为Tensor
        return img, label  # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


def generate_dataset():
    # torch.cuda.set_device(gpu_id)#使用GPU
    # learning_rate = 0.0001
    init_read_pose_annotation()

    # 数据集的设置**************************************************************************
    root = "dataset/txt_init/video30/"  # 调用图像

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
    train_data = MyDataset(txt=root + 'train.txt')
    test_data = MyDataset(txt=root + 'test.txt')
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
