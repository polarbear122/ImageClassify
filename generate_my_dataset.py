# 创建自己的数据集
import json

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image  # 导入PIL库
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data import SubsetRandomSampler
from torchvision.transforms import transforms

from config import config_csv_path, jaad_face_patch, config_dataset_root
from generate_txt import test_data_list, train_data_list
from log_config.log import logger as Log

pose_arr_position = [0]  # 记录每个视频pose的长度位置，取video_id位置的数据pose_arr_list[video_id-1,video_id]


# pose_arr_numpy = np.zeros((1, 1))  # 以numpy数组形式保存pose arr数组


# 标准化读取数据集
def normalize_read(_data_path, _data_list):
    # 先初始化向量
    _pose = pd.read_csv(_data_path + "data" + str(_data_list[0]) + ".csv", header=None, sep=',', encoding='utf-8')
    for v_id in _data_list[1:]:
        try:
            _pose_arr = pd.read_csv(_data_path + "data" + str(v_id) + ".csv", header=None, sep=',', encoding='utf-8')
            print("shape:", _pose_arr.shape)
            _pose = np.concatenate((_pose, _pose_arr), axis=0)
        except OSError:
            print("data ", v_id, "is not exist")
        else:
            print("data has been load ", v_id)
    return _pose


def init_read_pose_annotation():
    print("------------------------init_read_pose_annotation----------------------------------------------- ")
    data_path = config_csv_path

    l = []
    for _i in range(1, 347):
        l.append(_i)
    data_list = l

    return normalize_read(data_path, data_list)


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

    img = img_blank
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
    def __init__(self, txt, transform=None, target_transform=None, pose_arr_numpy=None):  # 初始化一些需要传入的参数
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
        self.pose_arr_numpy = pose_arr_numpy
        # print("pose arr numpy", self.pose_arr_numpy)
        self.loader = self.default_loader

    # 定义读取文件的格式
    # 仅读取脸部图像
    def default_loader(self, path):
        path_split = path.split('*')
        img_name = path_split[0]
        uuid_idx = path_split[1]
        uuid = int(uuid_idx.split('/')[0])
        id_in_video = int(uuid_idx.split('/')[1]) - 1

        pose_arr = self.pose_arr_numpy
        uuid_arr, v_id_arr, idx_arr, img_id_arr, label_arr = pose_arr[:, 0], pose_arr[:, 1], pose_arr[:, 2], \
                                                             pose_arr[:, 3], pose_arr[:, 86]
        imgsize_x, img_size_y = 15, 15
        # print(img_name)
        raw_img = cv2.imread(img_name)
        img_shape = (imgsize_x, img_size_y, 3, 1)
        img = np.resize(raw_img, img_shape)
        # img = cv2.resize(raw_img, (15, 15))
        # 往前追溯30帧
        u = uuid

        img_concat = img
        for i in range(10 - 1):
            pre = u - (i + 1) * 1  # 之前的帧，选择抽取1秒内的30帧
            id_in_v = id_in_video - (i + 1) * 1
            label = label_arr[u]
            # 如果视频id不正确，或第u帧之前无图像，或者前面i帧的idx和第u帧的idx不一致，都只添加0矩阵
            if v_id_arr[pre] != v_id_arr[u] or pre <= 0 or idx_arr[pre] != idx_arr[u] or id_in_v < 0:
                pose_temp = np.zeros(img_shape)
            else:
                pre_img_path = jaad_face_patch + str(int(v_id_arr[u])).zfill(4) + "/" + str(id_in_v) + ".jpg"
                pose_temp = cv2.imread(pre_img_path)
                pose_temp = np.resize(pose_temp, img_shape)
                label = np.max(label_arr[pre:u])
            img_concat = np.concatenate((img_concat, pose_temp), axis=3).astype(np.float32)
        return img_concat, int(label)

    # 读取人的全部范围的图像，加上特征点连线，组成6维向量
    def default_loader_all(self, path):
        path_split = path.split('*')
        img_name = path_split[0]
        # img = cv2.imread(img_name)  # 读取图片
        # img = cv2.resize(img, (50, 100))
        # image = img[: (img.shape[0] // 2), :]
        # return image

        # raw_img_resize =img.resize((50,20))
        # print(img_name)
        uuid_idx = path_split[1]
        uuid = int(uuid_idx.split('/')[0])
        pose = self.pose_arr_numpy[uuid]
        xtl, ytl, width, height = round(pose[82]), round(pose[83]), round(pose[84]), round(pose[85])
        # xbr, ybr = xtl + width, ytl + height
        points_float = pose[4:82]
        # print(uuid)
        raw_img = cv2.imread(img_name)
        img_height, img_width, img_shape = raw_img.shape
        points_limbs_blank = np.zeros((img_height, img_width, 3))  # 初始化一个0矩阵,存储特征点和肢体连线，彩色图像
        prints_rectify = rectify_keypoints(points_float, xtl, ytl, img_width, img_height)
        img_points_limbs = draw_pose(points_limbs_blank, prints_rectify)

        raw_img_numpy = np.concatenate((raw_img, img_points_limbs), axis=2)
        # 如果使用ndarray.resize扩展形状大小，空白部分用第一个元素补全，如果使用numpy.resize()
        # 扩展形状大小，空白部分依次用原数据的从头到尾的顺序填充。
        raw_img_resize = np.resize(raw_img_numpy, (50, 100, 6)).astype(np.float32)
        # print("raw_img_resize",raw_img_resize)

        # print(type(raw_img_resize))
        # print(raw_img_resize.shape)
        return raw_img_resize

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img, label = self.loader(fn)  # 按照路径读取图片
        # print("type(img) ", type(img))
        # if self.transform is not None:
        #     img = self.transform(img)  # 数据标签转换为Tensor
        return img, label  # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


def test_kfold(dataset):
    batch_size = 32
    num_workers = 16
    shuffle_dataset = True
    random_seed = 42
    validation_split = .2

    # Usage Example:
    num_epochs = 10  # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,
                                               num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler,
                                                    num_workers=num_workers)
    return train_loader, validation_loader


def generate_dataset():
    # torch.cuda.set_device(gpu_id)#使用GPU
    # learning_rate = 0.0001
    # 数据集的设置**************************************************************************
    root = config_dataset_root  # 调用图像

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
    pose_arr_numpy = init_read_pose_annotation()
    # pose_arr_numpy = []
    transform_method = transforms.ToTensor()
    train_data = MyDataset(txt=root + 'train.txt', transform=transform_method, pose_arr_numpy=pose_arr_numpy)
    test_data = MyDataset(txt=root + 'test.txt', transform=transform_method, pose_arr_numpy=pose_arr_numpy)
    # 然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
    # train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=16)
    # 处理数据不平衡
    from torch.utils.data.sampler import WeightedRandomSampler
    # 如果label为1，那么对应的该类别被取出来的概率是另外一个类别的2倍
    train_weights = [2 if label == 1 else 1 for data, label in train_data]
    dataset_size = len(train_data)
    train_sampler = WeightedRandomSampler(train_weights, num_samples=(dataset_size // 2), replacement=True)
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=False, num_workers=16, sampler=train_sampler)

    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=16)

    print('num_of_trainData:', len(train_data))
    print('num_of_testData:', len(test_data))
    Log.info('num_of_trainData:%d' % (len(train_data)))
    Log.info('num_of_testData:%d' % (len(test_data)))
    # all_data = MyDataset(txt=root + 'all.txt', transform=transforms.ToTensor(), pose_arr_numpy=pose_arr_numpy)
    # print('num_of_allData:', len(all_data))
    # train_loader, test_loader = test_kfold(all_data)
    return train_loader, test_loader


if __name__ == "__main__":
    generate_dataset()
