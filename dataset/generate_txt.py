import os
import pandas as pd


def generate_txt(__dir_path, __save_path):
    if not os.path.exists(__save_path):
        os.makedirs(__save_path)
    train_txt = open(__save_path + '/train.txt', 'a')  # 以追加方式打开文件
    test_txt = open(__save_path + '/test.txt', 'a')
    i = 1
    # 遍历每个子文件夹
    for video_id in range(1, 347):
        try:
            label_arr = pd.read_csv(label_path + "label" + str(video_id) + ".csv", header=None, sep=',',
                                    encoding='utf-8')
            length = len(label_arr)
            label_id = 0
            for img_id in range(length):
                img_path = __dir_path + str(video_id).zfill(4) + "/" + str(img_id) + ".jpg"
                # print("label 0:",label_arr[0][label_id])
                print(img_path)
                # print("label_arr[label_id]:", label_arr[label_id], label_id)
                img_label = int(label_arr[0][label_id])
                label_id += 1
                if i % 10 == 5:
                    test_txt.write(img_path + ' ' + str(img_label) + '\n')
                else:
                    train_txt.write(img_path + ' ' + str(img_label) + '\n')
                i = i + 1
        except OSError:
            print("data or label ", video_id, "is not exist")
        else:
            print("data has been load ", video_id)
    train_txt.close()
    test_txt.close()


if __name__ == "__main__":
    save_path = "txt"  # 保存的路径
    dir_path = "D:/CodeResp/IRBOPP/train/train_data/iou/data_by_video/image_patch/video_"  # 存放数据集的位置
    label_path = "D:/CodeResp/IRBOPP/train/train_data/iou/data_by_video/all_single/"
    # 训练集和测试集的比例9：1
    generate_txt(dir_path, save_path)
