import os
import pandas as pd


def generate_txt(__data_path, __save_path, __init_img_path):
    if not os.path.exists(__save_path):
        os.makedirs(__save_path)
    train_txt = open(__save_path + '/train.txt', 'w')  # 以追加方式打开文件
    test_txt = open(__save_path + '/test.txt', 'w')
    i = 1
    # 遍历每个子文件夹
    for video_id in range(1, 30):
        try:
            data_arr = pd.read_csv(__data_path + str(video_id) + ".csv", header=None, sep=',',
                                   encoding='utf-8')
            length = len(data_arr)
            for __i in range(length):
                img_id = round(data_arr[1][__i])  # 索引时先列后行，与平时不同
                img_path = __init_img_path + str(video_id).zfill(4) + "/" + str(img_id) + ".jpg"
                # print("label 0:",label_arr[0][label_id])
                print(img_path)
                # print("label_arr[label_id]:", label_arr[label_id], label_id)
                img_label = int(data_arr[84][__i])
                if i % 10 == 2 or i % 10 == 9 or i % 10 == 7 or i % 10 == 6:
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
    save_path = "txt_init_img/video30"  # 保存的路径
    data_path = "D:/CodeResp/IRBOPP/train/halpe26_data/data_by_video/all_single/data"  # 存放数据集的位置
    init_img_path = "E:/CodeResp/pycode/DataSet/JAAD_image/video_"
    generate_txt(data_path, save_path, init_img_path)
