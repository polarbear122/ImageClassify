import os
import pandas as pd

train_data_list = [80, 9, 203, 198, 101, 237, 244, 17, 261, 62, 242, 115, 220, 31, 65, 270, 185, 12, 172, 168, 180, 110,
                   150, 336,
                   294, 206, 116, 339, 119, 240, 184, 19, 98, 277, 137, 221, 128, 87, 170, 1, 78, 192, 288, 5, 189, 194,
                   287, 112,
                   122, 103, 274, 2, 120, 205, 15, 307, 164, 284, 36, 282, 304, 276, 81, 278, 285, 281, 318, 211, 230,
                   266, 217, 16,
                   68, 311, 233, 188, 182, 169, 236, 66, 154, 344, 85, 262, 24, 256, 72, 340, 271, 18, 293, 149, 152,
                   249, 207, 298,
                   191, 273, 268, 279, 329, 209, 303, 238, 323, 222, 156, 32, 136, 60, 187, 253, 25, 176, 113, 297, 37,
                   3, 196, 82,
                   159, 229, 13, 147, 105, 342, 286, 343, 138, 96, 160, 56, 324, 126, 50, 171, 38, 202, 314, 106, 94,
                   67, 58, 239,
                   210, 260, 208, 316, 46, 302, 111, 337, 43, 49, 83, 90, 177, 131, 86, 310, 280, 218, 228, 0, 320, 133,
                   291, 61,
                   132, 100, 47, 91, 199, 158, 22, 10, 144, 225, 251, 77, 296, 44, 195, 51, 88, 21, 272, 334, 26, 255,
                   123, 27, 263,
                   55, 163, 118, 226, 175, 254, 312, 257, 20, 248, 332, 79, 162, 148,346]

test_data_list = [140, 121, 259, 174, 167, 333, 41, 299, 42, 73, 63, 223, 246, 212, 6, 151, 345, 104, 40, 109, 327, 200,
                  28, 258, 135, 232, 267, 326, 141, 45, 57, 305, 75, 338, 231, 30, 153, 264, 215, 309, 54, 317, 295,
                  325, 283, 64, 53, 52, 183, 289, 193, 319, 335, 89, 99, 224, 76, 214, 197, 4, 179, 155, 322, 243, 7,
                  92, 14, 29, 157, 84, 130, 213, 321, 204, 108, 69, 290, 301, 331, 250, 39, 129, 190, 146, 134, 300,
                  216, 241, 93, 95, 275, 306, 227, 313, 166, 127, 292, 219, 107, 142, 315, 330, 145, 186, 71, 102, 114,
                  201, 143, 48, 33, 341, 59, 235, 124, 161, 139, 308, 247, 125, 74, 97, 35, 181, 328, 117, 269, 178,
                  265, 234, 23, 165, 11, 34, 70, 252, 8, 245, 173]


def generate_txt(__data_path, __save_path, __init_img_path):
    if not os.path.exists(__save_path):
        os.makedirs(__save_path)
    train_txt = open(__save_path + '/train.txt', 'w')  # 以只写方式打开文件
    test_txt = open(__save_path + '/test.txt', 'w')
    # 遍历每个子文件夹
    for video_id in range(347):
        try:
            data_arr = pd.read_csv(__data_path + str(video_id) + ".csv", header=None, sep=',',
                                   encoding='utf-8')
            length = len(data_arr)
            for __i in range(length):
                # 索引时先列后行，与平时不同
                uuid = round(data_arr[0][__i])
                v_id = round(data_arr[1][__i])
                idx = round(data_arr[2][__i])
                img_id = round(data_arr[3][__i])
                img_path = __init_img_path + str(video_id).zfill(4) + "/" + str(img_id) + ".jpg" + "*" + str(
                    uuid) + "/" + str(idx)
                # print("label 0:",label_arr[0][label_id])
                print(img_path)
                # print("label_arr[label_id]:", label_arr[label_id], label_id)
                img_label = int(data_arr[86][__i])
                if video_id in train_data_list:
                    train_txt.write(img_path + ' ' + str(img_label) + '\n')
                elif video_id in test_data_list:
                    test_txt.write(img_path + ' ' + str(img_label) + '\n')
                else:
                    print("error, video id is not in train or test list")
        except OSError:
            print("data or label ", video_id, "is not exist")
        else:
            print("data has been load ", video_id)
    train_txt.close()
    test_txt.close()


if __name__ == "__main__":
    save_path = "txt_init/lab3070/"  # 保存的路径
    data_path = "D:/CodeResp/IRBOPP/train/halpe26_reid/iou06/data"  # 存放数据集的位置
    init_img_path = "E:/CodeResp/pycode/DataSet/JAAD_image/video_"
    generate_txt(data_path, save_path, init_img_path)
