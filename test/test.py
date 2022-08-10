import cv2
import numpy as np
from PIL import Image


def test_img_convert():
    img_path = "E:/CodeResp/pycode/DataSet/JAAD_image/video_0004/15.jpg"
    img = cv2.imread(img_path)
    img = cv2.resize(img, (20, 50))
    print(img.shape)
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img


def test_numpy_resize():
    test_numpy_arr = np.ones((5, 5))
    print(test_numpy_arr)
    result = np.resize(test_numpy_arr, (10, 5))
    # result = test_numpy_arr.resize((10, 5))
    print(result)


def test_numpy_list():
    zeros1 = np.zeros((3, 2))
    ones1 = np.ones((4, 5))
    concat_list = [np.ones((3, 10))]
    concat_list.append(zeros1)
    print(concat_list)
    concat_list.append(ones1)
    print(concat_list)


if __name__ == "__main__":
    test_numpy_list()
