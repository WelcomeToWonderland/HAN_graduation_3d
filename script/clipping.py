'''
通过剪裁图像，将图像前两个维度中奇数转变为偶数
防止图像下采样之后，上采样的图形与原图像shape不一致
'''
import os
import cv2
import numpy as np

def clipping_img(path):
    '''
    将一个数据集文件夹中的所有图片的奇数维度减一，转化为偶数
    最后处理过的文件替换原文件
    '''
    path_dataset = path
    name_img_list = os.listdir(path_dataset)

    for name_img in name_img_list:
        img = cv2.imread(os.path.join(path_dataset, name_img))
        shape_img = np.shape(img)

        print(f"name:{name_img}, shape:{shape_img}")

        clipping = 0
        if shape_img[0] % 2 == 1:
            clipping = 1
            img = np.delete(img, 0, 0)
        if shape_img[1] % 2 == 1:
            clipping = 1
            img = np.delete(img, 0, 1)

        if clipping == 1:
            print(f"shape_img:{shape_img}, shape_clipping:{np.shape(img)}")
            cv2.imwrite(os.path.join(path_dataset, name_img), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def clipping_dat(path):
    type = np.uint8
    x = 495
    y = 615
    z = 752
    data = np.fromfile(path, dtype=type)
    data = data.reshape(x, y, z)
    data = np.delete(data, 0, 1)
    data = np.delete(data, 0, 0)
    data.tofile(path)


if __name__ == '__main__':

    path = r'D:\workspace\dataset\OABreast\clipping\Neg_47_Left\HR\MergedPhantom.DAT'
    clipping_dat(path)
    data = np.fromfile(path, dtype=np.uint8)
    data = data.reshape(494, 614, 752)
    print(data.shape)


