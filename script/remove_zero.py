import os
import numpy as np
from scipy import io
from script.common import delete_folder

def remove_zero(dir_original, dir_processed):
    # 创建存储文件夹
    # delete_folder(dir_processed)
    os.makedirs(dir_processed, exist_ok=True)
    # 遍历文件夹
    for filename in os.listdir(dir_original):
        # 加载数据
        file = io.loadmat(os.path.join(dir_original, filename))
        data = file['img']
        # 剔除元素全为零的切片
        print(f"\nbefore : {data.shape}")
        _, _, length = data.shape
        result = None
        for iz in range(length):
            temp = data[..., iz]
            temp = np.expand_dims(temp, axis=2)
            if not np.all(temp == 0):
                if result is None:
                    result = temp
                else:
                    result = np.concatenate((result, temp), axis=2)
        print(f"after : {result.shape}")
        # 存储经过处理的文件
        path_save = os.path.join(dir_processed, filename)
        file['img'] = result
        io.savemat(path_save, file)

if __name__ == '__main__':
    resolutions = ['HR', 'LR/X2', 'SR/X2']
    dataset_original = r'/workspace/datasets/OA-breast_correct/Neg_07_Left_test'
    dataset_processed = r'/workspace/datasets/OA-breast_correct/Neg_07_Left_test_remove_zero'
    for resolution in resolutions:
        dir_original = os.path.join(dataset_original, resolution)
        dir_processed = os.path.join(dataset_processed, resolution)
        remove_zero(dir_original, dir_processed)