import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import io
import os
from src.utility import get_3d
from script.common import delete_folder

def show_data_distribution(path, save_folder, more_accuate: bool = True):
    plt.clf()
    print(f"\npath : {path}")
    if path.endswith('.mat'):
        file = io.loadmat(path)
        data = file['f1']
    elif path.endswith('.DAT'):
        filename = os.path.basename(path)
        basename, _ = os.path.splitext(filename)
        x, y, z = get_3d(basename)
        data = np.fromfile(path)
        data = data.reshape(x, y, z)
    else:
        print(f'the path is wrong, program termnates!')
        return
    # 绘制直方图
    # hist, bin_edges = np.histogram(data, bins=10, range=(data.min(), data.max()))

    filename = os.path.basename(path)
    basename, _ = os.path.splitext(filename)

    os.makedirs(os.path.join('../script_results', save_folder), exist_ok=True)

    hist, bin_edges = np.histogram(data.ravel(), bins=10, range=(data.min(), data.max()))
    print(f"\nhist : {hist}")
    print(f"bin_edges : {bin_edges}")
    plt.hist(data.ravel(), bins=10, range=(data.min(), data.max()))
    plt.savefig(os.path.join('.', save_folder, basename+'_all.png'))
    plt.close()

    if more_accuate:
        plt.hist(data.ravel(), bins=20, range=(1450, 1560))
        plt.savefig(os.path.join('.', save_folder, basename+'_reduced.png'))
        plt.close()


    # plt.show()



if __name__ == '__main__':
    save_folder = 'distribution'
    # delete_folder(os.path.join('.', save_folder))
    delete_folder(os.path.join('../script_results', save_folder))
    path = r'D:\workspace\dataset\USCT\original\HR'
    for filename in os.listdir(path):
        show_data_distribution(os.path.join(path, filename), save_folder=save_folder)

    save_folder = 'distribution_processed'
    # delete_folder(os.path.join('.', save_folder))
    delete_folder(os.path.join('../script_results', save_folder))
    path = r'D:\workspace\dataset\USCT\clipping\pixel_translation\3d_to_oabreast\HR'
    for filename in os.listdir(path):
        show_data_distribution(os.path.join(path, filename), save_folder=save_folder, more_accuate=False)