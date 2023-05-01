from scipy import io

def USCT2OAbreast(path):
    # 加载数据
    file = io.loadmat(path)
    data = file['f1']
    # 数据转化：遍历所有元素
    x, y, z = data.shape
    for ix in range(x):
        for iy in range(y):
            for iz in range(z):
                if