import numpy as np
import os

def downing(original, modified):
    print(f"\noriginal : {original}")
    print(f"modified : {modified}")
    folder, _ = os.path.split(modified)
    print(f"modified folder : {folder}")
    os.makedirs(folder, exist_ok=True)
    file = np.fromfile(original, dtype=np.uint8)
    print("before downing")
    hist, bins = np.histogram(file, range(7))
    dentities, _ = np.histogram(file, range(7), density=True)
    print(f"hist : {hist}")
    print(f"dentities : {dentities}")
    print(f"bins : {bins}")
    for pixel in range(2, 6):
        file = np.where(file == pixel, pixel-1, file)
    print("after downing")
    hist, bins = np.histogram(file, range(7))
    dentities, _ = np.histogram(file, range(7), density=True)
    print(f"hist : {hist}")
    print(f"dentities : {dentities}")
    print(f"bins : {bins}")
    file.tofile(modified)

def uping(original, modified):
    print(f"\noriginal : {original}")
    print(f"modified : {modified}")
    folder, _ = os.path.split(modified)
    print(f"modified folder : {folder}")
    os.makedirs(folder, exist_ok=True)
    file = np.fromfile(original, dtype=np.uint8)
    print("before downing")
    hist, bins = np.histogram(file, range(7))
    dentities, _ = np.histogram(file, range(7), density=True)
    print(f"hist : {hist}")
    print(f"dentities : {dentities}")
    print(f"bins : {bins}")
    for pixel in range(1, 5):
        file = np.where(file == pixel, pixel+1, file)
    print("after downing")
    hist, bins = np.histogram(file, range(7))
    dentities, _ = np.histogram(file, range(7), density=True)
    print(f"hist : {hist}")
    print(f"dentities : {dentities}")
    print(f"bins : {bins}")
    file.tofile(modified)


if __name__ == '__main__':
    o1 = r"D:\workspace\dataset\OABreast\clipping\Neg_"
    o2 = r"_Left\HR\MergedPhantom.DAT"
    m1 = r"D:\workspace\dataset\OABreast\clipping\pixel_translation\downing\Neg_"
    m2 = r"_Left\HR\MergedPhantom.DAT"
    list = ['07', '35', '47']
    for idx in range(3):
        original = o1 + list[idx] + o2
        modified = m1 + list[idx] + m2
        downing(original, modified)