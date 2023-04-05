import numpy as np

def from0to1(original, modified):
    print(f"\noriginal : {original}")
    print(f"modified : {modified}")
    file_original = np.fromfile(original, dtype=np.uint8)
    file_modified = np.where(file_original == 0, 1, file_original)
    file_modified.tofile(modified)

def from1to0(original, modified):
    print(f"original : {original}")
    print(f"modified : {modified}")
    file_original = np.fromfile(original, dtype=np.uint8)
    file_modified = np.where(file_original == 1, 0, file_original)
    file_modified.tofile(modified)


if __name__ == '__main__':
    o1 = r"D:\workspace\dataset\OABreast\clipping\Neg_"
    o2 = r"_Left\HR\MergedPhantom.DAT"
    m1 = r"D:\workspace\dataset\OABreast\clipping\zero2one\Neg_"
    m2 = r"_Left\HR\MergedPhantom.DAT"
    list = ['07', '35', '47']
    for idx in range(3):
        original = o1 + list[idx] + o2
        modified = m1 + list[idx] + m2
        from0to1(original, modified)