import os

def delete_folder(path):
    if os.path.isdir(path):
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            os.remove(filepath)
        os.rmdir(path)
    elif os.path.isfile(path):
        os.remove(path)