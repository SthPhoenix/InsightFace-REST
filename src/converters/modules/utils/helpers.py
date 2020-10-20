import os

def prepare_folders(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)

