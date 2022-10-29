import os
from itertools import chain, islice
from distutils import util

def prepare_folders(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)

def to_chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))

def tobool(input):
    try:
        return bool(util.strtobool(input))
    except:
        return False

def parse_size(size=None, def_size='640,480'):
    if size is None:
        size = def_size
    size_lst = list(map(int, size.split(',')))
    return size_lst

