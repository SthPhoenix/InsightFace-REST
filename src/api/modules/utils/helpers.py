from itertools import chain, islice
from distutils import util
from json import JSONEncoder
import numpy as np

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


class NpEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

