import os
from distutils import util
from itertools import chain, islice

from api_trt.logger import logger


def prepare_folders(paths):
    """
    Creates a list of directories if they do not exist.

    Args:
        paths (list): A list of directory paths.

    Returns:
        None
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)

def to_chunks(iterable, size=10):
    """
    Splits an iterable into chunks of a specified size.

    Args:
        iterable: An iterable object.
        size (int): The size of each chunk. Defaults to 10.

    Yields:
        A chunk of the iterable.
    """
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))

def tobool(input):
    """
    Converts a string input to a boolean value.

    Args:
        input (str): The input string.

    Returns:
        A boolean value.
    """
    try:
        return bool(util.strtobool(input))
    except:
        return False

def toNone(input):
    """
    Converts a string input to None if it is empty or contains the words "none" or "null".

    Args:
        input (str): The input string.

    Returns:
        None or the original input.
    """
    if str(input).lower() in ['', 'none', 'null']:
        return None
    else:
        return input

def parse_size(size=None, def_size='640,480'):
    """
    Parses a size string into a list of integers. If no size is provided, uses the default size.

    Args:
        size (str): The size string.
        def_size (str): The default size string. Defaults to '640,480'.

    Returns:
        A list of integers representing the size.
    """
    if size is None:
        size = def_size
    size_lst = list(map(int, size.split(',')))
    return size_lst


def colorize_log(string, color):
    """
    Colors a log message with a specified color.

    Args:
        string (str): The log message.
        color (str): The color to use. Can be 'grey', 'yellow', 'red', 'bold_red', or 'green'.

    Returns:
        A colored log message.
    """
    colors = dict(
        grey="\x1b[38;21m",
        yellow="\x1b[33;21m",
        red="\x1b[31;21m",
        bold_red="\x1b[31;1m",
        green="\x1b[32;1m",
    )
    reset = "\x1b[0m"
    col = colors.get(color)
    if col is None:
        return string
    string = f"{col}{string}{reset}"
    return string

def validate_max_size(max_size):
    """
    Validates that an image's maximum size is a multiple of 32 and adjusts it if necessary.

    Args:
        max_size (list): The maximum size as a list of integers.

    Returns:
        A validated maximum size.
    """
    if max_size[0] % 32 != 0 or max_size[1] % 32 != 0:
        max_size[0] = max_size[0] // 32 * 32
        max_size[1] = max_size[1] // 32 * 32
        logger.warning(f'Input image dimensions should be multiples of 32. Max size changed to: {max_size}')
    return max_size