import os
from typing import Dict
import time
import traceback
import logging
import base64
import numpy as np
import cv2
import httpx
from modules.utils.helpers import tobool

from turbojpeg import TurboJPEG

if tobool(os.getenv('USE_NVJPEG', False)):
    try:
        from nvjpeg import NvJpeg

        jpeg = NvJpeg()
        print('Using nvJPEG for JPEG decoding')
    except:
        print('Using TurboJPEG for JPEG decoding')
        jpeg = TurboJPEG()
else:
    jpeg = TurboJPEG()

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
}

client = httpx.AsyncClient(headers=headers)

def read_as_bytes(path, **kwargs):
    with open(path, mode='rb') as fl:
        data = fl.read()
        _bytes = np.frombuffer(data, dtype='uint8')
        return _bytes


def b64_to_bytes(b64encoded, **kwargs):
    __bin = None
    try:
        __bin = b64encoded.split(",")[-1]
        __bin = base64.b64decode(__bin)
        __bin = np.frombuffer(__bin, dtype='uint8')

    except Exception:
        tb = traceback.format_exc()
        logging.warning(tb)
        return __bin, tb
    return __bin, None


def decode_img_bytes(im_bytes, **kwargs):
    t0 = time.perf_counter()
    try:
        _image = jpeg.decode(im_bytes)
    except:
        logging.debug('JPEG decoder failed, fallback to cv2.imdecode')
        _image = cv2.imdecode(im_bytes, cv2.IMREAD_COLOR)
    t1 = time.perf_counter()
    logging.debug(f'Decoding took: {(t1 - t0) * 1000:.3f} ms.')
    return _image


async def dl_image(path, **kwargs):
    __bin = None
    try:
        if path.startswith('http'):
            resp = await client.get(path)
            __bin = np.frombuffer(resp.content, dtype='uint8')
        else:
            if not os.path.exists(path):
                tb = f"File: '{path}' not found"
                return __bin, tb
            __bin = read_as_bytes(path)
    except Exception:
        tb = traceback.format_exc()
        logging.warning(tb)
        return __bin, tb
    return __bin, None

def make_im_data(__bin,tb, decode=True):
    traceback = None
    if tb is None:
        if decode:
            data = decode_img_bytes(__bin)
        else:
            data = __bin

        if isinstance(data, type(None)):
            tb = "Can't decode file, possibly not an image"

    if tb:
        data = None
        traceback = tb
        logging.warning(tb)

    im_data = dict(data=data,
                   traceback=traceback)
    return im_data


async def get_images(data: Dict[str, list], decode=True, **kwargs):
    images = []

    if data.get('urls') is not None:
        urls = data['urls']
        for url in urls:
            __bin, tb = await dl_image(url)
            im_data = make_im_data(__bin, tb, decode=decode)
            images.append(im_data)

    elif data.get('data') is not None:
        b64_images = data['data']
        images = []
        for b64_img in b64_images:
            __bin, tb = b64_to_bytes(b64_img)
            im_data = make_im_data(__bin, tb, decode=decode)
            images.append(im_data)

    return images