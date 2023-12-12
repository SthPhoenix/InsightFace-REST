import asyncio
import io
import os
from typing import Dict
import time
import traceback
import logging
import base64
import numpy as np
import cv2
import httpx
import aiofiles
import imageio

from modules.utils.helpers import tobool

from turbojpeg import TurboJPEG
import exifread

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

client = httpx.AsyncClient(headers=headers, follow_redirects=True)

def sniff_gif(data):
    try:
        if b"GIF" in data[:32]:
            print(f"Image is a GIF")
            sequence = imageio.get_reader(data, '.gif')
            binary = None
            for frame in sequence:
                outp = io.BytesIO()
                imageio.imwrite(outp, frame, format='jpeg')
                outp.seek(0)
                binary = outp.read()
                break
            return binary
        else:
            return data
    except:
        return data

def transposeImage(image, orientation):
    """See Orientation in https://www.exif.org/Exif2-2.PDF for details."""
    if orientation == None: return image
    val = orientation.values[0]
    if val == 1:
        return image
    elif val == 2:
        return np.fliplr(image)
    elif val == 3:
        return np.rot90(image, 2)
    elif val == 4:
        return np.flipud(image)
    elif val == 5:
        return np.rot90(np.flipud(image), -1)
    elif val == 6:
        return np.rot90(image, -1)
    elif val == 7:
        return np.rot90(np.flipud(image))
    elif val == 8:
        return np.rot90(image)
    else:
        return image


async def read_as_bytes(path, **kwargs):
    async with aiofiles.open(path, mode='rb') as fl:
        data = await fl.read()
        data = sniff_gif(data)
        _bytes = np.frombuffer(data, dtype='uint8')
        return _bytes


def b64_to_bytes(b64encoded, **kwargs):
    __bin = None
    try:
        __bin = b64encoded.split(",")[-1]
        __bin = base64.b64decode(__bin)
        __bin = sniff_gif(__bin)
        __bin = np.frombuffer(__bin, dtype='uint8')
    except Exception:
        tb = traceback.format_exc()
        logging.warning(tb)
        return __bin, tb
    return __bin, None





def decode_img_bytes(im_bytes, **kwargs):
    t0 = time.perf_counter()
    try:
        rot = exifread.process_file(io.BytesIO(im_bytes)).get('Image Orientation', None)
        _image = jpeg.decode(im_bytes)
        _image = transposeImage(_image, orientation=rot)
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
            data = sniff_gif(resp.content)
            __bin = np.frombuffer(data, dtype='uint8')
        else:
            if not os.path.exists(path):
                tb = f"File: '{path}' not found"
                return __bin, tb
            __bin = await read_as_bytes(path)
    except Exception:
        tb = traceback.format_exc()
        logging.warning(tb)
        return __bin, tb
    return __bin, None


def make_im_data(__bin, tb, decode=True):
    traceback_msg = None
    if tb is None:
        if decode:
            try:
                data = decode_img_bytes(__bin)
            except Exception:
                data = None
                tb = traceback.format_exc()
                logging.warning(tb)
        else:
            data = __bin

        if isinstance(data, type(None)):
            if tb is None:
                tb = "Can't decode file, possibly not an image"

    if tb:
        data = None
        traceback_msg = tb
        logging.warning(tb)

    im_data = dict(data=data,
                   traceback=traceback_msg)
    return im_data


async def get_images(data: Dict[str, list], decode=True, **kwargs):
    images = []

    if data.get('urls') is not None:
        urls = data['urls']
        tasks = []
        for url in urls:
            tasks.append(asyncio.ensure_future(dl_image(url)))

        results = await asyncio.gather(*tasks)
        for res in results:
            __bin, tb = res
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