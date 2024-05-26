import asyncio
import base64
import io
import logging
import os
import time
import traceback
from typing import Dict

import aiofiles
import aiohttp
import cv2
import exifread
import imageio
import numpy as np
from tenacity import retry, wait_exponential, stop_after_attempt, before_sleep_log, retry_if_not_exception_type
from turbojpeg import TurboJPEG

from api_trt.logger import logger
from api_trt.modules.utils.helpers import tobool
from api_trt.settings import Settings


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


settings = Settings()
headers = settings.defaults.img_req_headers


def resize_image(image, max_size: list = None):
    if max_size is None:
        max_size = [640, 640]

    cw = max_size[0]
    ch = max_size[1]
    h, w, _ = image.shape

    scale_factor = min(cw / w, ch / h)
    # If image is too small, it may contain only single face, which leads to decreased detection accuracy,
    # so we reduce scale factor by some factor
    if scale_factor > 2:
        scale_factor = scale_factor * 0.7

    if scale_factor <= 1.:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LINEAR

    if scale_factor == 1.:
        transformed_image = image
    else:
        transformed_image = cv2.resize(image, (0, 0), fx=scale_factor,
                                       fy=scale_factor,
                                       interpolation=interp)

    h, w, _ = transformed_image.shape

    if w < cw:
        transformed_image = cv2.copyMakeBorder(transformed_image, 0, 0, 0, cw - w,
                                               cv2.BORDER_CONSTANT)
    if h < ch:
        transformed_image = cv2.copyMakeBorder(transformed_image, 0, ch - h, 0, 0,
                                               cv2.BORDER_CONSTANT)

    return transformed_image, scale_factor


def sniff_gif(data):
    """
       Sniffs first 32 bytes of data and decodes first frame if it's GIF.

       Args:
           data (bytes): The input data to be processed.

       Returns:
           bytes: The binary image if it's a GIF, otherwise the original data.
       """
    try:
        if b"GIF" in data[:32]:
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
    """
    Transposes the image based on the given orientation.
    See Orientation in https://www.exif.org/Exif2-2.PDF for details.
    Args:
        image (np.ndarray): The input image.
        orientation (exifread.Image): The orientation of the image.

    Returns:
        np.ndarray: The transposed image.
    """
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
    """
    Asynchronously reads the file at the given path and returns it as a bytes object.

    Args:
        path (str): The path of the file to be read.

    Returns:
        np.ndarray: The file contents.
    """
    async with aiofiles.open(path, mode='rb') as fl:
        data = await fl.read()
        data = sniff_gif(data)
        _bytes = np.frombuffer(data, dtype='uint8')
        return _bytes


def b64_to_bytes(b64encoded, **kwargs):
    """
    Decodes the base64 encoded string and returns it as a bytes object.

    Args:
        b64encoded (str): The base64 encoded string to be decoded.

    Returns:
        tuple: A tuple containing the decoded bytes and any error message.
    """
    __bin = None
    try:
        __bin = b64encoded.split(",")[-1]
        __bin = base64.b64decode(__bin)
        __bin = sniff_gif(__bin)
        __bin = np.frombuffer(__bin, dtype='uint8')
    except Exception:
        tb = traceback.format_exc()
        logger.warning(tb)
        return __bin, tb
    return __bin, None


def decode_img_bytes(im_bytes, **kwargs):
    """
    Decodes the image bytes and returns it as a numpy array.
    Tries JPEG decoding with TurboJPEG or NVJpeg first.

    Args:
        im_bytes (bytes): The image bytes to be decoded.

    Returns:
        np.ndarray: The decoded image.
    """
    t0 = time.perf_counter()
    try:
        rot = exifread.process_file(io.BytesIO(im_bytes)).get('Image Orientation', None)
        _image = jpeg.decode(im_bytes)
        _image = transposeImage(_image, orientation=rot)
    except:
        logger.debug('JPEG decoder failed, fallback to cv2.imdecode')
        _image = cv2.imdecode(im_bytes, cv2.IMREAD_COLOR)
    t1 = time.perf_counter()
    logger.debug(f'Decoding took: {(t1 - t0) * 1000:.3f} ms.')
    return _image


@retry(wait=wait_exponential(min=0.5, max=5), stop=stop_after_attempt(5), reraise=True,
       before_sleep=before_sleep_log(logger, logging.WARNING),
       retry=retry_if_not_exception_type(ValueError))
async def make_request(url, session):
    """
    Makes a GET request to the given URL and returns the response. Retries on failure.

    Args:
        url (str): The URL of the request.
        session (aiohttp.ClientSession): The client session for making requests.

    Returns:
        aiohttp.ClientResponse: The response from the server.
    """
    resp = await session.get(url, allow_redirects=True)
    # Here we make an assumption that 404 and 403 codes shouldn't require retries.
    # Any other exception might be retried again.
    if resp.status in [404, 403]:
        raise ValueError(f"Failed to get data from {url}. Status code: {resp.status}")
    if resp.status >= 400:
        raise aiohttp.ClientResponseError(resp.request_info, status=resp.status, history=())
    return resp


async def dl_image(path, session: aiohttp.ClientSession = None, **kwargs):
    """
    Downloads the image from the given path and returns it as a bytes object.

    Args:
        path (str): The path of the file to be downloaded.
        session (aiohttp.ClientSession): The client session for making requests.

    Returns:
        tuple: A tuple containing the downloaded bytes and any error message.
    """
    __bin = None
    try:
        if path.startswith('http'):
            resp = await make_request(path, session)
            content = await resp.content.read()
            data = sniff_gif(content)
            __bin = np.frombuffer(data, dtype='uint8')
        else:
            if not os.path.exists(path):
                tb = f"File: '{path}' not found"
                return __bin, tb
            __bin = await read_as_bytes(path)
    except Exception:
        tb = traceback.format_exc()
        logger.warning(tb)
        return __bin, tb
    return __bin, None


def make_im_data(__bin, tb, decode=True):
    """
    Creates a dictionary containing the image data and any error message occurred.

    Args:
        __bin (np.ndarray): The image bytes.
        tb (str): The error message.
        decode (bool): Whether to decode the image or not.

    Returns:
        dict: A dictionary containing the image data and any error message.
    """
    traceback_msg = None
    if tb is None:
        if decode:
            try:
                data = decode_img_bytes(__bin)
            except Exception:
                data = None
                tb = traceback.format_exc()
                logger.warning(tb)
        else:
            data = __bin

        if isinstance(data, type(None)):
            if tb is None:
                tb = "Can't decode file, possibly not an image"

    if tb:
        data = None
        traceback_msg = tb
        logger.warning(tb)

    im_data = dict(data=data,
                   traceback=traceback_msg)
    return im_data


async def get_images(data: Dict[str, list], decode=True, session: aiohttp.ClientSession = None, **kwargs):
    """
    Downloads and decodes the images from the given data.

    Args:
        data (Dict[str, list]): The input data containing URLs or base64 encoded strings.
        decode (bool): Whether to decode the images or not. Defaults to True.
        session (aiohttp.ClientSession): The client session for making requests.

    Returns:
        list: A list of dictionaries containing the image data and any error message.
    """

    images = []

    if data.get('urls') is not None:
        urls = data['urls']
        tasks = []
        for url in urls:
            tasks.append(asyncio.ensure_future(dl_image(url, session=session)))

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
