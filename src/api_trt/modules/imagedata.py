import time
from typing import List
import cv2
from typing import Union
import numpy as np
import logging

class ImageData:
    def __init__(self, image, max_size: List[int] = None):

        if max_size is None:
            max_size = [640, 480]

        if len(max_size) == 1:
            max_size = [max_size[0]] * 2

        self.orig_image = image
        self.transformed_image = self.orig_image
        self.const_width = max_size[0]
        self.const_height = max_size[1]
        self.scale_factor = 1.0
        self.resize_ms = 0

    def resize_image(self, pad: bool = True, mode: str = 'pad'):
        t0 = time.perf_counter()
        cw = self.const_width
        ch = self.const_height

        h, w, _ = self.transformed_image.shape
        if mode == 'stretch':
            self.transformed_image = cv2.resize(self.transformed_image, dsize=(self.const_width, self.const_height))
        else:
            self.scale_factor = min(cw / w, ch / h)
            # If image is too small, it may contain only single face, which leads to decreased detection accuracy,
            # so we reduce scale factor by some factor
            if self.scale_factor > 3:
                self.scale_factor = self.scale_factor * 0.7

            self.transformed_image = cv2.resize(self.transformed_image, (0, 0), fx=self.scale_factor,
                                                fy=self.scale_factor,
                                                interpolation=cv2.INTER_LINEAR)
            if pad:
                # # Pad right and bottom with black border for fixed image proportions
                h, w, _ = self.transformed_image.shape
                if w < cw:
                    self.transformed_image = cv2.copyMakeBorder(self.transformed_image, 0, 0, 0, cw - w,
                                                                cv2.BORDER_CONSTANT)
                    self.left_border = cw - w
                if h < ch:
                    self.transformed_image = cv2.copyMakeBorder(self.transformed_image, 0, ch - h, 0, 0,
                                                                cv2.BORDER_CONSTANT)
                    self.bottom_border = ch - h
        self.resize_ms = (time.perf_counter() - t0) * 1000
        logging.debug(f'Resizing image took: {self.resize_ms:.3f} ms.')



def resize_image(image, max_size: list = None):

    if max_size is None:
        max_size = [640, 640]

    cw = max_size[0]
    ch = max_size[1]
    h, w, _ = image.shape

    scale_factor = min(cw / w, ch / h)
    # If image is too small, it may contain only single face, which leads to decreased detection accuracy,
    # so we reduce scale factor by some factor
    if scale_factor > 3:
        scale_factor = scale_factor * 0.7

    transformed_image = cv2.resize(image, (0, 0), fx=scale_factor,
                                             fy=scale_factor,
                                             interpolation=cv2.INTER_LINEAR)
    h, w, _ = transformed_image.shape

    if w < cw:
       transformed_image = cv2.copyMakeBorder(transformed_image, 0, 0, 0, cw - w,
                                                    cv2.BORDER_CONSTANT)
    if h < ch:
        transformed_image = cv2.copyMakeBorder(transformed_image, 0, ch - h, 0, 0,
                                                    cv2.BORDER_CONSTANT)

    return transformed_image, scale_factor
