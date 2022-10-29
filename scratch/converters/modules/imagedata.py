from typing import List
import cv2

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
        self.resize_times = 0
        self.scale_factor = 1.0

    def resize_image(self, pad: bool = True, mode: str = 'pad'):
        self.resize_times += 1
        cw = int(self.const_width / self.resize_times)
        ch = int(self.const_height / self.resize_times)
        h, w, _ = self.transformed_image.shape
        if mode == 'stretch':
            self.transformed_image = cv2.resize(self.transformed_image, dsize=(self.const_width, self.const_height))
        else:
            self.scale_factor = min(cw / w, ch / h)
            # If image is too small, it may contain only single face, which leads to decreased detection accuracy,
            # so we reduce scale factor by some factor
            if self.scale_factor > 5:
                self.scale_factor = self.scale_factor * 0.7

            self.transformed_image = cv2.resize(self.transformed_image, (0, 0), fx=self.scale_factor, fy=self.scale_factor,
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
