from __future__ import division
import numpy as np
import cv2
import time
import logging
from typing import Union
from numba import njit

from .common.nms import nms
from ..exec_backends.onnxrt_backend import DetectorInfer as DIO

# Since TensorRT and pycuda are optional dependencies it might be not available
try:
    from ..exec_backends.trt_backend import DetectorInfer as DIT
except:
    DIT = None


@njit()
def _exp(v):
    gate: int = 1
    base = np.exp(1)
    def a(v):
        if abs(v) < gate:
            return v * base
        if v > 0:
            return np.exp(v)
        else:
            return -np.exp(-v)

    return np.array([a(item) for item in v], v.dtype)


def max_pool2d(A, kernel_size=3, stride=1, padding=1):
    """2D Max Pooling
    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
    """
    A = np.pad(A, padding, mode='constant')
    output_shape = ((A.shape[0] - kernel_size) // stride + 1, (A.shape[1] - kernel_size) // stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = np.lib.stride_tricks.as_strided(A,
                                          shape=output_shape + kernel_size,
                                          strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)
    return A_w.max(axis=(1, 2)).reshape(output_shape)


def get_topk_score_indices(hm_pool, hm, k):
    ary = ((hm_pool == hm).astype(np.bool) * hm).reshape(-1)
    indices = ary.argsort()[::-1][:k]
    scores = ary[indices]
    return scores, indices

@njit()
def bx_lm(box, landmark, scores, threshold, xs, ys):
    size = xs.shape[0]
    stride = 4
    lms = np.zeros((size, 5, 2))
    boxes = np.zeros((size, 5))
    count = 0
    for i in range(size):
        if scores[i] < threshold:
            break
        x, y, r, b = box[:, ys[i], xs[i]]
        xyrbs = (np.array([xs[i], ys[i], xs[i], ys[i], 0]) + np.array([-x, -y, r, b, 0])) * stride
        xyrbs[4] = scores[i]
        x5y5 = landmark[:, ys[i], xs[i]]
        x5y5s = (_exp(x5y5 * 4) + np.array([xs[i]] * 5 + [ys[i]] * 5)) * stride
        box_landmark = np.dstack((x5y5s[:5], x5y5s[5:]))[0]
        boxes[count] = xyrbs
        lms[count] = box_landmark
        count += 1
    return boxes[:count], lms[:count]


@njit()
def prepare_image(img):
    mean = np.array([0.408, 0.447, 0.47], dtype=np.float32)
    std = np.array([0.289, 0.274, 0.278], dtype=np.float32)
    img = ((img / 255.0 - mean) / std).astype(np.float32)
    #img = ((img / 255.0)).astype(np.float32)
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)

    return img


class DBFace(object):
    def __init__(self, inference_backend: Union[DIT, DIO], landmarks=True):
        self.landmarks = landmarks
        self.net = inference_backend
        self.masks = False
        self.nms_threshold = 0.45
        self.input_shape = (1, 3, 480, 640)

    def prepare(self, nms_threshold: float = 0.45, **kwargs):
        self.nms_threshold = nms_threshold
        self.net.prepare()
        self.input_shape = self.net.input_shape

    def detect(self, imgs: Union[list, tuple], threshold: float = 0.4):
        if not isinstance(imgs, tuple):
            imgs = (imgs)

        det_list = []
        lmk_list = []
        for img in imgs:
            img = prepare_image(img)
            t0 = time.time()
            hm, box, landmark = self.net.run(img)
            t1 = time.time()
            logging.debug(f"DBFace inference took: {t1 - t0}")
            det, landmarks = self.postprocess(hm, box, landmark, threshold=threshold)
            det_list.append(det)
            lmk_list.append(landmarks)
        return det_list, lmk_list

    def postprocess(self, hm, box, landmark, threshold=0.35):
        t0 = time.time()
        hm_pool = max_pool2d(hm[0, 0, :, :], 3, 1, 1)
        hm_pool = np.expand_dims(np.expand_dims(hm_pool, 0), 0)
        scores, indices = get_topk_score_indices(hm_pool, hm, k=1000)

        hm_height, hm_width = hm.shape[2:]
        ys = indices // hm_width
        xs = indices % hm_width
        box = box.reshape(box.shape[1:])
        landmark = landmark.reshape(landmark.shape[1:])

        boxes, landmarks = bx_lm(box, landmark, scores, threshold, xs, ys)

        #tn0 = time.time()
        boxes = np.asarray(boxes, dtype=np.float32)
        keep = nms(boxes, self.nms_threshold)
        boxes = boxes[keep, :]
        lms = np.asarray(landmarks, dtype=np.float32)
        lms = lms[keep, :]
        #tn1 = time.time()
        #logging.debug(f"NMS took: {tn1 - tn0} ({1 / (tn1 - tn0)} im/sec)")
        t1 = time.time()
        logging.debug(f"DBFace postprocess took: {t1 - t0}")
        return boxes, lms

