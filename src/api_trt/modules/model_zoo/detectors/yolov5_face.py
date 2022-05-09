import time
from typing import Union
from functools import wraps
import logging

import cv2
import numpy as np
from numba import njit

from .common.nms import nms
from ..exec_backends.onnxrt_backend import DetectorInfer as DIO

# Since TensorRT and pycuda are optional dependencies it might be not available
try:
    import cupy as cp
    from ..exec_backends.trt_backend import DetectorInfer as DIT
except BaseException:
    DIT = None


@njit(cache=True, fastmath=True)
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    x[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    x[:, 2] = x[:, 0] + x[:, 2]  # bottom right x
    x[:, 3] = x[:, 1] + x[:, 3]  # bottom right y
    return x

@njit(cache=True, fastmath=True)
def _filter(dets, threshold, nms_threshold):
    order = np.where(dets[:, 4] >= threshold)[0]
    dets = dets[order, :]
    pre_det = dets[:, 0:5]
    lmks = dets[:, 5:15]
    pre_det = xywh2xyxy(pre_det)
    keep = nms(pre_det, thresh=nms_threshold)
    keep = np.asarray(keep)
    det_out = pre_det[keep, :]
    lmks = lmks[keep, :]
    lmks = lmks.reshape((lmks.shape[0], -1, 2))
    return det_out, lmks

def _normalize_on_device(input, stream, out):
    """
    Normalize image on GPU using inference backend preallocated buffers

    :param input: Raw image as nd.ndarray with HWC shape
    :param stream: Inference backend CUDA stream
    :param out: Inference backend pre-allocated input buffer
    :return: Image shape after preprocessing
    """

    allocate_place = np.prod(input.shape)
    with stream:
        g_img = cp.asarray(input)
        g_img = g_img[..., ::-1]
        g_img = cp.transpose(g_img, (0, 3, 1, 2))
        g_img = cp.divide(g_img, 255., dtype=cp.float32)
        out.device[:allocate_place] = g_img.flatten()
    return g_img.shape


class YoloV5:

    def __init__(self, inference_backend: Union[DIT, DIO]):
        self.session = inference_backend
        self.nms_threshold = 0.4
        self.masks = False
        self.out_shapes = None
        self.stream = None
        self.input_ptr = None

    def prepare(self, nms_treshold: float = 0.4, **kwargs):
        """
        Read network params and populate class parameters

        :param nms_treshold: Threshold for NMS IoU

        """

        self.nms_threshold = nms_treshold
        self.session.prepare()
        self.out_shapes = self.session.out_shapes
        self.input_shape = self.session.input_shape
        self.infer_shape = self.input_shape

        # Check if exec backend provides CUDA stream
        try:
            self.stream = self.session.stream
            self.input_ptr = self.session.input_ptr
        except BaseException:
            pass

    def detect(self, imgs, threshold=0.5):
        """
        Run detection pipeline for provided image

        :param img: Raw image as nd.ndarray with HWC shape
        :param threshold: Confidence threshold
        :return: Face bboxes with scores [t,l,b,r,score], and key points
        """

        if isinstance(imgs, list) or isinstance(imgs, tuple):
            if len(imgs) == 1:
                imgs = np.expand_dims(imgs[0], 0)
            else:
                imgs = np.stack(imgs)
        elif len(imgs.shape) == 3:
            imgs = np.expand_dims(imgs, 0)

        blob = self._preprocess(imgs)
        net_outs = self._forward(blob)

        dets_list, kpss_list = self._postprocess(net_outs, threshold)

        return dets_list, kpss_list

    def _preprocess(self, img):
        """
        Normalize image on CPU if backend can't provide CUDA stream,
        otherwise preprocess image on GPU using CuPy

        :param img: Raw image as np.ndarray with HWC shape
        :return: Preprocessed image or None if image was processed on device
        """

        blob = None
        if self.stream:
            self.infer_shape = _normalize_on_device(
                img, self.stream, self.input_ptr)
        else:
            input_size = tuple(img[0].shape[0:2][::-1])
            blob = cv2.dnn.blobFromImages(
                img, 1.0 / 255., input_size, 0., swapRB=True
            )
        return blob

    def _forward(self, blob):
        """
        Send input data to inference backend.

        :param blob: Preprocessed image of shape NCHW or None
        :return: network outputs
        """

        t0 = time.time()
        if self.stream:
            net_outs = self.session.run(
                from_device=True, infer_shape=self.infer_shape)
        else:
            net_outs = self.session.run(blob)
        t1 = time.time()
        logging.debug(f'Inference cost: {(t1 - t0) * 1000:.3f} ms.')
        return net_outs

    def _postprocess(self, net_outs, threshold=0.6):
        """
        Process network outputs

        :param net_outs: Network outputs
        :param threshold: Confidence threshold
        :return: filtered bboxes, keypoints and scores
        """

        batch_size = self.infer_shape[0]
        dets_list = []
        kpss_list = []

        for i in range(batch_size):

            dets = net_outs[0][i]
            det_out, lmks = _filter(dets, threshold, self.nms_threshold)
            dets_list.append(det_out)
            kpss_list.append(lmks)

        return dets_list, kpss_list


