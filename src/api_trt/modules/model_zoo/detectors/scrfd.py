# -*- coding: utf-8 -*-
# Based on Jia Guo reference implementation at
# https://github.com/deepinsight/insightface/blob/master/detection/scrfd/tools/scrfd.py


from __future__ import division
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

import asyncio

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        t0 = time.time()
        result = f(*args, **kw)
        took_ms = (time.time() - t0) * 1000
        logging.debug(f'func: "{f.__name__}" took: {took_ms:.4f} ms')
        return result

    return wrap


@njit(fastmath=True, cache=True)
def single_distance2bbox(point, distance, stride):
    """
    Fast conversion of single bbox distances to coordinates

    :param point: Anchor point
    :param distance: Bbox distances from anchor point
    :param stride: Current stride scale
    :return: bbox
    """
    distance[0] = point[0] - distance[0] * stride
    distance[1] = point[1] - distance[1] * stride
    distance[2] = point[0] + distance[2] * stride
    distance[3] = point[1] + distance[3] * stride
    return distance


@njit(fastmath=True, cache=True)
def single_distance2kps(point, distance, stride):
    """
    Fast conversion of single keypoint distances to coordinates

    :param point: Anchor point
    :param distance: Keypoint distances from anchor point
    :param stride: Current stride scale
    :return: keypoint
    """
    for ix in range(0, distance.shape[0], 2):
        distance[ix] = distance[ix] * stride + point[0]
        distance[ix + 1] = distance[ix + 1] * stride + point[1]
    return distance


@njit(fastmath=True, cache=True)
def generate_proposals(score_blob, bbox_blob, kpss_blob, stride, anchors, threshold, score_out, bbox_out, kpss_out,
                       offset):
    """
    Convert distances from anchors to actual coordinates on source image
    and filter proposals by confidence threshold.
    Uses preallocated np.ndarrays for output.

    :param score_blob: Raw scores for stride
    :param bbox_blob: Raw bbox distances for stride
    :param kpss_blob: Raw keypoints distances for stride
    :param stride: Stride scale
    :param anchors: Precomputed anchors for stride
    :param threshold: Confidence threshold
    :param score_out: Output scores np.ndarray
    :param bbox_out: Output bbox np.ndarray
    :param kpss_out: Output key points np.ndarray
    :param offset: Write offset for output arrays
    :return:
    """

    total = offset

    for ix in range(0, anchors.shape[0]):
        if score_blob[ix, 0] > threshold:
            score_out[total] = score_blob[ix]
            bbox_out[total] = single_distance2bbox(anchors[ix], bbox_blob[ix], stride)
            kpss_out[total] = single_distance2kps(anchors[ix], kpss_blob[ix], stride)
            total += 1

    return score_out, bbox_out, kpss_out, total


# @timing
@njit(fastmath=True, cache=True)
def filter(bboxes_list: np.ndarray, kpss_list: np.ndarray,
           scores_list: np.ndarray, nms_threshold: float = 0.4):
    """
    Filter postprocessed network outputs with NMS

    :param bboxes_list: List of bboxes (np.ndarray)
    :param kpss_list: List of keypoints (np.ndarray)
    :param scores_list: List of scores (np.ndarray)
    :return: Face bboxes with scores [t,l,b,r,score], and key points
    """

    pre_det = np.hstack((bboxes_list, scores_list))
    keep = nms(pre_det, thresh=nms_threshold)
    keep = np.asarray(keep)
    det = pre_det[keep, :]
    kpss = kpss_list[keep, :]
    kpss = kpss.reshape((kpss.shape[0], -1, 2))

    return det, kpss


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
        g_img = cp.subtract(g_img, 127.5, dtype=cp.float32)
        out.device[:allocate_place] = cp.multiply(g_img, 1 / 128).flatten()
    return g_img.shape


class SCRFD:

    def __init__(self, inference_backend: Union[DIT, DIO], ver=1):
        self.session = inference_backend
        self.center_cache = {}
        self.nms_threshold = 0.4
        self.masks = False
        self.ver = ver
        self.out_shapes = None
        self._anchor_ratio = 1.0
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
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

        # Preallocate reusable arrays for proposals
        max_prop_len = self._get_max_prop_len(self.input_shape,
                                              self._feat_stride_fpn,
                                              self._num_anchors)
        self.score_list = np.zeros((max_prop_len, 1), dtype='float32')
        self.bbox_list = np.zeros((max_prop_len, 4), dtype='float32')
        self.kpss_list = np.zeros((max_prop_len, 10), dtype='float32')

        # Check if exec backend provides CUDA stream
        try:
            self.stream = self.session.stream
            self.input_ptr = self.session.input_ptr
        except BaseException:
            pass

    # @timing
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

        input_height = imgs[0].shape[0]
        input_width = imgs[0].shape[1]
        blob = self._preprocess(imgs)
        net_outs = self._forward(blob)

        dets_list = []
        kpss_list = []

        bboxes_by_img, kpss_by_img, scores_by_img = self._postprocess(net_outs, input_height, input_width, threshold)

        for e in range(self.infer_shape[0]):
            det, kpss = filter(
                bboxes_by_img[e], kpss_by_img[e], scores_by_img[e], self.nms_threshold)

            dets_list.append(det)
            kpss_list.append(kpss)

        return dets_list, kpss_list

    @staticmethod
    def _get_max_prop_len(input_shape, feat_strides, num_anchors):
        """
        Estimate maximum possible number of proposals returned by network

        :param input_shape: maximum input shape of model (i.e (1, 3, 640, 640))
        :param feat_strides: model feature strides (i.e. [8, 16, 32])
        :param num_anchors: model number of anchors (i.e 2)
        :return:
        """

        ln = 0
        pixels = input_shape[2] * input_shape[3]
        for e in feat_strides:
            ln += pixels / (e * e) * num_anchors
        return int(ln)

    # @timing
    @staticmethod
    def _build_anchors(input_height, input_width, strides, num_anchors):
        """
        Precompute anchor points for provided image size

        :param input_height: Input image height
        :param input_width: Input image width
        :param strides: Model strides
        :param num_anchors: Model num anchors
        :return: box centers
        """

        centers = []
        for stride in strides:
            height = input_height // stride
            width = input_width // stride

            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))
            centers.append(anchor_centers)
        return centers

    # @timing
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
                img, 1.0 / 128, input_size, (127.5, 127.5, 127.5), swapRB=True)
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

    # @timing
    def _postprocess(self, net_outs, input_height, input_width, threshold):
        """
        Precompute anchor points for provided image size and process network outputs

        :param net_outs: Network outputs
        :param input_height: Input image height
        :param input_width: Input image width
        :param threshold: Confidence threshold
        :return: filtered bboxes, keypoints and scores
        """

        key = (input_height, input_width)

        if not self.center_cache.get(key):
            self.center_cache[key] = self._build_anchors(input_height, input_width, self._feat_stride_fpn,
                                                         self._num_anchors)
        anchor_centers = self.center_cache[key]
        bboxes, kpss, scores = self._process_strides(net_outs, threshold, anchor_centers)
        return bboxes, kpss, scores

    def _process_strides(self, net_outs, threshold, anchor_centers):
        """
        Process network outputs by strides and return results proposals filtered by threshold

        :param net_outs: Network outputs
        :param threshold: Confidence threshold
        :param anchor_centers: Precomputed anchor centers for all strides
        :return: filtered bboxes, keypoints and scores
        """

        offset = 0

        batch_size = self.infer_shape[0]
        bboxes_by_img = []
        kpss_by_img = []
        scores_by_img = []

        for n_img in range(batch_size):
            for idx, stride in enumerate(self._feat_stride_fpn):
                score_blob = net_outs[idx][n_img]
                bbox_blob = net_outs[idx + self.fmc][n_img]
                kpss_blob = net_outs[idx + self.fmc * 2][n_img]
                stride_anchors = anchor_centers[idx]
                self.score_list, self.bbox_list, self.kpss_list, total = generate_proposals(score_blob, bbox_blob,
                                                                                            kpss_blob, stride,
                                                                                            stride_anchors, threshold,
                                                                                            self.score_list,
                                                                                            self.bbox_list,
                                                                                            self.kpss_list, offset)
                offset = total

            bboxes_by_img.append(self.bbox_list[:offset])
            kpss_by_img.append(self.kpss_list[:offset])
            scores_by_img.append(self.score_list[:offset])

        return bboxes_by_img, kpss_by_img, scores_by_img
