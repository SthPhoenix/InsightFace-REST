# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : 

# Modified for InsightFace-REST

from __future__ import division
import numpy as np
from numba import njit

import time
import os.path as osp
import cv2
from typing import Union
import logging

from .common.nms import nms
from ..exec_backends.onnxrt_backend import DetectorInfer as DIO

# Since TensorRT and pycuda are optional dependencies it might be not available
try:
    from ..exec_backends.trt_backend import DetectorInfer as DIT
except:
    DIT = None


@njit(cache=True)
def distance2bbox(points, distance):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).

    Returns:
        Tensor: Decoded bboxes.
    """
    # WARNING! Don't try this at home, without Numba at least...
    # Here we use C-style function instead of Numpy matrix operations
    # since after Numba compilation code seems to work 2-4x times faster.

    for ix in range(0, distance.shape[0]):
        distance[ix, 0] = points[ix, 0] - distance[ix, 0]
        distance[ix, 1] = points[ix, 1] - distance[ix, 1]
        distance[ix, 2] = points[ix, 2] + distance[ix, 2]
        distance[ix, 3] = points[ix, 3] + distance[ix, 3]

    return distance


@njit(cache=True)
def distance2kps(points, distance):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).

    Returns:
        Tensor: Decoded bboxes.
    """
    # WARNING! Don't try this at home, without Numba at least...
    # Here we use C-style function instead of Numpy matrix operations
    # since after Numba compilation code seems to work 2-4x times faster.

    for ix in range(0, distance.shape[1], 2):
        for j in range(0, distance.shape[0]):
            distance[j, ix] += points[j, 0]
            distance[j, ix + 1] += points[j, 1]

    return distance


class SCRFD:
    def __init__(self, inference_backend: Union[DIT, DIO], ver=1):
        self.session = inference_backend
        self.center_cache = {}
        self.nms_threshold = 0.4
        self.masks = False
        self.ver = ver
        self._init_vars()

    def _init_vars(self):
        self.use_kps = False
        self.out_shapes = None
        self.batched = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True

    def prepare(self, nms_treshold: float = 0.45, **kwargs):
        self.nms_threshold = nms_treshold
        self.session.prepare()
        self.out_shapes = self.session.out_shapes
        if len(self.out_shapes[0]) == 3:
            self.batched = True
        self.input_shape = self.session.input_shape

    def preprocess(self, img):
        if self.ver == 2:
            blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            blob = np.transpose(blob, (2, 0, 1))
            blob = np.expand_dims(blob, axis=0).astype(self.session.input_dtype)
        else:
            input_size = tuple(img.shape[0:2][::-1])
            blob = cv2.dnn.blobFromImage(img, 1.0 / 128, input_size, (127.5, 127.5, 127.5), swapRB=True)
        return blob

    def forward(self, blob):
        t0 = time.time()
        net_outs = self.session.run(blob)
        t1 = time.time()
        logging.debug(f'Inference cost: {(t1 - t0) * 1000:.3f} ms.')
        return net_outs

    def postprocess(self, input_height, input_width, net_outs, threshold):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            kps_preds = None
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + fmc][0] * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2][0] * stride
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + fmc]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return bboxes_list, kpss_list, scores_list

    def filter(self, bboxes_list, kpss_list, scores_list, img_center, max_num, metric):
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list)
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = np.vstack(kpss_list)
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                              det[:, 1])

            if metric == 'max':
                values = area
            else:

                offsets = np.vstack([
                    (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                    (det[:, 1] + det[:, 3]) / 2 - img_center[0]
                ])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering

            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def detect(self, img, threshold=0.5, max_num=0, metric='default'):
        input_height = img.shape[0]
        input_width = img.shape[1]
        img_center = img.shape[0] // 2, img.shape[1] // 2
        blob = self.preprocess(img)
        net_outs = self.forward(blob)
        bboxes_list, kpss_list, scores_list = self.postprocess(input_height, input_width, net_outs, threshold)
        det, kpss = self.filter(bboxes_list, kpss_list, scores_list, img_center, max_num, metric)

        return det, kpss
