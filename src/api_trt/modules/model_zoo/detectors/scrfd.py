# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : 

# Modified for InsightFace-REST

from __future__ import division
import numpy as np

import time
import os.path as osp
import cv2
from typing import Union

from .common.nms import nms
from ..exec_backends.onnxrt_backend import DetectorInfer as DIO

# Since TensorRT and pycuda are optional dependencies it might be not available
try:
    from ..exec_backends.trt_backend import DetectorInfer as DIT
except:
    DIT = None


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

class SCRFD:
    def __init__(self, inference_backend: Union[DIT, DIO]):
        self.session = inference_backend
        self.center_cache = {}
        self.nms_threshold = 0.4
        self.masks = False
        self._init_vars()

    def _init_vars(self):
        # input_cfg = self.session.rec_model.get_inputs()[0]
        # input_shape = input_cfg.shape
        # if isinstance(input_shape[2], str):
        #     self.input_size = None
        # else:
        #     self.input_size = tuple(input_shape[2:4][::-1])
        # input_name = input_cfg.name
        # outputs = self.session.rec_model.get_outputs()
        # output_names = []
        # for o in outputs:
        #     output_names.append(o.name)
        # self.input_name = input_name
        # self.output_names = output_names
        self.use_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1
        # if len(outputs)==6:
        #     self.fmc = 3
        #     self._feat_stride_fpn = [8, 16, 32]
        #     self._num_anchors = 2
        #elif len(outputs)==9:
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True
        # elif len(outputs)==10:
        #     self.fmc = 5
        #     self._feat_stride_fpn = [8, 16, 32, 64, 128]
        #     self._num_anchors = 1
        # elif len(outputs)==15:
        #     self.fmc = 5
        #     self._feat_stride_fpn = [8, 16, 32, 64, 128]
        #     self._num_anchors = 1
        #     self.use_kps = True

    def prepare(self, nms: float = 0.45, **kwargs):
        self.nms_threshold = nms
        self.session.prepare()
        self.input_shape = self.session.input_shape


    def forward(self, img, threshold):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(img, 1.0/128, input_size, (127.5, 127.5, 127.5), swapRB=True)
        t0 = time.time()
        net_outs = self.session.run(blob)
        t1 = time.time()
        print('inference cost:', (t1 - t0))

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx+fmc]
            bbox_preds = bbox_preds * stride
            if self.use_kps:
                kps_preds = net_outs[idx+fmc*2] * stride
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                if self._num_anchors>1:
                    anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape( (-1,2) )
                if len(self.center_cache)<100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores>=threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def detect(self, img, threshold=0.5, max_num=0, metric='default'):

        scores_list, bboxes_list, kpss_list = self.forward(img, threshold)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list)
        if self.use_kps:
            kpss = np.vstack(kpss_list)
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
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
                img_center = img.shape[0] // 2, img.shape[1] // 2
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

