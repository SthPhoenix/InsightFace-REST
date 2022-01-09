import time
import numpy as np
import logging

from .common.nms import nms
from typing import Union
from ..exec_backends.onnxrt_backend import DetectorInfer as DIO

# Since TensorRT and pycuda are optional dependencies it might be not available
try:
    from ..exec_backends.trt_backend import DetectorInfer as DIT
except:
    DIT = None



class CenterFace(object):
    def __init__(self, inference_backend: Union[DIT, DIO], landmarks=True):
        self.landmarks = True
        self.net = inference_backend
        self.nms_threshold = 0.3
        self.masks = False
        self.input_shape = (1, 3, 480, 640)

    def __call__(self, img, threshold=0.5):
        return self.detect(img, threshold)

    def prepare(self, nms_threshold: float = 0.3, **kwargs):
        self.nms_threshold = nms_threshold
        self.net.prepare()
        self.input_shape = self.net.input_shape

    def detect(self, imgs: Union[list, tuple], threshold: float = 0.4):

        if not isinstance(imgs, tuple):
            imgs = (imgs)

        det_list = []
        lmk_list = []

        for img in imgs:
            h, w = img.shape[:2]
            blob = np.expand_dims(img[:, :, (2, 1, 0)].transpose(2, 0, 1), axis=0).astype("float32")
            t0 = time.time()
            heatmap, scale, offset, lms = self.net.run(blob)
            t1 = time.time()
            logging.debug(f"Inference took: {(t1 - t0)*1000:.3f} ms.")
            det, landmarks = self.postprocess(heatmap, lms, offset, scale, (h, w), threshold)
            det_list.append(det)
            lmk_list.append(landmarks)

        return det_list, lmk_list

    def postprocess(self, heatmap, lms, offset, scale, size, threshold):
        t0 = time.time()
        if self.landmarks:
            dets, lms = self.decode(heatmap, scale, offset, lms, size, threshold=threshold)
        else:
            dets = self.decode(heatmap, scale, offset, None, size, threshold=threshold)
        if len(dets) > 0:
            dets[:, 0:4:2], dets[:, 1:4:2] = dets[:, 0:4:2], dets[:, 1:4:2]
            if self.landmarks:
                lms[:, 0:10:2], lms[:, 1:10:2] = lms[:, 0:10:2], lms[:, 1:10:2]
        else:
            dets = np.empty(shape=[0, 5], dtype=np.float32)
            if self.landmarks:
                lms = np.empty(shape=[0, 10], dtype=np.float32)
        t1 = time.time()
        logging.debug(f"Postprocess took: {(t1 - t0)*1000:.3f} ms.")

        if self.landmarks:
            return dets, lms
        else:
            return dets

    def decode(self, heatmap, scale, offset, landmark, size, threshold=0.1):
        heatmap = np.squeeze(heatmap)
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        c0, c1 = np.where(heatmap > threshold)
        if self.landmarks:
            boxes, lms = [], []
        else:
            boxes = []
        if len(c0) > 0:
            for i in range(len(c0)):
                s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
                o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
                s = heatmap[c0[i], c1[i]]
                x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 / 2), max(0, (c0[i] + o0 + 0.5) * 4 - s0 / 2)
                x1, y1 = min(x1, size[1]), min(y1, size[0])
                boxes.append([x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), s])
                if self.landmarks:
                    lm = []
                    for j in range(5):
                        lm.append(landmark[0, j * 2 + 1, c0[i], c1[i]] * s1 + x1)
                        lm.append(landmark[0, j * 2, c0[i], c1[i]] * s0 + y1)
                    lms.append(lm)
            boxes = np.asarray(boxes, dtype=np.float32)
            keep = nms(boxes, self.nms_threshold)
            boxes = boxes[keep, :]
            if self.landmarks:
                lms = np.asarray(lms, dtype=np.float32)
                lms = lms[keep, :]
                lms = lms.reshape((-1, 5, 2))
        if self.landmarks:
            return boxes, lms
        else:
            return boxes
