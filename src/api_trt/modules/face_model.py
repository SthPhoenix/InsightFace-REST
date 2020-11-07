import collections
from typing import Dict, List, Optional
import numpy as np
from numpy.linalg import norm
import cv2
import logging

import time
from insightface.utils import face_align

from modules.model_zoo.getter import get_model
from modules.imagedata import ImageData

import asyncio


Face = collections.namedtuple("Face", ['bbox', 'landmark', 'det_score', 'embedding', 'gender', 'age', 'embedding_norm',
                                       'normed_embedding', 'facedata', 'scale', 'num_det'])

Face.__new__.__defaults__ = (None,) * len(Face._fields)

device2ctx = {
    'cpu': -1,
    'cuda': 0
}


# Wrapper for insightface detection model
class Detector:
    def __init__(self, device: str = 'cuda', det_name: str = 'retinaface_r50_v1', max_size=None,
                 backend_name: str = 'trt', force_fp16: bool = False):
        if max_size is None:
            max_size = [640, 480]

        if det_name == 'centerface' and backend_name == 'mxnet':
            backend_name = 'onnx'
        self.retina = get_model(det_name, backend_name=backend_name, force_fp16=force_fp16, im_size=max_size, root_dir='/models')
        self.retina.prepare(ctx_id=device2ctx[device], nms=0.35)

    def detect(self, data, threshold=0.3):
        bboxes, landmarks = self.retina.detect(data, threshold=threshold)
        boxes = bboxes[:, 0:4]
        probs = bboxes[:, 4]
        t1 = time.time()
        return boxes, probs, landmarks


# Wrapper for facenet_pytorch.MTCNN
class DetectorMTCNN:
    def __init__(self, device: str = 'cuda', select_largest: bool = True, post_process: bool = False,
                 keep_all: bool = True, min_face_size: int = 20,
                 factor: float = 0.709):
        # Importing MTCNN at startup causes GPU memory usage even if Retina is selected.
        from facenet_pytorch import MTCNN

        self.mtcnn = MTCNN(device=device, select_largest=select_largest, post_process=post_process, keep_all=keep_all,
                           min_face_size=min_face_size, factor=factor)

    def detect(self, data, threshold=0.6):
        img_rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = self.mtcnn.detect(img_rgb, landmarks=True)
        return boxes, probs, landmarks


class FaceAnalysis:
    def __init__(self, det_name: str = 'retinaface_r50_v1', rec_name: str = 'arcface_r100_v1',
                 ga_name: str = 'genderage_v1', device: str = 'cuda',
                 select_largest: bool = True, keep_all: bool = True, min_face_size: int = 20,
                 mtcnn_factor: float = 0.709, max_size=None, backend_name: str = 'mxnet', force_fp16: bool = False):

        if max_size is None:
            max_size = [640, 480]

        self.max_size = max_size

        if backend_name == 'mxnet':
            import mxnet as mx
            try:
                _ = mx.nd.array([1, 2, 3], ctx=mx.gpu(0))
            except:
                device = 'cpu'
                logging.info('MXNet was not compiled with cuda support, fallback to CPU')

        assert det_name is not None

        ctx = device2ctx[device]

        if det_name == 'mtcnn':
            self.det_model = DetectorMTCNN(device, select_largest=select_largest, keep_all=keep_all,
                                           min_face_size=min_face_size, factor=mtcnn_factor)
        else:
            self.det_model = Detector(det_name=det_name, device=device, max_size=self.max_size,
                                      backend_name=backend_name, force_fp16=force_fp16)

        if rec_name is not None:
            self.rec_model = get_model(rec_name, backend_name=backend_name,force_fp16=force_fp16)
            self.rec_model.prepare(ctx_id = ctx)
        else:
            self.rec_model = None

        if ga_name is not None:
            self.ga_model = get_model(ga_name, backend_name=backend_name,force_fp16=force_fp16)
            self.ga_model.prepare(ctx_id = ctx)
        else:
            self.ga_model = None

    def sort_boxes(self, boxes, probs, landmarks, img):
        # TODO implement sorting of bounding boxes with respect to size and position
        return boxes, probs, landmarks

        # Translate bboxes and landmarks from resized to original image size

    def reproject_points(self, dets, scale: float):
        if scale != 1.0:
            dets = dets / scale
        return dets

    # Process single image
    async def get(self, img, extract_embedding: bool = True, extract_ga: bool = True,
            return_face_data: bool = True, max_size: List[int] = None, threshold: float = 0.6):

        ts = time.time()

        t0 = time.time()
        # If detector has input_shape attribute, use it instead of provided value
        try:
            max_size = self.det_model.retina.input_shape[2:][::-1]
        except:
            pass
        img = ImageData(img, max_size=max_size)
        img.resize_image(mode='pad')
        t1 = time.time()
        logging.debug(f'Preparing image took: {t1 - t0}')

        t0 = time.time()
        boxes, probs, landmarks = self.det_model.detect(img.transformed_image, threshold=threshold)
        t1 = time.time()
        logging.debug(f'Detection took: {t1 - t0}')
        await asyncio.sleep(0)
        ret = []
        if not isinstance(boxes, type(None)):
            t0 = time.time()
            for i in range(len(boxes)):
                # Translate points to original image size
                bbox = self.reproject_points(boxes[i], img.scale_factor)
                landmark = self.reproject_points(landmarks[i], img.scale_factor)
                det_score = probs[i]
                # Crop faces from original image instead of resized to improve quality
                _crop = face_align.norm_crop(img.orig_image, landmark=landmark)
                facedata = None
                embedding = None
                embedding_norm = None
                normed_embedding = None
                gender = None
                age = None
                if return_face_data:
                    facedata = _crop

                if extract_embedding and self.rec_model:
                    embedding = self.rec_model.get_embedding(_crop).flatten()
                    embedding_norm = norm(embedding)
                    normed_embedding = embedding / embedding_norm

                if self.ga_model:
                    if extract_ga:
                        gender, age = self.ga_model.get(_crop)
                        gender = int(gender)

                face = Face(bbox=bbox, landmark=landmark, det_score=det_score, embedding=embedding, gender=gender,
                            age=age,
                            normed_embedding=normed_embedding, embedding_norm=embedding_norm, facedata=facedata,
                            num_det=i,
                            scale=img.scale_factor)

                ret.append(face)
            t1 = time.time()
            logging.debug(f'Embedding {len(boxes)} faces took: {t1 - t0}')
            await asyncio.sleep(0)

        tf = time.time()
        logging.debug(f'Full processing took: {tf - ts}')
        return ret
