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
from modules.utils.helpers import to_chunks

Face = collections.namedtuple("Face", ['bbox', 'landmark', 'det_score', 'embedding', 'gender', 'age', 'embedding_norm',
                                       'normed_embedding', 'facedata', 'scale', 'num_det', 'mask_prob'])

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

        self.retina = get_model(det_name, backend_name=backend_name, force_fp16=force_fp16, im_size=max_size,
                                root_dir='/models', download_model=False)
        self.retina.prepare(ctx_id=device2ctx[device], nms=0.35)

    def detect(self, data, threshold=0.3):
        bboxes, landmarks = self.retina.detect(data, threshold=threshold)
        boxes = bboxes[:, 0:4]
        probs = bboxes[:, 4]
        mask_probs = None
        try:
            if self.retina.masks == True:
                mask_probs = bboxes[:, 5]
        except:
            pass
        t1 = time.time()
        return boxes, probs, landmarks, mask_probs


class FaceAnalysis:
    def __init__(self, det_name: str = 'retinaface_r50_v1', rec_name: str = 'arcface_r100_v1',
                 ga_name: str = 'genderage_v1', device: str = 'cuda',
                 max_size=None, max_rec_batch_size: int = 1,
                 backend_name: str = 'mxnet', force_fp16: bool = False):

        if max_size is None:
            max_size = [640, 640]

        self.max_size = max_size
        self.max_rec_batch_size = max_rec_batch_size
        if backend_name != 'trt' and max_rec_batch_size != 1:
            logging.warning('Batch processing supported only for TensorRT backend. Fallback to 1.')
            self.max_rec_batch_size = 1

        assert det_name is not None

        ctx = device2ctx[device]

        self.det_model = Detector(det_name=det_name, device=device, max_size=self.max_size,
                                  backend_name=backend_name, force_fp16=force_fp16)

        if rec_name is not None:
            self.rec_model = get_model(rec_name, backend_name=backend_name, force_fp16=force_fp16,
                                       max_batch_size=self.max_rec_batch_size, download_model=False)
            self.rec_model.prepare(ctx_id=ctx)
        else:
            self.rec_model = None

        if ga_name is not None:
            self.ga_model = get_model(ga_name, backend_name=backend_name, force_fp16=force_fp16, download_model=False)
            self.ga_model.prepare(ctx_id=ctx)
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

    def process_faces(self, faces: List[Face], extract_embedding: bool = True, extract_ga: bool = True,
                      return_face_data: bool = False):
        chunked_faces = to_chunks(faces, self.max_rec_batch_size)
        for chunk in chunked_faces:
            chunk = list(chunk)
            crops = [e.facedata for e in chunk]
            total = len(crops)
            embeddings = [None] * total
            ga = [[None, None]] * total

            if extract_embedding:
                t0 = time.time()
                embeddings = self.rec_model.get_embedding(crops)
                t1 = time.time()
                took = t1-t0
                logging.debug(f'Embedding {total} faces took: {took} ({took/total} per face)')

            if extract_ga:
                t0 = time.time()
                ga = [self.ga_model.get(face.facedata) for face in chunk]
                t1 = time.time()
                logging.debug(f'Extracting g/a for {total} faces took: {t1 - t0}')

            for i, crop in enumerate(crops):
                embedding = None
                embedding_norm = None
                normed_embedding = None
                gender = None
                age = None

                embedding = embeddings[i]
                if extract_embedding:
                    embedding_norm = norm(embedding)
                    normed_embedding = embedding / embedding_norm
                _ga = ga[i]
                if extract_ga:
                    gender = int(_ga[0])
                    age = _ga[1]

                face = chunk[i]
                if return_face_data is False:
                    face = face._replace(facedata=None)

                face = face._replace(embedding=embedding, embedding_norm=embedding_norm,
                                         normed_embedding=normed_embedding, gender=gender, age=age)
                yield face

    # Process single image
    def get(self, img, extract_embedding: bool = True, extract_ga: bool = True,
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
        boxes, probs, landmarks, mask_probs = self.det_model.detect(img.transformed_image, threshold=threshold)
        t1 = time.time()
        logging.debug(f'Detection took: {t1 - t0}')
        faces = []
        if not isinstance(boxes, type(None)):
            t0 = time.time()
            for i in range(len(boxes)):
                # Translate points to original image size
                bbox = self.reproject_points(boxes[i], img.scale_factor)
                landmark = self.reproject_points(landmarks[i], img.scale_factor)
                det_score = probs[i]

                if not isinstance(mask_probs, type(None)):
                    mask_prob = mask_probs[i]
                else:
                    mask_prob = None

                # Crop faces from original image instead of resized to improve quality
                _crop = face_align.norm_crop(img.orig_image, landmark=landmark)
                face = Face(bbox=bbox, landmark=landmark, det_score=det_score,
                            num_det=i, scale=img.scale_factor, mask_prob=mask_prob, facedata=_crop)

                faces.append(face)

            t1 = time.time()
            logging.debug(f'Cropping {len(boxes)} faces took: {t1 - t0}')

            # Process detected faces
            faces = [e for e in self.process_faces(faces, extract_embedding=extract_embedding,
                                                   extract_ga=extract_ga, return_face_data=return_face_data)]

        tf = time.time()
        logging.debug(f'Full processing took: {tf - ts}')
        return faces
