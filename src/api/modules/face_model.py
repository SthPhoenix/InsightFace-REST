import collections
from typing import Dict, List, Optional
import numpy as np
from numpy.linalg import norm
import cv2

from insightface.model_zoo import model_zoo
from insightface.utils import face_align

from .utils.helpers import to_chunks

Face = collections.namedtuple("Face", ['bbox', 'landmark', 'det_score', 'embedding', 'gender', 'age', 'embedding_norm',
                                       'normed_embedding', 'facedata', 'scale', 'num_det'])

Face.__new__.__defaults__ = (None,) * len(Face._fields)

device2ctx = {
    'cpu': -1,
    'cuda': 0
}


# Wrapper for insightface detection model
class DetectorRetina:
    def __init__(self, device: str = 'cuda', det_name: str = 'retinaface_r50_v1'):
        ctx = device2ctx[device]
        self.retina = model_zoo.get_model(det_name)
        self.retina.prepare(ctx, 0.4)

    def detect(self, data, threshold=0.6):
        bboxes, landmarks = self.retina.detect(data, threshold=threshold)
        boxes = bboxes[:, 0:4]
        probs = bboxes[:, 4]
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

    def resize_image(self, pad: bool = True):
        self.resize_times += 1
        cw = int(self.const_width / self.resize_times)
        ch = int(self.const_height / self.resize_times)
        h, w, _ = self.transformed_image.shape
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


class FaceAnalysis:
    def __init__(self, det_name: str = 'retinaface_r50_v1', rec_name: str = 'arcface_r100_v1',
                 ga_name: str = 'genderage_v1', device: str = 'cuda',
                 select_largest: bool = True, keep_all: bool = True, min_face_size: int = 20,
                 mtcnn_factor: float = 0.709):

        assert det_name is not None

        ctx = device2ctx[device]

        if det_name == 'mtcnn':
            self.det_model = DetectorMTCNN(device, select_largest=select_largest, keep_all=keep_all,
                                           min_face_size=min_face_size, factor=mtcnn_factor)
        else:
            self.det_model = DetectorRetina(det_name=det_name, device=device)

        if rec_name is not None:
            self.rec_model = model_zoo.get_model(rec_name)
            self.rec_model.prepare(ctx)
        else:
            self.rec_model = None

        if ga_name is not None:
            self.ga_model = model_zoo.get_model(ga_name)
            self.ga_model.prepare(ctx)
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
    def get(self, img, extract_embedding: bool = True, extract_ga: bool = True,
            return_face_data: bool = True, max_size: List[int] = None, threshold: float = 0.6):

        if max_size is None:
            max_size = [640, 480]

        img = ImageData(img, max_size=max_size)
        img.resize_image(pad=True)

        boxes, probs, landmarks = self.det_model.detect(img.transformed_image, threshold=threshold)

        ret = []
        if not isinstance(boxes, type(None)):
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

        return ret
