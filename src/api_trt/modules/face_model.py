import asyncio
import collections
import logging
import time
from functools import partial
from typing import Dict, List, Union
import traceback

import cv2
import numpy as np
import base64
from modules.imagedata import resize_image
from modules.model_zoo.getter import get_model
from modules.utils import fast_face_align as face_align
from modules.utils.helpers import to_chunks, colorize_log, validate_max_size
from numpy.linalg import norm

Face = collections.namedtuple("Face", ['bbox', 'landmark', 'det_score', 'embedding', 'gender', 'age', 'embedding_norm',
                                       'normed_embedding', 'facedata', 'scale', 'num_det', 'mask', 'mask_probs'])

Face.__new__.__defaults__ = (None,) * len(Face._fields)


def serialize_face(_face_dict: dict, return_face_data: bool, return_landmarks: bool = False):
    if _face_dict.get('norm'):
        _face_dict.update(vec=_face_dict['vec'].tolist(),
                          norm=float(_face_dict['norm']))
    # Warkaround for embed_only flag
    if _face_dict.get('prob'):
        _face_dict.update(prob=float(_face_dict['prob']),
                          bbox=_face_dict['bbox'].astype(int).tolist(),
                          size=int(_face_dict['bbox'][2] - _face_dict['bbox'][0]))

    if return_landmarks:
        _face_dict['landmarks'] = _face_dict['landmarks'].astype(int).tolist()
    else:
        _face_dict.pop('landmarks', None)

    if return_face_data:
        _face_dict['facedata'] = base64.b64encode(cv2.imencode('.jpg', _face_dict['facedata'])[1].tostring()).decode(
            'ascii')
    else:
        _face_dict.pop('facedata', None)

    return _face_dict


# Wrapper for insightface detection model
class Detector:
    def __init__(self, det_name: str = 'retinaface_r50_v1', max_size=None,
                 backend_name: str = 'trt', force_fp16: bool = False, triton_uri=None, max_batch_size: int = 1,
                 root_dir='/models'):
        if max_size is None:
            max_size = [640, 480]

        self.retina = get_model(det_name, backend_name=backend_name, force_fp16=force_fp16, im_size=max_size,
                                root_dir=root_dir, download_model=False, triton_uri=triton_uri,
                                max_batch_size=max_batch_size)

        self.retina.prepare(nms=0.35)

    def detect(self, data, threshold=0.3):
        bboxes, landmarks = self.retina.detect(data, threshold=threshold)

        boxes = [e[:, 0:4] for e in bboxes]
        probs = [e[:, 4] for e in bboxes]

        return boxes, probs, landmarks


# Translate bboxes and landmarks from resized to original image size
def reproject_points(dets, scale: float):
    if scale != 1.0:
        dets = dets / scale
    return dets


class FaceAnalysis:
    def __init__(self,
                 det_name: str = 'retinaface_r50_v1',
                 rec_name: str = 'arcface_r100_v1',
                 ga_name: str = 'genderage_v1',
                 mask_detector: str = 'mask_detector',
                 max_size=None,
                 max_rec_batch_size: int = 1,
                 max_det_batch_size: int = 1,
                 backend_name: str = 'mxnet',
                 force_fp16: bool = False,
                 triton_uri=None,
                 root_dir: str = '/models',
                 **kwargs):

        if max_size is None:
            max_size = [640, 640]

        self.decode_required = True
        self.max_size = validate_max_size(max_size)
        self.max_rec_batch_size = max_rec_batch_size
        self.max_det_batch_size = max_det_batch_size
        self.det_name = det_name
        self.rec_name = rec_name
        if backend_name not in ('trt', 'triton') and max_rec_batch_size != 1:
            logging.warning('Batch processing supported only for TensorRT & Triton backend. Fallback to 1.')
            self.max_rec_batch_size = 1

        assert det_name is not None

        self.det_model = Detector(det_name=det_name, max_size=self.max_size,
                                  max_batch_size=self.max_det_batch_size, backend_name=backend_name,
                                  force_fp16=force_fp16, triton_uri=triton_uri, root_dir=root_dir)

        if rec_name is not None:
            self.rec_model = get_model(rec_name, backend_name=backend_name, force_fp16=force_fp16,
                                       max_batch_size=self.max_rec_batch_size, root_dir=root_dir,
                                       download_model=False, triton_uri=triton_uri)
            self.rec_model.prepare()
        else:
            self.rec_model = None

        if ga_name is not None:
            self.ga_model = get_model(ga_name, backend_name=backend_name, force_fp16=force_fp16,
                                      max_batch_size=self.max_rec_batch_size, root_dir=root_dir,
                                      download_model=False, triton_uri=triton_uri)
            self.ga_model.prepare()
        else:
            self.ga_model = None

        if mask_detector is not None:
            self.mask_model = get_model(mask_detector, backend_name=backend_name, force_fp16=force_fp16,
                                        max_batch_size=self.max_rec_batch_size, root_dir=root_dir,
                                        download_model=False, triton_uri=triton_uri)

            self.mask_model.prepare()
        else:
            self.mask_model = None

    def sort_boxes(self, boxes, probs, landmarks, shape, max_num=0):
        # Based on original InsightFace python package implementation
        if max_num > 0 and boxes.shape[0] > max_num:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            img_center = shape[0] // 2, shape[1] // 2
            offsets = np.vstack([
                (boxes[:, 0] + boxes[:, 2]) / 2 - img_center[1],
                (boxes[:, 1] + boxes[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            values = area  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]

            boxes = boxes[bindex, :]
            probs = probs[bindex]

            landmarks = landmarks[bindex, :]

        return boxes, probs, landmarks

    def process_faces(self,
                      faces: List[dict],
                      extract_embedding: bool = True,
                      extract_ga: bool = True,
                      return_face_data: bool = False,
                      detect_masks: bool = True,
                      mask_thresh: float = 0.89,
                      **kwargs):
        chunked_faces = to_chunks(faces, self.max_rec_batch_size)
        for chunk in chunked_faces:
            chunk = list(chunk)
            crops = [e['facedata'] for e in chunk]
            total = len(crops)
            embeddings = [None] * total
            ga = [[None, None]] * total

            if extract_embedding:
                t0 = time.perf_counter()
                embeddings = self.rec_model.get_embedding(crops)
                took = time.perf_counter() - t0
                logging.debug(
                    f'Embedding {total} faces took: {took * 1000:.3f} ms. ({(took / total) * 1000:.3f} ms. per face)')

            if extract_ga and self.ga_model:
                t0 = time.perf_counter()
                ga = self.ga_model.get(crops)
                t1 = time.perf_counter()
                took = t1 - t0
                logging.debug(
                    f'Extracting g/a for {total} faces took: {took * 1000:.3f} ms. ({(took / total) * 1000:.3f} ms. per face)')

            if detect_masks and self.mask_model:
                t0 = time.perf_counter()
                masks = self.mask_model.get(crops)
                t1 = time.perf_counter()
                took = t1 - t0
                logging.debug(
                    f'Detecting masks for  {total} faces took: {took * 1000:.3f} ms. ({(took / total) * 1000:.3f} ms. per face)')

            for i, crop in enumerate(crops):
                embedding_norm = None
                normed_embedding = None
                gender = None
                age = None
                mask = None
                mask_probs = None

                embedding = embeddings[i]
                if extract_embedding:
                    embedding_norm = norm(embedding)
                    normed_embedding = embedding / embedding_norm

                if extract_ga and self.ga_model:
                    _ga = ga[i]
                    gender = int(_ga[0])
                    age = _ga[1]

                if detect_masks and self.mask_model:
                    _masks = masks[i]
                    mask = False
                    mask_prob = float(_masks[0])
                    no_mask_prob = float(_masks[1])
                    if mask_prob > no_mask_prob and mask_prob >= mask_thresh:
                        mask = True
                    mask_probs = dict(mask=mask_prob,
                                      no_mask=no_mask_prob)

                face = chunk[i]
                if return_face_data is False:
                    face['facedata'] = None

                # face['raw_vec'] = embedding
                face['norm'] = embedding_norm
                face['vec'] = normed_embedding
                face['gender'] = gender
                face['age'] = age
                face['mask'] = mask
                face['mask_probs'] = mask_probs

                yield face

    # Process single image
    async def get(self, images,
                  extract_embedding: bool = True,
                  extract_ga: bool = True,
                  detect_masks: bool = True,
                  return_face_data: bool = True,
                  max_size: List[int] = None,
                  threshold: float = 0.6,
                  min_face_size: int = 0,
                  mask_thresh: float = 0.89,
                  limit_faces: int = 0,
                  use_rotation: bool = False,
                  **kwargs):

        ts = time.perf_counter()

        # If detector has input_shape attribute, use it instead of provided value
        try:
            max_size = self.det_model.retina.input_shape[2:][::-1]
        except:
            pass

        # Pre-assign max_size to resize function
        _partial_resize = partial(resize_image, max_size=max_size)
        # Pre-assign threshold to detect function
        _partial_detect = partial(self.det_model.detect, threshold=threshold)

        # Initialize resied images iterator
        res_images = map(_partial_resize, images)
        batches = to_chunks(res_images, self.max_det_batch_size)

        faces = []
        faces_per_img = {}

        for bid, batch in enumerate(batches):
            batch_imgs, scales = zip(*batch)
            t0 = time.perf_counter()
            det_predictions = zip(*_partial_detect(batch_imgs))
            t1 = time.perf_counter()
            logging.debug(f'Detection took: {(t1 - t0) * 1000:.3f} ms.')

            for idx, pred in enumerate(det_predictions):
                await asyncio.sleep(0)
                orig_id = (bid * self.max_det_batch_size) + idx
                boxes, probs, landmarks = pred
                faces_per_img[orig_id] = len(boxes)

                if not isinstance(boxes, type(None)):
                    t0 = time.perf_counter()
                    if limit_faces > 0:
                        boxes, probs, landmarks = self.sort_boxes(boxes, probs, landmarks,
                                                                  shape=batch_imgs[idx].shape,
                                                                  max_num=limit_faces)
                        faces_per_img[orig_id] = len(boxes)

                    # Translate points to original image size
                    boxes = reproject_points(boxes, scales[idx])
                    logging.debug(landmarks.shape)
                    landmarks = reproject_points(landmarks, scales[idx])
                    # Crop faces from original image instead of resized to improve quality
                    if extract_ga or extract_embedding or return_face_data or detect_masks:
                        crops = face_align.norm_crop_batched(images[orig_id], landmarks)
                    else:
                        crops = [None] * len(boxes)

                    for i, _crop in enumerate(crops):
                        face = dict(
                            bbox=boxes[i], landmarks=landmarks[i], prob=probs[i],
                            num_det=i, scale=scales[idx], facedata=_crop
                        )
                        if min_face_size > 0:
                            w = boxes[i][2] - boxes[i][0]
                            if w >= min_face_size:
                                faces.append(face)
                        else:
                            faces.append(face)

                    t1 = time.perf_counter()
                    logging.debug(f'Cropping {len(boxes)} faces took: {(t1 - t0) * 1000:.3f} ms.')

        # Process detected faces
        tps = time.perf_counter()
        if extract_ga or extract_embedding or detect_masks:
            faces = list(self.process_faces(faces,
                                            extract_embedding=extract_embedding,
                                            extract_ga=extract_ga,
                                            return_face_data=return_face_data,
                                            detect_masks=detect_masks, mask_thresh=mask_thresh))
        tpf = time.perf_counter()
        logging.debug(colorize_log(f'Processing faces took: {(tpf - tps) * 1000:.3f} ms.', 'green'))
        faces_by_img = []
        offset = 0

        for key in faces_per_img:
            value = faces_per_img[key]
            faces_by_img.append(faces[offset:offset + value])
            offset += value

        tf = time.perf_counter()

        logging.debug(colorize_log(f'Full processing took: {(tf - ts) * 1000:.3f} ms.', 'red'))
        return faces_by_img

    def __iterate_images(self, crops):
        for face in crops:
            if face.get('traceback') is None:
                face = face.get('data')
                yield face

    def embed_crops(self,
                    images,
                    extract_embedding: bool = True,
                    extract_ga: bool = True,
                    detect_masks: bool = False,
                    **kwargs):

        t0 = time.time()
        output = dict(took_ms=None, data=[], status="ok")

        iterator = self.__iterate_images(images)
        iterator = ({'facedata': e} for e in iterator)
        faces = self.process_faces(iterator, extract_embedding=extract_embedding, extract_ga=extract_ga,
                                   return_face_data=False, detect_masks=detect_masks)

        try:
            for image in images:
                if image.get('traceback') is not None:
                    _face_dict = dict(status='failed',
                                      traceback=image.get('traceback'))
                else:
                    _face_dict = serialize_face(_face_dict=next(faces), return_face_data=False,
                                                return_landmarks=False)
                    _face_dict['status'] = 'ok'
                output['data'].append(_face_dict)
        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
            output['status'] = 'failed'
            output['traceback'] = tb

        took = time.time() - t0
        output['took_ms'] = took * 1000
        return output

    async def embed(self,
                    images: Dict[str, list],
                    max_size: List[int] = None,
                    threshold: float = 0.6,
                    limit_faces: int = 0,
                    min_face_size: int = 0,
                    return_face_data: bool = False,
                    extract_embedding: bool = True,
                    extract_ga: bool = True,
                    return_landmarks: bool = False,
                    detect_masks: bool = False,
                    use_rotation: bool = False):

        _get = partial(self.get, max_size=max_size, threshold=threshold,
                       return_face_data=return_face_data,
                       extract_embedding=extract_embedding, extract_ga=extract_ga,
                       limit_faces=limit_faces,
                       min_face_size=min_face_size,
                       detect_masks=detect_masks,
                       use_rotation=use_rotation)

        _serialize = partial(serialize_face, return_face_data=return_face_data,
                             return_landmarks=return_landmarks)

        output = dict(took={}, data=[])

        imgs_iterable = self.__iterate_images(images)

        faces_by_img = (e for e in await _get([img for img in imgs_iterable]))

        for img in images:
            _faces_dict = dict(status='failed', took_ms=0., faces=[])
            try:
                if img.get('traceback') is not None:
                    _faces_dict['status'] = 'failed'
                    _faces_dict['traceback'] = img.get('traceback')
                else:
                    t0 = time.perf_counter()
                    faces = faces_by_img.__next__()
                    tss = time.perf_counter()
                    _faces_dict['faces'] = list(map(_serialize, faces))
                    tsf = time.perf_counter()
                    logging.debug(f'Serializing took: {(tsf - tss) * 1000} ms.')
                    took = time.perf_counter() - t0
                    _faces_dict['took_ms'] = took * 1000
                    _faces_dict['status'] = 'ok'
            except Exception as e:
                tb = traceback.format_exc()
                print(tb)
                _faces_dict['status'] = 'failed'
                _faces_dict['traceback'] = tb

            output['data'].append(_faces_dict)

        return output

    def draw_faces(self,
                   image,
                   faces,
                   draw_landmarks=True,
                   draw_scores=True,
                   draw_sizes=True):

        for face in faces:
            bbox = face["bbox"].astype(int)
            pt1 = tuple(bbox[0:2])
            pt2 = tuple(bbox[2:4])
            color = (0, 255, 0)
            x, y = pt1
            r, b = pt2
            w = r - x
            if face.get("mask") is False:
                color = (0, 0, 255)
            cv2.rectangle(image, pt1, pt2, color, 1)

            if draw_landmarks:
                lms = face["landmarks"].astype(int)
                pt_size = int(w * 0.05)
                cv2.circle(image, (lms[0][0], lms[0][1]), 1, (0, 0, 255), pt_size)
                cv2.circle(image, (lms[1][0], lms[1][1]), 1, (0, 255, 255), pt_size)
                cv2.circle(image, (lms[2][0], lms[2][1]), 1, (255, 0, 255), pt_size)
                cv2.circle(image, (lms[3][0], lms[3][1]), 1, (0, 255, 0), pt_size)
                cv2.circle(image, (lms[4][0], lms[4][1]), 1, (255, 0, 0), pt_size)

            if draw_scores:
                text = f"{face['prob']:.3f}"
                pos = (x + 3, y - 5)
                textcolor = (0, 0, 0)
                thickness = 1
                border = int(thickness / 2)
                cv2.rectangle(image, (x - border, y - 21, w + thickness, 21), color, -1, 16)
                cv2.putText(image, text, pos, 0, 0.5, color, 3, 16)
                cv2.putText(image, text, pos, 0, 0.5, textcolor, 1, 16)
            if draw_sizes:
                text = f"w:{w}"
                pos = (x + 3, b - 5)
                cv2.putText(image, text, pos, 0, 0.5, (0, 0, 0), 3, 16)
                cv2.putText(image, text, pos, 0, 0.5, (0, 255, 0), 1, 16)

        total = f'faces: {len(faces)} ({self.det_name})'
        bottom = image.shape[0]
        cv2.putText(image, total, (5, bottom - 5), 0, 1, (0, 0, 0), 3, 16)
        cv2.putText(image, total, (5, bottom - 5), 0, 1, (0, 255, 0), 1, 16)

        return image
