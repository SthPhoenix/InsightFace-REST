import base64
import io
import logging
import time
import traceback
from functools import partial
from typing import Dict, List, Union

import cv2
import numpy as np
from modules.utils.image_provider import get_images

from .face_model import FaceAnalysis


class Serializer:

    def serialize(self, data, api_ver: str = '1'):
        serializer = self.get_serializer(api_ver)
        return serializer(data)

    def get_serializer(self, api_ver):
        if api_ver == '1':
            return self._serializer_v1
        else:
            return self._serializer_v2

    def _serializer_v1(self, data):
        data = data.get('data', [])
        resp = [img.get('faces') for img in data]
        return resp

    def _serializer_v2(self, data):

        # Response data is by default in v2 format
        return data


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


class Processing:

    def __init__(self, det_name: str = 'retinaface_r50_v1', rec_name: str = 'arcface_r100_v1',
                 ga_name: str = 'genderage_v1', mask_detector: str = 'mask_detector',
                 max_size: List[int] = None,
                 backend_name: str = 'trt', max_rec_batch_size: int = 1, max_det_batch_size: int = 1,
                 force_fp16: bool = False, triton_uri=None, root_dir: str = '/models'):

        if max_size is None:
            max_size = [640, 480]

        self.max_rec_batch_size = max_rec_batch_size
        self.max_det_batch_size = max_det_batch_size
        self.det_name = det_name
        self.max_size = max_size
        self.model = FaceAnalysis(det_name=det_name,
                                  rec_name=rec_name,
                                  ga_name=ga_name,
                                  mask_detector=mask_detector,
                                  max_size=self.max_size,
                                  max_rec_batch_size=self.max_rec_batch_size,
                                  max_det_batch_size=self.max_det_batch_size,
                                  backend_name=backend_name,
                                  force_fp16=force_fp16,
                                  triton_uri=triton_uri,
                                  root_dir=root_dir
                                  )

    def __iterate_images(self, crops):
        for face in crops:
            if face.get('traceback') is None:
                face = face.get('data')
                yield face

    def embed_crops(self, images, extract_embedding: bool = True, extract_ga: bool = True, detect_masks: bool = False):

        t0 = time.time()
        output = dict(took_ms=None, data=[], status="ok")

        iterator = self.__iterate_images(images)
        iterator = ({'facedata': e} for e in iterator)
        faces = self.model.process_faces(iterator, extract_embedding=extract_embedding, extract_ga=extract_ga,
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

    async def embed(self, images: Dict[str, list], max_size: List[int] = None, threshold: float = 0.6,
                    limit_faces: int = 0, return_face_data: bool = False, extract_embedding: bool = True,
                    extract_ga: bool = True, return_landmarks: bool = False, detect_masks: bool = False):

        _get = partial(self.model.get, max_size=max_size, threshold=threshold,
                       return_face_data=return_face_data,
                       extract_embedding=extract_embedding, extract_ga=extract_ga,
                       limit_faces=limit_faces, detect_masks=detect_masks)

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

    async def extract(self, images: Dict[str, list], max_size: List[int] = None, threshold: float = 0.6,
                      limit_faces: int = 0, embed_only: bool = False, return_face_data: bool = False,
                      extract_embedding: bool = True, extract_ga: bool = True, return_landmarks: bool = False,
                      detect_masks: bool = False,
                      verbose_timings=True, api_ver: str = "1"):

        if not max_size:
            max_size = self.max_size

        t0 = time.time()

        tl0 = time.time()
        images = await get_images(images)
        tl1 = time.time()
        took_loading = tl1 - tl0
        logging.debug(f'Reading images took: {took_loading * 1000:.3f} ms.')
        serializer = Serializer()

        if embed_only:
            _faces_dict = self.embed_crops(images, extract_embedding=extract_embedding, extract_ga=extract_ga,
                                           detect_masks=detect_masks)
            return _faces_dict

        else:
            te0 = time.time()
            output = await self.embed(images, max_size=max_size, return_face_data=return_face_data, threshold=threshold,
                                      limit_faces=limit_faces, extract_embedding=extract_embedding,
                                      extract_ga=extract_ga, return_landmarks=return_landmarks,
                                      detect_masks=detect_masks
                                      )
            took_embed = time.time() - te0
            took = time.time() - t0
            output['took']['total_ms'] = took * 1000
            if verbose_timings:
                output['took']['read_imgs_ms'] = took_loading * 1000
                output['took']['embed_all_ms'] = took_embed * 1000

            return serializer.serialize(output, api_ver=api_ver)

    async def draw(self, images: Union[Dict[str, list], bytes], threshold: float = 0.6,
                   draw_landmarks: bool = True, draw_scores: bool = True, draw_sizes: bool = True, limit_faces=0,
                   detect_masks: bool = False,
                   multipart=False):

        if not multipart:
            images = await get_images(images)
            image = images[0].get('data')
        else:
            __bin = np.fromstring(images, np.uint8)
            image = cv2.imdecode(__bin, cv2.IMREAD_COLOR)

        faces = await self.model.get([image], threshold=threshold, return_face_data=False,
                                     extract_embedding=False, extract_ga=False, limit_faces=limit_faces,
                                     detect_masks=detect_masks)

        image = self.model.draw_faces(image, faces[0],
                                      draw_landmarks=draw_landmarks,
                                      draw_scores=draw_scores,
                                      draw_sizes=draw_sizes)

        is_success, buffer = cv2.imencode(".jpg", image)
        io_buf = io.BytesIO(buffer)
        return io_buf
