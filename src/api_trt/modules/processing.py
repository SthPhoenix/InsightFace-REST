from typing import Dict, List, Optional, Union
import requests
import traceback
import io
import base64
import time
import os
import logging

import numpy as np
import cv2
from turbojpeg import TurboJPEG

from .face_model import FaceAnalysis, Face

jpeg = TurboJPEG()

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
}
sess = requests.Session()
sess.headers.update(headers)

async def dl_image(path):

    im_data = dict(data=None,
                   traceback=None)

    try:
        if path.startswith('http'):
            resp = sess.get(path)
            __bin = bytearray(resp.content)
            try:
                _image = jpeg.decode(__bin)
            except:
                logging.debug('TurboJPEG failed, fallback to cv2.imdecode')
                _image = np.asarray(__bin, dtype="uint8")
                _image = cv2.imdecode(_image, cv2.IMREAD_COLOR)
        else:
            if not os.path.exists(path):
                im_data.update(traceback=f"File: '{path}' not found")
                return im_data
            try:
                in_file = open(path, 'rb')
                _image = jpeg.decode(in_file.read())
                in_file.close()
            except:
                logging.debug('TurboJPEG failed, fallback to cv2.imdecode')
                _image = cv2.imread(path, cv2.IMREAD_COLOR)
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        im_data.update(traceback=tb)
        return im_data
    im_data.update(data=_image)
    return im_data


async def decode_image(b64encoded):
    im_data = dict(data=None,
                   traceback=None)
    try:
        __bin = b64encoded.split(",")[-1]
        __bin = base64.b64decode(__bin)
        t0 = time.time()
        try:
            _image = jpeg.decode(__bin)
        except:
            logging.debug('TurboJPEG failed, fallback to cv2.imdecode')
            __bin = np.fromstring(__bin, np.uint8)
            _image = cv2.imdecode(__bin, cv2.IMREAD_COLOR)
        t1 = time.time()
        logging.info(f'Decoding took: {t1 - t0}')
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        im_data.update(traceback=tb)
        return im_data
    im_data.update(data=_image)
    return im_data


async def get_images(data: Dict[str, list]):
    images = []
    if data.get('urls') is not None:
        urls = data['urls']
        for url in urls:
            _image = await dl_image(url)
            images.append(_image)
    elif data.get('data') is not None:
        b64_images = data['data']
        images = []
        for b64_img in b64_images:
            _image = await decode_image(b64_img)
            images.append(_image)

    return images


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


def serialize_face(face: Face, return_face_data: bool, return_landmarks: bool = False):
    _face_dict = dict(
        det=face.num_det,
        prob=None,
        bbox=None,
        size=None,
        landmarks=None,
        gender=face.gender,
        age=face.age,
        mask_prob=None,
        norm=None,
        vec=None,
    )

    if face.embedding_norm:
        _face_dict.update(vec=face.normed_embedding.tolist(),
                          norm=float(face.embedding_norm))
    # Warkaround for embed_only flag
    if face.det_score:
        _face_dict.update(prob=float(face.det_score),
                          bbox=face.bbox.astype(int).tolist(),
                          size=int(face.bbox[2] - face.bbox[0]))

        if return_landmarks:
            _face_dict.update({
                'landmarks': face.landmark.astype(int).tolist()
            })

    if face.mask_prob:
        _face_dict.update(mask_prob=float(face.mask_prob))

    if return_face_data:
        _face_dict.update({
            'facedata': base64.b64encode(cv2.imencode('.jpg', face.facedata)[1].tostring()).decode(
                'utf-8')
        })

    return _face_dict


class Processing:

    def __init__(self, det_name: str = 'retinaface_r50_v1', rec_name: str = 'arcface_r100_v1',
                 ga_name: str = 'genderage_v1', device: str = 'cuda', max_size: List[int] = None,
                 backend_name: str = 'trt', max_rec_batch_size: int = 1,
                 force_fp16: bool = False, triton_uri=None):

        if max_size is None:
            max_size = [640, 480]

        self.max_rec_batch_size = max_rec_batch_size
        self.det_name = det_name

        self.max_size = max_size
        self.model = FaceAnalysis(det_name=det_name, rec_name=rec_name, ga_name=ga_name, device=device,
                                  max_size=self.max_size, max_rec_batch_size=self.max_rec_batch_size,
                                  backend_name=backend_name, force_fp16=force_fp16, triton_uri=triton_uri
                                  )

    def __iterate_faces(self, crops):
        for face in crops:
            if face.get('traceback') is None:
                face = Face(facedata=face.get('data'))
                yield face

    def embed_crops(self, images, extract_embedding: bool = True, extract_ga: bool = True):

        t0 = time.time()
        output = dict(took=None, data=[], status="ok")

        iterator = self.__iterate_faces(images)
        faces = self.model.process_faces(iterator, extract_embedding=extract_embedding, extract_ga=extract_ga,
                                         return_face_data=False)

        try:
            for image in images:
                if image.get('traceback') is not None:
                    _face_dict = dict(status='failed',
                                      traceback=image.get('traceback'))
                else:
                    _face_dict = serialize_face(face=next(faces), return_face_data=False,
                                                return_landmarks=False)
                output['data'].append(_face_dict)
        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
            output['status'] = 'failed'
            output['traceback'] = tb

        took = time.time() - t0
        output['took'] = took
        return output

    async def embed(self, images: Dict[str, list], max_size: List[int] = None, threshold: float = 0.6,
                    limit_faces: int = 0, return_face_data: bool = False, extract_embedding: bool = True,
                    extract_ga: bool = True, return_landmarks: bool = False):

        output = dict(took={}, data=[])

        for image_data in images:
            _faces_dict = dict(status=None, took=None, faces=[])
            try:
                t1 = time.time()
                if image_data.get('traceback') is not None:
                    _faces_dict['status'] = 'failed'
                    _faces_dict['traceback'] = image_data.get('traceback')
                else:
                    image = image_data.get('data')
                    faces = await self.model.get(image, max_size=max_size, threshold=threshold,
                                                 return_face_data=return_face_data,
                                                 extract_embedding=extract_embedding, extract_ga=extract_ga,
                                                 limit_faces=limit_faces)

                    for idx, face in enumerate(faces):
                        _face_dict = serialize_face(face=face, return_face_data=return_face_data,
                                                    return_landmarks=return_landmarks)
                        _faces_dict['faces'].append(_face_dict)
                    took_image = time.time() - t1
                    _faces_dict['took'] = took_image
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
                      verbose_timings=True, api_ver: str = "1"):

        if not max_size:
            max_size = self.max_size

        t0 = time.time()

        tl0 = time.time()
        images = await get_images(images)
        tl1 = time.time()
        took_loading = tl1 - tl0
        logging.debug(f'Reading images took: {took_loading} s.')
        serializer = Serializer()

        if embed_only:
            _faces_dict = self.embed_crops(images, extract_embedding=extract_embedding, extract_ga=extract_ga)
            return _faces_dict

        else:
            te0 = time.time()
            output = await self.embed(images, max_size=max_size, return_face_data=return_face_data, threshold=threshold,
                                      limit_faces=limit_faces, extract_embedding=extract_embedding,
                                      extract_ga=extract_ga, return_landmarks=return_landmarks
                                      )
            took_embed = time.time() - te0
            took = time.time() - t0
            output['took']['total'] = took
            if verbose_timings:
                output['took']['read_imgs'] = took_loading
                output['took']['embed_all'] = took_embed

            return serializer.serialize(output, api_ver=api_ver)

    async def draw(self, images: Union[Dict[str, list], bytes], threshold: float = 0.6,
                   draw_landmarks: bool = True, draw_scores: bool = True, draw_sizes: bool = True, limit_faces=0,
                   multipart=False):

        if not multipart:
            images = await get_images(images)
            image = images[0].get('data')
        else:
            __bin = np.fromstring(images, np.uint8)
            image = cv2.imdecode(__bin, cv2.IMREAD_COLOR)

        faces = await self.model.get(image, threshold=threshold, return_face_data=False,
                                     extract_embedding=False, extract_ga=False, limit_faces=limit_faces)

        for face in faces:
            bbox = face.bbox.astype(int)
            pt1 = tuple(bbox[0:2])
            pt2 = tuple(bbox[2:4])
            color = (0, 255, 0)
            x, y = pt1
            r, b = pt2
            w = r - x
            if face.mask_prob:
                if face.mask_prob >= 0.2:
                    color = (0, 255, 255)
            cv2.rectangle(image, pt1, pt2, color, 1)

            if draw_landmarks:
                lms = face.landmark.astype(int)
                pt_size = int(w * 0.05)
                cv2.circle(image, (lms[0][0], lms[0][1]), 1, (0, 0, 255), pt_size)
                cv2.circle(image, (lms[1][0], lms[1][1]), 1, (0, 255, 255), pt_size)
                cv2.circle(image, (lms[2][0], lms[2][1]), 1, (255, 0, 255), pt_size)
                cv2.circle(image, (lms[3][0], lms[3][1]), 1, (0, 255, 0), pt_size)
                cv2.circle(image, (lms[4][0], lms[4][1]), 1, (255, 0, 0), pt_size)

            if draw_scores:
                text = f"{face.det_score:.3f}"
                pos = (x + 3, y - 5)
                textcolor = (0, 0, 0)
                thickness = 1
                border = int(thickness / 2)
                cv2.rectangle(image, (x - border, y - 21, w + thickness, 21), color, -1, 16)
                cv2.putText(image, text, pos, 0, 0.5, (0, 255, 0), 3, 16)
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

        is_success, buffer = cv2.imencode(".jpg", image)
        io_buf = io.BytesIO(buffer)
        return io_buf
