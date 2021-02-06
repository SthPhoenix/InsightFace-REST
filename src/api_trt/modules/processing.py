from typing import Dict, List, Optional, Union
import urllib
import urllib.request
import traceback
import io
import base64

import numpy as np
import cv2

from .face_model import FaceAnalysis, Face


def get_image(data: Dict[str, list]):
    images = []
    if data.get('urls') is not None:
        urls = data['urls']
        for url in urls:
            try:
                if url.startswith('http'):
                    req = urllib.request.Request(
                        url,
                        data=None,
                        headers={
                            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
                        }
                    )
                    resp = urllib.request.urlopen(req)
                    _image = np.asarray(bytearray(resp.read()), dtype="uint8")
                    _image = cv2.imdecode(_image, cv2.IMREAD_COLOR)

                else:
                    _image = cv2.imread(url, cv2.IMREAD_COLOR)
            except Exception:
                tb = traceback.format_exc()
                print(tb)
                _image = np.zeros([3, 3], dtype=int)

            if _image is None:
                _image = np.zeros([3, 3], dtype=int)
            images.append(_image)
    elif data.get('data') is not None:
        _bin = data['data']
        if _bin is not None:
            if not isinstance(_bin, list):
                try:
                    _bin = _bin.split("base64,")[-1]
                    _bin = base64.b64decode(_bin)
                    _bin = np.fromstring(_bin, np.uint8)
                    image = cv2.imdecode(_bin, cv2.IMREAD_COLOR)
                    return [image]
                except:
                    return [np.zeros([3, 3], dtype=int)]
            else:
                images = []
                for __bin in _bin:
                    try:
                        __bin = __bin.split("base64,")[-1]
                        __bin = base64.b64decode(__bin)
                        __bin = np.fromstring(__bin, np.uint8)
                        _image = cv2.imdecode(__bin, cv2.IMREAD_COLOR)
                    except:
                        _image = np.zeros([3, 3], dtype=int)

                    images.append(_image)
    return images


class Serializer:

    def serialize(self, face: Face, return_face_data: bool = False, return_landmarks: bool = False, api_ver: str = '1'):
        serializer = self.get_serializer(api_ver)
        return serializer(face, return_face_data=return_face_data, return_landmarks=return_landmarks)

    def get_serializer(self, api_ver):
        if api_ver == '1':
            return self._serializer_v1
        else:
            return self._serializer_v1

    def _serializer_v1(self, face: Face, return_face_data: bool, return_landmarks: bool =False):
        _face_dict = dict(status='Ok',
                          det=face.num_det,
                          prob=float(face.det_score),
                          bbox=face.bbox.astype(int).tolist(),
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

        if face.mask_prob:
            _face_dict.update(mask_prob=float(face.mask_prob))

        if return_face_data:
            _face_dict.update({
                'facedata': base64.b64encode(cv2.imencode('.jpg', face.facedata)[1].tostring()).decode(
                    'utf-8')
            })
        if return_landmarks:
            _face_dict.update({
                'landmarks': face.landmark.astype(int).tolist()
            })

        return _face_dict


class Processing:

    def __init__(self, det_name: str = 'retinaface_r50_v1', rec_name: str = 'arcface_r100_v1',
                 ga_name: str = 'genderage_v1', device: str = 'cuda', max_size: List[int] = None,
                 backend_name: str = 'trt', max_rec_batch_size: int = 1,
                 force_fp16: bool = False):

        if max_size is None:
            max_size = [640, 480]

        self.max_rec_batch_size = max_rec_batch_size

        self.max_size = max_size
        self.model = FaceAnalysis(det_name=det_name, rec_name=rec_name, ga_name=ga_name, device=device,
                                  max_size=self.max_size, max_rec_batch_size=self.max_rec_batch_size,
                                  backend_name=backend_name, force_fp16=force_fp16
                                  )

    async def embed(self, images: Dict[str, list], max_size: List[int] = None, threshold: float = 0.6, return_face_data: bool = False,
              extract_embedding: bool = True, extract_ga: bool = True, return_landmarks: bool = False, api_ver: str = "1"):

        if not max_size:
            max_size = self.max_size

        images = get_image(images)
        output = []
        serializer = Serializer()
        for image in images:
            try:
                faces = await self.model.get(image, max_size=max_size, threshold=threshold, return_face_data=return_face_data,
                                       extract_embedding=extract_embedding, extract_ga=extract_ga)
                _faces_dict = []

                for idx, face in enumerate(faces):
                    _face_dict = serializer.serialize(face=face, return_face_data=return_face_data,
                                                      return_landmarks= return_landmarks, api_ver=api_ver)
                    _faces_dict.append(_face_dict)
            except Exception as e:
                tb = traceback.format_exc()
                print(tb)
                _faces_dict = []
            output.append(_faces_dict)

        return output

    async def draw(self, images: Dict[str, list], max_size: List[int] = None, threshold: float = 0.6, return_face_data: bool = False,
             extract_embedding: bool = True, extract_ga: bool = True):

        if not max_size:
            max_size = self.max_size

        image = get_image(images)[0]
        faces = await self.model.get(image, max_size=max_size, threshold=threshold, return_face_data=return_face_data,
                               extract_embedding=False, extract_ga=extract_ga)
        for face in faces:
            pt1 = tuple(map(int, face.bbox[0:2]))
            pt2 = tuple(map(int, face.bbox[2:4]))
            color = (0, 255, 0)
            if face.mask_prob:
                if face.mask_prob >= 0.2:
                    color = (0, 255, 255)
            cv2.rectangle(image, pt1, pt2, color, 1)
            lms = face.landmark
            cv2.circle(image, (lms[0][0], lms[0][1]), 1, (0, 0, 255), 4)
            cv2.circle(image, (lms[1][0], lms[1][1]), 1, (0, 255, 255), 4)
            cv2.circle(image, (lms[2][0], lms[2][1]), 1, (255, 0, 255), 4)
            cv2.circle(image, (lms[3][0], lms[3][1]), 1, (0, 255, 0), 4)
            cv2.circle(image, (lms[4][0], lms[4][1]), 1, (255, 0, 0), 4)

        is_success, buffer = cv2.imencode(".jpg", image)
        io_buf = io.BytesIO(buffer)
        return io_buf
