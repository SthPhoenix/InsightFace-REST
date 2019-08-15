from flask import Flask
import face_model
import argparse
import json
import base64
# import requests
import numpy as np
import urllib
import cv2
import time
import os
from itertools import chain, islice
from flask import Flask, render_template, request, jsonify

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

parser = argparse.ArgumentParser(description='do verification')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default=os.path.join(dir_path, 'models/model-r100-ii/model,0'),
                    help='path to load model.')
parser.add_argument('--gpu', default=-1, type=int, help='gpu id')
args = parser.parse_args()

model = face_model.FaceModel(args)

app = Flask(__name__)


def to_chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))


@app.route('/')
def hello_world():
    return 'InsightFace-REST'


def image_resize(image):
    m = min(image.shape[0], image.shape[1])
    f = 640.0 / m
    if f < 1.0:
        image = cv2.resize(image, (int(image.shape[1] * f), int(image.shape[0] * f)))
    return image


def get_image(data):
    image = None
    if 'url' in data:
        url = data['url']
        if url.startswith('http'):
            resp = urllib.urlopen(url)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(url, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif 'data' in data:
        _bin = data['data']
        if _bin is not None:
            if not isinstance(_bin, list):
                try:
                    _bin = base64.b64decode(_bin)
                    _bin = np.fromstring(_bin, np.uint8)
                    image = cv2.imdecode(_bin, cv2.IMREAD_COLOR)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # image = image_resize(image)
                    return [image]
                except:
                    return [np.zeros([3, 3], dtype=int)]
            else:
                image = []
                for __bin in _bin:
                    try:
                        __bin = base64.b64decode(__bin)
                        __bin = np.fromstring(__bin, np.uint8)
                        _image = cv2.imdecode(__bin, cv2.IMREAD_COLOR)
                        _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
                    except:
                        _image = np.zeros([3, 3], dtype=int)
                    image.append(_image)

    return image


@app.route('/ver', methods=['POST'])
def ver():
    try:
        data = request.data
        values = json.loads(data.decode('utf-8'))
        source_image = get_image(values['source'])[0]
        if source_image is None:
            print('source image is None')
            return '-1'
        assert not isinstance(source_image, list)
        target_image = get_image(values['target'])
        if target_image is None:
            print('target image is None')
            return '-1'
        if not isinstance(target_image, list):
            target_image = [target_image]
        ret = model.sim(source_image, target_image)
    except Exception as ex:
        print(ex)
        return '-1'
    return jsonify(ret)


@app.route('/extract', methods=['POST'])
def extract():
    data = request.data
    values = json.loads(data.decode('utf-8'))
    images = get_image(values['images'])
    chunks = to_chunks(images, 20)
    detections = []
    # Iterate through chunked images
    for chunk in chunks:
        chunk = list(chunk)
        detections += model.get_all_faces_bulk(chunk)

    def det_provider(dets):
        for id, img in enumerate(dets):
            if img is not None:
                for idx, det in enumerate(img):
                    yield ((id, idx), det)

    detections = det_provider(detections)
    det_chunks = to_chunks(detections, 50)
    output = []
    for det_chunk in det_chunks:
        pos, detections = zip(*det_chunk)
        aligned, meta = zip(*detections)
        batch = np.stack([e for e in aligned])
        embs = model.get_feature_bulk(batch, norm=True)
        to_out = zip(pos, embs, meta)
        for e in to_out:
            im = e[0][0]
            if len(output) < im + 1:
                for i in range((im + 1) - len(output)):
                    output.append([])
            emb = {"vec": e[1].tolist(), "det": e[0][1], "prob": e[2][1], 'bbox': [int(p) for p in e[2][0].tolist()]}
            output[im].append(emb)
    if len(output) < len(detections):
        for i in range(len(detections) - len(output)):
            output.append([])
    return jsonify(output)


if __name__ == '__main__':
    app.run('0.0.0.0', port=6000, debug=False)
