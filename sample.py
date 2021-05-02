import os
import base64
import requests
import glob
import time
import multiprocessing
import numpy as np
from itertools import chain, islice
import ujson
import logging
import shutil

from tqdm import tqdm

dir_path = os.path.dirname(os.path.realpath(__file__))
test_cat = os.path.join(dir_path, 'images')

session = requests.Session()
session.trust_env = False

logging.basicConfig(
    level='INFO',
    format='%(asctime)s %(levelname)s - %(message)s',
    datefmt='[%H:%M:%S]',
)


def file2base64(path):
    with open(path, mode='rb') as fl:
        encoded = base64.b64encode(fl.read()).decode('ascii')
        return encoded


def save_crop(data, name):
    img = base64.b64decode(data)
    with open(name, mode="wb") as fl:
        fl.write(img)
        fl.close()

def extract_vecs(task):
    target = task[0]
    server = task[1]
    images = dict(data=target)
    req = dict(images=images,
               threshold=0.6,
               extract_ga=True,
               extract_embedding=True,
               return_face_data=True,
               embed_only=False, # If set to true API expects each image to be 112x112 face crop
               limit_faces=0, # Limit maximum number of processed faces, 0 = no limit
               api_ver='2'
               )

    resp = session.post(server, json=req, timeout=120)

    content = ujson.loads(resp.content)
    took = content.get('took')
    status = content.get('status')
    images = content.get('data')
    counts = [len(e.get('faces', [])) for e in images]

    for im in images:
        faces = im.get('faces', [])
        for i, face in enumerate(faces):
            norm = face.get('norm', 0)
            prob = face.get('prob')
            size = face.get('size')
            facedata = face.get('facedata')
            if facedata:
                save_crop(facedata, f'crops/{i}_{size}_{norm:2.0f}_{prob}_{face["age"]}_{time.time()}.jpg')


def to_chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))


if __name__ == "__main__":

    ims = 'src/api_trt/test_images'
    server = 'http://localhost:18081/extract'

    if os.path.exists('crops'):
        shutil.rmtree('crops')
    os.mkdir('crops')

    speeds = []

    # Test different image types
    #files = glob.glob(ims + '*/*.jpg')
    # Test multiple faces per image
    #files = ['src/api_trt/test_images/lumia.jpg']
    # Test single face per image
    files = ['src/api_trt/test_images/Stallone.jpg']

    print(f"Total files detected: {len(files)}")
    multiply = 100
    t0 = time.time()
    target = [file2base64(fl) for fl in files] * multiply
    target = list(target)
    print(f"Total files per test: {len(target)}")
    t1 = time.time()
    print(f'Encoding images took {t1 - t0}')
    target_chunks = to_chunks(target, 1)
    task_set = [[list(chunk), server] for i, chunk in enumerate(target_chunks)]
    task_set = list(task_set)
    print('Encoding images.... Finished')
    pool = multiprocessing.Pool(2)
    for i in range(0, 10):
        t0 = time.time()
        r = pool.map(extract_vecs, task_set)
        t1 = time.time()
        took = t1 - t0
        speed = len(files) * multiply / took
        speeds.append(speed)
        print("Took: {} ({} im/sec)".format(took, speed))

    pool.close()
    mean = np.mean(speeds)
    median = np.median(speeds)

    print(f'mean: {mean} im/sec\n'
          f'median: {median}\n'
          f'min: {np.min(speeds)}\n'
          f'max: {np.max(speeds)}\n'
          )
