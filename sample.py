import os
import json
import base64
import requests
import glob
import time
import multiprocessing
import numpy as np
from itertools import chain, islice

dir_path = os.path.dirname(os.path.realpath(__file__))
test_cat = os.path.join(dir_path, 'images')

session = requests.Session()
session.trust_env = False


def file2base64(path):
    with open(path, mode='rb') as fl:
        encoded = base64.b64encode(fl.read()).decode('ascii')
        return encoded


def save_file(data, name):
    img = base64.b64decode(data)
    with open(name, mode="wb") as fl:
        fl.write(img)
        fl.close()


import ujson


def extract_vecs(task):
    target = task[0]
    server = task[1]
    req = dict(images=dict(data=target),
               max_size=[1280, 800],
               threshold=0.2,
               extract_ga=False,
               extract_embedding=True,
               return_face_data=True
               )

    resp = session.post(server, json=req)
    content = ujson.loads(resp.content)

    counts = [len(e) for e in content]
    print(counts)
    for im in content:
        for face in im:
            norm = face['norm']
            prob = face['prob']
            facedata = face['facedata']
            save_file(facedata, f'crops/{int(norm)}_{prob}_{face["age"]}_{time.time()}.jpg')


ims = 'test_images'


def to_chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))


if __name__ == "__main__":

    server = 'http://localhost:18081/extract'

    speeds = []
    for i in range(0, 2):
        files = glob.glob(ims + '*/*.jpg')
        print(f"Total files detected: {len(files)}")
        multiply = 1
        target = [file2base64(fl) for fl in files] * multiply
        target = list(target)
        target_chunks = to_chunks(target, 1)
        task_set = [[list(chunk), server] for i, chunk in enumerate(target_chunks)]
        task_set = list(task_set)
        print('Encoding images.... Finished')

        pool = multiprocessing.Pool(20)
        t0 = time.time()
        r = pool.map(extract_vecs, task_set)
        pool.close()
        t1 = time.time()
        took = t1 - t0
        speed = len(files) * multiply / took
        speeds.append(speed)
        print("Took: {} ({} im/sec)".format(took, speed))

    avg_speed = np.mean(speeds)
    print('Average speed: {} im/sec'.format(avg_speed))
