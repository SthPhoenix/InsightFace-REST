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
    # req = {'images': {"data": target},"max_size": 1024}

    req = dict(images=dict(data=target),
               max_size=[1024, 768],
               threshold=0.2,
               extract_ga=False,
               extract_embedding=True,
               return_face_data=True
               )

    resp = session.post(server, json=req)
    content = ujson.loads(resp.content)
    #print(content)
    #content_json = ujson.loads(content)

    counts = [len(e) for e in content]
    #print(counts)
    #print(content)
    for im in content:
        for face in im:
            norm = face['norm']
            prob = face['prob']
            facedata = face['facedata']
            #print("FG",face['gender'], face['age'])
            #if norm > 16 and norm < 20 :
            #print(norm)
            norm = 1
            save_file(facedata, 'crops/{}_{}_{}.jpg'.format(int(norm), prob, face['age']))

    #print(counts)
    # for i,e in enumerate(counts):
    #     if e == 0:
    #         save_file(target[i],"{}.jpg".format(i))

    # values = json.loads(content.decode('utf-8'))
    # return values


ims = '/home/dmitry/PycharmProjects/untitled/InsightFace-REST/images'


def to_chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))


import random
import shutil

if os.path.exists('crops'):
    shutil.rmtree('crops')

os.makedirs('crops')


if __name__ == "__main__":

    start = 0
    step = 9
    server = 'http://10.1.3.166:18080/extract'

    # files = glob.glob(ims+'*/*.jpg')[start:start+step]
    speeds = []
    for i in range(0, 10):
        files = glob.glob(ims + '*/*.jpg')
        print(f"Total files detected: {len(files)}")
        # dest = '/home/dmitry/PycharmProjects/scratch/camera_indexer/owl-sig/data/out'
        # files = glob.glob('{}/**/*.jpg'.format(dest), recursive=True)

        #random.shuffle(files)
        files = files
        #files = [e for e in files if "000388.jpg" in e]

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
