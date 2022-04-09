import os
import base64
import requests
import time
import glob
import multiprocessing
import numpy as np
from itertools import chain, islice, cycle
from functools import partial
from distutils import util
import ujson
import logging
import shutil
import argparse
import msgpack

dir_path = os.path.dirname(os.path.realpath(__file__))
test_cat = os.path.join(dir_path, 'images')

session = requests.Session()
session.trust_env = False

logging.basicConfig(
    level='INFO',
    format='%(asctime)s %(levelname)s - %(message)s',
    datefmt='[%H:%M:%S]',
)


def to_chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))


def file2base64(path):
    with open(path, mode='rb') as fl:
        encoded = base64.b64encode(fl.read()).decode('ascii')
        return encoded


def save_crop(data, name):
    img = base64.b64decode(data)
    with open(name, mode="wb") as fl:
        fl.write(img)
        fl.close()


def to_bool(input):
    try:
        return bool(util.strtobool(input))
    except:
        return False


class IFRClient:

    def __init__(self, host: str = 'http://localhost', port: int = '18081'):
        self.server = f'{host}:{port}'
        self.sess = requests.Session()

    def server_info(self, server: str = None, show=True):
        if server is None:
            server = self.server

        info_uri = f'{server}/info'
        info = self.sess.get(info_uri).json()

        if show:
            server_uri = self.server
            backend_name = info['models']['backend_name']
            det_name = info['models']['det_name']
            rec_name = info['models']['rec_name']
            rec_batch_size = info['models']['rec_batch_size']
            det_batch_size = info['models']['det_batch_size']

            print(f'Server: {server_uri}\n'
                  f'    Inference backend:      {backend_name}\n'
                  f'    Detection model:        {det_name}\n'
                  f'    Detection batch size:   {det_batch_size}\n'
                  f'    Recognition model:      {rec_name}\n'
                  f'    Recognition batch size: {rec_batch_size}')

        return info

    def extract(self, data: list, mode: str = 'paths', server: str = None,
                threshold: float = 0.6, extract_embedding=True, return_face_data=False,
                embed_only=False, limit_faces=0, use_msgpack=True):

        if server is None:
            server = self.server

        extract_uri = f'{server}/extract'

        if mode == 'data':
            images = dict(data=data)
        elif mode == 'paths':
            images = dict(urls=data)

        req = dict(images=images,
                   threshold=threshold,
                   extract_ga=False,
                   extract_embedding=extract_embedding,
                   return_face_data=return_face_data,
                   embed_only=embed_only,  # If set to true API expects each image to be 112x112 face crop
                   limit_faces=limit_faces,  # Limit maximum number of processed faces, 0 = no limit
                   api_ver='2',
                   msgpack=use_msgpack,
                   )

        resp = self.sess.post(extract_uri, json=req, timeout=120)
        if resp.headers['content-type'] == 'application/x-msgpack':
            content = msgpack.loads(resp.content)
        else:
            content = ujson.loads(resp.content)

        images = content.get('data')
        for im in images:
            status = im.get('status')
            if status != 'ok':
                print(content.get('traceback'))
                break
            faces = im.get('faces', [])
            for i, face in enumerate(faces):
                norm = face.get('norm', 0)
                prob = face.get('prob')
                size = face.get('size')
                facedata = face.get('facedata')
                if facedata:
                    save_crop(facedata, f'crops/{i}_{size}_{norm:2.0f}_{prob}.jpg')

        return content


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='defa')

    parser.add_argument('-p', '--port', default=18081, type=int, help='Port')
    parser.add_argument('-u', '--uri', default='http://localhost', type=str, help='Server hostname or ip with protocol')
    parser.add_argument('-i', '--iters', default=10, type=int, help='Number of iterations')
    parser.add_argument('-t', '--threads', default=10, type=int, help='Number of threads')
    parser.add_argument('-b', '--batch', default=1, type=int, help='Batch size')
    parser.add_argument('-d', '--dir', default=None, type=str, help='Path to directory with images')
    parser.add_argument('-n', '--num_files', default=1000, type=int, help='Number of files per test')
    parser.add_argument('-lf', '--limit_faces', default=0, type=int, help='Number of files per test')
    parser.add_argument('--embed', default='True', type=str, help='Extract embeddings, otherwise run detection only')
    parser.add_argument('--embed_only', default='False', type=str, help='Omit detection step. Expects already cropped 112x112 images')

    args = parser.parse_args()

    allowed_ext = '.jpeg .jpg .bmp .png .webp .tiff'.split()

    client = IFRClient(host=args.uri, port=args.port)

    if os.path.exists('crops'):
        shutil.rmtree('crops')
    os.mkdir('crops')

    print('---')
    client.server_info(show=True)
    print('Benchmark configs:')
    print(f"    Embed detected faces:        {args.embed}")
    print(f"    Run in embed only mode:      {args.embed_only}")
    print(f'    Request batch size:          {args.batch}')
    print(f"    Min. num. of files per iter: {args.num_files}")
    print(f"    Number of iterations:        {args.iters}")
    print(f"    Number of threads:           {args.threads}")
    print('---')

    mode = 'paths'
    if args.dir is None:
        # Test single face per image
        if to_bool(args.embed_only):
            files = ['test_images/TH.png']
        else:
            files = ['test_images/Stallone.jpg']
        print(f'No data directory provided. Using `{files[0]}` for testing.')
    else:
        files = glob.glob(os.path.join(args.dir, '*/*.*'))
        files = [file for file in files if os.path.splitext(file)[1].lower() in allowed_ext]
        if args.dir.startswith('src/api_trt/'):
            files = [file.replace('src/api_trt/', '') for file in files]
        else:
            print('Images will be sent in base64 encoding')
            mode = 'data'
            files = [file2base64(file) for file in files]

    print(f"Total files detected: {len(files)}")
    total = len(files)

    if total < args.num_files:
        print(f'Number of files is less than {args.num_files}. Files will be cycled.')
        total = args.num_files
        files = islice(cycle(files), total)

    im_batches = to_chunks(files, args.batch)
    im_batches = [list(chunk) for chunk in im_batches]

    _part_extract_vecs = partial(client.extract, extract_embedding=to_bool(args.embed), embed_only=to_bool(args.embed_only), mode=mode,
                                 limit_faces = args.limit_faces)

    pool = multiprocessing.Pool(args.threads)
    speeds = []

    print('\nRunning benchmark...')
    for i in range(0, args.iters):
        t0 = time.time()
        r = pool.map(_part_extract_vecs, im_batches)
        t1 = time.time()
        took = t1 - t0
        speed = total / took
        speeds.append(speed)
        print(f"    Iter {i + 1}/{args.iters} Took: {took:.3f} s. ({speed:.3f} im/sec)")

    pool.close()

    mean = np.mean(speeds)
    median = np.median(speeds)

    print(f'\nThroughput:\n'
          f'    mean:   {mean:.3f} im/sec\n'
          f'    median: {median:.3f} im/sec\n'
          f'    min:    {np.min(speeds):.3f} im/sec\n'
          f'    max:    {np.max(speeds):.3f} im/sec\n'
          )
