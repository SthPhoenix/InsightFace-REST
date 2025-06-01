import argparse
import glob
import logging
import multiprocessing
import os
import shutil
import time
from functools import partial
from itertools import islice, cycle
from typing import Union
import numpy as np
import requests

from ifr_clients.common_utils import to_chunks, to_bool, read_image
from ifr_clients import IFRClient

dir_path = os.path.dirname(os.path.realpath(__file__))
test_cat = os.path.join(dir_path, 'images')

session = requests.Session()
session.trust_env = False

logging.basicConfig(
    level='INFO',
    format='%(asctime)s %(levelname)s - %(message)s',
    datefmt='[%H:%M:%S]',
)


def run_benchmark(
        host: str = 'http://localhost',
        port: Union[str, int] = 18081,
        batch_size: int = 64,
        threads: int = 12,
        iterations: int = 10,
        num_files: int = 10_000,
        max_faces: int = 0,
        embed: bool = True,
        embed_only: bool = False,
        use_msgpack: bool = True,
        images_dir: Union[str, None] = None
):
    allowed_ext = '.jpeg .jpg .bmp .png .webp .tiff'.split()

    client = IFRClient(host=host, port=port)

    if os.path.exists('crops'):
        shutil.rmtree('crops')
    os.mkdir('crops')

    print('---')
    client.server_info(show=True)
    print('Benchmark configs:')
    print(f"    Embed detected faces:        {embed}")
    print(f"    Run in embed only mode:      {embed_only}")
    print(f'    Request batch size:          {batch_size}')
    print(f"    Min. num. of files per iter: {num_files}")
    print(f"    Number of iterations:        {iterations}")
    print(f"    Number of threads:           {threads}")
    print('---')

    mode = 'paths'
    if images_dir is None:
        # Test single face per image
        if to_bool(embed_only):
            files = ['test_images/TH.png']
        else:
            files = ['test_images/Stallone.jpg']
        print(f'No data directory provided. Using `{files[0]}` for testing.')
    else:
        files = glob.glob(os.path.join(dir_path, '*.*'))
        files = [file for file in files if os.path.splitext(file)[1].lower() in allowed_ext]
        files = [read_image(file, ) for file in files]

    print(f"Total files detected: {len(files)}")
    total = len(files)

    if total < num_files:
        print(f'Number of files is less than {num_files}. Files will be cycled.')
        total = num_files
        files = islice(cycle(files), total)

    im_batches = to_chunks(files, batch_size)
    im_batches = [list(chunk) for chunk in im_batches]

    _part_extract_vecs = partial(client.extract, extract_embedding=to_bool(embed),
                                 embed_only=to_bool(embed_only), mode=mode,
                                 limit_faces=max_faces, use_msgpack=use_msgpack)

    pool = multiprocessing.Pool(threads)
    speeds = []

    print('\nRunning benchmark...')
    for i in range(0, iterations):
        t0 = time.time()
        r = pool.map(_part_extract_vecs, im_batches)
        t1 = time.time()
        took = t1 - t0
        speed = total / took
        speeds.append(speed)
        print(f"    Iter {i + 1}/{iterations} Took: {took:.3f} s. ({speed:.3f} im/sec)")

    pool.close()

    mean = np.mean(speeds)
    median = np.median(speeds)

    print(f'\nThroughput:\n'
          f'    mean:   {mean:.3f} im/sec\n'
          f'    median: {median:.3f} im/sec\n'
          f'    min:    {np.min(speeds):.3f} im/sec\n'
          f'    max:    {np.max(speeds):.3f} im/sec\n'
          )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='defa')

    parser.add_argument('-p', '--port', default=18081, type=int, help='Port')
    parser.add_argument('-u', '--uri', default='http://localhost', type=str, help='Server hostname or ip with protocol')
    parser.add_argument('-i', '--iters', default=10, type=int, help='Number of iterations')
    parser.add_argument('-t', '--threads', default=12, type=int, help='Number of threads')
    parser.add_argument('-b', '--batch', default=64, type=int, help='Batch size')
    parser.add_argument('-d', '--dir', default=None, type=str, help='Path to directory with images')
    parser.add_argument('-n', '--num_files', default=10000, type=int, help='Number of files per test')
    parser.add_argument('-lf', '--limit_faces', default=0, type=int, help='Number of files per test')
    parser.add_argument('--embed', default='True', type=str, help='Extract embeddings, otherwise run detection only')
    parser.add_argument('--embed_only', default='False', type=str,
                        help='Omit detection step. Expects already cropped 112x112 images')
    parser.add_argument('--use_msgpack', default='True', type=str,
                        help='Use msgpack for data transfer')

    args = parser.parse_args()

    run_benchmark(host=args.uri,
                  port=args.port,
                  batch_size=args.batch,
                  threads=args.threads,
                  iterations=args.iters,
                  num_files=args.num_files,
                  max_faces=args.limit_faces,
                  embed=args.embed,
                  embed_only=args.embed_only,
                  use_msgpack=args.use_msgpack,
                  images_dir=args.dir
                  )
