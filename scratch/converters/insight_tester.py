import time
import os
import numpy as np
import cv2
from numpy.linalg import norm
import logging

from modules.model_zoo.getter import get_model
from modules.configs import Configs

from insightface import model_zoo

'''
ATTENTION!!! This script is for testing purposes only. Work in progress.
'''


def normalize(embedding):
    embedding_norm = norm(embedding)
    normed_embedding = embedding / embedding_norm
    return normed_embedding

logging.basicConfig(
    level='INFO',
    format='%(asctime)s %(levelname)s - %(message)s',
    datefmt='[%H:%M:%S]',
)

config = Configs('/models')

iters = 10
model_name = 'arcface_r100_v1'
backend = 'trt'  # or 'onnx'

model = get_model(model_name, backend, root_dir='/models', force_fp16=False)
model.prepare()

logging.info('Loading original MXNet model for comparison')
model_orig = model_zoo.get_model(model_name, root=config.mxnet_models_dir)
model_orig.prepare(-1)

im = cv2.imread('test_images/crop.jpg', cv2.IMREAD_COLOR)


paths = ['test_images/crop.jpg', 'test_images/TH.png']

logging.info('Measuring speed...')
t0 = time.time()
for i in range(iters):
    emb = model.get_embedding(im)[0].tolist()
    logging.debug(emb[:10])

t1 = time.time()
logging.info(f'Took {t1 - t0} s. ({iters/(t1 - t0)} faces/sec)')

logging.info('Comparing different faces...')
embs = []
for path in paths:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    emb = model.get_embedding(img)[0].tolist()
    embs.append(emb)
    logging.debug(emb[:10])

embs = normalize(embs)
sim = (1. + np.dot(embs[0], embs[1])) / 2.
logging.info(f"Faces similarity: {sim} (from 0 to 1 where 1 is absolute match)")


#
logging.info('Comparing embeddings of TRT model and MXNet model...')
emb1 = model.get_embedding(im)[0]
emb2 = model_orig.get_embedding(im)[0]
logging.debug(f"First 10 values of TRT embedding:\n{emb1.tolist()[:10]}")
logging.debug(f"First 10 values of MXNet embedding:\n{emb2.tolist()[:10]}")

emb1 = normalize(emb1)
emb2 = normalize(emb2)
sim = (1. + np.dot(emb1, emb2)) / 2.
logging.info(f"Similarity: {sim}")

