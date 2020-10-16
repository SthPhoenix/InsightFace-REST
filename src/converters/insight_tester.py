import time
import os
import numpy as np
import cv2
from insightface import model_zoo

from numpy.linalg import norm

from configs import Configs
from exec_backends.trt_backend import Arcface as InsightTRT
#from exec_backends.triton_backend import Arcface as InsightTriton
from exec_backends.onnxrt_backend import Arcface as InsightORT


'''
ATTENTION!!! This script is for testing purposes only. Work in progress.
'''


def normalize(embedding):
    embedding_norm = norm(embedding)
    normed_embedding = embedding / embedding_norm
    return normed_embedding

config = Configs()

model_name = 'arcface_r100_v1'

engine = config.build_model_paths(model_name, 'plan')[1]
model_onnx = config.build_model_paths(model_name, 'onnx')[1]

model = InsightTRT(rec_name=engine)
#model = InsightTriton(rec_name=engine)
#model = InsightORT(rec_name=model_onnx)
model.prepare()

model_orig = model_zoo.get_model(model_name, root=config.mxnet_models_dir)
model_orig.prepare(-1)

im = cv2.imread('test_images/crop.jpg', cv2.IMREAD_COLOR)
iters = 100

t0 = time.time()
for i in range(iters):
    emb = model.get_embedding(im)

t1 = time.time()



print(f'Took {t1 - t0} s. ({iters/(t1 - t0)} faces/sec)')
#
emb1 = model.get_embedding(im)[0]
emb2 = model_orig.get_embedding(im)[0]
emb1 = normalize(emb1)
emb2 = normalize(emb2)
sim = (1. + np.dot(emb1, emb2)) / 2.
print(sim)

