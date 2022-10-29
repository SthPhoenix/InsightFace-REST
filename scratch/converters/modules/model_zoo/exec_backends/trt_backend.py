import os
import cv2
import numpy as np
import time
import logging

from .trt_loader import TrtModel

class Arcface:

    def __init__(self, rec_name: str='/models/trt-engines/arcface_r100_v1/arcface_r100_v1.plan'):
        self.rec_model = TrtModel(rec_name)
        self.input_shape = None
        self.max_batch_size = 1

    # warmup
    def prepare(self, ctx_id=0):
        logging.info("Warming up ArcFace TensorRT engine...")
        self.rec_model.build()
        self.input_shape = self.rec_model.input_shapes[0]
        self.max_batch_size = self.rec_model.max_batch_size
        if self.input_shape[0] == -1:
            self.input_shape = (1,) + self.input_shape[1:]

        self.rec_model.run(np.zeros(self.input_shape, np.float32))
        logging.info(f"Engine warmup complete! Expecting input shape: {self.input_shape}. Max batch size: {self.max_batch_size}")

    def get_embedding(self, face_img):

        if not isinstance(face_img, list):
            face_img = [face_img]

        for i, img in enumerate(face_img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            #img = np.expand_dims(img, axis=0)
            face_img[i] = img
        #assert face_img.shape == self.rec_model.input_shapes[0]
        face_img = np.stack(face_img)
        embeddings = self.rec_model.run(face_img, deflatten=True)[0]
        return embeddings


class FaceGenderage:

    def __init__(self, rec_name: str='/models/trt-engines/genderage_v1/genderage_v1.plan'):
        self.rec_model = TrtModel(rec_name)
        self.input_shape = None

    # warmup
    def prepare(self, ctx_id=0):
        logging.info("Warming up GenderAge TensorRT engine...")
        self.rec_model.build()
        self.input_shape = self.rec_model.input_shapes[0]
        self.rec_model.run(np.zeros(self.input_shape, np.float32))
        logging.info(f"Engine warmup complete! Expecting input shape: {self.input_shape}")

    def get(self, face_img):
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = np.transpose(face_img, (2, 0, 1))
        face_img = np.expand_dims(face_img, axis=0)
        assert face_img.shape == self.rec_model.input_shapes[0]
        ret = self.rec_model.run(face_img, deflatten=True)[0]
        g = ret[:, 0:2].flatten()
        gender = np.argmax(g)
        a = ret[:, 2:202].reshape((100, 2))
        a = np.argmax(a, axis=1)
        age = int(sum(a))
        return gender, age

class DetectorInfer:

    def __init__(self, model='/models/trt-engines/centerface/centerface.plan',
                 output_order=None):
        self.rec_model = TrtModel(model)
        self.model_name = os.path.basename(model)
        self.input_shape = None
        self.output_order = output_order

    # warmup
    def prepare(self, ctx_id=0):
        logging.info(f"Warming up face detector TensorRT engine...")
        self.rec_model.build()
        self.input_shape = self.rec_model.input_shapes[0]
        if not self.output_order:
            self.output_order = self.rec_model.out_names
        self.rec_model.run(np.zeros(self.input_shape, np.float32))
        logging.info(f"Engine warmup complete! Expecting input shape: {self.input_shape}")

    def run(self, input):
        net_out = self.rec_model.run(input, deflatten=True, as_dict=True)
        net_out = [net_out[e] for e in self.output_order]
        return net_out