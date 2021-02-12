import onnxruntime
import cv2
import numpy as np
import logging

class Arcface:
    def __init__(self, rec_name='/models/onnx/arcface_r100_v1/arcface_r100_v1.onnx'):
        self.rec_model = onnxruntime.InferenceSession(rec_name)
        self.outputs = [e.name for e in self.rec_model.get_outputs()]

    # warmup
    def prepare(self, **kwargs):
        logging.info("Warming up ArcFace ONNX Runtime engine...")
        self.rec_model.run(self.outputs, {self.rec_model.get_inputs()[0].name: [np.zeros((3, 112, 112), np.float32)]})

    def get_embedding(self, face_img):
        if not isinstance(face_img, list):
            face_img = [face_img]

        for i, img in enumerate(face_img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            face_img[i] = img.astype(np.float32)

        face_img = np.stack(face_img)
        net_out = self.rec_model.run(self.outputs, {self.rec_model.get_inputs()[0].name: face_img})
        return net_out[0]


class FaceGenderage:

    def __init__(self, rec_name='/models/onnx/genderage_v1/genderage_v1.onnx', outputs=None):
        self.rec_model = onnxruntime.InferenceSession(rec_name)
        self.input = self.rec_model.get_inputs()[0]
        if outputs is None:
            outputs = [e.name for e in self.rec_model.get_outputs()]
        self.outputs = outputs

    # warmup
    def prepare(self, **kwargs):
        logging.info("Warming up GenderAge ONNX Runtime engine...")
        self.rec_model.run(self.outputs,
                           {self.rec_model.get_inputs()[0].name: [np.zeros(tuple(self.input.shape[1:]), np.float32)]})

    def get(self, face_img):
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = np.transpose(face_img, (2, 0, 1))
        face_img = np.expand_dims(face_img, axis=0)
        face_img = face_img.astype(np.float32)

        ret = self.rec_model.run(self.outputs, {self.input.name: face_img})[0]
        g = ret[:, 0:2].flatten()
        gender = np.argmax(g)
        a = ret[:, 2:202].reshape((100, 2))
        a = np.argmax(a, axis=1)
        age = int(sum(a))
        return gender, age


class DetectorInfer:

    def __init__(self, model='/models/onnx/centerface/centerface.onnx',
                 output_order=None):

        self.rec_model = onnxruntime.InferenceSession(model)
        self.input = self.rec_model.get_inputs()[0]

        if output_order is None:
            output_order = [e.name for e in self.rec_model.get_outputs()]
        self.output_order = output_order

        self.input_shape = tuple(self.input.shape)
        print(self.input_shape)

    # warmup
    def prepare(self, ctx=0):
        logging.info("Warming up face detection ONNX Runtime engine...")
        self.rec_model.run(self.output_order,
                           {self.rec_model.get_inputs()[0].name: [np.zeros(tuple(self.input.shape[1:]), np.float32)]})

    def run(self, input):
        net_out = self.rec_model.run(self.output_order, {self.input.name: input})
        return net_out