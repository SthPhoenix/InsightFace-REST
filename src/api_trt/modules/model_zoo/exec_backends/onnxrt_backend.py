import onnxruntime
import cv2
import numpy as np
import logging


class Arcface:
    def __init__(self, rec_name='/models/onnx/arcface_r100_v1/arcface_r100_v1.onnx',**kwargs):
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

class Cosface:
    def __init__(self, rec_name='/models/onnx/glintr100/glintr100.onnx',**kwargs):
        self.rec_model = onnxruntime.InferenceSession(rec_name)
        self.input_shape = None
        self.max_batch_size = 1
        self.input_mean = 127.5
        self.input_std = 127.5
        self.outputs = [e.name for e in self.rec_model.get_outputs()]

    # warmup
    def prepare(self, **kwargs):
        logging.info("Warming up ArcFace ONNX Runtime engine...")
        self.rec_model.run(self.outputs, {self.rec_model.get_inputs()[0].name: [np.zeros((3, 112, 112), np.float32)]})

    def get_embedding(self, face_img):
        if not isinstance(face_img, list):
            face_img = [face_img]

        for i, img in enumerate(face_img):
            input_size = tuple(img.shape[0:2][::-1])
            blob = cv2.dnn.blobFromImage(img, 1.0 / self.input_std, input_size,
                                         (self.input_mean, self.input_mean, self.input_mean), swapRB=True)[0]
            face_img[i] = blob
        face_img = np.stack(face_img)
        net_out = self.rec_model.run(self.outputs, {self.rec_model.get_inputs()[0].name: face_img})
        return net_out[0]

class FaceGenderage:

    def __init__(self, rec_name='/models/onnx/genderage_v1/genderage_v1.onnx', outputs=None,**kwargs):
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
        if not isinstance(face_img, list):
            face_img = [face_img]
        if not face_img[0].shape == (3, 112, 112):
            for i, img in enumerate(face_img):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.transpose(img, (2, 0, 1))
                face_img[i] = img.astype(np.float32)
        face_img = np.stack(face_img)

        _ga = []

        ret = self.rec_model.run(self.outputs, {self.input.name: face_img})[0]
        for e in ret:
            e = np.expand_dims(e, axis=0)
            g = e[:, 0:2].flatten()
            gender = np.argmax(g)
            a = e[:, 2:202].reshape((100, 2))
            a = np.argmax(a, axis=1)
            age = int(sum(a))
            _ga.append((gender, age))
        return _ga


class DetectorInfer:

    def __init__(self, model='/models/onnx/centerface/centerface.onnx',
                 output_order=None,**kwargs):
        self.rec_model = onnxruntime.InferenceSession(model)
        self.input = self.rec_model.get_inputs()[0]

        if output_order is None:
            output_order = [e.name for e in self.rec_model.get_outputs()]
        self.output_order = output_order

        self.input_shape = tuple(self.input.shape)
        print(self.input_shape)

    # warmup
    def prepare(self, **kwargs):
        logging.info("Warming up face detection ONNX Runtime engine...")
        self.rec_model.run(self.output_order,
                           {self.rec_model.get_inputs()[0].name: [np.zeros(tuple(self.input.shape[1:]), np.float32)]})

    def run(self, input):
        net_out = self.rec_model.run(self.output_order, {self.input.name: input})
        return net_out
