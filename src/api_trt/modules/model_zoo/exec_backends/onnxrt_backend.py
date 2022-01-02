import onnxruntime
import cv2
import numpy as np
import logging


class Arcface:
    def __init__(self, rec_name='/models/onnx/arcface_r100_v1/arcface_r100_v1.onnx',
                 input_mean: float = 0.,
                 input_std: float = 1.,
                 **kwargs):
        self.rec_model = onnxruntime.InferenceSession(rec_name)
        self.input_mean = input_mean
        self.input_std = input_std
        self.outputs = [e.name for e in self.rec_model.get_outputs()]

    # warmup
    def prepare(self, **kwargs):
        logging.info("Warming up ArcFace ONNX Runtime engine...")
        self.rec_model.run(self.outputs, {self.rec_model.get_inputs()[0].name: [np.zeros((3, 112, 112), np.float32)]})

    def get_embedding(self, face_img):
        if not isinstance(face_img, list):
            face_img = [face_img]

        face_img = np.stack(face_img)

        input_size = tuple(face_img[0].shape[0:2][::-1])
        blob = cv2.dnn.blobFromImages(face_img, 1.0 / self.input_std, input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)

        net_out = self.rec_model.run(self.outputs, {self.rec_model.get_inputs()[0].name: blob})
        return net_out[0]


class FaceGenderage:

    def __init__(self, rec_name='/models/onnx/genderage_v1/genderage_v1.onnx', outputs=None, **kwargs):
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

        face_img = np.stack(face_img)
        imgs = face_img.copy()


        if not face_img[0].shape == (3, 112, 112):
            imgs = imgs[..., ::-1]
            imgs = np.transpose(imgs, (0, 3, 1, 2))

        _ga = []

        ret = self.rec_model.run(self.outputs, {self.input.name: imgs})[0]
        for e in ret:
            e = np.expand_dims(e, axis=0)
            g = e[:, 0:2].flatten()
            gender = np.argmax(g)
            a = e[:, 2:202].reshape((100, 2))
            a = np.argmax(a, axis=1)
            age = int(sum(a))
            _ga.append((gender, age))
        return _ga

class MaskDetection:

    def __init__(self, rec_name='/models/onnx/genderage_v1/genderage_v1.onnx', outputs=None, **kwargs):
        self.rec_model = onnxruntime.InferenceSession(rec_name)
        self.input = self.rec_model.get_inputs()[0]
        if outputs is None:
            outputs = [e.name for e in self.rec_model.get_outputs()]
        self.outputs = outputs

    # warmup
    def prepare(self, **kwargs):
        logging.info("Warming up mask detection ONNX Runtime engine...")
        self.rec_model.run(self.outputs,
                           {self.rec_model.get_inputs()[0].name: [np.zeros(tuple(self.input.shape[1:]), np.float32)]})

    def get(self, face_img):
        if not isinstance(face_img, list):
            face_img = [face_img]
        if not self.input.shape[1:3] == [112, 112]:
            for i, img in enumerate(face_img):
                img = cv2.resize(img, (224, 224))
                face_img[i] = img
            face_img = np.stack(face_img)

        face_img = np.multiply(face_img, 1/127.5, dtype='float32') - 1.
        _mask = []

        ret = self.rec_model.run(self.outputs, {self.input.name: face_img})[0]
        for e in ret:
            mask = e[0]
            no_mask = e[1]
            _mask.append((mask, no_mask))
        return _mask


class DetectorInfer:

    def __init__(self, model='/models/onnx/centerface/centerface.onnx',
                 output_order=None, **kwargs):

        self.rec_model = onnxruntime.InferenceSession(model)
        logging.info('Detector started')
        self.input = self.rec_model.get_inputs()[0]
        self.input_dtype = self.input.type
        if self.input_dtype == 'tensor(float)':
            self.input_dtype = np.float32
        else:
            self.input_dtype = np.uint8

        self.output_order = output_order
        self.out_shapes = None
        self.input_shape = tuple(self.input.shape)

    # warmup
    def prepare(self, **kwargs):
        logging.info("Warming up face detection ONNX Runtime engine...")
        if self.output_order is None:
            self.output_order = [e.name for e in self.rec_model.get_outputs()]
        self.out_shapes = [e.shape for e in self.rec_model.get_outputs()]
        self.rec_model.run(self.output_order,
                           {self.rec_model.get_inputs()[0].name: [
                               np.zeros(tuple(self.input.shape[1:]), self.input_dtype)]})

    def run(self, input):
        net_out = self.rec_model.run(self.output_order, {self.input.name: input})
        return net_out
