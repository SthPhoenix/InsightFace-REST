import onnxruntime
import cv2
import numpy as np


class Arcface:
    def __init__(self, rec_name='/models/onnx/arcface_r100_v1/arcface_r100_v1.onnx'):
        self.rec_model = onnxruntime.InferenceSession(rec_name)
        self.outputs = [e.name for e in self.rec_model.get_outputs()]

    # warmup
    def prepare(self, **kwargs):
        print("Warming up ArcFace ONNX Runtime engine...")
        self.rec_model.run(self.outputs, {self.rec_model.get_inputs()[0].name: [np.zeros((3, 112, 112), np.float32)]})

    def get_embedding(self, face_img):
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = np.transpose(face_img, (2, 0, 1))
        face_img = np.expand_dims(face_img, axis=0)
        face_img = face_img.astype(np.float32)
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
        print("Warming up GenderAge ONNX Runtime engine...")
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


class RetinaInfer:

    def __init__(self, rec_name='/models/trt-engines/retinaface_mnet025_v2/retinaface_mnet025_v2.plan',
                 outputs=None):
        self.rec_model = onnxruntime.InferenceSession(rec_name)
        self.input = self.rec_model.get_inputs()[0]
        if outputs is None:
            outputs = [e.name for e in self.rec_model.get_outputs()]
        self.outputs = outputs

    # warmup
    def prepare(self, **kwargs):
        print("Warming up RetinaFace ONNX Runtime engine...")
        self.rec_model.run(self.outputs,
                           {self.rec_model.get_inputs()[0].name: [np.zeros(tuple(self.input.shape[1:]), np.float32)]})

    def run(self, input):
        net_out = self.rec_model.run(self.outputs, {self.input.name: input})
        return net_out


class CenterFaceInfer:

    def __init__(self):
        raise NotImplemented
