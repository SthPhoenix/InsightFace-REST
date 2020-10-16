from .trt_loader import TrtModel
import cv2
import numpy as np


class Arcface:

    def __init__(self, rec_name='/models/trt-engines/arcface_r100_v1/arcface_r100_v1.plan'):
        self.rec_model = TrtModel(rec_name)

    # warmup
    def prepare(self, ctx=0):
        print("Warming up TensorRT engine...")
        self.rec_model.build()
        self.rec_model.run(np.zeros(self.rec_model.input_shapes[0], np.float32))
        print(f"Engine warmup complete!\nExpecting input shapes: {self.rec_model.input_shapes}")

    def get_embedding(self, face_img):
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = np.transpose(face_img, (2, 0, 1))
        face_img = np.expand_dims(face_img, axis=0)
        assert face_img.shape == self.rec_model.input_shapes[0]
        embedding = self.rec_model.run(face_img, deflatten=True)[0]
        return embedding


class FaceGenderage:

    def __init__(self, rec_name='/models/trt-engines/genderage_v1/genderage_v1.plan'):
        self.rec_model = TrtModel(rec_name)

    # warmup
    def prepare(self, ctx=0):
        print("Warming up TensorRT engine...")
        self.rec_model.build()
        self.rec_model.run(np.zeros(self.rec_model.input_shapes[0], np.float32))
        print(f"Engine warmup complete!\nExpecting input shapes: {self.rec_model.input_shapes}")

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


class RetinaInfer:

    def __init__(self, rec_name='/models/trt-engines/retinaface_mnet025_v2/retinaface_mnet025_v2.plan',
                 output_order=None):
        self.rec_model = TrtModel(rec_name)
        retina_outputs = ['face_rpn_cls_prob_reshape_stride32',
                          'face_rpn_bbox_pred_stride32',
                          'face_rpn_landmark_pred_stride32',
                          'face_rpn_cls_prob_reshape_stride16',
                          'face_rpn_bbox_pred_stride16',
                          'face_rpn_landmark_pred_stride16',
                          'face_rpn_cls_prob_reshape_stride8',
                          'face_rpn_bbox_pred_stride8',
                          'face_rpn_landmark_pred_stride8']

        if not output_order:
            output_order = retina_outputs

        self.output_order = output_order

    # warmup
    def prepare(self, ctx=0):
        print("Warming up TensorRT engine...")
        self.rec_model.build()
        self.rec_model.run(np.zeros(self.rec_model.input_shapes[0], np.float32))
        print(f"Engine warmup complete!\nExpecting input shapes: {self.rec_model.input_shapes}")

    def run(self, input):
        net_out = self.rec_model.run(input, deflatten=True, as_dict=True)
        net_out = [net_out[e] for e in self.output_order]

        return net_out


class CenterFaceInfer:

    def __init__(self):
        raise NotImplemented
