from .trt_loader import TrtModel
import cv2
import numpy as np

class Arcface:
    def __init__(self, rec_name='/models/trt-engines/arcface_r100_v1/arcface_r100_v1.plan'):
        self.rec_model = TrtModel(rec_name)
    #warmup
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