from abc import ABC, abstractmethod


class AbstractArcFace(ABC):
    @abstractmethod
    def __init__(self, rec_name='/models/onnx/arcface_r100_v1/arcface_r100_v1.onnx',
                 input_mean: float = 0.,
                 input_std: float = 1.,
                 swapRB=True,
                 **kwargs
                 ):
        ...

    @abstractmethod
    def prepare(self, **kwargs):
        ...

    @abstractmethod
    def get_embedding(self, face_img):
        ...


class AbstractFaceGenderAge(ABC):
    @abstractmethod
    def __init__(self, rec_name='/models/onnx/genderage_v1/genderage_v1.onnx', outputs=None, **kwargs):
        ...

    # warmup
    @abstractmethod
    def prepare(self, **kwargs):
        ...

    @abstractmethod
    def get(self, face_img):
        ...


class AbstractMaskDetection:
    @abstractmethod
    def __init__(self, rec_name='/models/onnx/genderage_v1/genderage_v1.onnx', outputs=None, **kwargs):
        ...

    # warmup
    @abstractmethod
    def prepare(self, **kwargs):
        ...

    @abstractmethod
    def get(self, face_img):
        ...


class AbstractDetectorInfer:
    @abstractmethod
    def __init__(self, model='/models/onnx/centerface/centerface.onnx',
                 output_order=None, **kwargs):
        self.rec_model = None
        self.model_name = None
        self.stream = None
        self.input_ptr = None
        self.input_shape = None
        self.output_order = output_order

    # warmup
    @abstractmethod
    def prepare(self, **kwargs):
        ...

    @abstractmethod
    def run(self, input):
        ...
