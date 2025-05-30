from abc import ABC, abstractmethod


class AbstractDetector(ABC):
    @abstractmethod
    def __init__(self, inference_backend, landmarks=True):
        ...

    @abstractmethod
    def prepare(self, nms_treshold: float = 0.4, **kwargs):
        ...

    @abstractmethod
    def detect(self, imgs, threshold=0.5):
        ...
