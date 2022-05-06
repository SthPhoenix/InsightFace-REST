from .detectors.retinaface import RetinaFace
from .detectors.centerface import CenterFace
from .detectors.dbface import DBFace
from .detectors.scrfd import SCRFD
from .detectors.yolov5_face import YoloV5


def get_retinaface(model_path, backend, outputs, rac, masks=False, **kwargs):
    inference_backend = backend.DetectorInfer(model=model_path, output_order=outputs, **kwargs)
    model = RetinaFace(inference_backend=inference_backend, rac=rac, masks=masks)
    return model


def retinaface_r50_v1(model_path, backend, outputs, **kwargs):
    model = get_retinaface(model_path, backend, outputs, rac="net3", **kwargs)
    return model


def retinaface_mnet025_v1(model_path, backend, outputs, **kwargs):
    model = get_retinaface(model_path, backend, outputs, rac="net3", **kwargs)
    return model


def retinaface_mnet025_v2(model_path, backend, outputs, **kwargs):
    model = get_retinaface(model_path, backend, outputs, rac="net3l", **kwargs)
    return model


def mnet_cov2(model_path, backend, outputs, **kwargs):
    model = get_retinaface(model_path, backend, outputs, rac="net3l", masks=True, **kwargs)
    return model


def centerface(model_path, backend, outputs, **kwargs):
    inference_backend = backend.DetectorInfer(model=model_path, output_order=outputs, **kwargs)
    model = CenterFace(inference_backend=inference_backend)
    return model


def dbface(model_path, backend, outputs, **kwargs):
    inference_backend = backend.DetectorInfer(model=model_path, output_order=outputs, **kwargs)
    model = DBFace(inference_backend=inference_backend)
    return model


def scrfd(model_path, backend, outputs, **kwargs):
    inference_backend = backend.DetectorInfer(model=model_path, output_order=outputs, **kwargs)
    model = SCRFD(inference_backend=inference_backend)
    return model

def scrfd_v2(model_path, backend, outputs, **kwargs):
    inference_backend = backend.DetectorInfer(model=model_path, output_order=outputs, **kwargs)
    model = SCRFD(inference_backend=inference_backend, ver=2)
    return model

def yolov5_face(model_path, backend, outputs, **kwargs):
    inference_backend = backend.DetectorInfer(model=model_path, output_order=outputs, **kwargs)
    model = YoloV5(inference_backend=inference_backend)
    return model