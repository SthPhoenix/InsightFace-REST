from .detectors.retinaface import RetinaFace
from .detectors.centerface import CenterFace


def arcface_r100_v1(model_path, backend, outputs):
    model = backend.Arcface(rec_name=model_path)
    return model


def r50_arcface_msfdrop75(model_path, backend, outputs):
    model = backend.Arcface(rec_name=model_path)
    return model


def r100_arcface_msfdrop75(model_path, backend, outputs):
    model = backend.Arcface(rec_name=model_path)
    return model


def glint360k_r100FC_1_0(model_path, backend, outputs):
    model = backend.Arcface(rec_name=model_path)
    return model


def glint360k_r100FC_0_1(model_path, backend, outputs):
    model = backend.Arcface(rec_name=model_path)
    return model


def genderage_v1(model_path, backend, outputs):
    model = backend.FaceGenderage(rec_name=model_path)
    return model
