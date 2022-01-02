import os
import json
from json import JSONEncoder

from modules.utils.helpers import parse_size, tobool, toNone


class MyEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


class Defaults:
    def __init__(self):
        # Global parameters
        self.threshold = float(os.getenv('DET_THRESH', 0.6))
        self.max_size = parse_size(os.getenv('MAX_SIZE'))
        self.return_face_data = tobool(os.getenv('DEF_RETURN_FACE_DATA', False))
        self.return_landmarks = tobool(os.getenv('DEF_RETURN_LANDMARKS', False))
        self.extract_embedding = tobool(os.getenv('DEF_EXTRACT_EMBEDDING', True))
        self.extract_ga = tobool(os.getenv('DEF_EXTRACT_GA', False))
        self.detect_masks = tobool(os.getenv('DEF_DETECT_MASKS', False))
        self.api_ver = os.getenv('DEF_API_VER', "1")


class Models:
    def __init__(self):
        self.backend_name = os.getenv('INFERENCE_BACKEND', 'onnx')
        self.device = os.getenv("DEVICE", 'cuda')
        self.det_name = os.getenv("DET_NAME", "scrfd_10g_gnkps")
        self.rec_name = toNone(os.getenv("REC_NAME", "glintr100"))
        self.ga_name = toNone(os.getenv("GA_NAME", None))
        self.mask_detector = toNone(os.getenv("MASK_DETECTOR", None))
        self.rec_batch_size = int(os.getenv('REC_BATCH_SIZE', 1))
        self.det_batch_size = int(os.getenv('DET_BATCH_SIZE', 1))
        self.fp16 = tobool(os.getenv('FORCE_FP16', False))
        self.triton_uri = os.getenv("TRITON_URI", None)


class EnvConfigs:
    def __init__(self):
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.port = os.getenv('PORT', 18080)

        self.models = Models()
        self.defaults = Defaults()
