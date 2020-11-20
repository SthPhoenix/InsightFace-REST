import os
import json
from json import JSONEncoder

from modules.utils.helpers import parse_size, tobool


class MyEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

class MTCNNConf:
    def __init__(self):
        # MTCNN parameters
        self.select_largest = tobool(os.getenv("SELECT_LARGEST", True))
        self.keep_all = tobool(os.getenv("KEEP_ALL", True))
        self.min_face_size = int(os.getenv("MIN_FACE_SIZE", 20))
        self.mtcnn_factor = float(os.getenv("MTCNN_FACTOR", 0.709))

class Defaults:
    def __init__(self):
        # Global parameters
        self.threshold = float(os.getenv('DET_THRESH', 0.6))
        self.max_size = parse_size(os.getenv('MAX_SIZE'))
        self.return_face_data = tobool(os.getenv('DEF_RETURN_FACE_DATA', False))
        self.extract_embedding = tobool(os.getenv('DEF_EXTRACT_EMBEDDING', True))
        self.extract_ga = tobool(os.getenv('DEF_EXTRACT_GA', False))
        self.api_ver = os.getenv('DEF_API_VER', "1")

class Models:
    def __init__(self):
        self.backend_name = os.getenv('INFERENCE_BACKEND', 'trt')
        self.device = os.getenv("DEVICE", 'cuda')
        self.rec_name = os.getenv("REC_NAME", "arcface_r100_v1")
        self.det_name = os.getenv("DET_NAME", "retinaface_mnet025_v2")
        self.ga_name = os.getenv("GA_NAME", "genderage_v1")
        self.fp16 = tobool(os.getenv('FORCE_FP16', False))
        self.ga_ignore = tobool(os.getenv('GA_IGNORE', False))
        self.rec_ignore = tobool(os.getenv('REC_IGNORE', False))

        if self.rec_ignore:
            self.rec_name = None
        if self.ga_ignore:
            self.ga_name = None




class EnvConfigs:
    def __init__(self):
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.port = os.getenv('PORT', 18080)

        self.models = Models()
        self.defaults = Defaults()
        self.mtcnn = MTCNNConf()


