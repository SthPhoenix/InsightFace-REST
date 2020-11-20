import os
import logging

from modules.utils.helpers import parse_size, tobool
from modules.model_zoo.getter import prepare_backend
from modules.configs import Configs

log_level = os.getenv('LOG_LEVEL', 'INFO')

logging.basicConfig(
    level=log_level,
    format='%(asctime)s %(levelname)s - %(message)s',
    datefmt='[%H:%M:%S]',
)


def prepare_models(root_dir: str = '/models'):
    backend_name = os.getenv('INFERENCE_BACKEND', 'trt')
    rec_name = os.getenv("REC_NAME", "arcface_r100_v1")
    det_name = os.getenv("DET_NAME", "retinaface_mnet025_v2")
    ga_name = os.getenv("GA_NAME", "genderage_v1")

    force_fp16 = tobool(os.getenv('FORCE_FP16', False))

    max_size = parse_size(os.getenv('MAX_SIZE'))

    if max_size is None:
        max_size = [640, 480]

    config = Configs(models_dir=root_dir)

    for model in [rec_name, det_name, ga_name]:
        logging.info(f"Preparing '{model}' model...")
        prepare_backend(model, backend_name, max_size, force_fp16, config=config)


if __name__ == "__main__":
    prepare_models()
