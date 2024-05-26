import os
import logging

from api_trt.modules.utils.helpers import parse_size, tobool, validate_max_size
from api_trt.modules.model_zoo.getter import prepare_backend
from api_trt.modules.configs import config
from api_trt.settings import Settings
from api_trt.logger import logger
settings = Settings()

logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s %(levelname)s - %(message)s',
    datefmt='[%H:%M:%S]',
)





def prepare_models(root_dir: str = '/models'):
    """
    Bootstrap models before running API.

    Args:
      root_dir (str):  root_dir (str): The root directory for models. Defaults to '/models'.
    """
    model_configs = config

    rec_name = settings.models.rec_name
    det_name = settings.models.det_name
    ga_name = settings.models.ga_name
    mask_detector = settings.models.mask_detector

    max_size = settings.models.max_size

    max_size = validate_max_size(max_size)

    models = [model for model in [det_name, rec_name, ga_name, mask_detector] if model is not None]

    for model in models:
        batch_size = 1
        if model_configs.models[model].get('allow_batching'):
            if model == det_name:
                batch_size = settings.models.det_batch_size
            else:
                batch_size = settings.models.rec_batch_size
        logger.info(f"Preparing '{model}' model...")

        prepare_backend(model_name=model, backend_name=settings.models.inference_backend, im_size=max_size,
                        force_fp16=settings.models.force_fp16,
                        max_batch_size=batch_size, config=model_configs)

        logger.info(f"'{model}' model ready!")


if __name__ == "__main__":
    prepare_models()
