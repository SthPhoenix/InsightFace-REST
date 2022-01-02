import os
import logging

from modules.utils.helpers import parse_size, tobool, validate_max_size
from modules.model_zoo.getter import prepare_backend
from modules.configs import Configs
from env_parser import EnvConfigs

log_level = os.getenv('LOG_LEVEL', 'INFO')

logging.basicConfig(
    level=log_level,
    format='%(asctime)s %(levelname)s - %(message)s',
    datefmt='[%H:%M:%S]',
)


def prepare_models(root_dir: str = '/models'):
    model_configs = Configs(models_dir=root_dir)
    env_configs = EnvConfigs()
    rec_name = env_configs.models.rec_name
    det_name = env_configs.models.det_name
    ga_name = env_configs.models.ga_name
    mask_detector = env_configs.models.mask_detector

    max_size = env_configs.defaults.max_size

    if max_size is None:
        max_size = [640, 640]

    max_size = validate_max_size(max_size)

    models = [model for model in [det_name, rec_name, ga_name, mask_detector] if model is not None]

    for model in models:
        batch_size = 1
        if model_configs.models[model].get('allow_batching'):
            if model == det_name:
                batch_size = env_configs.models.det_batch_size
            else:
                batch_size = env_configs.models.rec_batch_size
        logging.info(f"Preparing '{model}' model...")

        prepare_backend(model_name=model, backend_name=env_configs.models.backend_name, im_size=max_size,
                        force_fp16=env_configs.models.fp16,
                        max_batch_size=batch_size, config=model_configs)

        logging.info(f"'{model}' model ready!")


if __name__ == "__main__":
    prepare_models()
