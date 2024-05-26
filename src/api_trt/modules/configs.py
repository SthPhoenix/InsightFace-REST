import json
import os

from api_trt.logger import logger


class Configs(object):
    def __init__(self, models_dir: str = '/models'):
        self.models_dir = self.__get_param('MODELS_DIR', models_dir)
        self.onnx_models_dir = os.path.join(self.models_dir, 'onnx')
        self.trt_engines_dir = os.path.join(self.models_dir, 'trt-engines')
        self.models = self.__read_models_file()
        self.type2path = dict(
            onnx=self.onnx_models_dir,
            engine=self.trt_engines_dir,
            plan=self.trt_engines_dir
        )

    def __read_models_file(self):
        models_default_path = os.path.join(self.models_dir, 'models.json')
        models_override_path = os.path.join(self.models_dir, 'models.override.json')
        models_conf = models_default_path
        if os.path.exists(models_override_path):
            models_conf = models_override_path
            logger.warning(f"Found '{models_override_path}', using this instead of default.")
        try:
            models = json.load(open(models_conf, mode='r'))
            return models
        except FileNotFoundError as e:
            e.strerror = f"The file `{models_conf}` doesn't exist"
            raise e
        except Exception as e:
            raise e

    def __get_param(self, ENV, default=None):
        return os.environ.get(ENV, default)

    def build_model_paths(self, model_name: str, ext: str):
        base = self.type2path[ext]
        parent = os.path.join(base, model_name)
        file = os.path.join(parent, f"{model_name}.{ext}")
        return parent, file

    def get_outputs_order(self, model_name):
        return self.models.get(model_name, {}).get('outputs')

    def get_shape(self, model_name):
        return self.models.get(model_name, {}).get('shape')

    def get_dl_link(self, model_name):
        return self.models.get(model_name, {}).get('link')

    def get_dl_type(self, model_name):
        return self.models.get(model_name, {}).get('dl_type')


config = Configs()
