import os
import logging
from typing import List

import onnx

from insightface.model_zoo import get_model as get_model_orig

from .face_detectors import *
from .face_processors import *

from ..converters.insight2onnx import convert_insight_model
from ..converters.reshape_onnx import reshape, reshape_onnx_input
from ..converters.remove_initializer_from_input import remove_initializer_from_input
from ..utils.helpers import prepare_folders
from ..utils.download import download
from ..utils.model_store import get_model_file

from ..configs import Configs

from .exec_backends import onnxrt_backend as onnx_backend

# Since TensorRT and PyCUDA are optional dependencies it might be not available
try:
    from .exec_backends import trt_backend
    from ..converters.onnx_to_trt import convert_onnx
except:
    trt_backend = None
    convert_onnx = None


models = {
    'arcface_r100_v1': arcface_r100_v1,
    'genderage_v1': genderage_v1,
    'retinaface_r50_v1': retinaface_r50_v1,
    'retinaface_mnet025_v1': retinaface_mnet025_v1,
    'retinaface_mnet025_v2': retinaface_mnet025_v2,
    'centerface': centerface,
}



def prepare_backend(model_name, backend_name, im_size: List[int] = None, config: Configs = None,  force_fp16: bool = False):
    if im_size is None:
        im_size = [640, 480]

    prepare_folders([config.mxnet_models_dir, config.onnx_models_dir, config.trt_engines_dir])

    in_package = config.in_official_package(model_name)
    reshape_allowed = config.mxnet_models[model_name].get('reshape')
    shape = config.get_shape(model_name)
    if reshape_allowed is True:
        shape = (1, 3) + tuple(im_size)[::-1]

    mxnet_symbol, mxnet_params = config.get_mxnet_model_paths(model_name)
    onnx_dir, onnx_path = config.build_model_paths(model_name, 'onnx')
    trt_dir, trt_path = config.build_model_paths(model_name, 'plan')

    if not os.path.exists(onnx_path):
        prepare_folders([onnx_dir])
        if in_package:
            print(f"Downloading model: {model_name}...")
            get_model_file(model_name, root=config.mxnet_models_dir)
            convert_insight_model(mxnet_symbol, mxnet_params, onnx_path, shape)
        else:
            download(config.get_dl_link(model_name), onnx_path)
            remove_initializer_from_input(onnx_path,onnx_path)


    if backend_name == 'onnx':
        model = onnx.load(onnx_path)
        if reshape_allowed is True:
            logging.info(f'Reshaping ONNX inputs to: {shape}')
            model = reshape(model, h=im_size[1], w=im_size[0])
        return model.SerializeToString()

    if backend_name == "trt":
        if reshape_allowed is True:
            trt_path = trt_path.replace('.plan', f'_{im_size[0]}_{im_size[1]}.plan')
        if not os.path.exists(trt_path):
            prepare_folders([trt_dir])
            temp_onnx_model = onnx_path + '.temp'
            if reshape_allowed is True:
                logging.info(f'Reshaping ONNX inputs to: {shape}')
                reshape_onnx_input(onnx_path, temp_onnx_model, im_size=im_size)
            else:
                temp_onnx_model = onnx_path

            logging.info(f"Building TRT engine for {model_name}...")
            convert_onnx(temp_onnx_model, engine_file_path=trt_path, force_fp16=force_fp16)
            os.remove(temp_onnx_model)
            logging.info('Building TRT engine complete!')
        return trt_path


def get_model(model_name: str, backend_name: str, im_size: List[int] = None, root_dir: str = "/models", **kwargs):

    if im_size is None:
        im_size = [640, 480]

    config = Configs(models_dir=root_dir)

    backends = {
        'onnx': onnx_backend,
        'trt': trt_backend,
        'mxnet': 'mxnet'
    }
    back2ext = {
        'onnx': 'onnx',
        'trt': 'plan',
    }

    if backend_name not in backends:
        logging.error(f"Unknown backend '{backend_name}' specified. Exiting.")
        exit(1)

    if model_name not in models:
        logging.error(f"Unknown model {model_name} specified."
                      f" Please select one of the following:\n"
                      f"{', '.join(list(models.keys()))}")
        exit(1)

    # Keep original InsightFace package available for a while for testing purposes.
    if backend_name == 'mxnet':
        return get_model_orig(model_name, root=config.mxnet_models_dir)

    backend = backends[backend_name]

    model_path = prepare_backend(model_name, backend_name, im_size=im_size, config=config)

    outputs = config.get_outputs_order(model_name)
    model = models[model_name](model_path=model_path, backend=backend, outputs=outputs)
    return model
