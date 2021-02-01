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
from .exec_backends import triton_backend as triton_backend

# Since TensorRT and PyCUDA are optional dependencies it might be not available
try:
    from .exec_backends import trt_backend
    from ..converters.onnx_to_trt import convert_onnx
except:
    trt_backend = None
    convert_onnx = None

# Map model names to corresponding functions
models = {
    'arcface_r100_v1': arcface_r100_v1,
    'r50-arcface-msfdrop75': r50_arcface_msfdrop75,
    'r100-arcface-msfdrop75': r100_arcface_msfdrop75,
    'glint360k_r100FC_1.0': glint360k_r100FC_1_0,
    'glint360k_r100FC_0.1': glint360k_r100FC_0_1,
    'genderage_v1': genderage_v1,
    'retinaface_r50_v1': retinaface_r50_v1,
    'retinaface_mnet025_v1': retinaface_mnet025_v1,
    'retinaface_mnet025_v2': retinaface_mnet025_v2,
    'mnet_cov2': mnet_cov2,
    'centerface': centerface,
    'dbface': dbface,
}


def prepare_backend(model_name, backend_name, im_size: List[int] = None,
                    max_batch_size: int = 1,
                    force_fp16: bool = False,
                    download_model: bool = True,
                    config: Configs = None):
    """
    Check if ONNX, MXNet and TensorRT models exist and download/create them otherwise.

    :param model_name: Name of required model. Must be one of keys in `models` dict.
    :param backend_name: Name of inference backend. (onnx, trt)
    :param im_size: Desired maximum size of image in W,H form. Will be overridden if model doesn't support reshaping.
    :param max_batch_size: Maximum batch size for inference, currently supported for ArcFace model only.
    :param force_fp16: Force use of FP16 precision, even if device doesn't support it. Be careful. TensorRT specific.
    :param download_model: Download MXNet or ONNX model if it not exist.
    :param config:  Configs class instance
    :return: ONNX model serialized to string, or path to TensorRT engine
    """

    prepare_folders([config.mxnet_models_dir, config.onnx_models_dir, config.trt_engines_dir])

    in_package = config.in_official_package(model_name)
    reshape_allowed = config.mxnet_models[model_name].get('reshape')
    shape = config.get_shape(model_name)
    if reshape_allowed is True and im_size is not None:
        shape = (1, 3) + tuple(im_size)[::-1]

    mxnet_symbol, mxnet_params = config.get_mxnet_model_paths(model_name)
    onnx_dir, onnx_path = config.build_model_paths(model_name, 'onnx')
    trt_dir, trt_path = config.build_model_paths(model_name, 'plan')

    if not os.path.exists(onnx_path) and download_model is True:
        prepare_folders([onnx_dir])
        if in_package:
            print(f"Downloading model: {model_name}...")
            get_model_file(model_name, root=config.mxnet_models_dir)
            convert_insight_model(mxnet_symbol, mxnet_params, onnx_path, shape)
        else:
            dl_link = config.get_dl_link(model_name)
            if dl_link:
                download(config.get_dl_link(model_name), onnx_path)
                remove_initializer_from_input(onnx_path, onnx_path)
            elif os.path.exists(mxnet_symbol) and os.path.exists(mxnet_params):
                convert_insight_model(mxnet_symbol, mxnet_params, onnx_path, shape)
            else:
                logging.error("You have requested non standard model, but haven't provided download link or "
                              "MXNet model. Place model to proper folder and change configs.py accordingly.")

    if backend_name == 'onnx':
        model = onnx.load(onnx_path)
        if reshape_allowed is True:
            logging.info(f'Reshaping ONNX inputs to: {shape}')
            model = reshape(model, h=im_size[1], w=im_size[0])
        return model.SerializeToString()

    if backend_name == "trt":
        if reshape_allowed is True:
            trt_path = trt_path.replace('.plan', f'_{shape[3]}_{shape[2]}.plan')
        if max_batch_size > 1:
            trt_path = trt_path.replace('.plan', f'_batch{max_batch_size}.plan')
        if force_fp16 is True:
            trt_path = trt_path.replace('.plan', '_fp16.plan')

        if not os.path.exists(trt_path):
            prepare_folders([trt_dir])

            if reshape_allowed is True or max_batch_size!=1:
                logging.info(f'Reshaping ONNX inputs to: {shape}')
                model = onnx.load(onnx_path)
                onnx_batch_size = 1
                if max_batch_size != 1:
                    onnx_batch_size = -1
                reshaped = reshape(model, n=onnx_batch_size, h=shape[2], w=shape[3])
                temp_onnx_model = reshaped.SerializeToString()

            else:
                temp_onnx_model = onnx_path

            logging.info(f"Building TRT engine for {model_name}...")
            convert_onnx(temp_onnx_model,
                         engine_file_path=trt_path,
                         max_batch_size=max_batch_size,
                         force_fp16=force_fp16)
            logging.info('Building TRT engine complete!')
        return trt_path


def get_model(model_name: str, backend_name: str, im_size: List[int] = None, max_batch_size: int = 1, force_fp16: bool = False,
              root_dir: str = "/models", download_model: bool = True, **kwargs):
    """
    Returns inference backend instance with loaded model.

    :param model_name: Name of required model. Must be one of keys in `models` dict.
    :param backend_name: Name of inference backend. (onnx, mxnet, trt)
    :param im_size: Desired maximum size of image in W,H form. Will be overridden if model doesn't support reshaping.
    :param max_batch_size: Maximum batch size for inference, currently supported for ArcFace model only.
    :param force_fp16: Force use of FP16 precision, even if device doesn't support it. Be careful. TensorRT specific.
    :param root_dir: Root directory where models will be stored.
    :param download_model: Download MXNet or ONNX model. Might be disabled if TRT model was already created.
    :param kwargs: Placeholder.
    :return: Inference backend with loaded model.
    """

    config = Configs(models_dir=root_dir)

    backends = {
        'onnx': onnx_backend,
        'trt': trt_backend,
        'mxnet': 'mxnet',
        'triton': triton_backend
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

    model_path = prepare_backend(model_name, backend_name, im_size=im_size, max_batch_size=max_batch_size, config=config, force_fp16=force_fp16,
                                 download_model=download_model)

    outputs = config.get_outputs_order(model_name)
    model = models[model_name](model_path=model_path, backend=backend, outputs=outputs)
    return model
