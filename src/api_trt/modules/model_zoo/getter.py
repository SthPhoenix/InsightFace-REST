import os
import logging
import json
from typing import List

import onnx

from .face_detectors import *
from .face_processors import *

# from ..converters.insight2onnx import convert_insight_model
from ..converters.reshape_onnx import reshape, reshape_onnx_input
from ..converters.remove_initializer_from_input import remove_initializer_from_input
from ..utils.helpers import prepare_folders
from ..utils.download import download
from ..utils.download_google import download_from_gdrive, check_hash

from ..configs import Configs

from .exec_backends import onnxrt_backend as onnx_backend

# Since TensorRT, TritonClient and PyCUDA are optional dependencies it might be not available
try:
    from .exec_backends import trt_backend
    from .exec_backends import triton_backend as triton_backend
    from ..converters.onnx_to_trt import convert_onnx
except Exception as e:
    print(e)
    trt_backend = None
    triton_backend = None
    convert_onnx = None

# Map function names to corresponding functions
func_map = {
    'genderage_v1': genderage_v1,
    'retinaface_r50_v1': retinaface_r50_v1,
    'retinaface_mnet025_v1': retinaface_mnet025_v1,
    'retinaface_mnet025_v2': retinaface_mnet025_v2,
    'mnet_cov2': mnet_cov2,
    'centerface': centerface,
    'dbface': dbface,
    'scrfd': scrfd,
    'scrfd_v2': scrfd_v2,
    'arcface_mxnet': arcface_mxnet,
    'arcface_torch': arcface_torch,
    'mask_detector': mask_detector,
    'yolov5_face': yolov5_face
}


def sniff_output_order(model_path, save_dir):
    model = onnx.load(model_path)
    output = [node.name for node in model.graph.output]
    with open(os.path.join(save_dir, 'output_order.json'), mode='w') as fl:
        fl.write(json.dumps(output))
    return output


def read_outputs_order(trt_dir):
    outputs = None
    outputs_file = os.path.join(trt_dir, 'output_order.json')
    if os.path.exists(outputs_file):
        with open(outputs_file, mode='r') as fl:
            outputs = json.loads(fl.read())
    return outputs


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

    prepare_folders([config.onnx_models_dir, config.trt_engines_dir])
    reshape_allowed = config.models[model_name].get('reshape')
    shape = config.get_shape(model_name)

    if reshape_allowed is True and im_size is not None:
        shape = (1, 3) + tuple(im_size)[::-1]

    onnx_dir, onnx_path = config.build_model_paths(model_name, 'onnx')
    trt_dir, trt_path = config.build_model_paths(model_name, 'plan')

    onnx_exists = os.path.exists(onnx_path)
    onnx_hash = config.models[model_name].get('md5')
    trt_rebuild_required = False
    if onnx_exists and onnx_hash:
        hashes_match = check_hash(onnx_path, onnx_hash, algo='md5')
        if not hashes_match:
            logging.warning('ONNX model hash mismatch, trying to download it again. '
                            'This behaviour is expected in current version for scrfd_*_gnkps models '
                            'due to models update.')
            onnx_exists = False
            trt_rebuild_required = True

    if not onnx_exists and download_model is True:
        prepare_folders([onnx_dir])
        dl_link = config.get_dl_link(model_name)
        dl_type = config.get_dl_type(model_name)
        if dl_link:
            if dl_type == 'google':
                download_from_gdrive(dl_link, onnx_path)
            else:
                download(dl_link, onnx_path)
            remove_initializer_from_input(onnx_path, onnx_path)
        else:
            logging.error("You have requested non standard model, but haven't provided download link or "
                          "ONNX model. Place model to proper folder and change configs.py accordingly.")
    if backend_name == 'triton':
        return model_name

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

        prepare_folders([trt_dir])

        if not config.get_outputs_order(model_name):
            logging.debug('No output order provided, trying to read it from ONNX model.')
            sniff_output_order(onnx_path, trt_dir)

        if not os.path.exists(trt_path) or trt_rebuild_required:

            if reshape_allowed is True or max_batch_size != 1:
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


def get_model(model_name: str, backend_name: str, im_size: List[int] = None, max_batch_size: int = 1,
              force_fp16: bool = False,
              root_dir: str = "/models", download_model: bool = True, triton_uri=None, **kwargs):
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

    if model_name not in config.models.keys():
        logging.error(f"Unknown model {model_name} specified."
                      f" Please select one of the following:\n"
                      f"{', '.join(list(config.models.keys()))}")
        exit(1)

    backend = backends[backend_name]

    model_path = prepare_backend(model_name, backend_name, im_size=im_size, max_batch_size=max_batch_size,
                                 config=config, force_fp16=force_fp16,
                                 download_model=download_model)

    outputs = config.get_outputs_order(model_name)
    if not outputs and backend_name == 'trt':
        logging.debug(f'No output order provided, for "{model_name}" trying to read it from "output_order.json"')
        trt_dir, trt_path = config.build_model_paths(model_name, 'plan')
        outputs = read_outputs_order(trt_dir)

    func = func_map[config.models[model_name].get('function')]
    model = func(model_path=model_path, backend=backend, outputs=outputs, triton_uri=triton_uri)
    return model
