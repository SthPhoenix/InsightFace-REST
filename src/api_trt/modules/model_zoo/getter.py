import json
import os
from typing import List
import logging
import onnx
from tenacity import retry, wait_exponential, stop_after_attempt, before_sleep_log, retry_if_not_exception_type
from api_trt.logger import logger
from api_trt.modules.configs import Configs, config
from api_trt.modules.converters.remove_initializer_from_input import remove_initializer_from_input
from api_trt.modules.converters.reshape_onnx import reshape
from api_trt.modules.model_zoo.exec_backends import onnxrt_backend as onnx_backend
from api_trt.modules.model_zoo.face_detectors import *
from api_trt.modules.model_zoo.face_processors import *
from api_trt.modules.utils.download import download
from api_trt.modules.utils.download_google import download_from_gdrive, check_hash
from api_trt.modules.utils.helpers import prepare_folders

# Since TensorRT, TritonClient and PyCUDA are optional dependencies it might be not available
try:
    from api_trt.modules.model_zoo.exec_backends import trt_backend
    from api_trt.modules.converters.onnx_to_trt import convert_onnx, check_fp16
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
    'adaface': adaface,
    'mask_detector': mask_detector,
    'yolov5_face': yolov5_face
}


def sniff_output_order(model_path, save_dir):
    """
    Sniffs the output order of a model.

    Args:
        model_path (str): The path to the ONNX model.
        save_dir (str): The directory where the output order will be saved.

    Returns:
        List[str]: A list of output names in the correct order.
    """
    outputs_file = os.path.join(save_dir, 'output_order.json')
    if not os.path.exists(outputs_file):
        model = onnx.load(model_path)
        output = [node.name for node in model.graph.output]
        with open(outputs_file, mode='w') as fl:
            fl.write(json.dumps(output))
    else:
        output = read_outputs_order(save_dir)
    return output


def read_outputs_order(trt_dir):
    """
    Reads the output order from a TRT directory.

    Args:
       trt_dir (str): The directory containing the TRT model.

    Returns:
       List[str]: A list of output names in the correct order.
    """
    outputs = None
    outputs_file = os.path.join(trt_dir, 'output_order.json')
    if os.path.exists(outputs_file):
        with open(outputs_file, mode='r') as fl:
            outputs = json.loads(fl.read())
    return outputs


@retry(wait=wait_exponential(min=0.5, max=5), stop=stop_after_attempt(5), reraise=True,
       before_sleep=before_sleep_log(logger, logging.WARNING),
       retry=retry_if_not_exception_type(ValueError))
def download_onnx(src, dst, dl_type='google', md5=None):
    if src not in [None, '']:
        if dl_type == 'google':
            download_from_gdrive(src, dst)
        else:
            download(src, dst)
        hashes_match = check_hash(dst, md5, algo='md5')
        if hashes_match:
            return dst
        else:
            logger.error(f"ONNX model hash mismatch after download")
            raise AssertionError
    else:
        logger.error(f"No download link provided for {dst}")
        raise ValueError


def prepare_backend(model_name, backend_name, im_size: List[int] = None,
                    max_batch_size: int = 1,
                    force_fp16: bool = False,
                    download_model: bool = True,
                    config: Configs = config):
    """
    Prepares the backend for a model.

    Args:
        model_name (str): The name of the model.
        backend_name (str): The name of the backend (onnx, trt).
        im_size (List[int]): The desired maximum size of the image in W, H form.
        max_batch_size (int): The maximum batch size for inference.
        force_fp16 (bool): Whether to force use of FP16 precision.
        download_model (bool): Whether to download the model if it doesn't exist.
        config (Configs): The configuration object.

    Returns:
        str: The path to the prepared backend model.
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
        logger.info(f"Checking model hash...")
        hashes_match = check_hash(onnx_path, onnx_hash, algo='md5')
        if not hashes_match:
            logger.warning('ONNX model hash mismatch, trying to download it again. ')
            onnx_exists = False
            trt_rebuild_required = True

    if not onnx_exists and download_model is True:
        prepare_folders([onnx_dir])
        dl_link = config.get_dl_link(model_name)
        dl_type = config.get_dl_type(model_name)
        if dl_link:
            try:
                download_onnx(dl_link, onnx_path, dl_type, onnx_hash)
            except AssertionError:
                logger.error(
                    f"Model download failed after multiple attempts. "
                    f"Try manually downloading model and placing it to `{onnx_path}`")
                exit(1)
            except Exception as e:
                logger.error(e)
                exit(1)
            remove_initializer_from_input(onnx_path, onnx_path)
        else:
            logger.error("You have requested non standard model, but haven't provided download link or "
                         "ONNX model. Place model to proper folder and change configs.py accordingly.")
            exit(1)

    if backend_name == 'triton':
        return model_name

    if backend_name == 'onnx':
        model = onnx.load(onnx_path)
        if reshape_allowed is True:
            logger.info(f'Reshaping ONNX inputs to: {shape}')
            model = reshape(model, h=im_size[1], w=im_size[0])
        return model.SerializeToString()

    if backend_name == "trt":
        has_fp16 = check_fp16()

        if reshape_allowed is True:
            trt_path = trt_path.replace('.plan', f'_{shape[3]}_{shape[2]}.plan')
        if max_batch_size > 1:
            trt_path = trt_path.replace('.plan', f'_batch{max_batch_size}.plan')
        if force_fp16 or has_fp16:
            trt_path = trt_path.replace('.plan', '_fp16.plan')

        prepare_folders([trt_dir])

        if not config.get_outputs_order(model_name):
            logger.debug('No output order provided, trying to read it from ONNX model.')
            sniff_output_order(onnx_path, trt_dir)

        if not os.path.exists(trt_path) or trt_rebuild_required:

            if reshape_allowed is True or max_batch_size != 1:
                logger.info(f'Reshaping ONNX inputs to: {shape}')
                model = onnx.load(onnx_path)
                onnx_batch_size = 1
                if max_batch_size != 1:
                    onnx_batch_size = -1
                reshaped = reshape(model, n=onnx_batch_size, h=shape[2], w=shape[3])
                temp_onnx_model = reshaped.SerializeToString()

            else:
                temp_onnx_model = onnx_path

            logger.info(f"Building TRT engine for {model_name}...")
            convert_onnx(temp_onnx_model,
                         engine_file_path=trt_path,
                         max_batch_size=max_batch_size,
                         force_fp16=force_fp16)
            logger.info('Building TRT engine complete!')
        return trt_path


def get_model(model_name: str, backend_name: str, im_size: List[int] = None, max_batch_size: int = 1,
              force_fp16: bool = False,
              root_dir: str = "/models", download_model: bool = True, triton_uri=None, **kwargs):
    """
    Returns an inference backend instance with a loaded model.

    Args:
        model_name (str): The name of the model.
        backend_name (str): The name of the backend (onnx, mxnet, trt).
        im_size (List[int]): The desired maximum size of the image in W, H form.
        max_batch_size (int): The maximum batch size for inference.
        force_fp16 (bool): Whether to force use of FP16 precision.
        root_dir (str): The root directory where models will be stored.
        download_model (bool): Whether to download the model if it doesn't exist.
        triton_uri (str): The URI of the Triton server.

    Returns:
        object: An inference backend instance with a loaded model.
    """

    backends = {
        'onnx': onnx_backend,
        'trt': trt_backend,
        'mxnet': 'mxnet',
        # 'triton': triton_backend
    }

    if backend_name not in backends:
        logger.error(f"Unknown backend '{backend_name}' specified. Exiting.")
        exit(1)

    if model_name not in config.models.keys():
        logger.error(f"Unknown model {model_name} specified."
                     f" Please select one of the following:\n"
                     f"{', '.join(list(config.models.keys()))}")
        exit(1)

    backend = backends[backend_name]

    model_path = prepare_backend(model_name, backend_name, im_size=im_size, max_batch_size=max_batch_size,
                                 config=config, force_fp16=force_fp16,
                                 download_model=download_model)

    outputs = config.get_outputs_order(model_name)
    if not outputs and backend_name == 'trt':
        logger.debug(f'No output order provided, for "{model_name}" trying to read it from "output_order.json"')
        trt_dir, trt_path = config.build_model_paths(model_name, 'plan')
        outputs = read_outputs_order(trt_dir)

    func = func_map[config.models[model_name].get('function')]
    model = func(model_path=model_path, backend=backend, outputs=outputs, triton_uri=triton_uri)
    return model
