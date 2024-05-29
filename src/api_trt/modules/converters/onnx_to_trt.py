import sys
from typing import Union

import tensorrt as trt

from api_trt.logger import logger

# Based on code from NVES_R's response at
# https://forums.developer.nvidia.com/t/segmentation-fault-when-creating-the-trt-builder-in-python-works-fine-with-trtexec/111376


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def _build_engine_onnx(input_onnx: Union[str, bytes], force_fp16: bool = False, max_batch_size: int = 1,
                       max_workspace: int = 1024):
    """
    Builds TensorRT engine from provided ONNX file

    Args:
        input_onnx (Union[str, bytes]): serialized ONNX model.
        force_fp16 (bool): Force use of FP16 precision, even if device doesn't support it. Be careful.
        max_batch_size (int): Define maximum batch size supported by engine. If >1 creates optimization profile.
        max_workspace (int): Maximum builder workspace in MB.

    Returns:
        TensorRT engine
    """

    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            builder.create_builder_config() as config, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        has_fp16 = builder.platform_has_fast_fp16
        if force_fp16 or has_fp16:
            logger.info('Building TensorRT engine with FP16 support.')
            if not has_fp16:
                logger.warning('Builder reports no fast FP16 support. Performance drop expected.')
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            logger.warning('Building engine in FP32 mode.')

        trt10 = not hasattr(config, 'max_workspace_size')

        if trt10:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace * 1024 * 1024)
        else:
            config.max_workspace_size = max_workspace * 1024 * 1024
        if not parser.parse(input_onnx):
            print('ERROR: Failed to parse the ONNX')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            sys.exit(1)

        if max_batch_size != 1:
            logger.warning('Batch size !=1 is used. Ensure your inference code supports it.')
        profile = builder.create_optimization_profile()
        # Get input name and shape for building optimization profile
        input = network.get_input(0)
        inp_shape = list(input.shape)
        inp_shape[0] = 1
        min_opt_shape = tuple(inp_shape)
        inp_shape[0] = max_batch_size
        max_shape = tuple(inp_shape)
        input_name = input.name
        profile.set_shape(input_name, min_opt_shape, min_opt_shape, max_shape)
        config.add_optimization_profile(profile)
        if trt10:
            return builder.build_serialized_network(network, config), trt10
        else:
            return builder.build_engine(network, config=config), trt10


def check_fp16():
    """
    Check if the device supports FP16 precision.

    Returns:
        bool: True if device supports FP16 precision, False otherwise.
    """
    builder = trt.Builder(TRT_LOGGER)
    has_fp16 = builder.platform_has_fast_fp16
    return has_fp16


def convert_onnx(input_onnx: Union[str, bytes], engine_file_path: str, force_fp16: bool = False,
                 max_batch_size: int = 1):
    """
    Creates TensorRT engine and serializes it to disk

    Args:
        input_onnx (Union[str, bytes]): Path to ONNX file on disk or serialized ONNX model.
        engine_file_path (str): Path where TensorRT engine should be saved.
        force_fp16 (bool): Force use of FP16 precision, even if device doesn't support it. Be careful.
        max_batch_size (int): Define maximum batch size supported by engine. If >1 creates optimization profile.

    Returns:
        None
    """

    onnx_obj = None
    if isinstance(input_onnx, str):
        with open(input_onnx, "rb") as f:
            onnx_obj = f.read()
    elif isinstance(input_onnx, bytes):
        onnx_obj = input_onnx

    engine, trt10 = _build_engine_onnx(input_onnx=onnx_obj,
                                force_fp16=force_fp16, max_batch_size=max_batch_size)

    assert not isinstance(engine, type(None))

    with open(engine_file_path, "wb") as f:
        if trt10:
            f.write(engine)
        else:
            f.write(engine.serialize())
