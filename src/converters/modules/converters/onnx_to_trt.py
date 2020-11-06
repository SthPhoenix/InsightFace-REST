import os
import tensorrt as trt
import sys
from typing import Tuple
import logging

# Based on code from NVES_R's response at
# https://forums.developer.nvidia.com/t/segmentation-fault-when-creating-the-trt-builder-in-python-works-fine-with-trtexec/111376


TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def _build_engine_onnx(onnx_path: str, force_fp16: bool = False, max_batch_size: int = 1, im_size: Tuple[int] = None):
    '''
    Builds TensorRT engine from provided ONNX file

    :param onnx_path: Path to ONNX file on disk
    :param force_fp16: Force use of FP16 precision, even if device doesn't support it. Be careful.
    :param max_batch_size: Define maximum batch size supported by engine. If >1 creates optimization profile.
    :param im_size: Required if max_batch_size > 1. Used for creation of optimization profile.
    :return: TensorRT engine
    '''

    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            builder.create_builder_config() as config, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        if builder.platform_has_fast_fp16 or force_fp16 is True:
            builder.fp16_mode = True
            builder.strict_type_constraints = True

        builder.max_workspace_size = 1 << 20

        if max_batch_size != 1 and im_size is not None:
            logging.warning('Batch size !=1 is used. Ensure your inference code supports it.')
            profile = builder.create_optimization_profile()
            profile.set_shape('data', (1, 3) + im_size, (max_batch_size, 3) + im_size, (max_batch_size, 3) + im_size)
            config.add_optimization_profile(profile)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                print('ERROR: Failed to parse the ONNX file: {}'.format(onnx_path))
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                sys.exit(1)

            if max_batch_size != 1:
                return builder.build_engine(network, config=config)
            else:
                return builder.build_cuda_engine(network)


def convert_onnx(onnx_path: str, engine_file_path: str, force_fp16: bool = False, max_batch_size: int = 1,
                 im_size: Tuple[int] = None):
    '''
    Creates TensorRT engine and serializes it to disk

    :param onnx_path: Path to ONNX file on disk
    :param engine_file_path: Path where TensorRT engine should be saved.
    :param force_fp16: Force use of FP16 precision, even if device doesn't support it. Be careful.
    :param max_batch_size: Define maximum batch size supported by engine. If >1 creates optimization profile.
    :param im_size: Required if max_batch_size > 1. Used for creation of optimization profile.
    :return: None
    '''

    engine = _build_engine_onnx(onnx_path=onnx_path,
                                force_fp16=force_fp16, max_batch_size=max_batch_size,
                                im_size=im_size)

    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())
