import os
import tensorrt as trt
import sys

# Based on code from NVES_R's response at
# https://forums.developer.nvidia.com/t/segmentation-fault-when-creating-the-trt-builder-in-python-works-fine-with-trtexec/111376


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def _build_engine_onnx(onnx_path: str, allow_fp16: bool = False):
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            builder.create_builder_config() as config, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        if builder.platform_has_fast_fp16 and allow_fp16 is True:
            builder.fp16_mode = True

        builder.max_workspace_size = 1 << 20

        # profile = builder.create_optimization_profile()
        # profile.set_shape('data',(1,3,112,112),(2,3,112,112),(5,3,112,112))
        # config.add_optimization_profile(profile)


        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                print('ERROR: Failed to parse the ONNX file: {}'.format(onnx_path))
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                sys.exit(1)
            return builder.build_cuda_engine(network)


def convert_onnx(onnx_path: str, engine_file_path: str, allow_fp16: bool = False):
    engine = _build_engine_onnx(onnx_path=onnx_path,
                                allow_fp16=allow_fp16)
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())
