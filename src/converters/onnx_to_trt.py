import os
import tensorrt as trt
import sys
batch_size = 1

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

# Based on code from NVES_R's response at
# https://forums.developer.nvidia.com/t/segmentation-fault-when-creating-the-trt-builder-in-python-works-fine-with-trtexec/111376

def build_engine_onnx(model_file, fp16=False):
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            builder.create_builder_config() as config, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        if builder.platform_has_fast_fp16:
            builder.fp16_mode = True

        builder.max_workspace_size = 1 << 30
        if fp16:
            builder.fp16_mode = True
            builder.strict_type_constraints = True

        with open(model_file, "rb") as f:
            if not parser.parse(f.read()):
                print('ERROR: Failed to parse the ONNX file: {}'.format(model_file))
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                sys.exit(1)
            return builder.build_cuda_engine(network)


def convert_onnx(onnx_path, engine_file_path):
    engine = build_engine_onnx(onnx_path)
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())