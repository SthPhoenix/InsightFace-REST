import os

from modules.converters.onnx_to_trt import convert_onnx
from modules.configs import Configs
from modules.utils.model_store import get_model_file
from modules.converters.insight2onnx import convert_insight_model
from modules.converters.reshape_onnx import reshape_onnx_input

'''
ATTENTION!!! This script is for testing purposes only.
'''


def prepare_folders(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def prepare_insight_engine(symbol: str,
                           params: str,
                           onnx_path: str,
                           engine_path: str,
                           model_name: str,
                           mxnet_models_dir: str,
                           input_shape,
                           max_batch_size: int = 1,
                           force: bool = False):
    # Set explicit batch size for ONNX
    onnx_batch_size = 1
    if max_batch_size != 1:
        onnx_batch_size = -1

    if not os.path.exists(engine_path) or force is True:
        if not os.path.exists(onnx_path):
            if not os.path.exists(symbol) and configs.in_official_package(model_name):
                get_model_file(model_name, mxnet_models_dir)
            print("Converting MXNet model to ONNX...")
            convert_insight_model(symbol, params, onnx_path, input_shape=input_shape)
        temp_onnx_model = onnx_path + '.tmp'
        reshape_onnx_input(onnx_path, temp_onnx_model, im_size=list(input_shape[2:]), batch_size=onnx_batch_size)
        print("Building TensorRT engine...")
        convert_onnx(temp_onnx_model, engine_path, max_batch_size=max_batch_size, im_size=input_shape[2:])
        os.remove(temp_onnx_model)
    print("TensorRT model ready.")


if __name__ == '__main__':
    configs = Configs(models_dir='/models')
    model_name = 'arcface_r100_v1'
    # ATTENTION inference with max_batch_size !=1 is not implemented yet
    max_batch_size = 1

    prepare_folders([configs.mxnet_models_dir, configs.onnx_models_dir, configs.trt_engines_dir])

    mx_symbol, mx_params = configs.get_mxnet_model_paths(model_name)

    output_onnx_model_path, output_onnx = configs.build_model_paths(model_name, 'onnx')
    output_trt_engine_path, output_engine = configs.build_model_paths(model_name, 'plan')

    prepare_folders([output_onnx_model_path, output_trt_engine_path])
    input_shape = configs.mxnet_models[model_name]['shape']
    prepare_insight_engine(mx_symbol, mx_params, output_onnx, output_engine, model_name, configs.mxnet_models_dir,
                           input_shape, max_batch_size=max_batch_size, force=True)
