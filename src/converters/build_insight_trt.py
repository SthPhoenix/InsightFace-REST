import os

from modules.converters.onnx_to_trt import convert_onnx
from modules.configs import Configs
from modules.utils.model_store import get_model_file
from modules.converters.insight2onnx import convert_insight_model

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
                           force: bool = False):

    if not os.path.exists(engine_path) or force is True:
        if not os.path.exists(onnx_path) or force is True:
            if not os.path.exists(symbol)  and configs.in_official_package(model_name):
                get_model_file(model_name, mxnet_models_dir)
            print("Converting MXNet model to ONNX...")
            convert_insight_model(symbol, params, onnx_path,input_shape=input_shape)
        print("Building TensorRT engine...")
        convert_onnx(onnx_path, engine_path)
    print("TensorRT model ready.")


if __name__ == '__main__':

    configs = Configs(models_dir='/models')
    model_name = 'arcface_r100_v1'

    prepare_folders([configs.mxnet_models_dir, configs.onnx_models_dir, configs.trt_engines_dir])

    mx_symbol, mx_params = configs.get_mxnet_model_paths(model_name)

    output_onnx_model_path, output_onnx = configs.build_model_paths(model_name, 'onnx')
    output_trt_engine_path, output_engine = configs.build_model_paths(model_name, 'plan')

    prepare_folders([output_onnx_model_path,output_trt_engine_path])
    input_shape = configs.mxnet_models[model_name]['shape']
    prepare_insight_engine(mx_symbol, mx_params, output_onnx, output_engine, model_name, configs.mxnet_models_dir,
                           input_shape, force=True)
