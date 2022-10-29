import os
import shutil
from modules.converters.onnx_to_trt import convert_onnx
from modules.configs import Configs
from modules.utils.helpers import prepare_folders
from modules.converters.reshape_onnx import reshape
import onnx

'''
ATTENTION!!! This script is for testing purposes only. 
'''

def reshape_onnx_input(onnx_path , out_path, im_size):
    model = onnx.load(onnx_path)
    reshaped = reshape(model,h = im_size[1], w = im_size[0], mode='centerface')

    with open(out_path, "wb") as file_handle:
        serialized = reshaped.SerializeToString()
        file_handle.write(serialized)



if __name__ == '__main__':

    configs = Configs(models_dir='/models')
    model_name = 'centerface'
    im_size = [1280, 1024] # W, H

    prepare_folders([configs.onnx_models_dir, configs.trt_engines_dir])

    _, onnx_model = configs.build_model_paths(model_name,'onnx')
    temp_onnx_model = f'{onnx_model}.temp'
    reshape_onnx_input(onnx_model, temp_onnx_model, im_size=im_size)


    output_trt_engine_path, output_engine = configs.build_model_paths(model_name, 'plan')

    convert_onnx(temp_onnx_model, output_engine)

    os.remove(temp_onnx_model)
