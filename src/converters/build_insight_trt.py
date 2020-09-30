import os

from utils.model_store import get_model_file
from convert_insight_onnx import convert_insight
from onnx_to_trt import convert_onnx

models_dir = '/models'
mxnet_models_dir = os.path.join(models_dir, 'mxnet')
onnx_models_dir = os.path.join(models_dir, 'onnx')
trt_engines_dir = os.path.join(models_dir,'trt-engines')
model_name = 'arcface_r100_v1'

for path in [mxnet_models_dir, onnx_models_dir,trt_engines_dir]:
    if not os.path.exists(path):
        os.makedirs(path)



def prepare_insight_engine(symbol,params,onnx_path,engine_path, model_name=model_name, force=False):
    if not os.path.exists(engine_path):
        if not os.path.exists(onnx_path):
            if not os.path.exists(symbol):
                get_model_file(model_name, mxnet_models_dir)
            print("Converting MXNet model to ONNX...")
            convert_insight(symbol, params, onnx_path)
        print("Building TensorRT engine...")
        convert_onnx(onnx_path, engine_path)
    print("TensorRT model ready.")

if __name__ == '__main__':
    ##Default path useed by insightface api for model downloads
    input_mxnet_symbol = os.path.join(mxnet_models_dir, model_name, 'model-symbol.json')
    input_mxnet_params = os.path.join(mxnet_models_dir, model_name, 'model-0000.params')

    output_onnx_model = os.path.join(onnx_models_dir, f"{model_name}.onnx")
    output_trt_engine = os.path.join(trt_engines_dir, f"{model_name}.engine")

    prepare_insight_engine(input_mxnet_symbol, input_mxnet_params,output_onnx_model, output_trt_engine)