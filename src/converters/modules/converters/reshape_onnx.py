import onnx
import math
import os
from typing import List


def reshape(model, n: int = 1, h: int = 480, w: int = 640, mode='auto'):
    '''
    :param model: Input ONNX model object
    :param n: Batch size dimension
    :param h: Height dimension
    :param w: Width dimension
    :param mode: Set `retinaface` to reshape RetinaFace model, otherwise reshape Centerface
    :return: ONNX model with reshaped input and outputs
    '''
    if mode == 'auto':
        # Assert that retinaface models have outputs containing word 'stride' in their names

        out_name = model.graph.output[0].name
        if 'stride' in out_name.lower():
            mode = 'retinaface'
        elif out_name.lower() == 'fc1':
            mode = 'arcface'
        else:
            mode = 'centerface'

    d = model.graph.input[0].type.tensor_type.shape.dim
    d[0].dim_value = n
    if mode != 'arcface':
        d[2].dim_value = h
        d[3].dim_value = w
    divisor = 4
    for output in model.graph.output:
        if mode == 'retinaface':
            divisor = int(output.name.split('stride')[-1])
        d = output.type.tensor_type.shape.dim
        d[0].dim_value = n
        if mode != 'arcface':
            d[2].dim_value = math.ceil(h / divisor)
            d[3].dim_value = math.ceil(w / divisor)
    return model


def reshape_onnx_input(onnx_path: str, out_path: str, im_size: List[int] = None, batch_size: int = 1,
                       mode: str = 'auto'):
    '''
    Reshape ONNX file input and output for different image sizes. Only applicable for MXNet Retinaface models
    and official Centerface models.

    :param onnx_path: Path to input ONNX file
    :param out_path: Path to output ONNX file
    :param im_size: Desired output image size in W, H format. Default: [640, 480]
    :param mode: Available modes: retinaface, centerface, auto (try to detect if input model is retina- or centerface)
    :return:
    '''

    if im_size is None:
        im_size = [640, 480]

    model = onnx.load(onnx_path)
    reshaped = reshape(model, n=batch_size, h=im_size[1], w=im_size[0], mode=mode)

    with open(out_path, "wb") as file_handle:
        serialized = reshaped.SerializeToString()
        file_handle.write(serialized)
