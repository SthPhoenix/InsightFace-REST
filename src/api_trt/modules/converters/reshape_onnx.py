import math
from typing import List

import onnx

from api_trt.logger import logger


def reshape(model, n: int = 1, h: int = 480, w: int = 640, mode='auto'):
    """
    Reshape ONNX model input and output for different image sizes. Only applicable for MXNet Retinaface models
    and official Centerface models.

    Args:
        model (onnx.ModelProto): Input ONNX model object
        n (int): Batch size dimension
        h (int): Height dimension
        w (int): Width dimension
        mode (str): Set `retinaface` to reshape RetinaFace model, otherwise reshape Centerface

    Returns:
        onnx.ModelProto: ONNX model with reshaped input and outputs
    """
    if mode == 'auto':
        # Assert that retinaface models have outputs containing word 'stride' in their names

        out_name = model.graph.output[0].name
        if 'stride' in out_name.lower():
            mode = 'retinaface'
        elif out_name.lower() == 'fc1':
            mode = 'arcface'
        else:
            mode = 'centerface'

        input_name = model.graph.input[0].name
        out_shape = model.graph.output[0].type.tensor_type.shape.dim[1].dim_value
        
        dyn_size = False
        if model.graph.input[0].type.tensor_type.shape.dim[2].dim_param == '?':
            dyn_size = True

        if input_name == 'input.1' and dyn_size is True:
            mode = 'scrfd'
        elif input_name == 'input.1' and out_shape == 512:
            mode = 'arcface'
        elif input_name == 'input' and model.graph.input[0].type.tensor_type.shape.dim[2].dim_value == 112:
            mode = 'arcface'
        if  model.graph.input[0].type.tensor_type.shape.dim[3].dim_value == 3:
            mode = 'mask_detector'
        if len(model.graph.output) == 1 and len(model.graph.output[0].type.tensor_type.shape.dim) == 3:
            mode = 'yolov5-face'


    d = model.graph.input[0].type.tensor_type.shape.dim
    d[0].dim_value = n
    logger.debug(f"In shape: {d}")
    if mode != 'arcface':
        d[2].dim_value = h
        d[3].dim_value = w
    divisor = 4
    logger.debug(f"Mode: {mode}")
    if mode == 'yolov5-face':
        d = model.graph.output[0].type.tensor_type.shape.dim
        mx = (h * w) / 16
        s = mx - mx / 64
        d[0].dim_value = n
        d[1].dim_value = int(s)
        d[2].dim_value = 16
    elif mode != 'scrfd':
        for output in model.graph.output:
            if mode == 'retinaface':
                divisor = int(output.name.split('stride')[-1])
            d = output.type.tensor_type.shape.dim
            d[0].dim_value = n
            if mode not in  ('arcface', 'mask_detector'):
                d[2].dim_value = math.ceil(h / divisor)
                d[3].dim_value = math.ceil(w / divisor)
    logger.debug(f"Out shape: {d}")
    return model


def reshape_onnx_input(onnx_path: str, out_path: str, im_size: List[int] = None, batch_size: int = 1,
                       mode: str = 'auto'):
    """
    Reshape ONNX file input and output for different image sizes. Only applicable for MXNet Retinaface models
    and official Centerface models.

    Args:
        onnx_path (str): Path to input ONNX file
        out_path (str): Path to output ONNX file
        im_size (List[int]): Desired output image size in W, H format. Default: [640, 480]
        mode (str): Available modes: retinaface, centerface, auto (try to detect if input model is retina- or centerface)

    Returns:
        None
    """

    if im_size is None:
        im_size = [640, 480]

    model = onnx.load(onnx_path)
    reshaped = reshape(model, n=batch_size, h=im_size[1], w=im_size[0], mode=mode)

    with open(out_path, "wb") as file_handle:
        serialized = reshaped.SerializeToString()
        file_handle.write(serialized)
