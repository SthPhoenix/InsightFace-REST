import onnx
import math
import os

def reshape(model, n: int = 1, h: int = 480, w: int = 640, mode='retina'):
    '''
    :param model: Input ONNX model object
    :param n: Batch size dimension
    :param h: Height dimension
    :param w: Width dimension
    :param mode: Set `retina` to reshape RetinaFace model, otherwise reshape Centerface
    :return: ONNX model with reshaped input and outputs
    '''

    d = model.graph.input[0].type.tensor_type.shape.dim
    d[0].dim_value = n
    d[2].dim_value = h
    d[3].dim_value = w
    divisor = 4
    for output in model.graph.output:
        if mode == 'retina':
            divisor = int(output.name.split('stride')[-1])
        d = output.type.tensor_type.shape.dim
        print(d)
        d[0].dim_value = 1
        d[2].dim_value = math.ceil(h / divisor)
        d[3].dim_value = math.ceil(w / divisor)
    return model
