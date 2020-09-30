import mxnet as mx
import numpy as np
from mxnet.contrib import onnx as onnx_mxnet
import onnx
from onnx import checker
import os
import logging

print('mxnet version:', mx.__version__)
print('onnx version:', onnx.__version__)

assert onnx.__version__ == '1.2.1'


# Based on
# https://github.com/onnx/models/issues/156#issuecomment-690847276

def convert_insight(symbol, params, output_path, input_shape=(1,3,112,112)):
    intermediate_onnx_model = f'{output_path}.temp'
    onnx_mxnet.export_model(symbol, params, [input_shape], np.float32, intermediate_onnx_model)

    model = onnx.load(intermediate_onnx_model)
    onnx_processed_nodes = []
    onnx_processed_inputs = []
    onnx_processed_outputs = []
    onnx_processed_initializers = []

    reshape_node = []

    for ind, node in enumerate(model.graph.node):
        if node.op_type == "PRelu":
            input_node = node.input
            input_bn = input_node[0]
            input_relu_gamma = input_node[1]
            output_node = node.output[0]

            input_reshape_name = "reshape{}".format(ind)
            slope_number = "slope{}".format(ind)

            node_reshape = onnx.helper.make_node(
                op_type="Reshape",
                inputs=[input_relu_gamma, input_reshape_name],
                outputs=[slope_number],
                name=slope_number
            )

            reshape_node.append(input_reshape_name)
            node_relu = onnx.helper.make_node(
                op_type="PRelu",
                inputs=[input_bn, slope_number],
                outputs=[output_node],
                name=output_node
            )
            onnx_processed_nodes.extend([node_reshape, node_relu])

        else:
            # If "spatial = 0" does not work for "BatchNormalization", change "spatial=1"
            # else comment this "if" condition
            if node.op_type == "BatchNormalization":
                for attr in node.attribute:
                    if (attr.name == "spatial"):
                        attr.i = 1
            onnx_processed_nodes.append(node)

    list_new_inp = []
    list_new_init = []
    for name_rs in reshape_node:
        new_inp = onnx.helper.make_tensor_value_info(
            name=name_rs,
            elem_type=onnx.TensorProto.INT64,
            shape=[4]
        )
        new_init = onnx.helper.make_tensor(
            name=name_rs,
            data_type=onnx.TensorProto.INT64,
            dims=[4],
            vals=[1, -1, 1, 1]
        )

        list_new_inp.append(new_inp)
        list_new_init.append(new_init)

    for k, inp in enumerate(model.graph.input):
        if "relu0_gamma" in inp.name or "relu1_gamma" in inp.name:  # or "relu_gamma" in inp.name:
            new_reshape = list_new_inp.pop(0)
            onnx_processed_inputs.extend([inp, new_reshape])
        else:
            onnx_processed_inputs.extend([inp])

    for k, outp in enumerate(model.graph.output):
        onnx_processed_outputs.extend([outp])

    for k, init in enumerate(model.graph.initializer):
        if "relu0_gamma" in init.name or "relu1_gamma" in init.name:
            new_reshape = list_new_init.pop(0)
            onnx_processed_initializers.extend([init, new_reshape])
        else:
            onnx_processed_initializers.extend([init])

    graph = onnx.helper.make_graph(
        onnx_processed_nodes,
        "mxnet_converted_model",
        onnx_processed_inputs,
        onnx_processed_outputs
    )

    graph.initializer.extend(onnx_processed_initializers)

    # Check graph
    checker.check_graph(graph)

    onnx_model = onnx.helper.make_model(graph)

    with open(output_path, "wb") as file_handle:
        serialized = onnx_model.SerializeToString()
        file_handle.write(serialized)
        logging.info("Input shape of the model %s ", input_shape)
        logging.info("Exported ONNX file %s saved to disk", output_path)

    print('Removing intermediate model...')
    os.remove(intermediate_onnx_model)

    print("Done!!!")
