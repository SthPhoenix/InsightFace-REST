import numpy as np
import cv2
import os
import sys
import argparse
import numpy as np
import logging

# import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
from tritonclient.utils import InferenceServerException
# import tritonclient.utils.shared_memory as shm
import tritonclient.utils.cuda_shared_memory as cudashm

# from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc as grpcclient

# import tritonclient.grpc.model_config_pb2 as mc


FLAGS = None


def parse_model_grpc(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    print(len(model_metadata.outputs))
    if len(model_metadata.inputs) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata.inputs)))

    if len(model_config.input) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config.input)))

    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]
    output_names = [out.name for out in model_metadata.outputs]
    out_shapes = [out.shape for out in model_metadata.outputs]
    max_batch_size = 0
    if model_config.max_batch_size:
        max_batch_size = model_config.max_batch_size

    # Model input must have 3 dims (not counting the batch dimension),
    # either CHW or HWC
    input_batch_dim = (max_batch_size > 0)
    # Workaround for SCRFD model.
    if max_batch_size == 0 and input_metadata.shape[0] <= 1:
        input_batch_dim = True
    expected_input_dims = 3 + (1 if input_batch_dim else 0)
    if len(input_metadata.shape) != expected_input_dims:
        raise Exception(
            "expecting input to have {} dimensions, model '{}' input has {}".
                format(expected_input_dims, model_metadata.name,
                       len(input_metadata.shape)))

    if input_config.format == "FORMAT_NHWC":
        h = input_metadata.shape[1 if input_batch_dim else 0]
        w = input_metadata.shape[2 if input_batch_dim else 1]
        c = input_metadata.shape[3 if input_batch_dim else 2]
    else:
        c = input_metadata.shape[1 if input_batch_dim else 0]
        h = input_metadata.shape[2 if input_batch_dim else 1]
        w = input_metadata.shape[3 if input_batch_dim else 2]

    return (max_batch_size, input_metadata.name, output_names, c,
            h, w, input_config.format, input_metadata.datatype, out_shapes)


url = 'localhost:8001'
model_name = 'arcface_r100_v1'
model_version = '1'


class Arcface:
    def __init__(self, rec_name='arcface_r100_v1', triton_uri='localhost:8001', model_version='1',
                 input_mean: float = 0.,
                 input_std: float = 1.,
                 **kwargs):
        self.model_name = rec_name
        self.model_version = model_version
        self.url = triton_uri
        self.input_shape = None
        self.max_batch_size = 1
        self.input_mean = input_mean
        self.input_std = input_std
        self.triton_client = grpcclient.InferenceServerClient(url=triton_uri)

    # warmup
    def prepare(self, **kwargs):
        concurrency = 10
        # Make sure the model matches our requirements, and get some
        # properties of the model that we need for preprocessing
        print("Model metadata:", self.model_name, self.model_version)
        try:
            model_metadata = self.triton_client.get_model_metadata(
                model_name=self.model_name, model_version=self.model_version)
        except InferenceServerException as e:
            print("failed to retrieve the metadata: " + str(e))
            sys.exit(1)

        try:
            model_config = self.triton_client.get_model_config(
                model_name=self.model_name, model_version=self.model_version)
        except InferenceServerException as e:
            print("failed to retrieve the config: " + str(e))
            sys.exit(1)

        self.max_batch_size, self.input_name, self.output_name, self.c, self.h, self.w, self.format, self.dtype, self.out_shapes = parse_model_grpc(
            model_metadata, model_config.config)

        self.in_handle_name = f'{self.model_name}_data_{os.getpid()}'
        self.input_bytesize = 12 * self.w * self.h * self.max_batch_size
        self.in_handle = cudashm.create_shared_memory_region(
            self.in_handle_name, self.input_bytesize, 0)

        self.out_handle_name = f'{self.model_name}_data_out_{os.getpid()}'
        self.out_bytesize = 12 * 512 * self.max_batch_size
        self.out_handle = cudashm.create_shared_memory_region(
            self.out_handle_name, self.out_bytesize, 0)

        self.triton_client.unregister_cuda_shared_memory(self.in_handle_name)
        self.triton_client.unregister_cuda_shared_memory(self.out_handle_name)

        self.triton_client.register_cuda_shared_memory(
            self.in_handle_name, cudashm.get_raw_handle(self.in_handle), 0,
            self.input_bytesize)

        self.triton_client.register_cuda_shared_memory(
            self.out_handle_name, cudashm.get_raw_handle(self.out_handle), 0,
            self.out_bytesize)

    def get_embedding(self, face_img):
        if not isinstance(face_img, list):
            face_img = [face_img]

        face_img = np.stack(face_img)

        input_size = tuple(face_img[0].shape[0:2][::-1])
        blob = cv2.dnn.blobFromImages(face_img, 1.0 / self.input_std, input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)

        blob = blob.astype(triton_to_np_dtype(self.dtype))

        inputs = []
        inputs.append(grpcclient.InferInput(self.input_name, [blob.shape[0], self.c, self.h, self.w], "FP32"))
        # inputs[0].set_data_from_numpy(face_img)

        cudashm.set_shared_memory_region(self.in_handle, [blob])
        input_bytesize = 12 * blob.shape[0] * self.w * self.h
        inputs[-1].set_shared_memory(self.in_handle_name, input_bytesize)

        outputs = []
        out_bytesize = 12 * 512 * self.max_batch_size
        outputs.append(grpcclient.InferRequestedOutput(self.output_name[0]))
        outputs[-1].set_shared_memory(self.out_handle_name, out_bytesize)

        out = self.triton_client.infer(self.model_name,
                                       inputs,
                                       model_version=self.model_version,
                                       outputs=outputs)

        out = [cudashm.get_contents_as_numpy(self.out_handle, triton_to_np_dtype(self.dtype), [blob.shape[0], 512])]
        # out = [out.as_numpy(e) for e in self.output_name]

        return out[0]


class DetectorInfer:

    def __init__(self, model='retinaface_r50_v1', output_order=None, triton_uri='localhost:8001', model_version='1',
                 **kwargs):
        self.model_name = model
        self.model_version = model_version
        self.url = triton_uri
        self.input_shape = (1, 3, 640, 640)
        self.input_dtype = np.float32
        self.output_order = output_order
        self.triton_client = grpcclient.InferenceServerClient(url=triton_uri)

    # warmup
    def prepare(self, ctx_id=0):
        concurrency = 2
        # Make sure the model matches our requirements, and get some
        # properties of the model that we need for preprocessing

        try:
            model_metadata = self.triton_client.get_model_metadata(
                model_name=self.model_name, model_version=self.model_version)
        except InferenceServerException as e:
            print("failed to retrieve the metadata: " + str(e))
            sys.exit(1)

        logging.info(model_metadata)

        try:
            model_config = self.triton_client.get_model_config(
                model_name=self.model_name, model_version=self.model_version)
        except InferenceServerException as e:
            print("failed to retrieve the config: " + str(e))
            sys.exit(1)

        self.max_batch_size, self.input_name, self.output_name, self.c, self.h, self.w, self.format, self.dtype, self.out_shapes = parse_model_grpc(
            model_metadata, model_config.config)

        self.input_shape = (1, self.c, self.h, self.w)
        self.input_dtype = triton_to_np_dtype(self.dtype)

        self.in_handle_name = f'{self.model_name}_data_{os.getpid()}'

        if self.max_batch_size <= 0:
            self.max_batch_size = 1
        self.input_bytesize = 12 * self.w * self.h * 1

        self.in_handle = cudashm.create_shared_memory_region(
            self.in_handle_name, self.input_bytesize, 0)

        self.triton_client.unregister_cuda_shared_memory(self.in_handle_name)
        self.triton_client.register_cuda_shared_memory(
            self.in_handle_name, cudashm.get_raw_handle(self.in_handle), 0,
            self.input_bytesize)

    def run(self, input):
        inputs = []
        outputs = [grpcclient.InferRequestedOutput(e) for e in self.output_order]
        inputs.append(grpcclient.InferInput(self.input_name, [1, self.c, self.h, self.w], self.dtype))
        # inputs[0].set_data_from_numpy(input)
        cudashm.set_shared_memory_region(self.in_handle, [input])
        inputs[-1].set_shared_memory(self.in_handle_name, self.input_bytesize)

        out = self.triton_client.infer(self.model_name,
                                       inputs,
                                       model_version=self.model_version,
                                       outputs=outputs)

        out = [out.as_numpy(e) for e in self.output_order]

        return out
