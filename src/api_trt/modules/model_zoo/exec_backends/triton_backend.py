import numpy as np
import cv2
import os
import sys
import argparse
import numpy as np
import logging

#import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
from tritonclient.utils import InferenceServerException
#import tritonclient.utils.shared_memory as shm
#import tritonclient.utils.cuda_shared_memory as cudashm

#from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc as grpcclient
#import tritonclient.grpc.model_config_pb2 as mc



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
    output_names = [model_metadata.outputs[i].name for i in range(len(model_metadata.outputs))]

    max_batch_size = 0
    if  model_config.max_batch_size:
        max_batch_size = model_config.max_batch_size

    # Model input must have 3 dims (not counting the batch dimension),
    # either CHW or HWC
    input_batch_dim = (max_batch_size > 0)
    # Workaround for SCRFD model.
    if max_batch_size == 0 and input_metadata.shape[0] < 0:
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
            h, w, input_config.format, input_metadata.datatype)

url = 'localhost:8001'
model_name = 'arcface_r100_v1'
model_version = '1'

class Arcface:

    def __init__(self, rec_name = 'arcface_r100_v1', model_version='1', triton_uri='localhost:8001', **kwargs):
        self.model_name = rec_name
        self.model_version = model_version
        self.url = triton_uri
        self.triton_client = grpcclient.InferenceServerClient(url=triton_uri)


    def prepare(self,**kwargs):
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

        self.max_batch_size, self.input_name, self.output_name, self.c, self.h, self.w, self.format, self.dtype = parse_model_grpc(
            model_metadata, model_config.config)




    def get_embedding(self,face_img):

        if not isinstance(face_img, list):
            face_img = [face_img]
        for i, img in enumerate(face_img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            face_img[i] = img

        face_img = np.stack(face_img)

        face_img = face_img.astype(triton_to_np_dtype(self.dtype))

        inputs = []
        inputs.append(grpcclient.InferInput(self.input_name, [face_img.shape[0], self.c, self.h, self.w], "FP32"))
        inputs[0].set_data_from_numpy(face_img)

        out = self.triton_client.infer(self.model_name,
                            inputs,
                            model_version=self.model_version,
                            outputs=None)

        out = [out.as_numpy(e) for e in self.output_name]

        return out[0]


class Cosface:
    def __init__(self, rec_name = 'arcface_r100_v1', triton_uri='localhost:8001', model_version='1',**kwargs):
        self.model_name = rec_name
        self.model_version = model_version
        self.url = triton_uri
        self.input_shape = None
        self.max_batch_size = 1
        self.input_mean = 127.5
        self.input_std = 127.5
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

        self.max_batch_size, self.input_name, self.output_name, self.c, self.h, self.w, self.format, self.dtype = parse_model_grpc(
            model_metadata, model_config.config)

    def get_embedding(self, face_img):
        if not isinstance(face_img, list):
            face_img = [face_img]

        for i, img in enumerate(face_img):
            input_size = tuple(img.shape[0:2][::-1])
            blob = cv2.dnn.blobFromImage(img, 1.0 / self.input_std, input_size,
                                         (self.input_mean, self.input_mean, self.input_mean), swapRB=True)[0]
            face_img[i] = blob

        face_img = np.stack(face_img)
        face_img = face_img.astype(triton_to_np_dtype(self.dtype))
        inputs = []
        inputs.append(grpcclient.InferInput(self.input_name, [face_img.shape[0], self.c, self.h, self.w], "FP32"))
        inputs[0].set_data_from_numpy(face_img)

        out = self.triton_client.infer(self.model_name,
                                       inputs,
                                       model_version=self.model_version,
                                       outputs=None)

        out = [out.as_numpy(e) for e in self.output_name]

        return out[0]


class DetectorInfer:

    def __init__(self, model = 'retinaface_r50_v1', output_order=None, triton_uri='localhost:8001', model_version='1',**kwargs):
        self.model_name = model
        self.model_version = model_version
        self.url = triton_uri
        self.input_shape = (1, 3, 640, 640)
        self.output_order = output_order
        self.triton_client = grpcclient.InferenceServerClient(url=triton_uri)

    # warmup
    def prepare(self, ctx_id=0):
        concurrency = 2
        # Make sure the model matches our requirements, and get some
        # properties of the model that we need for preprocessing

        #self.triton_client.unregister_system_shared_memory()
        #self.triton_client.unregister_cuda_shared_memory()

        #self.in_handle_name = f'data_{os.getpid()}'
        #self.input_bytesize = 4915200
        #self.in_handle = cudashm.create_shared_memory_region(
        #    self.in_handle_name, self.input_bytesize, 0)

        #self.triton_client.register_cuda_shared_memory(
        #    self.in_handle_name, cudashm.get_raw_handle(self.in_handle), 0,
        #    self.input_bytesize)



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

        self.max_batch_size, self.input_name, self.output_name, self.c, self.h, self.w, self.format, self.dtype = parse_model_grpc(
            model_metadata, model_config.config)

        self.input_shape = (1, self.c, self.h, self.w)

    def run(self, input):
        inputs = []
        outputs = [grpcclient.InferRequestedOutput(e) for e in self.output_order]
        inputs.append(grpcclient.InferInput(self.input_name, [1, self.c, self.h, self.w], "FP32"))
        inputs[0].set_data_from_numpy(input)
        #cudashm.set_shared_memory_region(self.in_handle, [input])
        #inputs[-1].set_shared_memory(self.in_handle_name, self.input_bytesize)

        out = self.triton_client.infer(self.model_name,
                                       inputs,
                                       model_version=self.model_version,
                                       outputs=outputs)

        out = [out.as_numpy(e) for e in self.output_order]

        return out