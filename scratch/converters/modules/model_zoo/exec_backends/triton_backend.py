import numpy as np
import cv2

import sys
import argparse
import numpy as np

import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
from tritonclient.utils import InferenceServerException

from numpy.linalg import norm


def normalize(embedding):
    embedding_norm = norm(embedding)
    normed_embedding = embedding / embedding_norm
    return normed_embedding

# Callback function used for async_run()
def completion_callback(input_filenames, user_data, infer_ctx, request_id):
    user_data._completed_requests.put((request_id, input_filenames))

FLAGS = None

def parse_model_http(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata['inputs']) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata['inputs'])))

    if len(model_config['input']) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config['input'])))

    input_metadata = model_metadata['inputs'][0]
    input_config = model_config['input'][0]
    output_names = [e['name'] for e in model_metadata['outputs']]

    max_batch_size = 0
    if 'max_batch_size' in model_config:
        max_batch_size = model_config['max_batch_size']

    # Model input must have 3 dims (not counting the batch dimension),
    # either CHW or HWC
    input_batch_dim = (max_batch_size > 0)
    expected_input_dims = 3 + (1 if input_batch_dim else 0)
    if len(input_metadata['shape']) != expected_input_dims:
        raise Exception(
            "expecting input to have {} dimensions, model '{}' input has {}".
            format(expected_input_dims, model_metadata['name'],
                   len(input_metadata['shape'])))


    if input_config['format'] == "FORMAT_NHWC":
        h = input_metadata['shape'][1 if input_batch_dim else 0]
        w = input_metadata['shape'][2 if input_batch_dim else 1]
        c = input_metadata['shape'][3 if input_batch_dim else 2]
    else:
        c = input_metadata['shape'][1 if input_batch_dim else 0]
        h = input_metadata['shape'][2 if input_batch_dim else 1]
        w = input_metadata['shape'][3 if input_batch_dim else 2]

    return (max_batch_size, input_metadata['name'],output_names, c,
            h, w, input_config['format'], input_metadata['datatype'])

url = 'localhost:8001'
model_name = 'arcface_r100_v1'
model_version = '1'

class Arcface:

    def __init__(self, rec_name = 'arcface_r100_v1', model_version='1',url='localhost:8001'):
        self.model_name = 'arcface_r100_v1'
        self.model_version = model_version
        self.url = url
        self.triton_client = httpclient.InferenceServerClient(url=url, concurrency=1)


    def prepare(self,ctx=0):
        concurrency = 1
        # Make sure the model matches our requirements, and get some
        # properties of the model that we need for preprocessing
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

        self.max_batch_size, self.input_name, self.output_name, self.c, self.h, self.w, self.format, self.dtype = parse_model_http(
            model_metadata, model_config)


    def get_embedding(self,face_img):
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = np.transpose(face_img, (2, 0, 1))
        face_img = np.expand_dims(face_img, axis=0)
        face_img = face_img.astype(triton_to_np_dtype(self.dtype))
        inputs = []
        inputs.append(httpclient.InferInput(self.input_name, [1, self.c, self.h,self.w], "FP32"))
        inputs[0].set_data_from_numpy(face_img)

        out = self.triton_client.infer(self.model_name,
                            inputs,
                            model_version=self.model_version,
                            outputs=None)
        out = [out.as_numpy(e)[0] for e in self.output_name]
        #print(output.get_output(self.output_name)['data'])
        return out
