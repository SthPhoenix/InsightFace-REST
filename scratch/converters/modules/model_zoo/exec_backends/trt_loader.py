import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

import tensorrt as trt

TRT_LOGGER = trt.Logger()

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    out_shapes = []
    input_shapes = []
    out_names = []
    max_batch_size = engine.get_profile_shape(0, 0)[2][0]
    for binding in engine:
        binding_shape = engine.get_binding_shape(binding)
        #Fix -1 dimension for proper memory allocation for batch_size > 1
        if binding_shape[0] == -1:
            binding_shape = (1,) + binding_shape[1:]
        size = trt.volume(binding_shape) * max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
            input_shapes.append(engine.get_binding_shape(binding))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
            #Collect original output shapes and names from engine
            out_shapes.append(engine.get_binding_shape(binding))
            out_names.append(binding)
    return inputs, outputs, bindings, stream, input_shapes, out_shapes, out_names, max_batch_size

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

class TrtModel(object):
    def __init__(self, model):
        self.engine_file = model
        self.engine = None
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.stream = None
        self.context = None
        self.input_shapes = None
        self.out_shapes = None
        self.max_batch_size = 1

    def build(self):
        with open(self.engine_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.inputs, self.outputs, self.bindings, self.stream, self.input_shapes, self.out_shapes, self.out_names, self.max_batch_size = allocate_buffers(
            self.engine)

        self.context = self.engine.create_execution_context()
        self.context.active_optimization_profile = 0

    def run(self, input, deflatten: bool = True, as_dict=False):
        # lazy load implementation
        if self.engine is None:
            self.build()

        input = np.asarray(input)
        batch_size = input.shape[0]
        allocate_place = np.prod(input.shape)
        self.inputs[0].host[:allocate_place] = input.flatten(order='C').astype(np.float32)
        self.context.set_binding_shape(0, input.shape)
        trt_outputs = do_inference(
            self.context, bindings=self.bindings,
            inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        #Reshape TRT outputs to original shape instead of flattened array
        if deflatten:
            trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, self.out_shapes)]
        if as_dict:
            return {name: trt_outputs[i] for i, name in enumerate(self.out_names)}

        return [trt_output[:batch_size] for trt_output in trt_outputs]

