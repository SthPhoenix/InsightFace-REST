import cupy as cp
import numpy as np
import os
import tensorrt as trt
import cupyx


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, size, dtype):
        self.size = size
        self.dtype = dtype
        self.host = cupyx.zeros_pinned(size, dtype)
        self.device = cp.zeros(size, dtype)

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

    @property
    def nbytes(self):
        return self.host.nbytes

    @property
    def hostptr(self):
        return self.host.ctypes.data

    @property
    def devptr(self):
        return self.device.data.ptr

    def copy_htod_async(self, stream):
        self.device.data.copy_from_host_async(self.hostptr, self.nbytes, stream)

    def copy_dtoh_async(self, stream):
        self.device.data.copy_to_host_async(self.hostptr, self.nbytes, stream)


class TrtModel(object):
    TRT_LOGGER = trt.Logger()
    trt.init_libnvinfer_plugins(None, "")

    def __init__(self, engine_file):
        self.engine_file = engine_file
        self.batch_size = 1
        self.engine = None
        self.context = None
        self.stream = None
        self.bindings = []
        self.outputs = []
        self.out_shapes = []
        self.out_names = []
        self.input_shapes = []
        self.input = None
        self.max_batch_size = 1

    def build(self):
        assert os.path.exists(self.engine_file), "Engine file doesn't exist"

        runtime = trt.Runtime(TrtModel.TRT_LOGGER)
        with open(self.engine_file, 'rb') as engine_file:
            self.engine = runtime.deserialize_cuda_engine(engine_file.read())

        if self.engine is None:
            raise RuntimeError('Unable to load the engine file')

        self.context = self.engine.create_execution_context()
        self.stream = cp.cuda.Stream(non_blocking=True)

        self.max_batch_size = self.engine.get_profile_shape(0, 0)[2][0]
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            if shape[0] == -1:
                shape = (self.max_batch_size,) + shape[1:]

            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            buffer = HostDeviceMem(size, dtype)

            self.bindings.append(buffer.devptr)
            if self.engine.binding_is_input(binding):
                self.input = buffer
                self.input_shapes.append(self.engine.get_binding_shape(binding))
            else:
                self.outputs.append(buffer)
                self.out_shapes.append(self.engine.get_binding_shape(binding))
                self.out_names.append(binding)

        assert self.input is not None

        self.start = cp.cuda.Event()
        self.end = cp.cuda.Event()

    def __del__(self):
        if hasattr(self, 'context'):
            del self.context
        if hasattr(self, 'engine'):
            del self.engine

    def run(self, input=None, deflatten: bool = True, as_dict=False, from_device=False, infer_shape=None):
        if not from_device:
            allocate_place = np.prod(input.shape)
            with self.stream:
                g_img = cp.asarray(input)
                self.input.device[:allocate_place] = cp.asarray(input, order='C').flatten()
                infer_shape = g_img.shape

        self.context.set_binding_shape(0, infer_shape)

        self.start.record(self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.ptr)

        for out in self.outputs:
            out.copy_dtoh_async(self.stream)

        self.end.record(self.stream)
        self.stream.synchronize()
        trt_outputs = [out.host for out in self.outputs]

        if deflatten:
            trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, self.out_shapes)]
        if as_dict:
            return {name: trt_outputs[i][:infer_shape[0]] for i, name in enumerate(self.out_names)}

        return [trt_output[:infer_shape[0]] for trt_output in trt_outputs]