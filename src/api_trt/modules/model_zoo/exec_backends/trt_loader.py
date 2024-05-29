import cupy as cp
import numpy as np
import os
import tensorrt as trt
import cupyx


class HostDeviceMem(object):
    """
    A simple helper class for managing host and device memory buffers.

    Attributes:
        size (int): The size of the buffer in bytes.
        dtype (numpy.dtype): The data type of the elements in the buffer.
        host (cupy.ndarray): A pinned host array for CPU-side operations.
        device (cupy.ndarray): A device array for GPU-side operations.

    Methods:
        __str__(): Returns a string representation of the host and device arrays.
        __repr__(): Returns the string representation of the object.
        @property nbytes(): Returns the size of the buffer in bytes.
        @property hostptr(): Returns the pointer to the host array data.
        @property devptr(): Returns the device pointer for the device array data.
        copy_htod_async(stream): Copies data from the host to the device asynchronously using a given CUDA stream.
        copy_dtoh_async(stream): Copies data from the device to the host asynchronously using a given CUDA stream.
    """

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
        """
        Copies data from the host to the device asynchronously using a given CUDA stream.

        Args:
            stream (cupy.cuda.Stream): The CUDA stream for asynchronous operations.
        """
        self.device.data.copy_from_host_async(self.hostptr, self.nbytes, stream)

    def copy_dtoh_async(self, stream):
        """
        Copies data from the device to the host asynchronously using a given CUDA stream.

        Args:
            stream (cupy.cuda.Stream): The CUDA stream for asynchronous operations.
        """
        self.device.data.copy_to_host_async(self.hostptr, self.nbytes, stream)


class TrtModel(object):
    """
    A wrapper for TensorRT engine that provides methods for loading and executing the model.

    Attributes:
        TRT_LOGGER (tensorrt.Logger): The logger instance used by TensorRT.
        engine_file (str): The file path to the serialized TensorRT engine.
        batch_size (int): The batch size of the input data.
        engine (tensorrt.ICudaEngine): The deserialized TensorRT engine.
        context (tensorrt.IExecutionContext): The execution context for running inference on the engine.
        stream (cupy.cuda.Stream): A CUDA stream for asynchronous operations.
        bindings (list): A list of pointers to input and output buffers.
        outputs (list): A list of HostDeviceMem objects representing the model outputs.
        out_shapes (list): A list of shapes of the model outputs.
        out_names (list): A list of names of the model outputs.
        input_shapes (list): A list of shapes of the model inputs.
        input (HostDeviceMem): The HostDeviceMem object representing the input buffer.
        max_batch_size (int): The maximum batch size supported by the engine.

    Methods:
        __init__(engine_file): Initializes the TrtModel instance with the path to the TensorRT engine file.
        build(): Loads and deserializes the TensorRT engine, creates an execution context, and initializes buffers.
        run(input=None, deflatten: bool = True, as_dict=False, from_device=False, infer_shape=None): Runs inference on the model with given input data.
    """

    TRT_LOGGER = trt.Logger()
    trt.init_libnvinfer_plugins(None, "")

    def __init__(self, engine_file):
        self.engine_file = engine_file
        self.batch_size = 1
        self.trt10 = False
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
        """
        Loads and deserializes the TensorRT engine, creates an execution context, and initializes buffers.
        """
        assert os.path.exists(self.engine_file), "Engine file doesn't exist"

        runtime = trt.Runtime(TrtModel.TRT_LOGGER)
        with open(self.engine_file, 'rb') as engine_file:
            self.engine = runtime.deserialize_cuda_engine(engine_file.read())

        if self.engine is None:
            raise RuntimeError('Unable to load the engine file')

        self.trt10 = not hasattr(self.engine, "num_bindings")
        self.context = self.engine.create_execution_context()
        self.stream = cp.cuda.Stream(non_blocking=True)



        if self.trt10:
            self.input_tensor_name = self.engine.get_tensor_name(0)
            self.max_batch_size = self.engine.get_tensor_profile_shape(self.input_tensor_name, 0)[2][0]
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(tensor_name)
                if shape[0] == -1:
                    shape = (self.max_batch_size,) + shape[1:]
                size =  trt.volume(shape)
                dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                buffer = HostDeviceMem(size, dtype)
                self.bindings.append(buffer.devptr)
                if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    self.input = buffer
                    self.input_shapes.append(self.engine.get_tensor_shape(tensor_name))
                else:
                    self.outputs.append(buffer)
                    self.out_shapes.append(self.engine.get_tensor_shape(tensor_name))
                    self.out_names.append(tensor_name)
        else:
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
        """
        Deletes the execution context and engine when the TrtModel instance is destroyed.
        """
        if hasattr(self, 'context'):
            del self.context
        if hasattr(self, 'engine'):
            del self.engine

    def run(self, input=None, deflatten: bool = True, as_dict=False, from_device=False, infer_shape=None):
        """
        Runs inference on the model with given input data.

        Args:
            input (numpy.ndarray): The input data to be fed into the model.
            deflatten (bool): Whether to reshape the output tensors back to their original shapes. Defaults to True.
            as_dict (bool): Whether to return the outputs as a dictionary with names as keys. Defaults to False.
            from_device (bool): Whether the input data is already on the device. Defaults to False.
            infer_shape (tuple): The shape of the input data if it's different from the default one.

        Returns:
            list or dict: A list of output tensors or a dictionary with names as keys, depending on the value of `as_dict`.
        """

        if not from_device:
            allocate_place = np.prod(input.shape)
            with self.stream:
                g_img = cp.asarray(input)
                self.input.device[:allocate_place] = cp.asarray(input, order='C').flatten()
                infer_shape = g_img.shape


        if self.trt10:
            self.context.set_input_shape(self.engine.get_tensor_name(0), infer_shape)
        else:
            self.context.set_binding_shape(0, infer_shape)

        self.start.record(self.stream)


        if self.trt10:
            # Setup tensor address
            for i in range(self.engine.num_io_tensors):
                self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])

        # Run inference
        if self.trt10:
            self.context.execute_async_v3(stream_handle=self.stream.ptr)
        else:
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
