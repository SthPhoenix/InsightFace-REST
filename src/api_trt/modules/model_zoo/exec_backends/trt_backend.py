import os
import cv2
import numpy as np
import logging
import cupy as cp
import time
from api_trt.modules.model_zoo.exec_backends.trt_loader import TrtModel
from api_trt.logger import logger

def _normalize_on_device(input, stream, out, mean=0., std=1., swapRB=True):
    """
    Normalizes the input image on the device using CUDA.

    Args:
        input (np.ndarray): The input image as a NumPy array.
        stream (cupy.cuda.Stream): The CUDA stream for memory operations.
        out (cupy.ndarray): The output array to store the normalized image.
        mean (float, optional): The mean value for normalization. Defaults to 0.
        std (float, optional): The standard deviation for normalization. Defaults to 1.
        swapRB (bool, optional): Whether to swap the red and blue channels. Defaults to True.

    Returns:
        tuple: A tuple containing the shape of the normalized image.
    """
    allocate_place = np.prod(input.shape)
    with stream:
        g_img = cp.asarray(input)
        if swapRB:
            g_img = g_img[..., ::-1]
        g_img = cp.transpose(g_img, (0, 3, 1, 2))
        g_img = cp.subtract(g_img, mean, dtype=cp.float32)
        out.device[:allocate_place] = cp.multiply(g_img, 1 / std).flatten()
    return g_img.shape


def _normalize_on_device_masks(input, stream, out):
    """
    Normalizes the input mask image on the device using CUDA.

    Args:
        input (np.ndarray): The input mask image as a NumPy array.
        stream (cupy.cuda.Stream): The CUDA stream for memory operations.
        out (cupy.ndarray): The output array to store the normalized mask image.

    Returns:
        tuple: A tuple containing the shape of the normalized mask image.
    """
    allocate_place = np.prod(input.shape)
    with stream:
        g_img = cp.asarray(input)
        g_img = g_img[..., ::-1]
        g_img = cp.multiply(g_img, 1 / 127.5, dtype=cp.float32)
        out.device[:allocate_place] = cp.subtract(g_img, 1.).flatten()
    return g_img.shape


class Arcface:
    """
    Class for performing face recognition using an ArcFace model loaded from a TensorRT engine.

    Args:
        rec_name (str): The file path to the TensorRT engine for the ArcFace model.
        input_mean (float, optional): The mean value for input normalization. Defaults to 0.
        input_std (float, optional): The standard deviation for input normalization. Defaults to 1.
        swapRB (bool, optional): Whether to swap the red and blue channels in input images. Defaults to False.

    Attributes:
        rec_model (TrtModel): The TensorRT model instance.
        input_mean (float): The mean value for input normalization.
        input_std (float): The standard deviation for input normalization.
        input_shape (tuple): The shape of the expected input images.
        swapRB (bool): Whether to swap the red and blue channels in input images.
        max_batch_size (int): The maximum batch size supported by the TensorRT engine.
        stream (cupy.cuda.Stream): The CUDA stream for memory operations.
        input_ptr (cupy.ndarray): The input array pointer for the TensorRT model.

    Methods:
        prepare(): Warm up the ArcFace TensorRT engine by running a dummy inference.
        get_embedding(face_img): Get the embedding vector for a given face image.
    """
    def __init__(self, rec_name: str = '/models/trt-engines/arcface_r100_v1/arcface_r100_v1.plan',
                 input_mean: float = 0.,
                 input_std: float = 1.,
                 swapRB=False,
                 **kwargs):
        self.rec_model = TrtModel(rec_name)
        self.input_mean = input_mean
        self.input_std = input_std
        self.input_shape = None
        self.swapRB=swapRB
        self.max_batch_size = 1
        self.stream = None
        self.input_ptr = None

    # warmup
    def prepare(self, **kwargs):
        """
        Warm up the ArcFace TensorRT engine by running a dummy inference.
        """
        logger.info("Warming up ArcFace TensorRT engine...")
        self.rec_model.build()
        self.input_shape = self.rec_model.input_shapes[0]
        self.max_batch_size = self.rec_model.max_batch_size
        self.stream = self.rec_model.stream
        self.input_ptr = self.rec_model.input
        if self.input_shape[0] == -1:
            self.input_shape = (1,) + self.input_shape[1:]

        self.rec_model.run(np.zeros(self.input_shape, np.float32))
        logger.info(
            f"Engine warmup complete! Expecting input shape: {self.input_shape}. Max batch size: {self.max_batch_size}")

    def get_embedding(self, face_img):
        """
        Get the embedding vector for a given face image.

        Args:
            face_img (np.ndarray or list of np.ndarray): The input face image as a NumPy array or a list of such arrays.

        Returns:
            np.ndarray: The embedding vector for the given face image(s).
        """
        if not isinstance(face_img, list):
            face_img = [face_img]
        face_img = np.stack(face_img)

        t0 = time.perf_counter()
        infer_shape = _normalize_on_device(face_img, self.stream, self.input_ptr, mean=self.input_mean,
                                           std=self.input_std, swapRB=self.swapRB)

        embeddings = self.rec_model.run(deflatten=True, from_device=True, infer_shape=infer_shape)[0]
        took = time.perf_counter() - t0
        logger.debug(f'Rec inference cost: {took*1000:.3f} ms.')
        return embeddings


class FaceGenderage:
    """
    Class for performing gender and age estimation using a GenderAge model loaded from a TensorRT engine.

    Args:
        rec_name (str): The file path to the TensorRT engine for the GenderAge model.

    Attributes:
        rec_model (TrtModel): The TensorRT model instance.
        input_shape (tuple): The shape of the expected input images.

    Methods:
        prepare(): Warm up the GenderAge TensorRT engine by running a dummy inference.
        get(face_img): Get the gender and age for a given face image.
    """
    def __init__(self, rec_name: str = '/models/trt-engines/genderage_v1/genderage_v1.plan', **kwargs):
        self.rec_model = TrtModel(rec_name)
        self.input_shape = None

    # warmup
    def prepare(self, **kwargs):
        """
        Warm up the GenderAge TensorRT engine by running a dummy inference.
        """
        logger.info("Warming up GenderAge TensorRT engine...")
        self.rec_model.build()
        self.input_shape = self.rec_model.input_shapes[0]
        self.max_batch_size = self.rec_model.max_batch_size
        if self.input_shape[0] == -1:
            self.input_shape = (1,) + self.input_shape[1:]

        self.rec_model.run(np.zeros(self.input_shape, np.float32))
        logger.info(
            f"Engine warmup complete! Expecting input shape: {self.input_shape}. Max batch size: {self.max_batch_size}")

    def get(self, face_img):
        """
        Get the gender and age for a given face image.

        Args:
            face_img (np.ndarray or list of np.ndarray): The input face image as a NumPy array or a list of such arrays.

        Returns:
            list of tuples: A list containing tuples of gender and age for each input face image.
        """
        if not isinstance(face_img, list):
            face_img = [face_img]

        face_img = np.stack(face_img)
        imgs = face_img.copy()

        if not face_img[0].shape == (3, 112, 112):
            imgs = imgs[..., ::-1]
            imgs = np.transpose(imgs, (0, 3, 1, 2))

        _ga = []
        ret = self.rec_model.run(imgs, deflatten=True)[0]
        for e in ret:
            e = np.expand_dims(e, axis=0)
            g = e[:, 0:2].flatten()
            gender = np.argmax(g)
            a = e[:, 2:202].reshape((100, 2))
            a = np.argmax(a, axis=1)
            age = int(sum(a))
            _ga.append((gender, age))
        return _ga


class MaskDetection:
    """
    Class for performing mask detection using a Mask Detection model loaded from a TensorRT engine.

    Args:
        rec_name (str): The file path to the TensorRT engine for the Mask Detection model.

    Attributes:
        rec_model (TrtModel): The TensorRT model instance.
        input_shape (tuple): The shape of the expected input images.

    Methods:
        prepare(): Warm up the mask detection TensorRT engine by running a dummy inference.
        get(face_img): Get the mask and no-mask probabilities for a given face image.
    """
    def __init__(self, rec_name: str = '/models/trt-engines/mask_detection/mask_detection.plan', **kwargs):
        self.rec_model = TrtModel(rec_name)
        self.input_shape = None

    # warmup
    def prepare(self, **kwargs):
        """
        Warm up the mask detection TensorRT engine by running a dummy inference.
        """
        logger.info("Warming up mask detection TensorRT engine...")
        self.rec_model.build()
        self.input_shape = self.rec_model.input_shapes[0]
        self.max_batch_size = self.rec_model.max_batch_size
        self.stream = self.rec_model.stream
        self.input_ptr = self.rec_model.input

        if self.input_shape[0] == -1:
            self.input_shape = (1,) + self.input_shape[1:]

        self.rec_model.run(np.zeros(self.input_shape, np.float32))
        logger.info(
            f"Mask detection engine warmup complete! Expecting input shape: {self.input_shape}. Max batch size: {self.max_batch_size}")

    def get(self, face_img):
        """
        Get the mask and no-mask probabilities for a given face image.

        Args:
            face_img (np.ndarray or list of np.ndarray): The input face image as a NumPy array or a list of such arrays.

        Returns:
            list of tuples: A list containing tuples of mask and no-mask probabilities for each input face image.
        """
        if not isinstance(face_img, list):
            face_img = [face_img]

        if not self.input_shape[1:3] == (112, 112):
            for i, img in enumerate(face_img):
                img = cv2.resize(img, (224, 224))
                face_img[i] = img

        face_img = np.stack(face_img)

        _mask = []
        infer_shape = _normalize_on_device_masks(face_img, self.stream, self.input_ptr)
        ret = self.rec_model.run(deflatten=True, from_device=True, infer_shape=infer_shape)[0]
        for e in ret:
            mask = e[0]
            no_mask = e[1]
            _mask.append((mask, no_mask))
        return _mask


class DetectorInfer:
    """
    Class for performing face detection using a detector model loaded from a TensorRT engine.

    Args:
        model (str): The file path to the TensorRT engine for the detector model.
        output_order (list, optional): The order of output tensors as they appear in the model. Defaults to None.

    Attributes:
        rec_model (TrtModel): The TensorRT model instance.
        model_name (str): The name of the detector model file.
        stream (cupy.cuda.Stream): The CUDA stream for memory operations.
        input_ptr (cupy.ndarray): The input array pointer for the TensorRT model.
        input_shape (tuple): The shape of the expected input images.
        output_order (list): The order of output tensors as they appear in the model.

    Methods:
        prepare(): Warm up the face detector TensorRT engine by running a dummy inference.
        run(input=None, from_device=False, infer_shape=None): Run the detector on input images and return the results.
    """
    def __init__(self, model='/models/trt-engines/centerface/centerface.plan',
                 output_order=None, **kwargs):

        self.rec_model = TrtModel(model)
        self.model_name = os.path.basename(model)
        self.stream = None
        self.input_ptr = None
        self.input_shape = None
        self.output_order = output_order

    # warmup
    def prepare(self, **kwargs):
        """
        Warm up the face detector TensorRT engine by running a dummy inference.
        """
        logger.info(f"Warming up face detector TensorRT engine...")
        self.rec_model.build()
        self.input_shape = self.rec_model.input_shapes[0]
        self.out_shapes = self.rec_model.out_shapes
        self.input_dtype = np.uint8
        self.max_batch_size = self.rec_model.max_batch_size

        self.stream = self.rec_model.stream
        self.input_ptr = self.rec_model.input

        if self.input_shape[0] == -1:
            self.input_shape = (1,) + self.input_shape[1:]

        if not self.output_order:
            self.output_order = self.rec_model.out_names
        self.rec_model.run(np.zeros(self.input_shape, np.float32))
        logger.info(f"Engine warmup complete! Expecting input shape: {self.input_shape}")

    def run(self, input=None, from_device=False, infer_shape=None, **kwargs):
        """
        Run the detector on input images and return the results.

        Args:
            input (np.ndarray or list of np.ndarray): The input image as a NumPy array or a list of such arrays.
            from_device (bool, optional): Whether to run the inference from device memory. Defaults to False.
            infer_shape (tuple, optional): The shape for inference. Defaults to None.

        Returns:
            list: A list containing the results of the detector inference.
        """
        net_out = self.rec_model.run(input, deflatten=True, as_dict=True, from_device=from_device,
                                     infer_shape=infer_shape)
        net_out = [net_out[e] for e in self.output_order]
        return net_out
