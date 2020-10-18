
import cv2
from imagedata import ImageData

import time

from configs import Configs

from detectors.retina import FaceDetector
from exec_backends.trt_backend import RetinaInfer as RetinaInferTRT
from exec_backends.onnxrt_backend import RetinaInfer as RetinaInferORT


'''
ATTENTION!!! This script is for testing purposes only. Work in progress.
'''


config = Configs(models_dir='/models')

iters = 100

#model_name = 'retinaface_mnet025_v2'
model_name = 'retinaface_r50_v1'
input_shape = [1024, 768]

# 'plan' for TensorRT or 'onnx' for ONNX
backend = 'plan'

retina_backends = {
    'onnx': RetinaInferORT,
    'plan': RetinaInferTRT
}

model_dir, model_path = config.build_model_paths(model_name, backend)

retina_backend = retina_backends[backend](rec_name=model_path)

detector = FaceDetector(inference_backend=retina_backend)

detector.prepare(fix_image_size=(input_shape[1], input_shape[0]))

tt0 = time.time()
image = cv2.imread('test_images/Stallone.jpg.jpg', cv2.IMREAD_COLOR)
image = ImageData(image, input_shape)
image.resize_image(mode='pad')
tt1 = time.time()
print(f"Preparing image took: {tt1 - tt0}")

t0 = time.time()
for i in range(iters):

    tf0 = time.time()
    detections = detector.detect(image.transformed_image, threshold=0.3)
    tf1 = time.time()
    print(f"Full detection  took: {tf1 - tf0}")

t1 = time.time()
print(f'Took {t1 - t0} s. ({iters / (t1 - t0)} im/sec)')

for det in detections[0]:
    if det[4] > 0.5:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    # print(res[4])
    pt1 = tuple(map(int, det[0:2]))
    pt2 = tuple(map(int, det[2:4]))
    cv2.rectangle(image.transformed_image, pt1, pt2, color, 1)

cv2.imwrite('test_retina.jpg', image.transformed_image)


