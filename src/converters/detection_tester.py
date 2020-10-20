import os
import cv2
import time
import logging

from modules.model_zoo.getter import get_model
from modules.imagedata import ImageData


'''
ATTENTION!!! This script is for testing purposes only. Work in progress.

This demo script is intended for testing face detection modules.
Following steps will be executed upon launch:

1. Download official MXNet Retinaface model
2. Convert model to ONNX, applying fixes to MXNet model and resulting ONNX
3. Reshape ONNX inputs/outputs to specified image W, H
4. Build TensorRT engine from ONNX
5. Run inference with TensorRT

If Centerface model is selected, it will be downloaded and than steps 3-4 applied.

'''

logging.basicConfig(
    level='INFO',
    format='%(asctime)s %(levelname)s - %(message)s',
    datefmt='[%H:%M:%S]',
)

iters = 10
model_name = 'retinaface_mnet025_v2'
#model_name = 'retinaface_r50_v1'
#model_name = 'centerface'

backend = 'trt'
im_size = [640, 480]

detector = get_model(model_name, backend, im_size=im_size, root_dir='/models')
detector.prepare(nms=0.35)
input_shape = detector.input_shape[2:][::-1]
logging.info(f"Images will be resized to WxH : {input_shape}")

tt0 = time.time()
image = cv2.imread('test_images/lumia.jpg', cv2.IMREAD_COLOR)
image = ImageData(image, input_shape)
image.resize_image(mode='stretch')
tt1 = time.time()

print(f"Preparing image took: {tt1 - tt0}")

# Since we are using Numba JIT compilation, first run will cause
# compilation on NMS module
logging.info('Warming up NMS jit...')
detections = detector.detect(image.transformed_image, threshold=0.1)
logging.info('Warming up complete!')



det_count = 0

t0 = time.time()
for i in range(iters):
    tf0 = time.time()
    detections = detector.detect(image.transformed_image, threshold=0.35)
    tf1 = time.time()
    det_count = len(detections[0])
    logging.debug(f"Full detection  took: {tf1 - tf0}")

t1 = time.time()
print(f'Took {t1 - t0} s. ({iters / (t1 - t0)} im/sec)')
print(f"Total faces detected: {det_count}")

for det in detections[0]:
    if det[4] > 0.5:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    # print(res[4])
    pt1 = tuple(map(int, det[0:2]))
    pt2 = tuple(map(int, det[2:4]))
    cv2.rectangle(image.transformed_image, pt1, pt2, color, 1)

cv2.imwrite(f'{model_name}.jpg', image.transformed_image)
