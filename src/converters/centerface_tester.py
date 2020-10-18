import time
import cv2
from configs import Configs
from exec_backends.trt_backend import CenterFaceInfer
from detectors.centerface import CenterFace
from imagedata import  ImageData



config = Configs(models_dir='/models')

iters = 100

backend = 'plan'
model_name = 'centerface'
input_shape = [1024, 768]

model_dir, model_path = config.build_model_paths(model_name, backend)

tt0 = time.time()
image = cv2.imread('test_images/Stallone.jpg.jpg', cv2.IMREAD_COLOR)
image = ImageData(image, input_shape)
image.resize_image(mode='pad')
tt1 = time.time()
print(f"Preparing image took: {tt1 - tt0}")

backend = CenterFaceInfer(rec_name=model_path)
backend.prepare()
cf = CenterFace(inference_backend=backend)

t0 = time.time()
for i in range(iters):

    tf0 = time.time()
    detections = cf(image.transformed_image, input_shape[1], input_shape[0], threshold=0.35)
    tf1 = time.time()
    print(f"Full detection  took: {tf1 - tf0}")
t1 = time.time()
print(f'Took {t1 - t0} s. ({iters / (t1 - t0)} im/sec)')

for det in detections[0]:
    pt1 = tuple(map(int, det[0:2]))
    pt2 = tuple(map(int, det[2:4]))
    cv2.rectangle(image.transformed_image, pt1, pt2, (0, 255, 0), 1)
cv2.imwrite('test_centerface.jpg', image.transformed_image)