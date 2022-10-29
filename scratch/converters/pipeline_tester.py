from modules.face_model import FaceAnalysis
import logging
import cv2
import time

logging.basicConfig(
    level='DEBUG',
    format='%(asctime)s %(levelname)s - %(message)s',
    datefmt='[%H:%M:%S]',
)


model = FaceAnalysis(max_size=[640, 640], backend_name='trt', det_name='retinaface_mnet025_v1', max_rec_batch_size=64)

# Warmup
image_path = 'test_images/Stallone.jpg'
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
faces = model.get(image)


#Test
iters = 50

t0 = time.time()
image_path = 'test_images/lumia.jpg'
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
for i in range(iters):
    tf0 = time.time()
    faces = model.get(image)
    tf1 = time.time()
    logging.debug(f"Full detection  took: {tf1 - tf0}")
t1 = time.time()

#print(faces)

print(f'Took {t1 - t0} s. ({iters / (t1 - t0)} im/sec)')
