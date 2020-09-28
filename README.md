## UPDATE 2020-09-28
Added GPU support and switched to FastApi, more details in changelog.


# InsightFace-REST

InsightFace REST API for easy deployment of face recognition services.
Code is heavily based on API [code](https://github.com/deepinsight/insightface/tree/master/python-package)
in official DeepInsight InsightFace [repository](https://github.com/deepinsight/insightface). 

This repository provides source code for building face recognition REST API
and Dockerfiles for fast deployment.

## Prerequesites:


1. MXNet
2. PyTorch
3. OpenCV
4. FastApi
5. Docker


## Usage:
### `/extract` endpoint

Extract endpoint accepts list of images and return faces bounding boxes with corresponding 
embeddings.

API accept JSON in following format:
```
{
  "images":{
      "data":[
          base64_encoded_image1,  
          base64_encoded_image2
      ]
  },
  "max_size":[640,480]
}
```

Where `max_size` is maximum image dimension, images with dimensions greater than `max_size`
will be downsized to provided value.

If `max_size` is set to **0**, image won't be resized.

To call API from Python you can use following sample code:

```python
import os
import json
import base64
import requests

def file2base64(path):
    with open(path, mode='rb') as fl:
        encoded = base64.b64encode(fl.read()).decode('ascii')
        return encoded


def extract_vecs(ims,max_size=[640,480]):
    target = [file2base64(im) for im in ims]
    req = {"images": {"data": target},"max_size":max_size}
    resp = requests.post('http://localhost:18081/extract', json=req)
    data = resp.json()
    return data
    
images_path = 'src/api/test_images'
images = os.path.listdir(images_path)
data = extract_vecs(images)

```
Response is in following format:

```json
[
    [
        {"vec": [0.322431242,0.53545632,], "det": 0, "prob": 0.999, "bbox": [100,100,200,200]},
        {"vec": [0.235334567,-0.2342546,], "det": 1, "prob": 0.998, "bbox": [200,200,300,300]},
    ],
    [
        {"vec": [0.322431242,0.53545632,], "det": 0, "prob": 0.999, "bbox": [100,100,200,200]},
        {"vec": [0.235334567,-0.2342546,], "det": 1, "prob": 0.998, "bbox": [200,200,300,300]},
    ]
]
```
First level is list in order the images were sent, second level are faces detected per each image as 
dictionary containing face embedding, bounding box, detection probability and detection number.  

## Run with Docker:

1. Clone repo.
2. Execute `deploy.sh` from repo's root.
3. Go to http://localhost:18081 to access documentation and try API

If you have multiple GPU's with enough GPU memory you can try running multiple containers by
editing *n_gpu* and *n_con* parameters in `deploy.sh`.

You would need load balancer like HAProxy to work with multiple containers,
example HAProxy config will be added later.


## Changelist:
### 2020-09-28
- REST API code refactored to FastAPI
- Detection/Recognition code is now based on official Insightface Python package.
- TensorFlow MTCNN replaced with PyTorch version
- Added RetinaFace detector
- Added InsightFace gender/age detector
- Added support for GPU inference
- Resize function refactored for fixed image proportions (significant speed increase and memory usage optimization)


