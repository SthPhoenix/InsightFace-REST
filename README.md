# InsightFace-REST

InsightFace REST API for easy deployment of face recognition services.
Code is heavily based on API [code](https://github.com/deepinsight/insightface/tree/master/src/api)
in official DeepInsight InsightFace [repository](https://github.com/deepinsight/insightface). 

This repository provides source code for building face recognition REST API
and Dockerfiles for fast deployment.

Currently this repository contains Dockerfiles for CPU inference.


## Prerequesites:

1. Tensorflow
2. MXNet
3. OpenCV
4. Flask
5. *Docker (optionally)*


## Usage:

API accept requests in JSON in following format:
```
{
  "images":{
      "data":[
          base64_encoded_image1,  base64_encoded_image2
      ]
  },
  "max_size":640
}
```

Where `max_size` is maximum image diemension, images with diemensions greater than `max_size` 
will be downsized to provided value, i.e if `max_size` is **640** (default value), then 1024x768 image will
be resized to 640x384.

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


def extract_vecs(ims,max_size=640):
    target = [file2base64(im) for im in ims]
    req = {"images": {"data": target},"max_size":max_size}
    resp = requests.post('/extract', json=req)
    data = resp.json()
    return data
    
images_path = 'src/api/test_images'
images = os.path.listdir(images_path)
data = exctract_vecs(images)

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


## Run:
1. Clone repo
2. Download model **LResNet100E-IR,ArcFace@ms1m-refine-v2** from 
DeepInsight [Model Zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo)
([dropbox](https://www.dropbox.com/s/tj96fsm6t6rq8ye/model-r100-arcface-ms1m-refine-v2.zip?dl=0))
3. Unzip downloaded model to `src/api/models`
2. Run `src/api/app.py`

## Run with Docker:

1. Follow steps 1-3 from above.
2. Execute `build.sh` from `docker_tf_opencv` folder to build base image
`tensorflow-opencv:preconf`
3. Execute `deploy.sh` from repo root folder to build  and start `insightface-rest:v0.1` image


## Known issues:
1. Docker container requires at least 4GB RAM.
2. MTCNN thresholds was a bit tuned for faster inference, but it has 
side effect of producing wrong face probability scores above 0.0-1.0 range.
