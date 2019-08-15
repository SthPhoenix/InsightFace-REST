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
Currently you can see usage examples in `src/api/tests.py`, API documentation
will be added later.

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
