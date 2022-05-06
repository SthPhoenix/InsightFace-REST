#! /bin/bash

IMAGE='insightface-rest'
TAG='v0.7.3.0'

# Change InsightFace-REST logging level (DEBUG,INFO,WARNING,ERROR)
log_level=INFO

# When starting multiple containers this will be port assigned to first container
START_PORT=18081

# Set number of GPU's availiable in your system
n_gpu=1

# Set how many app instances you want to run per GPU, ensure you have enough GPU
# memory for desired number. Try running with n_workers=1 to estimate memory consumption
# per instance.
# Take note: larger number won't speed up single image inference time, it'll increase
# concurrent throughput.
n_workers=1

# Maximum image size (W,H). If your input images has fixed image size set this
# value proportional or equal to it. Otherwise select value based on your
# performance/accuracy needs.
# If input images may have both album/portrait orientation it's recommended to
# set square dimensions, like 640x640 for better accuracy.
# ATTENTION: For TensorRT backend this size currently can't be set during
# runtime.
max_size=640,640

# Force FP16 mode for building TensorRT engines, even if it's not supported.
# Please check that your GPU supports FP16, otherwise performance may drop.
# For GPUs supporting it gives about 2x performance boost.
force_fp16=False


# DET MODELS:
## retinaface_mnet025_v1, retinaface_mnet025_v2, retinaface_r50_v1, centerface
## scrfd_500m_bnkps, scrfd_2.5g_bnkps, scrfd_10g_bnkps
## scrfd_500m_gnkps, scrfd_2.5g_gnkps, scrfd_10g_gnkps
## yolov5l-face, yolov5m-face, yolov5s-face, yolov5n-face, yolov5n-0.5
## Note: SCRFD family models requires input image shape dividable by 32, i.e 640x640, 1024x768.
det_model=scrfd_10g_gnkps

## Maximum batch size for detection model
det_batch_size=1

# REC MODELS:
## None, arcface_r100_v1, glintr100, w600k_r50, w600k_mbf
rec_model=glintr100

## Maximum batch size for recognition model (this value also applies for GA and mask detection models)
rec_batch_size=1


# Mask detection models
## None, mask_detector, mask_detector112
mask_detector=None

# GENDER/AGE MODELS:
## None, genderage_v1
ga_model=None

# Triton Inference Server GRPC uri:port (optional)
# Should be updated when INFERENCE_BACKEND=triton
triton_uri='localhost:8001'

# Default settings for inference requests, can be overridden inside
# request body.

## Return base64 encoded face crops.
return_face_data=False
## Get faces embeddings. Otherwise only bounding boxes will be returned.
extract_embeddings=True
## Estimate gender/age
detect_ga=False
##Face detection probability threshold
det_thresh=0.6


# DEPLOY CONTAINERS

# Create directory to store downloaded models
mkdir -p models


docker build -t $IMAGE:$TAG -f src/Dockerfile_trt src/.

echo "Starting $((n_gpu * n_workers)) workers on $n_gpu GPUs ($n_workers workers per GPU)";
echo "Containers port range: $START_PORT - $(($START_PORT + ($n_gpu) - 1))"


p=0

for i in $(seq 0 $(($n_gpu - 1)) ); do
    device='"device='$i'"';
    port=$((START_PORT + $p));
    name=$IMAGE-gpu$i-trt;

    docker stop $name;
    docker rm $name;
    echo --- Starting container $name  with $device  at port $port;
    ((p++));
    docker run  -p $port:18080\
        --gpus $device\
        -d\
        -e LOG_LEVEL=$log_level\
        -e USE_NVJPEG=False\
        -e PYTHONUNBUFFERED=0\
        -e PORT=18080\
        -e NUM_WORKERS=$n_workers\
        -e INFERENCE_BACKEND=trt\
        -e FORCE_FP16=$force_fp16\
        -e DET_NAME=$det_model\
        -e DET_THRESH=$det_thresh\
        -e REC_NAME=$rec_model\
        -e REC_IGNORE=$rec_ignore\
        -e MASK_DETECTOR=$mask_detector\
        -e MASK_IGNORE=$mask_ignore\
        -e REC_BATCH_SIZE=$rec_batch_size\
        -e DET_BATCH_SIZE=$det_batch_size\
        -e GA_NAME=$ga_model\
        -e GA_IGNORE=$ga_ignore\
        -e TRITON_URI=$triton_uri\
        -e KEEP_ALL=True\
        -e MAX_SIZE=$max_size\
        -e DEF_RETURN_FACE_DATA=$return_face_data\
        -e DEF_EXTRACT_EMBEDDING=$extract_embeddings\
        -e DEF_EXTRACT_GA=$detect_ga\
        -e DEF_API_VER='2'\
        -v $PWD/models:/models\
        -v $PWD/src/api_trt:/app\
        --health-cmd='curl -f http://localhost:18080/info || exit 1'\
        --health-interval=1m\
        --health-timeout=10s\
        --health-retries=3\
        --name=$name\
        $IMAGE:$TAG
done

