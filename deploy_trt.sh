#! /bin/bash

IMAGE='insightface-rest'
TAG='v0.5.5'

#When starting multiple containers this will be port assigned to first container
START_PORT=18081

#Set number of GPU's availiable in your system
n_gpu=1
#Set how many containers you want to run per GPU
n_con=1


# Calculate number of CPU cores required to run desired number of containers.
# By default every container will be assigned to 1 core.
ncpu=$((n_gpu * n_con))
cpus=$(( ncpu / (n_gpu * n_con) ))


#Create directory to store downloaded models
mkdir -p models

echo "Starting $((n_gpu * n_con)) containers on $n_gpu GPUs ($n_con containers per GPU)";
echo "Containers port range: $START_PORT - $(($START_PORT + ($n_gpu* $n_con) - 1))"

docker build -t $IMAGE:$TAG -f src/Dockerfile_trt src/.

p=0
for i in $(seq 0 $(($n_gpu - 1)) ); do
    device='"device='$i'"';
    for j in $(seq 0 $(($n_con - 1))); do

        port=$((START_PORT + $p));
        name=$IMAGE-g$i-$j-trt;

        docker stop $name;
        docker rm $name;
        echo --- Starting container $name  with $device  at port $port;
        ((p++));
        docker run  -p $port:18080\
            --gpus $device\
            -d\
            -e PYTHONUNBUFFERED=0\
            -e PORT=18080\
            -e DET_NAME=retinaface_mnet025_v2\
            -e REC_NAME=arcface_r100_v1\
            -e GA_NAME=genderage_v1\
            -e GA_IGNORE=False\
            -e KEEP_ALL=True\
            -e MIN_FACE_SIZE=20\
            -e MTCNN_FACTOR=0.700\
            -e REC_IGNORE=False\
            -e MAX_SIZE=640,640\
            -e DEF_RETURN_FACE_DATA=False\
            -e DEF_EXTRACT_EMBEDDING=True\
            -e DEF_EXTRACT_GA=False\
            -e DEF_API_VER='1'\
            -v $PWD/models:/models\
            -v $PWD/src/api_trt:/app\
            --name=$name\
            $IMAGE:$TAG
    done
done

#DET MODELS:
#mtcnn, retinaface_mnet025_v1, retinaface_mnet025_v2, retinaface_r50_v1, centerface