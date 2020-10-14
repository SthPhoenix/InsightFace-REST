#! /bin/bash

IMAGE='insightface-rest'
TAG='v0.5'

# If set to 1 slows down startup time, but might increase inference speed
AUTOTUNE=1

# NTHREADS=1 AND --cpuset-cpus=0-0 shows best performance, but you can try different values in your environment
WORKER_NTHREADS=1

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

docker build -t $IMAGE:$TAG -f src/Dockerfile src/.

p=0
for i in $(seq 0 $(($n_gpu - 1)) ); do
    device='"device='$i'"';
    for j in $(seq 0 $(($n_con - 1))); do
        port=$((START_PORT + $p));
        name=$IMAGE-g$i-$j;

        docker stop $name;
        docker rm $name;

        cpu_from=$(( (p) * cpus ))
        cpu_to=$(( cpu_from + cpus - 1 ))
        CPU_SET=$cpu_from-$cpu_to
        echo --- Starting container $name  with $device  at port $port CPU set=$CPU_SET;
        ((p++));
        docker run  -p $port:18080\
            --gpus $device\
            -d\
            -e PYTHONUNBUFFERED=0\
            -e PORT=18080\
            -e MXNET_CUDNN_AUTOTUNE_DEFAULT=$AUTOTUNE\
            -e MXNET_CPU_WORKER_NTHREADS=$WORKER_NTHREADS\
            -e MXNET_ENGINE_TYPE=ThreadedEnginePerDevice\
            -e MXNET_MKLDNN_CACHE_NUM=0\
            -e DET_NAME=retinaface_mnet025_v1\
            -e REC_NAME=arcface_r100_v1\
            -e GA_NAME=genderage_v1\
            -e GA_IGNORE=True\
            -e KEEP_ALL=True\
            -e MIN_FACE_SIZE=20\
            -e MTCNN_FACTOR=0.700\
            -e REC_IGNORE=False\
            -e MAX_SIZE=1280,800\
            -e DEF_RETURN_FACE_DATA=False\
            -e DEF_EXTRACT_EMBEDDING=True\
            -e DEF_EXTRACT_GA=False\
            -e DEF_API_VER='1'\
            -v $PWD/models/mxnet:/root/.insightface/models\
            --name=$name\
            --cpuset-cpus=$CPU_SET\
            $IMAGE:$TAG
    done
done

#DET MODELS:
#mtcnn, retinaface_mnet025_v1, retinaface_mnet025_v2, retinaface_r50_v1