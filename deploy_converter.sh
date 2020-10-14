#! /bin/bash

IMAGE='insightface-converter'
TAG='v0.1'

docker rm $(docker stop $(docker ps -a -q --filter ancestor=$IMAGE:$TAG --format="{{.ID}}"))

#docker image rm $IMAGE:$TAG
docker build -t $IMAGE:$TAG -f src/Dockerfile.converter src/.

docker run\
    --gpus all\
    -it\
    -e PYTHONUNBUFFERED=0\
    -v $PWD/src/converters:/app\
    -v $PWD/models:/models\
    --name=$IMAGE\
    $IMAGE:$TAG

