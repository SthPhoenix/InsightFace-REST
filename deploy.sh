#! /bin/bash

IMAGE='insightface-rest'
TAG='v0.1.2'

docker rm $(docker stop $(docker ps -a -q --filter ancestor=$IMAGE:$TAG --format="{{.ID}}"))
docker image rm $IMAGE:$TAG

docker build -t $IMAGE:$TAG .

docker run  -p 18080:18080\
    -e PYTHONUNBUFFERED=0\
    -e PORT=18080\
    -e MAX_SIZE=640\
     $IMAGE:$TAG

