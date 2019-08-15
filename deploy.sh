#! /bin/bash

IMAGE='insightface-rest'
TAG='v0.1'

docker rm $(docker stop $(docker ps -a -q --filter ancestor=$IMAGE:$TAG --format="{{.ID}}"))
docker image rm $IMAGE:$TAG

docker build -t $IMAGE:$TAG .

docker run  -p 6000:6000 $IMAGE:$TAG

