#! /bin/bash

IMAGE='tensorflow-opencv'
TAG='preconf'

docker rm $(docker stop $(docker ps -a -q --filter ancestor=$IMAGE:$TAG --format="{{.ID}}"))
docker image rm $IMAGE:$TAG
docker build -t $IMAGE:$TAG .


