#! /bin/bash

TAG=$(date +%s)

sudo docker build \
    -f deploy/DockerFile \
    --tag matchadl:$TAG \
    .

sudo docker image tag matchadl:$TAG matchadl:latest

sudo docker image tag matchadl:latest liscluster:5000/matchadl

sudo docker push liscluster:5000/matchadl