#! /bin/bash

TAG=$(date +%s)

sudo docker build \
    -f deploy/DockerFile \
    --tag matchadl:$TAG \
    .

sudo docker image tag matchadl:$TAG matchadl:latest

sudo docker image tag matchadl:latest liseda-cluster.lasige.di.fc.ul.pt:5000/matchadl

sudo docker push liseda-cluster.lasige.di.fc.ul.pt:5000/matchadl