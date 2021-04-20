#!/bin/sh
docker stop medscan
docker container rm medscan
docker build -t medscan .
docker create -p 5000:5000 --name medscan medscan
docker start medscan
