#!/bin/bash
docker build -f compose/$1/Dockerfile -t mlbench_$1:latest .
docker tag mlbench_$1:latest localhost:5000/mlbench_$1:latest
docker push localhost:5000/mlbench_$1:latest