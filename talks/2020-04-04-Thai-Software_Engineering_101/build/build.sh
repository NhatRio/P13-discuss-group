#!/bin/bash
cd ..
docker build -f build/user-service/Dockerfile -t user-service .
docker build -f build/channel-service/Dockerfile -t channel-service .

cd build
docker-compose up -d