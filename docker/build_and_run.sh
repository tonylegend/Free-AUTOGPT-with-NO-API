#!/bin/sh

# Change to the project home directory
cd "$(dirname "$0")/.." || exit

# Build the Docker image
#docker build -t autogpt -f docker/Dockerfile .

# Run the Docker container
#docker run -p 8501:8501 autogpt

docker tag autogpt test-docker-reg:5000/autogpt:latest
# docker push 192.168.0.2:5000/autogpt:latest
#docker buildx build -f docker/Dockerfile --platform linux/amd64,linux/arm64 -t 192.168.0.2:5000/autogpt:latest --load .
docker buildx build -f docker/Dockerfile --platform linux/amd64 -t test-docker-reg:5000/autogpt:latest --push .