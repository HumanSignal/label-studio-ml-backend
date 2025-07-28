#!/bin/bash

# Check if docker works without sudo
if ! docker info >/dev/null 2>&1; then
  echo "Docker does not work without sudo. Please configure your user permissions."
  exit 1
else
  echo "Docker works without sudo."
fi

# Check if CUDA can be run through docker
if docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi >/dev/null 2>&1; then
  echo "CUDA is available in Docker."
else
  echo "CUDA is NOT available in Docker. Please check your NVIDIA Docker setup."
fi

# Run docker compose for label_studio
echo "Starting docker compose for label_studio.docker-compose.yml..."
docker compose -f ./label-studio.docker-compose.yml up -d

# Run docker compose for grounding_sam
cd ./label_studio_ml/examples/grounding_sam || { echo "Directory not found: ./label_studio_ml/examples/grounding_sam"; exit 1; }
echo "Starting docker compose for grounding_sam..."
docker compose up -d

echo "Docker compose up signals have been sent for both services."
echo "Monitor containers independently with 'docker ps' and 'docker logs <container>' as needed."