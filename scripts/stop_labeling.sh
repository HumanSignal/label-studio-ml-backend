#!/bin/bash

# Stop docker compose from local file
docker compose -f ./label-studio.docker-compose.yml down
if [ $? -eq 0 ]; then
  echo "Stopped docker compose from ./label-studio.docker-compose.yml successfully."
else
  echo "Failed to stop docker compose from ./label-studio.docker-compose.yml."
fi

# Stop docker compose from the examples directory
docker compose -f ./label_studio_ml/examples/grounding_sam/docker-compose.yml down
if [ $? -eq 0 ]; then
  echo "Stopped docker compose from ./label_studio_ml/examples/grounding_sam/docker-compose.yml successfully."
else
  echo "Failed to stop docker compose from ./label_studio_ml/examples/grounding_sam/docker-compose.yml."
fi