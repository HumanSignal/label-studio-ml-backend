#!/bin/bash

DUMMY_DOCKERFILE_CONTENT="FROM alpine\nCOPY . /app"
TEMP_IMAGE_BASE="temp_context_image"

# Loop through all directories containing a .dockerignore file
for dir in ../../label_studio_ml/examples/*/; do
  if [ -f "$dir/.dockerignore" ]; then
    echo "Checking directory: $dir"

    # Navigate into the directory
    pushd "$dir" >/dev/null

    # Create a temporary dummy Dockerfile
    echo -e "$DUMMY_DOCKERFILE_CONTENT" >Dockerfile.tmp

    # Define a unique temporary image name using directory name to avoid conflicts
    TEMP_IMAGE="${TEMP_IMAGE_BASE}_$(basename "$dir")"

    # Build the temporary image and get its ID
    docker build -q -f Dockerfile.tmp -t "$TEMP_IMAGE" . >/dev/null

    # Remove the temporary Dockerfile
    rm -f Dockerfile.tmp

    # List all files excluding the temporary Dockerfile, sorted for comparison
    LOCAL_FILES=$(find . -mindepth 1 -type f -not -name "Dockerfile.tmp" | grep -v '.dockerignore\|README.md' | sort)

    # Use a Docker container to list all files included in the build context, simulating .dockerignore application
    INCLUDED_FILES=$(docker run --rm -w /app "${TEMP_IMAGE}" find . -mindepth 1 -type f | grep -v '.dockerignore\|README.md' | sort)

    # Pop back to the parent directory
    popd >/dev/null

    # Compare the lists to find files not ignored by .dockerignore (included in Docker context)
    echo "Files ignored by .dockerignore:"
    comm -23 <(echo "$LOCAL_FILES") <(echo "$INCLUDED_FILES")
    echo "---------------------------------------------------------------------------------------------------------------------------"
  else
    echo "No .dockerignore found in $dir"
  fi
done
