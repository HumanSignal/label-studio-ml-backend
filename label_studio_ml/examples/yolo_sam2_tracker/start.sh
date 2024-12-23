#!/bin/bash

# Export environment variables
export LABEL_STUDIO_URL=
export LABEL_STUDIO_API_KEY=
export LOG_LEVEL="DEBUG"

# Start the label-studio-ml backend
label-studio-ml start ./../yolo_sam2_tracker
