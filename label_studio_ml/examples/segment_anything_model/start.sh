#!/bin/bash

# Check if the file specified in the ONNX_CHECKPOINT environment variable exists
if [ ! -f "$ONNX_CHECKPOINT" ]; then
  # Run the python onnxconverter.py script if the file is not found
  python onnxconverter.py
else
  # Otherwise, print a message to the console
  echo "ONNX checkpoint found in $ONNX_CHECKPOINT, skipping conversion"
fi

# Execute the gunicorn command
exec gunicorn --preload --bind :$PORT --workers 1 --threads 8 --timeout 0 _wsgi:app
