#!/bin/bash

# Run converter for ONNX
if [ "$SAM_CHOICE" == "ONNX" ]; then
  python3 onnxconverter.py
fi

# Execute the gunicorn command
exec gunicorn --preload --bind :$PORT --workers 1 --threads 8 --timeout 0 _wsgi:app
